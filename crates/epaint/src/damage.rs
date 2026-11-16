//! Finding the part of the screen that changed between two frames, so a backend can repaint just that.

use alloc::{sync::Arc, vec::Vec};

use ahash::HashSet;
use emath::Rect;
use smallvec::SmallVec;

use crate::{ClippedShape, Shape, TextureId, textures::TexturesDelta};

/// How many shapes ahead [`DamageTracker`] looks to re-align two frames after shapes were inserted or removed.
const LOOKAHEAD: usize = 16;

/// Coordinates (in points) beyond this are limited to it: no screen is this large.
const FAR: f32 = 1.0e6;

/// Damaged rectangles closer than this (in points) are merged, painting the gap costs less than another pass.
const MERGE_GAP: f32 = 16.0;

/// What changed on screen between two frames.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Damage {
    /// Nothing visible changed.
    None,

    /// Only pixels within these rectangles may differ.
    Partial(DamageRegion),

    /// Anything may have changed.
    Full,
}

impl Damage {
    /// Everything that changed in either `self` or `other`, e.g. over the frames since a back buffer was last painted.
    #[must_use]
    pub fn union(self, other: Self) -> Self {
        match (self, other) {
            (Self::Full, _) | (_, Self::Full) => Self::Full,
            (Self::None, damage) | (damage, Self::None) => damage,
            (Self::Partial(mut region), Self::Partial(other)) => {
                for rect in other.rects {
                    region.add(rect);
                }
                Self::Partial(region)
            }
        }
    }
}

/// A few non-empty rectangles in points, at most [`Self::MAX_RECTS`].
///
/// Rectangles close to each other are merged into their bounding rectangle, and so are the two that grow the least
/// when there would be too many.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DamageRegion {
    rects: SmallVec<[Rect; Self::MAX_RECTS]>,
}

impl DamageRegion {
    /// A backend paints each rectangle separately, so more of them cost more draw calls than they save.
    pub const MAX_RECTS: usize = 4;

    /// The rectangles, never empty and never overlapping each other.
    pub fn rects(&self) -> &[Rect] {
        &self.rects
    }

    /// The smallest rectangle containing the whole region.
    pub fn bounding_rect(&self) -> Rect {
        self.rects
            .iter()
            .fold(Rect::NOTHING, |union, rect| union | *rect)
    }

    fn from_rects(rects: impl IntoIterator<Item = Rect>) -> Option<Self> {
        let mut region = Self {
            rects: SmallVec::new(),
        };
        for rect in rects {
            region.add(rect);
        }
        (!region.rects.is_empty()).then_some(region)
    }

    fn add(&mut self, rect: Rect) {
        if !rect.is_positive() {
            return;
        }
        let mut rect = rect;
        while let Some(index) = self
            .rects
            .iter()
            .position(|other| other.expand(MERGE_GAP).intersects(rect))
        {
            rect |= self.rects.swap_remove(index);
        }
        if self.rects.len() == Self::MAX_RECTS {
            let cheapest = (0..self.rects.len())
                .min_by(|&a, &b| {
                    growth(self.rects[a], rect).total_cmp(&growth(self.rects[b], rect))
                })
                .unwrap_or_default();
            let merged = self.rects.swap_remove(cheapest) | rect;
            self.add(merged);
        } else {
            self.rects.push(rect);
        }
    }
}

fn growth(a: Rect, b: Rect) -> f32 {
    (a | b).area() - a.area() - b.area()
}

/// Compares the shapes of each frame to the previous one, to find what part of the screen changed.
///
/// A shape matched to an identical one of the previous frame keeps its pixels, as long as the texture it samples
/// was not updated. Matching keeps the paint order, so a pixel outside every unmatched shape is covered by the
/// same shapes in the same order in both frames and needs no repaint.
///
/// Shapes inserted or removed in a few places, like an animated spinner or a list scrolling within a clip rect,
/// cost only their own area.
///
/// [`Shape::Callback`] always counts as changed, since there is no way to tell what it paints.
#[derive(Default)]
pub struct DamageTracker {
    previous: Option<Frame>,

    /// Textures that changed outside the [`TexturesDelta`] of the next [`Self::update`].
    invalidated: HashSet<TextureId>,

    /// The tessellator's feathering in physical pixels ([`Self::set_feathering`]).
    feathering: f32,
}

struct Frame {
    shapes: Vec<ClippedShape>,
    pixels_per_point: f32,
}

impl DamageTracker {
    /// Compares `shapes` to those of the previous call and remembers them for the next one.
    ///
    /// `textures_delta` are the texture changes that will be applied before painting `shapes`.
    /// The first call, and any call with another `pixels_per_point`, returns [`Damage::Full`].
    pub fn update(
        &mut self,
        shapes: &[ClippedShape],
        pixels_per_point: f32,
        textures_delta: &TexturesDelta,
    ) -> Damage {
        profiling::function_scope!();
        let damage = match &self.previous {
            Some(previous) if previous.pixels_per_point == pixels_per_point => {
                let mut changed_textures = Self::changed_textures(textures_delta);
                changed_textures.extend(self.invalidated.drain());
                let areas = changed_shapes(&previous.shapes, shapes, &changed_textures)
                    .into_iter()
                    .map(|clipped| painted_area(clipped, pixels_per_point, self.feathering));
                DamageRegion::from_rects(areas).map_or(Damage::None, Damage::Partial)
            }
            _ => Damage::Full,
        };
        self.invalidated.clear();
        self.previous = Some(Frame {
            shapes: shapes.to_vec(),
            pixels_per_point,
        });
        damage
    }

    /// Forget the previous frame, e.g. when its pixels were lost. The next [`Self::update`] returns [`Damage::Full`].
    pub fn reset(&mut self) {
        // the feathering stays: it describes the tessellator, not the previous frame
        self.previous = None;
        self.invalidated.clear();
    }

    /// Textures whose pixels change with `textures_delta`, ignoring glyphs added to the font atlas: they land in
    /// unused space.
    pub fn changed_textures(textures_delta: &TexturesDelta) -> HashSet<TextureId> {
        textures_delta
            .set
            .iter()
            .filter(|(id, deltas)| {
                **id != TextureId::default() || deltas.iter().any(|delta| delta.pos.is_none())
            })
            .map(|(id, _)| *id)
            .collect()
    }

    /// How wide the tessellator feathers edges, in physical pixels (`TessellationOptions::feathering_size_in_pixels`,
    /// or zero without feathering): an edge is painted that far beyond its shape. Without this call one pixel is
    /// assumed, the tessellator's default.
    pub fn set_feathering(&mut self, pixels: f32) {
        self.feathering = if pixels.is_finite() {
            pixels.max(0.0)
        } else {
            0.0
        };
    }

    /// These textures changed in a [`TexturesDelta`] this tracker is not given, e.g. one applied while another
    /// window was painted: the shapes sampling them count as changed in the next [`Self::update`].
    pub fn invalidate_textures(&mut self, textures: impl IntoIterator<Item = TextureId>) {
        if self.previous.is_some() {
            self.invalidated.extend(textures);
        }
    }
}

/// The shapes of both frames that are not matched to an identical one in the other, or sample a changed texture.
fn changed_shapes<'a>(
    previous: &'a [ClippedShape],
    current: &'a [ClippedShape],
    changed_textures: &HashSet<TextureId>,
) -> Vec<&'a ClippedShape> {
    let mut changed = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < previous.len() && j < current.len() {
        if same(&previous[i], &current[j]) {
            if samples_any(&current[j].shape, changed_textures)
                || samples_user_texture(&current[j].shape)
            {
                changed.push(&current[j]);
            }
            i += 1;
            j += 1;
        } else if let Some(inserted) = (1..=LOOKAHEAD).find(|&n| {
            current
                .get(j + n)
                .is_some_and(|now| same(&previous[i], now))
        }) {
            changed.extend(&current[j..j + inserted]);
            j += inserted;
        } else if let Some(removed) = (1..=LOOKAHEAD).find(|&n| {
            previous
                .get(i + n)
                .is_some_and(|before| same(before, &current[j]))
        }) {
            changed.extend(&previous[i..i + removed]);
            i += removed;
        } else {
            changed.extend([&previous[i], &current[j]]);
            i += 1;
            j += 1;
        }
    }
    changed.extend(&previous[i..]);
    changed.extend(&current[j..]);
    changed
}

fn samples_any(shape: &Shape, textures: &HashSet<TextureId>) -> bool {
    match shape {
        Shape::Vec(shapes) => shapes.iter().any(|shape| samples_any(shape, textures)),
        _ => textures.contains(&shape.texture_id()),
    }
}

/// Whether `shape` samples a texture the backend owns ([`TextureId::User`]): its pixels can change without a
/// [`TexturesDelta`], so the shape always counts as changed.
fn samples_user_texture(shape: &Shape) -> bool {
    match shape {
        Shape::Vec(shapes) => shapes.iter().any(samples_user_texture),
        _ => matches!(shape.texture_id(), TextureId::User(_)),
    }
}

fn same(before: &ClippedShape, now: &ClippedShape) -> bool {
    before.clip_rect == now.clip_rect
        && before.transform_after_tessellation == now.transform_after_tessellation
        && same_shape(&before.shape, &now.shape)
}

/// Like `==`, but first by pointer for text and meshes: comparing their content costs about as much as painting it.
fn same_shape(before: &Shape, now: &Shape) -> bool {
    match (before, now) {
        (Shape::Callback(_), _) | (_, Shape::Callback(_)) => false,
        (Shape::Vec(before), Shape::Vec(now)) => {
            before.len() == now.len() && before.iter().zip(now).all(|(a, b)| same_shape(a, b))
        }
        (Shape::Text(before), Shape::Text(now)) => {
            (Arc::ptr_eq(&before.galley, &now.galley) || before.galley == now.galley)
                && before.pos == now.pos
                && before.underline == now.underline
                && before.fallback_color == now.fallback_color
                && before.override_text_color == now.override_text_color
                && before.opacity_factor == now.opacity_factor
                && before.angle == now.angle
        }
        (Shape::Mesh(before), Shape::Mesh(now)) => Arc::ptr_eq(before, now) || before == now,
        _ => before == now,
    }
}

/// Where `clipped` may touch pixels, in points.
fn painted_area(clipped: &ClippedShape, pixels_per_point: f32, feathering: f32) -> Rect {
    // feathering spreads its width beyond the shape (one pixel by default), pixel snapping and the rounded
    // scissor rect up to half a pixel more
    let feathering = feathering.max(1.0) / pixels_per_point;
    let margin = feathering + 1.0 / pixels_per_point;
    let bounds = shape_bounds(&clipped.shape, feathering);
    if bounds.is_negative() {
        return Rect::NOTHING;
    }
    // The tessellator paints a shape with infinite bounds (`Rect::EVERYTHING`) over all it is clipped to, so such
    // a shape damages that, not nothing. Limited here, so the rectangles below stay finite.
    let far = Rect::from_min_max(emath::pos2(-FAR, -FAR), emath::pos2(FAR, FAR));
    let bounds = bounds.intersect(far);
    let area = (clipped.transform_after_tessellation * bounds.expand(margin))
        .intersect(clipped.clip_rect.expand(margin))
        .intersect(far);
    if area.is_negative() {
        Rect::NOTHING
    } else {
        area
    }
}

/// Where `shape` may touch pixels before feathering, which is `feathering` points wide.
fn shape_bounds(shape: &Shape, feathering: f32) -> Rect {
    let visual = shape.visual_bounding_rect();
    match shape {
        Shape::Vec(shapes) => shapes
            .iter()
            .map(|shape| shape_bounds(shape, feathering))
            .fold(Rect::NOTHING, Rect::union),
        Shape::Text(text) => text
            .galley
            .mesh_bounds
            .union(text.galley.rect)
            // the underline is stroked along the bottom of each row, half its width to either side
            .expand(text.underline.width.max(0.0))
            .rotate_bb(emath::Rot2::from_angle(text.angle))
            .translate(text.pos.to_vec2()),
        _ if visual.is_negative() => Rect::NOTHING,
        // The joins of a stroke, and of a fill's feathered edge, are mitered: a sharp corner reaches beyond
        // the stroke's own width.
        Shape::Path(path) => {
            let reach =
                miter_reach(&path.points, path.closed) * (path.stroke.width.max(0.0) + feathering);
            if reach.is_finite() {
                Rect::from_points(&path.points).expand(reach)
            } else {
                Rect::EVERYTHING
            }
        }
        // Turned into paths only when tessellated (a band into one outline for each run), so where their mitered
        // corners are is not known here.
        Shape::QuadraticBezier(_) | Shape::CubicBezier(_) | Shape::Band(_) => Rect::EVERYTHING,
        // Stroked outside the outline, where the visual bounds count half the stroke.
        Shape::Circle(circle) => visual.expand(circle.stroke.width.max(0.0) * 0.5),
        Shape::Ellipse(ellipse) => visual.expand(ellipse.stroke.width.max(0.0) * 0.5),
        _ => visual,
    }
}

/// How far the stroke of a path may reach from its points, in stroke widths: at least one (the stroke may lie
/// wholly on one side), more at a corner sharper than a right angle, where the tessellator's miter join extends
/// by the stroke's half width over the sine of half the corner's angle. Infinite where the path turns back on
/// itself.
fn miter_reach(points: &[emath::Pos2], closed: bool) -> f32 {
    let n = points.len();
    let mut reach: f32 = 1.0;
    if n < 3 {
        return reach;
    }
    let direction = |from: usize, to: usize| (points[to] - points[from]).normalized();
    let corners = if closed { 0..n } else { 1..n - 1 };
    for i in corners {
        let before = direction((i + n - 1) % n, i);
        let after = direction(i, (i + 1) % n);
        if before == emath::Vec2::ZERO || after == emath::Vec2::ZERO {
            // a repeated point: the tessellator joins it squarely
            continue;
        }
        let half = ((before + after) * 0.5).length();
        reach = reach.max(if half > 0.0 {
            1.0 / half
        } else {
            f32::INFINITY
        });
    }
    reach
}

#[cfg(test)]
mod tests {
    use emath::{Pos2, TSTransform, pos2, vec2};

    use super::*;
    use crate::{
        Color32, ColorImage, FontId, Fonts, ImageDelta, PaintCallback, RectShape, Stroke,
        StrokeKind, TextOptions, text::FontDefinitions, textures::TextureOptions,
    };

    const SCREEN: Rect = Rect::from_min_max(Pos2::ZERO, pos2(1000.0, 1000.0));

    fn rect_at(x: f32, y: f32) -> ClippedShape {
        ClippedShape::new(
            SCREEN,
            RectShape::filled(
                Rect::from_min_size(pos2(x, y), vec2(10.0, 10.0)),
                0.0,
                Color32::RED,
            ),
        )
    }

    fn row(count: usize) -> Vec<ClippedShape> {
        (0..count).map(|i| rect_at(20.0 * i as f32, 0.0)).collect()
    }

    fn region(damage: &Damage) -> &DamageRegion {
        match damage {
            Damage::Partial(region) => region,
            other => panic!("expected a partial damage, got {other:?}"),
        }
    }

    fn tracker_after(shapes: &[ClippedShape]) -> DamageTracker {
        let mut tracker = DamageTracker::default();
        assert_eq!(
            tracker.update(shapes, 1.0, &TexturesDelta::default()),
            Damage::Full,
            "the first frame has nothing to compare to"
        );
        tracker
    }

    fn assert_covers_only(damage: &Damage, changed: &[Rect], untouched: &[Rect]) {
        let rects = region(damage).rects();
        for rect in changed {
            assert!(
                rects.iter().any(|area| area.contains_rect(*rect)),
                "{rects:?} misses changed {rect:?}"
            );
        }
        for rect in untouched {
            assert!(
                !rects.iter().any(|area| area.intersects(*rect)),
                "{rects:?} covers untouched {rect:?}"
            );
        }
    }

    fn bounds(clipped: &ClippedShape) -> Rect {
        clipped.shape.visual_bounding_rect()
    }

    #[test]
    fn distant_changes_are_separate_rects() {
        let before = row(40);
        let mut after = before.clone();
        after[1] = rect_at(20.0, 30.0);
        after[38] = rect_at(760.0, 30.0);
        let mut tracker = tracker_after(&before);
        let damage = tracker.update(&after, 1.0, &TexturesDelta::default());
        assert_eq!(region(&damage).rects().len(), 2, "{damage:?}");
        assert_covers_only(
            &damage,
            &[bounds(&after[1]), bounds(&after[38])],
            &[bounds(&before[20])],
        );
    }

    #[test]
    fn region_merges_close_rects_and_keeps_at_most_max_rects() {
        let square = |x: f32, y: f32| Rect::from_min_size(pos2(x, y), vec2(10.0, 10.0));
        let close = DamageRegion::from_rects([square(0.0, 0.0), square(15.0, 0.0)]).unwrap();
        assert_eq!(
            close.rects(),
            &[Rect::from_min_max(pos2(0.0, 0.0), pos2(25.0, 10.0))]
        );

        let spread: Vec<Rect> = (0..10)
            .map(|i| square(100.0 * i as f32, 100.0 * (i % 3) as f32))
            .collect();
        let region = DamageRegion::from_rects(spread.iter().copied()).unwrap();
        assert!(
            region.rects().len() <= DamageRegion::MAX_RECTS,
            "{region:?}"
        );
        for rect in &spread {
            assert!(
                region.rects().iter().any(|area| area.contains_rect(*rect)),
                "{rect:?} lost"
            );
        }
        for (i, a) in region.rects().iter().enumerate() {
            for b in &region.rects()[i + 1..] {
                assert!(
                    !a.expand(MERGE_GAP).intersects(*b),
                    "{a:?} and {b:?} should have merged"
                );
            }
        }
        assert_eq!(
            DamageRegion::from_rects([Rect::NOTHING, square(0.0, 0.0).shrink(5.0)]),
            None
        );
    }

    #[test]
    fn union_of_damages() {
        let square = |x: f32| {
            Damage::Partial(
                DamageRegion::from_rects([Rect::from_min_size(pos2(x, 0.0), vec2(10.0, 10.0))])
                    .unwrap(),
            )
        };
        assert_eq!(Damage::None.union(square(0.0)), square(0.0));
        assert_eq!(square(0.0).union(Damage::Full), Damage::Full);
        assert_eq!(
            square(0.0).union(square(500.0)).union(Damage::None),
            Damage::Partial(
                DamageRegion::from_rects([
                    Rect::from_min_size(pos2(0.0, 0.0), vec2(10.0, 10.0)),
                    Rect::from_min_size(pos2(500.0, 0.0), vec2(10.0, 10.0)),
                ])
                .unwrap()
            )
        );
    }

    #[test]
    fn identical_frames_change_nothing() {
        let shapes = row(5);
        let mut tracker = tracker_after(&shapes);
        assert_eq!(
            tracker.update(&shapes, 1.0, &TexturesDelta::default()),
            Damage::None,
            "the same shapes paint the same pixels"
        );
    }

    #[test]
    fn a_moved_shape_damages_where_it_was_and_where_it_is() {
        let before = row(5);
        let mut after = before.clone();
        after[2] = rect_at(40.0, 100.0);
        let mut tracker = tracker_after(&before);
        let damage = tracker.update(&after, 1.0, &TexturesDelta::default());
        assert_covers_only(
            &damage,
            &[bounds(&before[2]), bounds(&after[2])],
            &[bounds(&before[0]), bounds(&before[4])],
        );
    }

    #[test]
    fn inserted_and_removed_shapes_do_not_damage_the_rest() {
        let before = row(10);
        let mut after = before.clone();
        let inserted = rect_at(500.0, 500.0);
        after.insert(3, inserted.clone());
        after.remove(8);
        let mut tracker = tracker_after(&before);
        let damage = tracker.update(&after, 1.0, &TexturesDelta::default());
        let untouched: Vec<Rect> = [0, 1, 2, 3, 6, 9]
            .iter()
            .map(|&i| bounds(&before[i]))
            .collect();
        assert_covers_only(
            &damage,
            &[bounds(&inserted), bounds(&before[7])],
            &untouched,
        );
    }

    #[test]
    fn rows_scrolling_within_a_clip_rect_damage_only_the_clip_rect() {
        let clip = Rect::from_min_size(pos2(300.0, 300.0), vec2(50.0, 60.0));
        let rows = |offset: f32| -> Vec<ClippedShape> {
            (0..20)
                .map(|i| pos2(300.0, 300.0 + 20.0 * i as f32 - offset))
                .filter(|pos| pos.y + 20.0 > clip.top() && pos.y < clip.bottom())
                .map(|pos| {
                    ClippedShape::new(
                        clip,
                        RectShape::filled(
                            Rect::from_min_size(pos, vec2(50.0, 18.0)),
                            0.0,
                            Color32::GRAY,
                        ),
                    )
                })
                .collect()
        };
        let mut before = row(10);
        before.extend(rows(0.0));
        before.extend(row(10));
        let mut after = row(10);
        after.extend(rows(13.0));
        after.extend(row(10));
        assert_ne!(before.len(), after.len(), "a row scrolled in");
        let mut tracker = tracker_after(&before);
        let damage = tracker.update(&after, 1.0, &TexturesDelta::default());
        let margin = 2.0;
        assert!(
            clip.expand(margin)
                .contains_rect(region(&damage).bounding_rect()),
            "{damage:?} reaches outside the clip rect {clip:?}"
        );
    }

    #[test]
    fn a_reordered_pair_damages_only_the_pair() {
        let mut before = row(6);
        before[2] = rect_at(200.0, 200.0);
        before[3] = rect_at(205.0, 205.0);
        let mut after = before.clone();
        after.swap(2, 3);
        let mut tracker = tracker_after(&before);
        let damage = tracker.update(&after, 1.0, &TexturesDelta::default());
        let overlap = bounds(&before[2]).intersect(bounds(&before[3]));
        assert_covers_only(
            &damage,
            &[overlap],
            &[bounds(&before[0]), bounds(&before[5])],
        );
    }

    #[test]
    fn a_shape_outside_its_clip_rect_damages_nothing() {
        let before = row(3);
        let mut after = before.clone();
        after.push(ClippedShape::new(
            Rect::from_min_size(pos2(0.0, 500.0), vec2(10.0, 10.0)),
            RectShape::filled(
                Rect::from_min_size(pos2(900.0, 900.0), vec2(10.0, 10.0)),
                0.0,
                Color32::RED,
            ),
        ));
        let mut tracker = tracker_after(&before);
        assert_eq!(
            tracker.update(&after, 1.0, &TexturesDelta::default()),
            Damage::None
        );
    }

    #[test]
    fn transform_after_tessellation_moves_the_damage() {
        let before = row(2);
        let mut after = before.clone();
        let mut moved = rect_at(0.0, 0.0);
        moved.transform_after_tessellation = TSTransform::from_translation(vec2(400.0, 400.0));
        after.push(moved);
        let mut tracker = tracker_after(&before);
        let damage = tracker.update(&after, 1.0, &TexturesDelta::default());
        assert_covers_only(
            &damage,
            &[Rect::from_min_size(pos2(400.0, 400.0), vec2(10.0, 10.0))],
            &[bounds(&before[0])],
        );
    }

    #[test]
    fn a_shape_with_an_updated_texture_is_damaged() {
        let texture = TextureId::User(7);
        let image = ClippedShape::new(
            SCREEN,
            RectShape::filled(
                Rect::from_min_size(pos2(300.0, 300.0), vec2(10.0, 10.0)),
                0.0,
                Color32::WHITE,
            )
            .with_texture(texture, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0))),
        );
        let mut shapes = row(3);
        shapes.push(image.clone());
        let mut tracker = tracker_after(&shapes);

        let mut delta = TexturesDelta::default();
        delta.set.insert(
            texture,
            [ImageDelta::partial(
                [0, 0],
                ColorImage::filled([1, 1], Color32::BLUE),
                TextureOptions::LINEAR,
            )]
            .into(),
        );
        let damage = tracker.update(&shapes, 1.0, &delta);
        assert_covers_only(&damage, &[bounds(&image)], &[bounds(&shapes[0])]);
        delta.clear();
    }

    #[test]
    fn new_glyphs_in_the_font_atlas_damage_nothing_but_a_new_atlas_damages_everything() {
        let shapes = row(3);
        let mut tracker = tracker_after(&shapes);
        let glyph = ImageDelta::partial(
            [10, 10],
            ColorImage::filled([2, 2], Color32::WHITE),
            TextureOptions::LINEAR,
        );
        let mut delta = TexturesDelta::default();
        delta.set.insert(TextureId::default(), [glyph].into());
        assert_eq!(tracker.update(&shapes, 1.0, &delta), Damage::None);

        let atlas = ImageDelta::full(
            ColorImage::filled([8, 8], Color32::WHITE),
            TextureOptions::LINEAR,
        );
        delta.set.insert(TextureId::default(), [atlas].into());
        assert_covers_only(
            &tracker.update(&shapes, 1.0, &delta),
            &[bounds(&shapes[0]), bounds(&shapes[2])],
            &[],
        );
        delta.clear();
    }

    #[test]
    fn paint_callbacks_are_always_damaged() {
        let callback = ClippedShape::new(
            SCREEN,
            PaintCallback {
                rect: Rect::from_min_size(pos2(600.0, 600.0), vec2(50.0, 50.0)),
                callback: Arc::new(()),
            },
        );
        let mut shapes = row(3);
        shapes.push(callback);
        shapes.extend(row(3));
        let mut tracker = tracker_after(&shapes);
        let damage = tracker.update(&shapes, 1.0, &TexturesDelta::default());
        assert_covers_only(
            &damage,
            &[Rect::from_min_size(pos2(600.0, 600.0), vec2(50.0, 50.0))],
            &[bounds(&shapes[0])],
        );
    }

    #[test]
    fn changed_text_damages_its_galley() {
        let mut fonts = Fonts::new(TextOptions::default(), FontDefinitions::default());
        let mut text = |content: &str| {
            ClippedShape::new(
                SCREEN,
                Shape::text(
                    &mut fonts.with_pixels_per_point(1.0),
                    pos2(100.0, 400.0),
                    emath::Align2::LEFT_TOP,
                    content,
                    FontId::proportional(14.0),
                    Color32::BLACK,
                ),
            )
        };
        let (short, long) = (text("1"), text("1234567"));
        let mut tracker = tracker_after(core::slice::from_ref(&short));
        let damage = tracker.update(core::slice::from_ref(&long), 1.0, &TexturesDelta::default());
        assert_covers_only(&damage, &[bounds(&short), bounds(&long)], &[]);
        assert_eq!(
            tracker.update(&[text("1234567")], 1.0, &TexturesDelta::default()),
            Damage::None,
            "an equal galley laid out again is not a change"
        );
    }

    #[test]
    fn a_sharp_corner_of_a_stroke_is_inside_the_damage() {
        // A 20 point stroke around a thin triangle: the tips reach far beyond the points.
        let points = vec![pos2(200.0, 500.0), pos2(300.0, 500.0), pos2(250.0, 100.0)];
        let triangle = ClippedShape::new(
            SCREEN,
            Shape::closed_line(points, Stroke::new(20.0, Color32::BLACK)),
        );
        let mut tracker = tracker_after(&[]);
        let damage = tracker.update(&[triangle], 1.0, &TexturesDelta::default());
        let tip = Rect::from_min_max(pos2(240.0, 30.0), pos2(260.0, 100.0));
        assert_covers_only(&damage, &[tip], &[]);
    }

    #[test]
    fn a_band_damages_what_it_is_clipped_to() {
        // A band narrowing to a point: its stroke's miter reaches far beyond the samples.
        let band = crate::BandShape::new(
            vec![
                crate::BandPoint::new(100.0, 90.0..=110.0),
                crate::BandPoint::new(150.0, 99.0..=101.0),
            ],
            Color32::RED,
            Stroke::new(20.0, Color32::BLACK),
        );
        let mut tracker = tracker_after(&[]);
        let damage = tracker.update(
            &[ClippedShape::new(SCREEN, band)],
            1.0,
            &TexturesDelta::default(),
        );
        assert_covers_only(&damage, &[SCREEN], &[]);
    }

    #[test]
    fn the_stroke_of_a_circle_is_inside_the_damage() {
        let circle = ClippedShape::new(
            SCREEN,
            Shape::circle_stroke(pos2(100.0, 100.0), 20.0, Stroke::new(20.0, Color32::BLACK)),
        );
        let mut tracker = tracker_after(&[]);
        let damage = tracker.update(&[circle], 1.0, &TexturesDelta::default());
        let stroke = Rect::from_min_max(pos2(59.5, 59.5), pos2(140.5, 140.5));
        assert_covers_only(&damage, &[stroke], &[]);
    }

    #[test]
    fn a_sharp_corner_of_a_fill_is_inside_the_damage() {
        // Filled, not stroked: the feathered edge is mitered at the thin corner all the same.
        let points = vec![pos2(50.0, 50.0), pos2(150.0, 50.0), pos2(50.0, 55.0)];
        let triangle = ClippedShape::new(
            SCREEN,
            Shape::convex_polygon(points, Color32::RED, Stroke::NONE),
        );
        let mut tracker = tracker_after(&[]);
        let damage = tracker.update(&[triangle], 1.0, &TexturesDelta::default());
        let tip = Rect::from_min_max(pos2(150.0, 49.0), pos2(170.0, 52.0));
        assert_covers_only(&damage, &[tip], &[]);
    }

    #[test]
    fn a_wider_feathering_is_inside_the_damage() {
        let shapes = [rect_at(100.0, 100.0)];
        let mut tracker = tracker_after(&[]);
        tracker.set_feathering(16.0);
        let damage = tracker.update(&shapes, 1.0, &TexturesDelta::default());
        let feathered = Rect::from_min_size(pos2(100.0, 100.0), vec2(10.0, 10.0)).expand(8.0);
        assert_covers_only(&damage, &[feathered], &[]);
    }

    #[test]
    fn a_texture_changed_for_another_window_damages_its_shapes() {
        let id = TextureId::Managed(7);
        let mut mesh = crate::Mesh::with_texture(id);
        let at = Rect::from_min_size(pos2(40.0, 40.0), vec2(20.0, 20.0));
        mesh.add_rect_with_uv(
            at,
            Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
            Color32::WHITE,
        );
        let shapes = [
            rect_at(0.0, 0.0),
            ClippedShape::new(SCREEN, Shape::mesh(mesh)),
        ];
        let mut tracker = tracker_after(&shapes);
        tracker.invalidate_textures([id]);
        let damage = tracker.update(&shapes, 1.0, &TexturesDelta::default());
        assert_covers_only(
            &damage,
            &[at],
            &[Rect::from_min_size(pos2(0.0, 0.0), vec2(10.0, 10.0))],
        );
        assert_eq!(
            tracker.update(&shapes, 1.0, &TexturesDelta::default()),
            Damage::None,
            "an invalidation counts once"
        );
    }

    #[test]
    fn a_shape_with_infinite_bounds_damages_what_it_is_clipped_to() {
        let everything =
            |color| ClippedShape::new(SCREEN, RectShape::filled(Rect::EVERYTHING, 0.0, color));
        let mut tracker = tracker_after(&[everything(Color32::RED)]);
        let damage = tracker.update(&[everything(Color32::BLUE)], 1.0, &TexturesDelta::default());
        assert_covers_only(&damage, &[SCREEN], &[]);
        assert!(region(&damage).bounding_rect().is_finite());
    }

    #[test]
    fn a_shape_with_a_backend_texture_is_always_damaged() {
        let mut mesh = crate::Mesh::with_texture(TextureId::User(7));
        let at = Rect::from_min_size(pos2(40.0, 40.0), vec2(20.0, 20.0));
        mesh.add_rect_with_uv(
            at,
            Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
            Color32::WHITE,
        );
        let shapes = [
            rect_at(0.0, 0.0),
            ClippedShape::new(SCREEN, Shape::mesh(mesh)),
        ];
        let mut tracker = tracker_after(&shapes);
        let damage = tracker.update(&shapes, 1.0, &TexturesDelta::default());
        assert_covers_only(
            &damage,
            &[at],
            &[Rect::from_min_size(pos2(0.0, 0.0), vec2(10.0, 10.0))],
        );
    }

    #[test]
    fn a_thick_underline_is_inside_the_damage() {
        let mut fonts = Fonts::new(TextOptions::default(), FontDefinitions::default());
        let galley = fonts.with_pixels_per_point(1.0).layout_no_wrap(
            "...".to_owned(),
            FontId::proportional(14.0),
            Color32::BLACK,
        );
        let underlined = ClippedShape::new(
            SCREEN,
            crate::TextShape::new(pos2(100.0, 100.0), Arc::clone(&galley), Color32::BLACK)
                .with_underline(Stroke::new(12.0, Color32::BLACK)),
        );
        let mut tracker = tracker_after(&[]);
        let damage = tracker.update(&[underlined], 1.0, &TexturesDelta::default());
        let underline = Rect::from_min_size(
            pos2(100.0, 100.0) + vec2(0.0, galley.rect.bottom() - 6.0),
            vec2(galley.rect.width(), 12.0),
        );
        assert_covers_only(&damage, &[underline], &[]);
    }

    #[test]
    fn underline_is_inside_the_damage() {
        let mut fonts = Fonts::new(TextOptions::default(), FontDefinitions::default());
        let galley = fonts.with_pixels_per_point(1.0).layout_no_wrap(
            "...".to_owned(),
            FontId::proportional(14.0),
            Color32::BLACK,
        );
        let underlined = ClippedShape::new(
            SCREEN,
            crate::TextShape::new(pos2(100.0, 100.0), Arc::clone(&galley), Color32::BLACK)
                .with_underline(Stroke::new(1.0, Color32::BLACK)),
        );
        let mut tracker = tracker_after(&[]);
        let damage = tracker.update(&[underlined], 1.0, &TexturesDelta::default());
        let underline = Rect::from_min_size(
            pos2(100.0, 100.0) + vec2(0.0, galley.rect.bottom() - 1.0),
            vec2(galley.rect.width(), 1.0),
        );
        assert_covers_only(&damage, &[underline], &[]);
    }

    #[test]
    fn another_pixels_per_point_damages_everything() {
        let shapes = row(3);
        let mut tracker = tracker_after(&shapes);
        assert_eq!(
            tracker.update(&shapes, 2.0, &TexturesDelta::default()),
            Damage::Full
        );
        assert_eq!(
            tracker.update(&shapes, 2.0, &TexturesDelta::default()),
            Damage::None
        );
    }

    #[test]
    fn reset_damages_everything() {
        let shapes = row(3);
        let mut tracker = tracker_after(&shapes);
        tracker.reset();
        assert_eq!(
            tracker.update(&shapes, 1.0, &TexturesDelta::default()),
            Damage::Full
        );
    }

    #[test]
    fn stroke_kind_outside_is_inside_the_damage() {
        let outlined = ClippedShape::new(
            SCREEN,
            RectShape::stroke(
                Rect::from_min_size(pos2(100.0, 100.0), vec2(10.0, 10.0)),
                0.0,
                Stroke::new(4.0, Color32::BLACK),
                StrokeKind::Outside,
            ),
        );
        let mut tracker = tracker_after(&[]);
        let damage = tracker.update(&[outlined], 1.0, &TexturesDelta::default());
        assert_covers_only(
            &damage,
            &[Rect::from_min_size(pos2(96.0, 96.0), vec2(18.0, 18.0))],
            &[],
        );
    }

    #[test]
    fn many_unrelated_changes_still_find_unchanged_shapes() {
        let before = row(40);
        let after: Vec<ClippedShape> = before
            .iter()
            .enumerate()
            .map(|(i, shape)| {
                if i % 4 == 0 {
                    rect_at(20.0 * i as f32, 50.0)
                } else {
                    shape.clone()
                }
            })
            .collect();
        let changed = changed_shapes(&before, &after, &HashSet::default());
        assert_eq!(
            changed.len(),
            20,
            "only the 10 moved shapes, before and after"
        );
        let mut tracker = tracker_after(&before);
        tracker.update(&after, 1.0, &TexturesDelta::default());
        assert_eq!(
            tracker.update(&after, 1.0, &TexturesDelta::default()),
            Damage::None,
            "the remembered frame is the latest"
        );
    }
}
