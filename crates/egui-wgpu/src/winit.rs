#![expect(clippy::missing_errors_doc)]
#![expect(clippy::undocumented_unsafe_blocks)]
#![expect(clippy::unwrap_used)] // TODO(emilk): avoid unwraps
#![expect(unsafe_code)]

use crate::{RenderState, SurfaceConfig, SurfaceErrorAction, WgpuConfiguration, renderer};
use crate::{RendererOptions, capture::CaptureState};
use core::num::NonZeroU32;
use egui::{Context, ViewportId, ViewportIdMap, ViewportIdSet};
use std::sync::Arc;

struct SurfaceState {
    surface: wgpu::Surface<'static>,
    alpha_mode: wgpu::CompositeAlphaMode,
    width: u32,
    height: u32,
    // `width` and `height` are the size the surface is configured at,
    // which `surface_fit` keeps within the device's `max_texture_dimension_2d`; these are the
    // window's own size, and `render_scale` is how much smaller the surface is drawn.
    window_width: u32,
    window_height: u32,
    render_scale: f32,
    resizing: bool,
    needs_reconfigure: bool,
    needs_recreate: bool,
    /// Partial repaint applies to this surface ([`crate::PartialRepaint`]): the adapter fits the
    /// option, there is no multisampling, and the surface can be copied to (it is configured with
    /// [`wgpu::TextureUsages::COPY_DST`]).
    partial_repaint: bool,
    /// The last frame and its shapes, while repainting only what changed.
    retained: Option<RetainedFrame>,
    /// Whether the log already says why partial repaint can't apply to this surface.
    fallback_logged: bool,
}

/// The last frame a surface showed, kept for [`crate::PartialRepaint`]: the surface's own
/// textures are not kept between frames.
struct RetainedFrame {
    tracker: epaint::DamageTracker,
    texture: wgpu::Texture,
    stats: RepaintStats,
    /// How the frame was tessellated: other options paint the same shapes differently.
    tessellation: Option<epaint::TessellationOptions>,
}

/// Why a frame was repainted whole.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FullRepaint {
    /// Nothing to compare to: the first frame, a new size or scale, or after a frame that
    /// could not be shown (a hidden window).
    NothingToCompare,
    /// The surface was configured anew.
    Surface,
    /// The tessellation options changed.
    Options,
    /// A paint callback or a clip debug option, which paint beyond what is tracked.
    Unbounded,
}

/// How partial repaint fares, logged at debug level (target `egui_wgpu::partial_repaint`)
/// every few seconds, so a report can show whether frames are skipped or repainted whole,
/// why, and how long painting them took.
#[derive(Default)]
struct RepaintStats {
    since: Option<web_time::Instant>,
    skipped: u32,
    partial: u32,
    full: u32,
    /// The share of the window the partial repaints covered, summed.
    partial_share: f64,
    /// The whole repaints by [`FullRepaint`], in its order.
    reasons: [u32; 4],
    /// Seconds spent tessellating and painting the partial and the whole repaints, summed, and
    /// the longest of either. Without the wait for the surface and its presentation, which
    /// with vsync is mostly a wait for the display.
    partial_seconds: f64,
    full_seconds: f64,
    longest_seconds: f64,
    /// That wait, summed over the painted frames. A software rasterizer may do its work in it.
    waiting_seconds: f64,
    /// The stages of the painted frames, summed.
    cost: FrameCost,
    /// The frames that got a record of their own.
    slow: u32,
}

impl RepaintStats {
    fn record(
        &mut self,
        damage: &epaint::Damage,
        reason: Option<FullRepaint>,
        screen: egui::Rect,
        pixels: [u32; 2],
    ) {
        let now = web_time::Instant::now();
        let since = *self.since.get_or_insert(now);
        if now.duration_since(since).as_secs_f32() >= 5.0 {
            let per = |sum: f64, count: u32| {
                if count > 0 {
                    sum / f64::from(count)
                } else {
                    0.0
                }
            };
            let [nothing, surface, options, unbounded] = self.reasons;
            log::debug!(
                target: "egui_wgpu::partial_repaint",
                "last {:.1} s at {}x{} px: {} frames skipped, {} partial (on average {:.0}% of the window, {:.1} ms), {} full ({:.1} ms; {nothing} with nothing to compare to, {surface} for the surface, {options} for options, {unbounded} for callbacks); longest {:.1} ms; waiting for the surface {:.1} ms a painted frame (acquire {:.1}, submit {:.1}, present {:.1} ms); uploads {:.1} ms and {:.0} kB, buffers {:.1} ms a painted frame",
                now.duration_since(since).as_secs_f32(),
                pixels[0],
                pixels[1],
                self.skipped,
                self.partial,
                100.0 * per(self.partial_share, self.partial),
                1000.0 * per(self.partial_seconds, self.partial),
                self.full,
                1000.0 * per(self.full_seconds, self.full),
                1000.0 * self.longest_seconds,
                1000.0 * per(self.waiting_seconds, self.partial + self.full),
                1000.0 * per(self.cost.acquire_seconds, self.partial + self.full),
                1000.0 * per(self.cost.submit_seconds, self.partial + self.full),
                1000.0 * per(self.cost.present_seconds, self.partial + self.full),
                1000.0 * per(self.cost.upload_seconds, self.partial + self.full),
                per(self.cost.upload_bytes as f64, self.partial + self.full) / 1000.0,
                1000.0 * per(self.cost.buffers_seconds, self.partial + self.full),
            );
            *self = Self {
                since: Some(now),
                ..Self::default()
            };
        }
        match damage {
            epaint::Damage::None => self.skipped += 1,
            epaint::Damage::Partial(region) => {
                self.partial += 1;
                let area: f32 = region
                    .rects()
                    .iter()
                    .map(|r| r.intersect(screen).area().max(0.0))
                    .sum();
                self.partial_share += f64::from(area / screen.area().max(1.0));
            }
            epaint::Damage::Full => {
                self.full += 1;
                if let Some(count) = reason.and_then(|reason| self.reasons.get_mut(reason as usize))
                {
                    *count += 1;
                }
            }
        }
    }

    /// The frame last recorded took `seconds` to tessellate and paint, and `waiting` more
    /// for the surface; `cost` is that by stage. A frame that took long gets a record of
    /// its own, with what it filled (`fill`, computed only for such a frame) and the share
    /// of the window it repainted.
    fn painted(
        &mut self,
        whole: bool,
        seconds: f64,
        waiting: f64,
        cost: &FrameCost,
        share: f32,
        fill: impl FnOnce() -> FrameFill,
    ) {
        self.cost.add(cost);
        if seconds + waiting >= SLOW_FRAME_SECONDS && self.slow < SLOW_FRAMES_LOGGED {
            self.slow += 1;
            let fill = fill();
            log::debug!(
                target: "egui_wgpu::partial_repaint",
                "slow frame: {:.0}% of the window; {} meshes, {} vertices, {} textures, filling {:.2} MP ({:.2} MP of images); tessellating and encoding {:.1} ms; uploads {:.1} ms and {:.0} kB; buffers {:.1} ms; acquire {:.1} ms; submit {:.1} ms; present {:.1} ms",
                100.0 * share,
                fill.meshes,
                fill.vertices,
                fill.textures,
                fill.pixels / 1.0e6,
                fill.image_pixels / 1.0e6,
                1000.0 * seconds,
                1000.0 * cost.upload_seconds,
                cost.upload_bytes as f64 / 1000.0,
                1000.0 * cost.buffers_seconds,
                1000.0 * cost.acquire_seconds,
                1000.0 * cost.submit_seconds,
                1000.0 * cost.present_seconds,
            );
        }
        self.waiting_seconds += waiting;
        if whole {
            self.full_seconds += seconds;
        } else {
            self.partial_seconds += seconds;
        }
        self.longest_seconds = self.longest_seconds.max(seconds);
    }
}

/// What painting one frame cost, by stage, for the statistics.
///
/// A software rasterizer does the drawing itself in one of the last three stages, whichever
/// its driver waits in, so those say what the frame's pixels cost.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct FrameCost {
    /// Uploading new and changed textures, and how many bytes of them.
    upload_seconds: f64,
    upload_bytes: u64,
    /// Uploading the vertices and indices.
    buffers_seconds: f64,
    /// `Surface::get_current_texture`.
    acquire_seconds: f64,
    /// `Queue::submit`.
    submit_seconds: f64,
    /// `Queue::present`.
    present_seconds: f64,
}

impl FrameCost {
    fn add(&mut self, other: &Self) {
        self.upload_seconds += other.upload_seconds;
        self.upload_bytes = self.upload_bytes.saturating_add(other.upload_bytes);
        self.buffers_seconds += other.buffers_seconds;
        self.acquire_seconds += other.acquire_seconds;
        self.submit_seconds += other.submit_seconds;
        self.present_seconds += other.present_seconds;
    }
}

/// How much a frame's meshes fill, for the statistics: an estimate of the pixels the
/// rasterizer shades, which with overlapping meshes is more than the pixels painted.
/// An estimate: a mesh counts for its triangles' area, at most the part of its bounds that
/// is repainted.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct FrameFill {
    meshes: u32,
    vertices: u64,
    /// Pixels filled by all meshes, and by those that draw a texture of the app's (an
    /// image) rather than egui's own (text and plain shapes).
    pixels: f64,
    image_pixels: f64,
    /// How many different textures the meshes draw.
    textures: u32,
}

impl FrameFill {
    /// The fill of `primitives` within the rectangles of `damage` (all of `screen` for
    /// `None`, a whole repaint). A mesh fills its triangles' area, at most what its clip
    /// rectangle shows.
    fn of(
        primitives: &[epaint::ClippedPrimitive],
        damage: Option<&[egui::Rect]>,
        screen: egui::Rect,
        pixels_per_point: f32,
    ) -> Self {
        let mut fill = Self::default();
        let mut textures: Vec<epaint::TextureId> = Vec::new();
        for epaint::ClippedPrimitive {
            clip_rect,
            primitive,
        } in primitives
        {
            let epaint::Primitive::Mesh(mesh) = primitive else {
                continue;
            };
            // What the mesh can cover at all: its vertices' bounds within its clip.
            let mut bounds = egui::Rect::NOTHING;
            for vertex in &mesh.vertices {
                if vertex.pos.x.is_finite() && vertex.pos.y.is_finite() {
                    bounds.extend_with(vertex.pos);
                }
            }
            let clip = clip_rect.intersect(screen).intersect(bounds);
            let shown: f32 = match damage {
                None => clip.area().max(0.0),
                Some(rects) => rects
                    .iter()
                    .map(|r| {
                        let r = r.intersect(clip);
                        if r.is_positive() { r.area() } else { 0.0 }
                    })
                    .sum(),
            };
            if !clip.is_positive() || shown <= 0.0 {
                continue;
            }
            let mut area = 0.0_f64;
            for triangle in mesh.indices.chunks_exact(3) {
                let corner = |i: usize| {
                    triangle
                        .get(i)
                        .and_then(|index| mesh.vertices.get(*index as usize))
                        .map(|v| v.pos)
                };
                if let (Some(a), Some(b), Some(c)) = (corner(0), corner(1), corner(2)) {
                    let doubled = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
                    if doubled.is_finite() {
                        area += f64::from(doubled.abs()) / 2.0;
                    }
                }
            }
            let pixels =
                area.min(f64::from(shown)) * f64::from(pixels_per_point * pixels_per_point);
            fill.meshes += 1;
            fill.vertices = fill.vertices.saturating_add(mesh.vertices.len() as u64);
            fill.pixels += pixels;
            if mesh.texture_id != epaint::TextureId::default() {
                fill.image_pixels += pixels;
            }
            if !textures.contains(&mesh.texture_id) {
                textures.push(mesh.texture_id);
            }
        }
        fill.textures = textures.len() as u32;
        fill
    }
}

/// A frame whose stages took this long together gets a record of its own.
const SLOW_FRAME_SECONDS: f64 = 0.05;

/// How many such records one interval of the statistics may write.
const SLOW_FRAMES_LOGGED: u32 = 12;

/// Whether the statistics are being logged, so what only they need is worth computing.
fn stats_logged() -> bool {
    log::log_enabled!(target: "egui_wgpu::partial_repaint", log::Level::Debug)
}

/// Whether `shape` is, or holds, a paint callback.
fn has_callback(shape: &epaint::Shape) -> bool {
    match shape {
        epaint::Shape::Vec(shapes) => shapes.iter().any(has_callback),
        epaint::Shape::Callback(_) => true,
        _ => false,
    }
}

impl RetainedFrame {
    fn new(device: &wgpu::Device, format: wgpu::TextureFormat, width: u32, height: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("egui_retained_frame"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        Self {
            tracker: epaint::DamageTracker::default(),
            texture,
            stats: RepaintStats::default(),
            tessellation: None,
        }
    }

    fn fits(&self, format: wgpu::TextureFormat, width: u32, height: u32) -> bool {
        let size = self.texture.size();
        self.texture.format() == format && size.width == width && size.height == height
    }
}

/// The size to configure a window's surface at, and the factor egui
/// draws it at, so that neither side exceeds `max_side` (the device's
/// `max_texture_dimension_2d`).
///
/// `Surface::configure` rejects a larger surface, and wgpu reports that as an uncaptured
/// validation error, which panics: a window bigger than the GPU's limit (restored at another
/// monitor's scale, stretched across monitors, …) closed the app before it opened. Both sides
/// shrink by the same factor and egui draws at that factor, so the whole window is drawn into
/// the surface. Where the swapchain is stretched over the window (DX12 uses
/// `DXGI_SCALING_STRETCH`) it fills the window and lines up with the pointer, only softer;
/// wgpu's GL backend copies it unscaled into a corner instead. Degraded either way, but not a
/// crash. A window within the limit is returned unchanged, with a factor of 1. The longer side
/// becomes exactly `max_side`, and a side that isn't zero never becomes zero (an empty surface
/// can't be configured either).
pub fn surface_fit(width: u32, height: u32, max_side: u32) -> (u32, u32, f32) {
    let max_side = max_side.max(1);
    if width <= max_side && height <= max_side {
        return (width, height, 1.0);
    }
    let longest = width.max(height);
    let scale = (f64::from(max_side) / f64::from(longest)).min(1.0);
    // `as` saturates and `scale` is finite and in (0, 1]: no overflow, NaN or panic. The longer
    // side is set rather than computed, so floating-point rounding can't leave it a pixel short.
    let fit = |side: u32| {
        if side == 0 {
            0
        } else if side == longest {
            max_side
        } else {
            ((f64::from(side) * scale).floor() as u32).clamp(1, max_side)
        }
    };
    (fit(width), fit(height), scale as f32)
}

/// Runs `f` and returns an error it raises on `device`, instead of
/// handing it to wgpu's uncaptured-error handler, which panics by default.
///
/// eframe configures a window's surface before the app is created, so no app code can install
/// a handler in time, and a surface the device can't configure closed the app before its window
/// appeared: wgpu's GL backend on a 2015 Intel driver timed out waiting for the GPU to go idle
/// (`ConfigureSurfaceError::GpuWaitTimeout`). Returned as an error instead, eframe returns
/// `Error::Wgpu` and the app can fall back (for example to OpenGL).
///
/// Validation, out-of-memory and internal errors are caught, one scope each; each scope keeps
/// the first error of its kind, and a validation error is returned before an out-of-memory one,
/// which is returned before an internal one. Not caught: a lost device (wgpu reports that only
/// to the device-lost callback, so `f` seems to succeed), a panic inside `f`, and errors raised
/// on other threads (wgpu's error scopes belong to the calling thread).
pub async fn catch_errors<T>(
    device: &wgpu::Device,
    f: impl FnOnce() -> T,
) -> Result<T, wgpu::Error> {
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let out_of_memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    // Should `f` panic, the guards pop their scopes as they drop, innermost first.
    let value = f();
    // Popped innermost first, as wgpu requires; each pop takes effect at once.
    let internal = internal.pop();
    let out_of_memory = out_of_memory.pop();
    let validation = validation.pop();
    let (validation, out_of_memory, internal) =
        (validation.await, out_of_memory.await, internal.await);
    match validation.or(out_of_memory).or(internal) {
        Some(error) => Err(error),
        None => Ok(value),
    }
}

/// Everything you need to paint egui with [`wgpu`] on [`winit`].
///
/// Alternatively you can use [`crate::Renderer`] directly.
///
/// NOTE: all egui viewports share the same painter.
pub struct Painter {
    context: Context,
    config: WgpuConfiguration,
    options: RendererOptions,
    support_transparent_backbuffer: bool,
    screen_capture_state: Option<CaptureState>,

    instance: wgpu::Instance,
    render_state: Option<RenderState>,

    // Per viewport/window:
    depth_texture_view: ViewportIdMap<wgpu::TextureView>,
    msaa_texture_view: ViewportIdMap<wgpu::TextureView>,
    surfaces: ViewportIdMap<SurfaceState>,

    /// What the frame painted last cost, for the partial-repaint statistics.
    frame_cost: FrameCost,
}

impl Painter {
    /// Manages [`wgpu`] state, including surface state, required to render egui.
    ///
    /// Only the [`wgpu::Instance`] is initialized here. Device selection and the initialization
    /// of render + surface state is deferred until the painter is given its first window target
    /// via [`set_window()`](Self::set_window). (Ensuring that a device that's compatible with the
    /// native window is chosen)
    ///
    /// Before calling [`paint_and_update_textures()`](Self::paint_and_update_textures) a
    /// [`wgpu::Surface`] must be initialized (and corresponding render state) by calling
    /// [`set_window()`](Self::set_window) once you have
    /// a [`winit::window::Window`] with a valid `.raw_window_handle()`
    /// associated.
    pub async fn new(
        context: Context,
        config: WgpuConfiguration,
        support_transparent_backbuffer: bool,
        options: RendererOptions,
    ) -> Self {
        let instance = config.wgpu_setup.new_instance().await;

        Self {
            context,
            config,
            options,
            support_transparent_backbuffer,
            screen_capture_state: None,
            frame_cost: FrameCost::default(),

            instance,
            render_state: None,

            depth_texture_view: Default::default(),
            surfaces: Default::default(),
            msaa_texture_view: Default::default(),
        }
    }

    /// Get the [`RenderState`].
    ///
    /// Will return [`None`] if the render state has not been initialized yet.
    pub fn render_state(&self) -> Option<RenderState> {
        self.render_state.clone()
    }

    /// The surface configuration (present mode, frame latency) the surfaces are configured with.
    pub fn surface_config(&self) -> SurfaceConfig {
        self.config.surface
    }

    /// Change the surface configuration (present mode, frame latency) at runtime.
    ///
    /// All surfaces are reconfigured before the next paint.
    ///
    /// This is cheap to call every frame.
    pub fn set_surface_config(&mut self, surface_config: SurfaceConfig) {
        if self.config.surface == surface_config {
            return;
        }

        self.config.surface = surface_config;

        #[expect(clippy::iter_over_hash_type)]
        for surface in self.surfaces.values_mut() {
            surface.needs_reconfigure = true;
        }
    }

    fn configure_surface(
        surface_state: &SurfaceState,
        render_state: &RenderState,
        config: &SurfaceConfig,
    ) {
        profiling::function_scope!();

        let SurfaceConfig {
            present_mode,
            desired_maximum_frame_latency,
        } = *config;

        // Transaction presentation can hold a drawable during AppKit live resize. Keep the
        // configured low-latency path normally, but use three Metal drawables while resizing.
        #[cfg(all(target_os = "macos", feature = "macos-window-resize-jitter-fix"))]
        let desired_maximum_frame_latency = if surface_state.resizing {
            Some(desired_maximum_frame_latency.unwrap_or(2).max(2))
        } else {
            desired_maximum_frame_latency
        };

        let width = surface_state.width;
        let height = surface_state.height;

        let usage = if surface_state.partial_repaint {
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_DST
        } else {
            wgpu::TextureUsages::RENDER_ATTACHMENT
        };
        let mut surf_config = wgpu::SurfaceConfiguration {
            usage,
            format: render_state.target_format,
            present_mode,
            alpha_mode: surface_state.alpha_mode,
            view_formats: vec![render_state.target_format],
            ..surface_state
                .surface
                .get_default_config(&render_state.adapter, width, height)
                .expect("The surface isn't supported by this adapter")
        };

        if let Some(desired_maximum_frame_latency) = desired_maximum_frame_latency {
            surf_config.desired_maximum_frame_latency = desired_maximum_frame_latency;
        }

        surface_state
            .surface
            .configure(&render_state.device, &surf_config);
    }

    /// Drop the existing [`wgpu::Surface`] for `viewport_id` and create a fresh one for the
    /// given window via [`wgpu::Instance::create_surface`], then configure it.
    ///
    /// Used to recover from [`wgpu::CurrentSurfaceTexture::Lost`], where reconfiguring the
    /// existing surface object cannot recover.
    fn recreate_surface(
        &mut self,
        viewport_id: ViewportId,
        window: &Arc<winit::window::Window>,
    ) -> Result<(), crate::WgpuError> {
        profiling::function_scope!();

        let Some(old_state) = self.surfaces.remove(&viewport_id) else {
            return Ok(());
        };

        let surface = self.instance.create_surface(Arc::clone(window))?;
        // The window's size; `width` and `height` may have been fitted.
        self.install_surface(
            surface,
            viewport_id,
            old_state.window_width,
            old_state.window_height,
            old_state.resizing,
        );
        Ok(())
    }

    /// Updates (or clears) the [`winit::window::Window`] associated with the [`Painter`]
    ///
    /// This creates a [`wgpu::Surface`] for the given Window (as well as initializing render
    /// state if needed) that is used for egui rendering.
    ///
    /// This must be called before trying to render via
    /// [`paint_and_update_textures`](Self::paint_and_update_textures)
    ///
    /// # Portability
    ///
    /// _In particular it's important to note that on Android a it's only possible to create
    /// a window surface between `Resumed` and `Paused` lifecycle events, and Winit will panic on
    /// attempts to query the raw window handle while paused._
    ///
    /// On Android [`set_window`](Self::set_window) should be called with `Some(window)` for each
    /// `Resumed` event and `None` for each `Paused` event. Currently, on all other platforms
    /// [`set_window`](Self::set_window) may be called with `Some(window)` as soon as you have a
    /// valid [`winit::window::Window`].
    ///
    /// # Errors
    /// If the provided wgpu configuration does not match an available device.
    pub async fn set_window(
        &mut self,
        viewport_id: ViewportId,
        window: Option<Arc<winit::window::Window>>,
    ) -> Result<(), crate::WgpuError> {
        profiling::scope!("Painter::set_window"); // profile_function gives bad names for async functions

        if let Some(window) = window {
            let size = window.inner_size();
            if !self.surfaces.contains_key(&viewport_id) {
                let surface = self.instance.create_surface(window)?;
                self.add_surface(surface, viewport_id, size).await?;
            }
        } else {
            log::warn!("No window - clearing all surfaces");
            self.surfaces.clear();
        }
        Ok(())
    }

    /// Updates (or clears) the [`winit::window::Window`] associated with the [`Painter`] without taking ownership of the window.
    ///
    /// Like [`set_window`](Self::set_window) except:
    ///
    /// # Safety
    /// The user is responsible for ensuring that the window is alive for as long as it is set.
    pub async unsafe fn set_window_unsafe(
        &mut self,
        viewport_id: ViewportId,
        window: Option<&winit::window::Window>,
    ) -> Result<(), crate::WgpuError> {
        profiling::scope!("Painter::set_window_unsafe"); // profile_function gives bad names for async functions

        if let Some(window) = window {
            let size = window.inner_size();
            if !self.surfaces.contains_key(&viewport_id) {
                let surface = unsafe {
                    self.instance
                        .create_surface_unsafe(wgpu::SurfaceTargetUnsafe::from_window(&window)?)?
                };
                self.add_surface(surface, viewport_id, size).await?;
            }
        } else {
            log::warn!("No window - clearing all surfaces");
            self.surfaces.clear();
        }
        Ok(())
    }

    async fn add_surface(
        &mut self,
        surface: wgpu::Surface<'static>,
        viewport_id: ViewportId,
        size: winit::dpi::PhysicalSize<u32>,
    ) -> Result<(), crate::WgpuError> {
        // The device is kept for `catch_errors` below.
        let render_state = match self.render_state.take() {
            Some(render_state) => render_state,
            None => {
                RenderState::create(&self.config, &self.instance, Some(&surface), self.options)
                    .await?
            }
        };
        let device = render_state.device.clone();
        self.render_state = Some(render_state);
        // A surface the device can't configure is returned as an error
        // instead of wgpu's default handler panicking; for the root window at start-up, eframe
        // returns it as `Error::Wgpu` before the app is created, and the app can fall back.
        // Nothing is kept for the window. A window whose size is zero here is first configured
        // later, on resize, outside this.
        let installed = catch_errors(&device, || {
            self.install_surface(surface, viewport_id, size.width, size.height, false);
        })
        .await;
        if let Err(error) = installed {
            self.surfaces.remove(&viewport_id);
            self.depth_texture_view.remove(&viewport_id);
            self.msaa_texture_view.remove(&viewport_id);
            return Err(crate::WgpuError::ConfigureSurface(error));
        }
        Ok(())
    }

    /// Inserts a freshly created surface into [`Self::surfaces`] and configures it.
    ///
    /// Render state must already be initialised before calling this.
    // NOTE: The same assumption is already required by `resize_and_generate_depth_texture_view_and_msaa_view`.
    fn install_surface(
        &mut self,
        surface: wgpu::Surface<'static>,
        viewport_id: ViewportId,
        width: u32,
        height: u32,
        resizing: bool,
    ) {
        let (alpha_mode, partial_repaint) = {
            // Panic: We use the same failure mode as `resize_and_generate_depth_texture_view_and_msaa_view`
            let render_state = self
                .render_state
                .as_ref()
                .expect("install_surface called before render_state initialization");
            let wanted = self
                .config
                .partial_repaint
                .applies_to(render_state.adapter.get_info().device_type);
            let copyable = surface
                .get_capabilities(&render_state.adapter)
                .usages
                .contains(wgpu::TextureUsages::COPY_DST);
            let partial_repaint = wanted && self.options.msaa_samples <= 1 && copyable;
            if wanted {
                if partial_repaint {
                    log::info!("Partial repaint is on");
                } else {
                    log::warn!(
                        "Partial repaint can't apply (multisampling: {}, surface can be copied to: {copyable}): every frame is repainted whole",
                        self.options.msaa_samples
                    );
                }
            }
            let alpha_mode = {
                if self.support_transparent_backbuffer {
                    let supported_alpha_modes =
                        surface.get_capabilities(&render_state.adapter).alpha_modes;
                    // Prefer pre multiplied over post multiplied!
                    if supported_alpha_modes.contains(&wgpu::CompositeAlphaMode::PreMultiplied) {
                        wgpu::CompositeAlphaMode::PreMultiplied
                    } else if supported_alpha_modes
                        .contains(&wgpu::CompositeAlphaMode::PostMultiplied)
                    {
                        wgpu::CompositeAlphaMode::PostMultiplied
                    } else {
                        log::warn!(
                            "Transparent window was requested, but the active wgpu surface does not support a `CompositeAlphaMode` with transparency."
                        );
                        wgpu::CompositeAlphaMode::Auto
                    }
                } else {
                    wgpu::CompositeAlphaMode::Auto
                }
            };
            (alpha_mode, partial_repaint)
        };
        // Store the fitted size, so nothing configures the window's own.
        // `render_scale` starts at 1 so that the resize below, which sets it, logs the fit once.
        let (fit_width, fit_height, _) = surface_fit(width, height, self.max_surface_side());
        self.surfaces.insert(
            viewport_id,
            SurfaceState {
                surface,
                width: fit_width,
                height: fit_height,
                window_width: width,
                window_height: height,
                render_scale: 1.0,
                alpha_mode,
                resizing,
                needs_reconfigure: false,
                needs_recreate: false,
                partial_repaint,
                retained: None,
                fallback_logged: false,
            },
        );
        let Some(width) = NonZeroU32::new(width) else {
            log::debug!("The window width was zero; skipping generate textures");
            return;
        };
        let Some(height) = NonZeroU32::new(height) else {
            log::debug!("The window height was zero; skipping generate textures");
            return;
        };
        self.resize_and_generate_depth_texture_view_and_msaa_view(viewport_id, width, height);
    }

    /// Returns the maximum texture dimension supported if known
    ///
    /// This API will only return a known dimension after `set_window()` has been called
    /// at least once, since the underlying device and render state are initialized lazily
    /// once we have a window (that may determine the choice of adapter/device).
    pub fn max_texture_side(&self) -> Option<usize> {
        self.render_state
            .as_ref()
            .map(|rs| rs.device.limits().max_texture_dimension_2d as usize)
    }

    /// The largest side a surface may be configured at, or no limit
    /// before the device exists (nothing is configured then).
    fn max_surface_side(&self) -> u32 {
        self.render_state
            .as_ref()
            .map_or(u32::MAX, |rs| rs.device.limits().max_texture_dimension_2d)
    }

    fn resize_and_generate_depth_texture_view_and_msaa_view(
        &mut self,
        viewport_id: ViewportId,
        width_in_pixels: NonZeroU32,
        height_in_pixels: NonZeroU32,
    ) {
        profiling::function_scope!();

        // Configure the surface, and size the depth and MSAA textures,
        // within the device's limit; `surface_fit` keeps a non-zero side non-zero.
        let max_side = self.max_surface_side();
        let (window_width, window_height) = (width_in_pixels.get(), height_in_pixels.get());
        let (width, height, render_scale) = surface_fit(window_width, window_height, max_side);

        let render_state = self.render_state.as_ref().unwrap();
        let surface_state = self.surfaces.get_mut(&viewport_id).unwrap();

        if render_scale < 1.0 && surface_state.render_scale >= 1.0 {
            log::warn!(
                "the window is {window_width} × {window_height} pixels, more than this GPU draws \
                 ({max_side} a side): drawing it at {width} × {height} and scaling it up"
            );
        }
        surface_state.width = width;
        surface_state.height = height;
        surface_state.window_width = window_width;
        surface_state.window_height = window_height;
        surface_state.render_scale = render_scale;

        Self::configure_surface(surface_state, render_state, &self.config.surface);

        if let Some(depth_format) = self.options.depth_stencil_format {
            self.depth_texture_view.insert(
                viewport_id,
                render_state
                    .device
                    .create_texture(&wgpu::TextureDescriptor {
                        label: Some("egui_depth_texture"),
                        size: wgpu::Extent3d {
                            width,
                            height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: self.options.msaa_samples.max(1),
                        dimension: wgpu::TextureDimension::D2,
                        format: depth_format,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                            | wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[depth_format],
                    })
                    .create_view(&wgpu::TextureViewDescriptor::default()),
            );
        }

        if let Some(render_state) = (self.options.msaa_samples > 1)
            .then_some(self.render_state.as_ref())
            .flatten()
        {
            let texture_format = render_state.target_format;
            self.msaa_texture_view.insert(
                viewport_id,
                render_state
                    .device
                    .create_texture(&wgpu::TextureDescriptor {
                        label: Some("egui_msaa_texture"),
                        size: wgpu::Extent3d {
                            width,
                            height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: self.options.msaa_samples.max(1),
                        dimension: wgpu::TextureDimension::D2,
                        format: texture_format,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                        view_formats: &[texture_format],
                    })
                    .create_view(&wgpu::TextureViewDescriptor::default()),
            );
        }
    }

    /// Handles changes of the resizing state.
    ///
    /// Should be called prior to the first [`Painter::on_window_resized`] call and after the last in
    /// the chain. Used to apply platform-specific logic, e.g. OSX Metal window resize jitter fix.
    pub fn on_window_resize_state_change(&mut self, viewport_id: ViewportId, resizing: bool) {
        profiling::function_scope!();

        let Some(state) = self.surfaces.get_mut(&viewport_id) else {
            return;
        };
        if state.resizing == resizing {
            if resizing {
                log::debug!(
                    "Painter::on_window_resize_state_change() redundant call while resizing"
                );
            } else {
                log::debug!(
                    "Painter::on_window_resize_state_change() redundant call after resizing"
                );
            }
            return;
        }

        // Set before reconfiguring so macOS live resize uses the temporary latency bump above.
        state.resizing = resizing;

        // Resizing is a bit tricky on macOS.
        // It requires enabling ["present_with_transaction"](https://developer.apple.com/documentation/quartzcore/cametallayer/presentswithtransaction)
        // flag to avoid jittering during the resize. Even though resize jittering on macOS
        // is common across rendering backends, the solution for wgpu/metal is known.
        //
        // See https://github.com/emilk/egui/issues/903
        #[cfg(all(target_os = "macos", feature = "macos-window-resize-jitter-fix"))]
        {
            // SAFETY: `as_hal::<Metal>()` returns `None` unless this surface is backed by wgpu's
            // Metal backend.
            unsafe {
                if let (Some(render_state), Some(hal_surface)) = (
                    self.render_state.as_ref(),
                    state.surface.as_hal::<wgpu::hal::api::Metal>(),
                ) {
                    hal_surface
                        .render_layer()
                        .lock()
                        .setPresentsWithTransaction(resizing);

                    Self::configure_surface(state, render_state, &self.config.surface);
                }
            }
        }
    }

    pub fn on_window_resized(
        &mut self,
        viewport_id: ViewportId,
        width_in_pixels: NonZeroU32,
        height_in_pixels: NonZeroU32,
    ) {
        profiling::function_scope!();

        if self.surfaces.contains_key(&viewport_id) {
            self.resize_and_generate_depth_texture_view_and_msaa_view(
                viewport_id,
                width_in_pixels,
                height_in_pixels,
            );
        } else {
            log::warn!(
                "Ignoring window resize notification with no surface created via Painter::set_window()"
            );
        }
    }

    /// Returns two things:
    ///
    /// The approximate number of seconds spent on vsync-waiting (if any),
    /// and the captures captured screenshot if it was requested.
    ///
    /// If `capture_data` isn't empty, a screenshot will be captured.
    #[expect(clippy::too_many_arguments)]
    pub fn paint_and_update_textures(
        &mut self,
        viewport_id: ViewportId,
        pixels_per_point: f32,
        clear_color: [f32; 4],
        clipped_primitives: &[epaint::ClippedPrimitive],
        textures_delta: &mut epaint::textures::TexturesDelta,
        capture_data: Vec<egui::ScreenshotCallback>,
        window: &Arc<winit::window::Window>,
    ) -> f32 {
        // What the retained frame holds is no longer what the window shows.
        if let Some(surface_state) = self.surfaces.get_mut(&viewport_id) {
            surface_state.retained = None;
        }
        self.invalidate_textures_elsewhere(viewport_id, textures_delta);
        self.paint(
            viewport_id,
            pixels_per_point,
            clear_color,
            clipped_primitives,
            textures_delta,
            capture_data,
            window,
            None,
        )
    }

    /// The texture changes of `textures_delta` are applied while `viewport_id` is painted, and
    /// then gone: tell the retained frames of the other windows, which sample the same textures.
    fn invalidate_textures_elsewhere(
        &mut self,
        viewport_id: ViewportId,
        textures_delta: &epaint::textures::TexturesDelta,
    ) {
        if textures_delta.set.is_empty() {
            return;
        }
        let changed = epaint::DamageTracker::changed_textures(textures_delta);
        #[expect(clippy::iter_over_hash_type)] // Order doesn't matter here
        for (id, surface_state) in &mut self.surfaces {
            if *id != viewport_id
                && let Some(retained) = &mut surface_state.retained
            {
                retained
                    .tracker
                    .invalidate_textures(changed.iter().copied());
            }
        }
    }

    /// Like [`Self::paint_and_update_textures`], but from the frame's shapes, which it
    /// tessellates itself: with [`crate::PartialRepaint`] it compares them to the previous
    /// frame's, skips a frame that looks the same, and repaints only what changed.
    ///
    /// Returns the approximate number of seconds spent on vsync-waiting.
    #[expect(clippy::too_many_arguments)]
    pub fn paint_shapes_and_update_textures(
        &mut self,
        viewport_id: ViewportId,
        pixels_per_point: f32,
        clear_color: [f32; 4],
        mut shapes: Vec<epaint::ClippedShape>,
        textures_delta: &mut epaint::textures::TexturesDelta,
        capture_data: Vec<egui::ScreenshotCallback>,
        window: &Arc<winit::window::Window>,
    ) -> f32 {
        profiling::function_scope!();
        self.invalidate_textures_elsewhere(viewport_id, textures_delta);
        let Some(surface_state) = self.surfaces.get(&viewport_id) else {
            return 0.0;
        };
        // A screenshot is painted into its own texture; a translucent clear colour can't be
        // painted over the old pixels (it would blend with them).
        let partial = surface_state.partial_repaint
            && capture_data.is_empty()
            && clear_color[3] >= 1.0
            && surface_state.width > 0
            && surface_state.height > 0;
        if !partial {
            if surface_state.partial_repaint
                && clear_color[3] < 1.0
                && let Some(surface_state) = self.surfaces.get_mut(&viewport_id)
                && !surface_state.fallback_logged
            {
                surface_state.fallback_logged = true;
                log::warn!(
                    "Partial repaint needs an opaque clear colour (`App::clear_color`), \
                     got alpha {}: every frame is repainted whole",
                    clear_color[3]
                );
            }
            let clipped_primitives = self.context.tessellate(shapes, pixels_per_point);
            return self.paint_and_update_textures(
                viewport_id,
                pixels_per_point,
                clear_color,
                &clipped_primitives,
                textures_delta,
                capture_data,
                window,
            );
        }
        let Some(render_state) = self.render_state.as_ref() else {
            return 0.0;
        };
        let Some(surface_state) = self.surfaces.get_mut(&viewport_id) else {
            return 0.0;
        };
        let (width, height) = (surface_state.width, surface_state.height);
        let screen_pixels_per_point = pixels_per_point * surface_state.render_scale;

        // The clear colour as a shape under all others: painted within the damage like the
        // rest, it clears just that, the same way in partial and whole repaints.
        let screen = egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(width as f32, height as f32) / screen_pixels_per_point,
        )
        .expand(1.0);
        let [r, g, b, a] = clear_color;
        // The colour the clear it stands in for leaves in the target. A clear writes its values
        // as they are; a shape's colour is a gamma one, which the shader of an sRGB target
        // makes linear first, so there the values go in as linear ones.
        let background = if render_state.target_format.is_srgb() {
            egui::Color32::from(egui::Rgba::from_rgba_premultiplied(r, g, b, a))
        } else {
            let byte = |value: f32| (value.clamp(0.0, 1.0) * 255.0).round() as u8;
            egui::Color32::from_rgba_premultiplied(byte(r), byte(g), byte(b), byte(a))
        };
        shapes.insert(
            0,
            epaint::ClippedShape {
                clip_rect: screen,
                shape: epaint::Shape::rect_filled(screen, 0.0, background),
                transform_after_tessellation: egui::emath::TSTransform::IDENTITY,
            },
        );

        let format = render_state.target_format;
        if surface_state
            .retained
            .as_ref()
            .is_some_and(|retained| !retained.fits(format, width, height))
        {
            surface_state.retained = None;
        }
        let retained = surface_state
            .retained
            .get_or_insert_with(|| RetainedFrame::new(&render_state.device, format, width, height));
        let tessellation = self.context.tessellation_options(|options| *options);
        retained.tracker.set_feathering(if tessellation.feathering {
            tessellation.feathering_size_in_pixels
        } else {
            0.0
        });
        let tracked = retained
            .tracker
            .update(&shapes, screen_pixels_per_point, textures_delta);
        // The debug options paint beyond the clip rects the damage is limited to, and a callback
        // can paint anywhere, scissor or not, even one the tracker matched to the last frame's.
        let unbounded = tessellation.debug_ignore_clip_rects
            || tessellation.debug_paint_clip_rects
            || shapes.iter().any(|clipped| has_callback(&clipped.shape));
        let reason = if surface_state.needs_reconfigure {
            Some(FullRepaint::Surface)
        } else if retained.tessellation != Some(tessellation) && retained.tessellation.is_some() {
            Some(FullRepaint::Options)
        } else if unbounded {
            Some(FullRepaint::Unbounded)
        } else if tracked == epaint::Damage::Full || retained.tessellation.is_none() {
            Some(FullRepaint::NothingToCompare)
        } else {
            None
        };
        let mut damage = if reason.is_some() {
            epaint::Damage::Full
        } else {
            tracked
        };
        retained.tessellation = Some(tessellation);
        retained
            .stats
            .record(&damage, reason, screen, [width, height]);
        let started = web_time::Instant::now();

        if damage == epaint::Damage::None {
            // Nothing visible changed: no render and no present. New textures are still
            // uploaded (the next frame may use them), and freed ones freed.
            profiling::scope!("skip unchanged frame");
            let mut renderer = render_state.renderer.write();
            #[expect(clippy::iter_over_hash_type)] // Order doesn't matter here
            for (id, image_deltas) in textures_delta.set.drain() {
                for image_delta in image_deltas {
                    renderer.update_texture(
                        &render_state.device,
                        &render_state.queue,
                        id,
                        &image_delta,
                    );
                }
            }
            render_state.queue.submit([]);
            #[expect(clippy::iter_over_hash_type)] // Order doesn't matter here
            for id in textures_delta.free.drain() {
                renderer.free_texture(&id);
            }
            return 0.0;
        }

        let clipped_primitives = self.context.tessellate(shapes, pixels_per_point);
        let callback = clipped_primitives
            .iter()
            .any(|p| matches!(p.primitive, epaint::Primitive::Callback(_)));
        if callback {
            // A callback can paint anywhere, scissor or not.
            damage = epaint::Damage::Full;
        }
        let vsync_sec = self.paint(
            viewport_id,
            pixels_per_point,
            clear_color,
            &clipped_primitives,
            textures_delta,
            capture_data,
            window,
            Some(&damage),
        );
        if let Some(surface_state) = self.surfaces.get_mut(&viewport_id)
            && let Some(retained) = &mut surface_state.retained
        {
            let waiting = f64::from(vsync_sec.max(0.0));
            let share = match &damage {
                epaint::Damage::None => 0.0,
                epaint::Damage::Full => 1.0,
                epaint::Damage::Partial(region) => {
                    let area: f32 = region
                        .rects()
                        .iter()
                        .map(|r| r.intersect(screen).area().max(0.0))
                        .sum();
                    area / screen.area().max(1.0)
                }
            };
            if stats_logged() {
                retained.stats.painted(
                    damage == epaint::Damage::Full,
                    (started.elapsed().as_secs_f64() - waiting).max(0.0),
                    waiting,
                    &self.frame_cost,
                    share,
                    || {
                        let rects = match &damage {
                            epaint::Damage::Partial(region) => Some(region.rects()),
                            _ => None,
                        };
                        FrameFill::of(&clipped_primitives, rects, screen, screen_pixels_per_point)
                    },
                );
            }
            if callback {
                // What it painted outside its rectangle stays when the callback goes: the
                // frame after one is repainted whole too.
                retained.tracker.reset();
            }
        }
        vsync_sec
    }

    /// Paint `clipped_primitives`: the whole window, or, with `damage`, into the surface's
    /// retained frame (only the damage when partial), which is then copied to the window.
    #[expect(clippy::too_many_arguments)]
    fn paint(
        &mut self,
        viewport_id: ViewportId,
        pixels_per_point: f32,
        clear_color: [f32; 4],
        clipped_primitives: &[epaint::ClippedPrimitive],
        textures_delta: &mut epaint::textures::TexturesDelta,
        capture_data: Vec<egui::ScreenshotCallback>,
        window: &Arc<winit::window::Window>,
        damage: Option<&epaint::Damage>,
    ) -> f32 {
        profiling::function_scope!();

        /// Guard to ensure that commands are always submitted to the renderer queue
        /// so that calls to [`write_buffer()`](https://docs.rs/wgpu/latest/wgpu/struct.Queue.html#method.write_buffer)
        /// are completed even if we take a codepath which doesn't submit commands and avoids
        /// internal buffers growing indefinitely.
        ///
        /// This may happen, for example, if no output frame is resolved.
        /// See <https://github.com/emilk/egui/pull/7928> for full context.
        struct RendererQueueGuard<'q> {
            queue: &'q wgpu::Queue,
            commands_submitted: bool,
        }

        impl Drop for RendererQueueGuard<'_> {
            fn drop(&mut self) {
                // Only submit an empty command buffer array if no commands were
                // explicitly submitted.
                if !self.commands_submitted {
                    self.queue.submit([]);
                }
            }
        }

        let capture = !capture_data.is_empty();
        let mut vsync_sec = 0.0;
        self.frame_cost = FrameCost::default();

        // If the previous frame produced `CurrentSurfaceTexture::Lost`, the action match
        // below set `needs_recreate`. Recreate the surface now, before re-borrowing
        // `self.render_state` / `self.surfaces` for the rest of the paint.
        if self
            .surfaces
            .get(&viewport_id)
            .is_some_and(|s| s.needs_recreate)
            && let Err(err) = self.recreate_surface(viewport_id, window)
        {
            log::error!("Failed to recreate surface for {viewport_id:?}: {err}");
            return vsync_sec;
        }

        let Some(render_state) = self.render_state.as_mut() else {
            return vsync_sec;
        };

        let mut render_queue_guard = RendererQueueGuard {
            queue: &render_state.queue,
            commands_submitted: false,
        };

        let Some(surface_state) = self.surfaces.get_mut(&viewport_id) else {
            return vsync_sec;
        };

        let mut encoder =
            render_state
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("encoder"),
                });

        // Upload all resources for the GPU.
        // A surface fitted within the GPU's limit is drawn smaller.
        let screen_descriptor = renderer::ScreenDescriptor {
            size_in_pixels: [surface_state.width, surface_state.height],
            pixels_per_point: pixels_per_point * surface_state.render_scale,
        };

        let user_cmd_bufs = {
            let mut renderer = render_state.renderer.write();
            let start = web_time::Instant::now();
            #[expect(clippy::iter_over_hash_type)] // Order doesn't matter here
            for (id, image_deltas) in textures_delta.set.drain() {
                for image_delta in image_deltas {
                    let [width, height] = image_delta.image.size();
                    let bytes = width
                        .saturating_mul(height)
                        .saturating_mul(image_delta.image.bytes_per_pixel());
                    self.frame_cost.upload_bytes =
                        self.frame_cost.upload_bytes.saturating_add(bytes as u64);
                    renderer.update_texture(
                        &render_state.device,
                        &render_state.queue,
                        id,
                        &image_delta,
                    );
                }
            }
            self.frame_cost.upload_seconds = start.elapsed().as_secs_f64();

            let start = web_time::Instant::now();
            let user_cmd_bufs = renderer.update_buffers(
                &render_state.device,
                &render_state.queue,
                &mut encoder,
                clipped_primitives,
                &screen_descriptor,
            );
            self.frame_cost.buffers_seconds = start.elapsed().as_secs_f64();
            user_cmd_bufs
        };

        if surface_state.needs_reconfigure {
            Self::configure_surface(surface_state, render_state, &self.config.surface);
            surface_state.needs_reconfigure = false;
        }

        let output_frame = {
            profiling::scope!("get_current_texture");
            // This is what vsync-waiting happens on my Mac.
            let start = web_time::Instant::now();
            let output_frame = surface_state.surface.get_current_texture();
            self.frame_cost.acquire_seconds = start.elapsed().as_secs_f64();
            vsync_sec += start.elapsed().as_secs_f32();
            output_frame
        };

        // `None` when there is no frame to present to. A capture is rendered to its own
        // texture and copied back to the cpu, so it is served even then — that is how a hidden
        // window, which the compositor gives no surface texture, is screenshotted at all.
        let output_frame = match output_frame {
            wgpu::CurrentSurfaceTexture::Success(frame) => Some(frame),
            wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                surface_state.needs_reconfigure = true;
                Some(frame)
            }
            other => {
                match (*self.config.on_surface_status)(&other) {
                    SurfaceErrorAction::Reconfigure => {
                        Self::configure_surface(surface_state, render_state, &self.config.surface);
                        self.context.request_repaint_of(viewport_id);
                    }
                    SurfaceErrorAction::RecreateSurface => {
                        // Because of ownership, I could not find an easy way to do a full recovery here,
                        // as that would involve dropping the old surface and creating a new one.
                        // For now, we defer the recreation to the beginning of the next frame (which
                        // we ensure to arrive via `request_repaint_of`). A cleaner solution would be
                        // to untangle the ownership of `RenderState`.
                        surface_state.needs_recreate = true;
                        self.context.request_repaint_of(viewport_id);
                    }
                    SurfaceErrorAction::SkipFrame => {}
                }
                if !capture {
                    // The retained frame wasn't painted: compare the next frame to nothing.
                    if let Some(retained) = &mut surface_state.retained {
                        retained.tracker.reset();
                    }
                    return vsync_sec;
                }
                None
            }
        };

        let mut capture_buffer = None;
        {
            let renderer = render_state.renderer.read();

            let target_texture = if capture {
                // The frame we would have presented decides the capture, when there is one.
                // Without it we have only the surface we configured, which is the same thing
                // as long as no resize is in flight.
                let (size, format) = match &output_frame {
                    Some(output_frame) => {
                        (output_frame.texture.size(), output_frame.texture.format())
                    }
                    None => (
                        wgpu::Extent3d {
                            width: surface_state.width,
                            height: surface_state.height,
                            depth_or_array_layers: 1,
                        },
                        render_state.target_format,
                    ),
                };
                if size.width == 0 || size.height == 0 {
                    log::warn!("Cannot capture a screenshot of a zero-sized surface");
                    return vsync_sec;
                }

                let capture_state = self
                    .screen_capture_state
                    .get_or_insert_with(|| CaptureState::new(&render_state.device, size, format));
                capture_state.update(&render_state.device, size);

                &capture_state.texture
            } else if let (Some(_), Some(retained)) = (&damage, &surface_state.retained) {
                &retained.texture
            } else if let Some(output_frame) = &output_frame {
                &output_frame.texture
            } else {
                // Unreachable: without a frame to present to, and with nothing to capture,
                // we returned above.
                return vsync_sec;
            };
            let target_view = target_texture.create_view(&wgpu::TextureViewDescriptor::default());

            let (view, resolve_target) = (self.options.msaa_samples > 1)
                .then_some(self.msaa_texture_view.get(&viewport_id))
                .flatten()
                .map_or((&target_view, None), |texture_view| {
                    (texture_view, Some(&target_view))
                });

            let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui_render"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: if matches!(damage, Some(epaint::Damage::Partial(_))) {
                            // Only the damage is painted; the rest keeps the last frame.
                            wgpu::LoadOp::Load
                        } else {
                            wgpu::LoadOp::Clear(wgpu::Color {
                                r: clear_color[0] as f64,
                                g: clear_color[1] as f64,
                                b: clear_color[2] as f64,
                                a: clear_color[3] as f64,
                            })
                        },
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: self.depth_texture_view.get(&viewport_id).map(|view| {
                    wgpu::RenderPassDepthStencilAttachment {
                        view,
                        depth_ops: self
                            .options
                            .depth_stencil_format
                            .is_some_and(|depth_stencil_format| {
                                depth_stencil_format.has_depth_aspect()
                            })
                            .then_some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                // It is very unlikely that the depth buffer is needed after egui finished rendering
                                // so no need to store it. (this can improve performance on tiling GPUs like mobile chips or Apple Silicon)
                                store: wgpu::StoreOp::Discard,
                            }),
                        stencil_ops: self
                            .options
                            .depth_stencil_format
                            .is_some_and(|depth_stencil_format| {
                                depth_stencil_format.has_stencil_aspect()
                            })
                            .then_some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(0),
                                store: wgpu::StoreOp::Discard,
                            }),
                    }
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            // Forgetting the pass' lifetime means that we are no longer compile-time protected from
            // runtime errors caused by accessing the parent encoder before the render pass is dropped.
            // Since we don't pass it on to the renderer, we should be perfectly safe against this mistake here!
            if let Some(epaint::Damage::Partial(region)) = &damage {
                renderer.render_damaged(
                    &mut render_pass.forget_lifetime(),
                    clipped_primitives,
                    &screen_descriptor,
                    region.rects(),
                );
            } else {
                renderer.render(
                    &mut render_pass.forget_lifetime(),
                    clipped_primitives,
                    &screen_descriptor,
                );
            }

            // The retained frame is the window's new content: copy all of it, since the
            // surface texture doesn't keep the last frame's pixels.
            if damage.is_some()
                && let (Some(retained), Some(output_frame)) =
                    (&surface_state.retained, &output_frame)
            {
                let size = retained.texture.size();
                let frame_size = output_frame.texture.size();
                encoder.copy_texture_to_texture(
                    retained.texture.as_image_copy(),
                    output_frame.texture.as_image_copy(),
                    wgpu::Extent3d {
                        width: size.width.min(frame_size.width),
                        height: size.height.min(frame_size.height),
                        depth_or_array_layers: 1,
                    },
                );
            }

            if capture && let Some(capture_state) = &mut self.screen_capture_state {
                capture_buffer = Some(capture_state.copy_textures(
                    &render_state.device,
                    output_frame.as_ref(),
                    &mut encoder,
                ));
            }
        }

        let encoded = {
            profiling::scope!("CommandEncoder::finish");
            encoder.finish()
        };

        // Submit the commands: both the main buffer and user-defined ones.
        {
            profiling::scope!("Queue::submit");
            // wgpu doesn't document where vsync can happen. Maybe here?
            let start = web_time::Instant::now();
            render_state
                .queue
                .submit(core::iter::chain(user_cmd_bufs, [encoded]));
            self.frame_cost.submit_seconds = start.elapsed().as_secs_f64();
            vsync_sec += start.elapsed().as_secs_f32();
        };

        // Ensure that the queue guard does not do unnecessary work when dropped
        render_queue_guard.commands_submitted = true;

        // Free textures marked for destruction **after** queue submit since they might still be used in the current frame.
        // Calling `wgpu::Texture::destroy` on a texture that is still in use would invalidate the command buffer(s) it is used in.
        // However, once we called `wgpu::Queue::submit`, it is up for wgpu to determine how long the underlying gpu resource has to live.
        {
            let mut renderer = render_state.renderer.write();
            #[expect(clippy::iter_over_hash_type)] // Order doesn't matter here
            for id in textures_delta.free.drain() {
                renderer.free_texture(&id);
            }
        }

        if let Some(capture_buffer) = capture_buffer
            && let Some(screen_capture_state) = &mut self.screen_capture_state
        {
            screen_capture_state.read_screen_rgba(capture_buffer, capture_data);
        }

        if let Some(output_frame) = output_frame {
            window.pre_present_notify();

            profiling::scope!("present");
            // wgpu doesn't document where vsync can happen. Maybe here?
            let start = web_time::Instant::now();
            render_state.queue.present(output_frame);
            self.frame_cost.present_seconds = start.elapsed().as_secs_f64();
            vsync_sec += start.elapsed().as_secs_f32();
        }

        vsync_sec
    }

    /// The next frame of `viewport_id` is painted whole, e.g. because the window was hidden
    /// and what it shows can't be trusted. Only matters with [`crate::PartialRepaint`].
    pub fn forget_last_frame(&mut self, viewport_id: ViewportId) {
        if let Some(retained) = self
            .surfaces
            .get_mut(&viewport_id)
            .and_then(|surface_state| surface_state.retained.as_mut())
        {
            retained.tracker.reset();
        }
    }

    pub fn gc_viewports(&mut self, active_viewports: &ViewportIdSet) {
        self.surfaces.retain(|id, _| active_viewports.contains(id));
        self.depth_texture_view
            .retain(|id, _| active_viewports.contains(id));
        self.msaa_texture_view
            .retain(|id, _| active_viewports.contains(id));
    }

    #[expect(clippy::needless_pass_by_ref_mut, clippy::unused_self)]
    pub fn destroy(&mut self) {
        // TODO(emilk): something here?
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_frames_fill_is_its_triangles_within_the_damage_and_the_clip() {
        use super::{FrameCost, FrameFill};
        let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(100.0, 100.0));
        let quad = |rect: egui::Rect, texture_id| {
            let mut mesh = epaint::Mesh::with_texture(texture_id);
            let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
            mesh.add_rect_with_uv(rect, uv, egui::Color32::WHITE);
            mesh
        };
        let image = epaint::TextureId::User(7);
        let rect = |x: f32, y: f32, w: f32, h: f32| {
            egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(w, h))
        };
        let primitives = vec![
            // 40x50 points of an image, all shown.
            epaint::ClippedPrimitive {
                clip_rect: screen,
                primitive: epaint::Primitive::Mesh(quad(rect(0.0, 0.0, 40.0, 50.0), image)),
            },
            // A plain 100x100 quad, of which the clip shows 10x10.
            epaint::ClippedPrimitive {
                clip_rect: rect(50.0, 50.0, 10.0, 10.0),
                primitive: epaint::Primitive::Mesh(quad(screen, epaint::TextureId::default())),
            },
            // Wholly outside the window: not counted.
            epaint::ClippedPrimitive {
                clip_rect: rect(200.0, 0.0, 10.0, 10.0),
                primitive: epaint::Primitive::Mesh(quad(rect(200.0, 0.0, 10.0, 10.0), image)),
            },
        ];
        let whole = FrameFill::of(&primitives, None, screen, 2.0);
        assert_eq!((whole.meshes, whole.vertices, whole.textures), (2, 8, 2));
        // At two pixels a point: 2000 and 100 square points.
        assert_eq!(whole.pixels, 8400.0);
        assert_eq!(whole.image_pixels, 8000.0);

        // Only the left 20 points are damaged: half the image, none of the plain quad.
        let damage = [rect(0.0, 0.0, 20.0, 100.0)];
        let part = FrameFill::of(&primitives, Some(&damage), screen, 1.0);
        assert_eq!(
            (part.meshes, part.pixels, part.image_pixels),
            (1, 1000.0, 1000.0)
        );

        assert_eq!(
            FrameFill::of(&primitives, Some(&[]), screen, 1.0),
            FrameFill::default()
        );

        // Indices that point nowhere are skipped, not a panic.
        let mut broken = quad(screen, image);
        broken.indices.push(99);
        broken.indices.push(98);
        broken.indices.push(97);
        let broken = [epaint::ClippedPrimitive {
            clip_rect: screen,
            primitive: epaint::Primitive::Mesh(broken),
        }];
        assert_eq!(FrameFill::of(&broken, None, screen, 1.0).pixels, 10000.0);

        let mut sum = FrameCost::default();
        let one = FrameCost {
            upload_seconds: 0.5,
            upload_bytes: 10,
            buffers_seconds: 0.25,
            acquire_seconds: 1.0,
            submit_seconds: 2.0,
            present_seconds: 3.0,
        };
        sum.add(&one);
        sum.add(&one);
        assert_eq!(sum.upload_bytes, 20);
        assert_eq!(sum.present_seconds, 6.0);
    }

    use super::surface_fit;

    #[test]
    fn surface_fit_keeps_windows_within_the_limit() {
        assert_eq!(surface_fit(1920, 1080, 8192), (1920, 1080, 1.0));
        assert_eq!(surface_fit(8192, 100, 8192), (8192, 100, 1.0));

        let (w, h, scale) = surface_fit(16384, 4000, 8192);
        assert_eq!((w, h), (8192, 2000));
        assert!((scale - 0.5).abs() < 1e-6);

        // A thin side never becomes zero, and an empty side stays empty.
        assert_eq!(surface_fit(100_000, 1, 8192).1, 1);
        assert_eq!(surface_fit(100_000, 0, 8192).1, 0);
        // A zero limit is treated as one pixel.
        assert_eq!(surface_fit(10, 10, 0), (1, 1, 0.1));
    }
}
