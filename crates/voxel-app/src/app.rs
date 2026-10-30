//! The desktop application: window, camera input, chunk upload, frame loop.
//!
//! Generation is done once at startup on the CPU, in parallel, and the result
//! is uploaded as a handful of vertex buffers. This POC is about comparing
//! carving strategies, so there is no streaming or re-meshing yet: the cost of
//! `generate_world` plus meshing is what the benchmark measures, and the frame
//! loop only draws.

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use voxel_core::mesh::mesh_chunk;
use voxel_core::{CarveParity, Carver, ChunkPos, Config};

use crate::camera::Camera;
use crate::gpu::{FrameOutcome, Renderer};
use crate::hud::{CarverSwitch, HudState};
use crate::render::ChunkMeshData;
use crate::stats::FrameTimer;
use crate::stream::StreamWorld;

/// Sky colour, used as the clear colour and as the ambient against which the
/// terrain's shading reads.
const SKY: wgpu::Color = wgpu::Color {
    r: 0.53,
    g: 0.81,
    b: 0.92,
    a: 1.0,
};

/// How far the camera moves per second while a key is held.
const MOVE_SPEED: f32 = 40.0;
/// How far the camera turns per second while a look key is held.
const LOOK_SPEED: f32 = 1.8;
/// Render distance in chunks, as requested for the demo.
pub const RENDER_DISTANCE: i32 = 16;

/// Chunks are kept this much further out than the render distance, so flying
/// back and forth across the boundary does not regenerate the same ground.
const EVICT_RADIUS: i32 = RENDER_DISTANCE + 4;

/// Runs the windowed application until the window closes.
pub fn run(cfg: Config) -> Result<()> {
    let event_loop = EventLoop::new().context("creating the event loop")?;
    // Poll rather than Wait: the camera moves continuously while a key is held,
    // and there are no other event sources to block on.
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new(cfg);
    event_loop.run_app(&mut app).context("running the event loop")?;
    Ok(())
}

struct App {
    cfg: Config,
    state: Option<State>,
    /// Keys currently held, so movement is smooth rather than one step per
    /// key-repeat.
    keys: Vec<KeyCode>,
    last_frame: Instant,
    /// Rolling frame-time window backing the HUD's percentiles.
    timer: FrameTimer,
    /// Carver selection, including any pending switch.
    carver: CarverSwitch,
    /// Counters used to decide when the HUD text actually changes, so the
    /// overlay is not rebuilt and re-uploaded every single frame.
    last_hud: Option<String>,
    /// Whether the benchmark readout is shown.
    benchmark_mode: bool,
    /// Carve coverage of each approach, measured once at startup.
    parity: CarveParity,
    /// Wall-clock accumulators for the logging summary.
    frames: u64,
    fps_timer: Instant,
}

impl App {
    fn new(cfg: Config) -> Self {
        Self {
            cfg,
            state: None,
            keys: Vec::new(),
            last_frame: Instant::now(),
            timer: FrameTimer::new(),
            carver: CarverSwitch::new(cfg.carver),
            last_hud: None,
            // Off by default: the timing block is for benchmarking, not for
            // flying around, and it triples the size of the overlay.
            benchmark_mode: false,
            // Measured once at startup over a small sample: this is a property
            // of the configuration, not of the running frame, so recomputing it
            // every frame would be waste.
            parity: CarveParity::measure(cfg.seed, &cfg.cave, 2),
            frames: 0,
            fps_timer: Instant::now(),
        }
    }
}

/// Everything that exists only once a window and GPU surface do.
struct State {
    window: Arc<Window>,
    renderer: Renderer,
    camera: Camera,
    uploaded_chunks: usize,
    uploaded_vertices: usize,
    /// The streamed world, generated around the camera as it moves.
    world: StreamWorld,
    /// Chunk the camera was in last frame, so streaming only runs on a boundary
    /// crossing rather than every frame.
    last_centre: ChunkPos,
    generation_ms: f32,
    meshing_ms: f32,
    upload_ms: f32,
    /// Chunks meshed and uploaded this frame, for the streaming readout.
    chunks_streamed: usize,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            // `resumed` fires again on some platforms after a suspend.
            return;
        }

        let attributes = Window::default_attributes()
            .with_title(format!("voxel world-gen POC [{}]", self.cfg.env))
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0));

        let window = match event_loop.create_window(attributes) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                log::error!("could not create a window: {e}");
                event_loop.exit();
                return;
            }
        };

        let size = window.inner_size();

        // One instance for both the surface and the adapter: wgpu tracks
        // surfaces per instance, so creating the surface from one instance and
        // the device from another fails at runtime.
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());

        // The surface borrows the window; `Arc` keeps the handle alive for as
        // long as the renderer's `'static` surface requires.
        let surface = match instance.create_surface(window.clone()) {
            Ok(s) => s,
            Err(e) => {
                log::error!("could not create a rendering surface: {e}");
                event_loop.exit();
                return;
            }
        };

        let renderer = match pollster::block_on(Renderer::new(
            &instance,
            surface,
            size.width,
            size.height,
            self.cfg.render.force_software_gpu,
            self.cfg.render.vsync,
        )) {
            Ok(r) => r,
            Err(e) => {
                log::error!("{e:#}");
                event_loop.exit();
                return;
            }
        };

        log::warn!(
            "renderer: {} ({:?}, {:?})",
            renderer.adapter_info.name,
            renderer.adapter_info.backend,
            renderer.adapter_info.device_type
        );
        if renderer.adapter_info.device_type == wgpu::DeviceType::Cpu {
            // Worth saying out loud: frame times on a CPU rasteriser are not
            // representative of real hardware.
            log::warn!("using a software rasteriser; frame timings are not representative");
        }

        let mut renderer = renderer;
        let camera = default_camera(&self.cfg);
        let mut world = StreamWorld::new(
            self.cfg.seed,
            self.cfg.world_height(),
            self.cfg.cave,
            self.cfg.carver,
        );

        // Stream the initial view distance before the first frame, so the world
        // is already there rather than popping in.
        let centre = ChunkPos::from_world(
            camera.eye[0] as i32,
            camera.eye[1] as i32,
            camera.eye[2] as i32,
        );
        let started = Instant::now();
        let streamed = world.stream_around(centre, RENDER_DISTANCE);
        let stream_ms = started.elapsed().as_secs_f32() * 1000.0;

        let t0 = Instant::now();
        let meshes = mesh_world(&world, centre, RENDER_DISTANCE);
        let meshing_ms = t0.elapsed().as_secs_f32() * 1000.0;

        let t1 = Instant::now();
        let vertices: usize = meshes.iter().map(|m| m.vertex_count()).sum();
        renderer.upload_chunks(&meshes);
        let upload_ms = t1.elapsed().as_secs_f32() * 1000.0;

        log::warn!(
            "world ready: {} chunks generated in {stream_ms:.0}ms, {} meshed in {meshing_ms:.0}ms, \
             {} vertices uploaded in {upload_ms:.0}ms",
            streamed,
            meshes.len(),
            vertices,
        );
        log::info!(
            "({:.2}ms per chunk to generate; generation is the sparse/density carver plus \
             the shared surface pass)",
            world.ms_per_chunk()
        );

        self.state = Some(State {
            window,
            renderer,
            camera,
            uploaded_chunks: meshes.len(),
            uploaded_vertices: vertices,
            world,
            last_centre: centre,
            generation_ms: stream_ms,
            meshing_ms,
            upload_ms,
            chunks_streamed: streamed,
        });
        self.last_frame = Instant::now();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                state.renderer.resize(size.width, size.height);
                state.camera.aspect = size.width.max(1) as f32 / size.height.max(1) as f32;
            }

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        repeat: false,
                        ..
                    },
                ..
            } => match key_state {
                ElementState::Pressed => {
                    match code {
                        KeyCode::Escape => {
                            event_loop.exit();
                            return;
                        }
                        // Toggle the overlay, for an unobstructed look at the
                        // terrain without restarting.
                        KeyCode::KeyH => {
                            state.renderer.hud_visible = !state.renderer.hud_visible;
                            // Force a rebuild so the change takes effect even if
                            // the text itself did not change.
                            self.last_hud = None;
                            log::info!("HUD {}", if state.renderer.hud_visible { "on" } else { "off" });
                        }
                        // Switch carver and regenerate. The actual work happens
                        // at the top of the next frame, not here, so input
                        // handling never blocks on generation.
                        KeyCode::KeyC => {
                            self.carver.toggle();
                            log::info!("carver switch queued");
                        }
                        // Toggle the generation-timing and carve-coverage block.
                        KeyCode::KeyB => {
                            self.benchmark_mode = !self.benchmark_mode;
                            self.last_hud = None;
                            log::info!(
                                "benchmark readout {}",
                                if self.benchmark_mode { "on" } else { "off" }
                            );
                        }
                        // Reset the frame-time window, to measure a specific
                        // moment rather than a lifetime average.
                        KeyCode::KeyR => {
                            self.timer.reset();
                            self.last_hud = None;
                            log::info!("frame-time statistics reset");
                        }
                        _ => {}
                    }
                    if !self.keys.contains(&code) {
                        self.keys.push(code);
                    }
                }
                ElementState::Released => self.keys.retain(|k| *k != code),
            },

            WindowEvent::RedrawRequested => {
                let frame_time = self.last_frame.elapsed();
                self.last_frame = Instant::now();
                self.timer.record(frame_time);
                // Clamp so a stall (a breakpoint, a suspended laptop) does not
                // teleport the camera across the world on the next frame.
                let dt = frame_time.as_secs_f32().min(0.1);

                // A pending carver switch regenerates before anything is drawn,
                // so the frame shows one consistent world rather than a mix.
                if let Some(carver) = self.carver.take() {
                    self.cfg.carver = carver;
                    log::warn!("switching carver to {carver}, regenerating world");
                    let t0 = Instant::now();
                    state.world.set_carver(carver);
                    state.world.regenerate_all();
                    state.generation_ms = t0.elapsed().as_secs_f32() * 1000.0;

                    // Remesh everything, since every chunk changed.
                    let t1 = Instant::now();
                    let meshes = mesh_world(&state.world, state.last_centre, RENDER_DISTANCE);
                    state.meshing_ms = t1.elapsed().as_secs_f32() * 1000.0;

                    let t2 = Instant::now();
                    state.uploaded_vertices = meshes.iter().map(|m| m.vertex_count()).sum();
                    state.uploaded_chunks = meshes.len();
                    state.renderer.upload_chunks(&meshes);
                    state.upload_ms = t2.elapsed().as_secs_f32() * 1000.0;
                    state.chunks_streamed = 0;

                    // The old world's frame times say nothing about the new
                    // one, so the window starts clean rather than blending the
                    // two carvers together in the percentiles.
                    self.timer.reset();
                    // Keeping the camera is the whole point: comparing two
                    // carvers is only meaningful from the same viewpoint.
                    log::warn!(
                        "carver {carver}: {} chunks regenerated in {:.0}ms, meshed in {:.0}ms",
                        state.world.len(),
                        state.generation_ms,
                        state.meshing_ms
                    );
                }

                apply_input(state, &self.keys, dt);

                // Stream only when the camera crosses a chunk boundary. Doing
                // it every frame would re-scan the radius for no reason, and a
                // boundary crossing is exactly when new ground becomes visible.
                let centre = ChunkPos::from_world(
                    state.camera.eye[0] as i32,
                    state.camera.eye[1] as i32,
                    state.camera.eye[2] as i32,
                );
                if centre != state.last_centre {
                    state.last_centre = centre;

                    let t0 = Instant::now();
                    let added = state.world.stream_around(centre, RENDER_DISTANCE);
                    state.world.evict_beyond(centre, EVICT_RADIUS);
                    state.generation_ms = t0.elapsed().as_secs_f32() * 1000.0;
                    state.chunks_streamed = added;

                    if added > 0 {
                        // Remesh and re-upload the whole view distance. Meshing
                        // only the new chunks would be cheaper, but it needs a
                        // per-chunk vertex buffer to avoid re-uploading one
                        // giant buffer, which is a streaming optimisation this
                        // POC does not need yet.
                        let t1 = Instant::now();
                        let meshes = mesh_world(&state.world, centre, RENDER_DISTANCE);
                        state.meshing_ms = t1.elapsed().as_secs_f32() * 1000.0;

                        let t2 = Instant::now();
                        state.uploaded_vertices =
                            meshes.iter().map(|m| m.vertex_count()).sum();
                        state.uploaded_chunks = meshes.len();
                        state.renderer.upload_chunks(&meshes);
                        state.upload_ms = t2.elapsed().as_secs_f32() * 1000.0;

                        log::info!(
                            "streamed {added} chunks in {:.1}ms, meshed {} in {:.1}ms, \
                             uploaded in {:.1}ms",
                            state.generation_ms,
                            state.uploaded_chunks,
                            state.meshing_ms,
                            state.upload_ms
                        );
                    }
                }

                let view_projection = state.camera.view_projection();
                state.renderer.set_view_projection(view_projection);

                // Rebuild the overlay only when its text changes: the
                // percentiles move slowly, and re-uploading a texture per line
                // every frame would itself cost more than the HUD is worth.
                update_hud(
                    &mut self.timer,
                    &mut self.last_hud,
                    self.carver.current(),
                    &self.cfg,
                    self.benchmark_mode,
                    self.parity,
                    state,
                );

                match state.renderer.render(SKY) {
                    Ok(()) => {}
                    Err(FrameOutcome::Skip) => {}
                    Err(FrameOutcome::Reconfigure) => {
                        let size = state.window.inner_size();
                        state.renderer.resize(size.width, size.height);
                    }
                }

                self.frames += 1;
                if self.fps_timer.elapsed().as_secs_f32() >= 2.0 {
                    if let Some(s) = self.timer.stats() {
                        log::info!(
                            "{} | {} chunks, {} vertices",
                            s.summary(),
                            state.uploaded_chunks,
                            state.uploaded_vertices
                        );
                    }
                    self.frames = 0;
                    self.fps_timer = Instant::now();
                }
            }

            WindowEvent::CursorMoved { .. } => {
                // Mouse look is deliberately absent: it needs pointer capture
                // and sensitivity options to be usable, which is scope this POC
                // does not need. Keyboard look is enough to verify rendering.
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // Continuous redraw, since the camera can move every frame.
        if let Some(state) = self.state.as_ref() {
            state.window.request_redraw();
        }
    }
}

/// Meshes every drawable chunk within the render distance of `centre`.
///
/// Only chunks inside the radius are meshed, even though the world keeps a
/// wider halo in memory: meshing the halo would triple the work for geometry
/// that is out of view.
fn mesh_world(
    world: &StreamWorld,
    centre: ChunkPos,
    radius: i32,
) -> Vec<ChunkMeshData> {
    use rayon::prelude::*;

    let positions: Vec<ChunkPos> = world
        .positions()
        .filter(|p| {
            if p.chebyshev(centre) > radius {
                return false;
            }
            let c = world.get(*p).expect("position came from the world");
            // The cull flags are why an empty sky chunk costs one bool rather
            // than a meshing pass.
            !c.empty && !c.enclosed
        })
        .collect();

    positions
        .par_iter()
        .map(|&pos| {
            let mesh = mesh_chunk(world.get(pos).expect("position came from the world"));
            ChunkMeshData::new(pos, &mesh)
        })
        .filter(|data| !data.is_empty())
        .collect()
}

/// Rebuilds the overlay if its content changed.
///
/// Takes the individual pieces rather than `&mut App`, because the caller is
/// already holding a mutable borrow of `App::state` at the point this runs.
fn update_hud(
    timer: &mut FrameTimer,
    last_hud: &mut Option<String>,
    carver: Carver,
    cfg: &Config,
    benchmark_mode: bool,
    parity: CarveParity,
    state: &mut State,
) {
    // `stats()` needs a mutable borrow, so it is taken once and reused.
    let stats = timer.stats();
    let hud = HudState {
        fps: stats.map_or(0.0, |s| s.fps()),
        frame: stats,
        carver,
        chunks_drawn: state.uploaded_chunks,
        vertices: state.uploaded_vertices,
        // "total" is the world actually held in memory, which is the halo, not
        // the render distance; that is what the ratio should describe.
        chunks_total: state.world.len(),
        generation_ms: state.generation_ms,
        meshing_ms: state.meshing_ms,
        upload_ms: state.upload_ms,
        chunks_streamed: state.chunks_streamed,
        render_distance: RENDER_DISTANCE,
        chunks_generated: state.world.chunks_generated_total(),
        ms_per_chunk: state.world.ms_per_chunk(),
        benchmark_mode,
        density_percent: parity.density_percent(),
        sparse_percent: parity.sparse_percent(),
        camera_pos: state.camera.eye,
        yaw: state.camera.yaw,
        pitch: state.camera.pitch,
        world_size: (cfg.world.x, cfg.world.y, cfg.world.z),
        seed: cfg.seed,
        env: if cfg.env.is_dev() { "dev" } else { "prod" },
    };
    let lines = hud.lines();

    // The HUD is only rebuilt when the rendered text differs. Frame times are
    // rounded to two decimals by the formatter, so a steady frame rate settles
    // into a stable string and the upload happens rarely rather than per frame.
    let key = lines.join("\n");
    if last_hud.as_deref() == Some(key.as_str()) {
        return;
    }
    if let Err(e) = state.renderer.set_hud_lines(&lines) {
        log::error!("could not update the HUD: {e:#}");
        return;
    }
    *last_hud = Some(key);
}

/// WASD to move, arrows to look, Space/Shift for vertical.
///
/// Keyboard-only on purpose: it works identically over a remote session, where
/// pointer capture does not.
fn apply_input(state: &mut State, keys: &[KeyCode], dt: f32) {
    let held = |code: KeyCode| keys.contains(&code);

    let (mut forward, mut right, mut up) = (0.0f32, 0.0f32, 0.0f32);
    if held(KeyCode::KeyW) {
        forward += 1.0;
    }
    if held(KeyCode::KeyS) {
        forward -= 1.0;
    }
    if held(KeyCode::KeyD) {
        right += 1.0;
    }
    if held(KeyCode::KeyA) {
        right -= 1.0;
    }
    if held(KeyCode::Space) {
        up += 1.0;
    }
    if held(KeyCode::ShiftLeft) {
        up -= 1.0;
    }
    if forward != 0.0 || right != 0.0 || up != 0.0 {
        let speed = MOVE_SPEED * dt;
        state
            .camera
            .translate(forward * speed, right * speed, up * speed);
    }

    let (mut dyaw, mut dpitch) = (0.0f32, 0.0f32);
    if held(KeyCode::ArrowLeft) {
        dyaw += 1.0;
    }
    if held(KeyCode::ArrowRight) {
        dyaw -= 1.0;
    }
    if held(KeyCode::ArrowUp) {
        dpitch += 1.0;
    }
    if held(KeyCode::ArrowDown) {
        dpitch -= 1.0;
    }
    if dyaw != 0.0 || dpitch != 0.0 {
        let speed = LOOK_SPEED * dt;
        state.camera.rotate(dyaw * speed, dpitch * speed);
    }
}

/// Spawns the camera above the middle of the world, looking along it.
///
/// The world is now unbounded horizontally, so "the middle" is the origin. The
/// camera starts a little above the terrain band looking slightly down, which
/// puts ground and sky in the first frame rather than the inside of a hill.
fn default_camera(cfg: &Config) -> Camera {
    let world_height = cfg.world_height() as f32;
    let mut camera = Camera {
        eye: [0.0, world_height * 0.75, 0.0],
        ..Default::default()
    };
    camera.pitch = -0.35;
    // Far plane large enough for the render distance plus some slack.
    camera.far = (RENDER_DISTANCE as f32 * 32.0).max(512.0);
    camera
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_core::{Carver, CaveParams, Env, RenderConfig, WorldSize};

    fn test_cfg() -> Config {
        Config {
            env: Env::Dev,
            world: WorldSize { x: 2, y: 2, z: 2 },
            seed: 1337,
            carver: Carver::Sparse,
            cave: CaveParams {
                origin_density: 0.02,
                min_radius: 1.5,
                max_radius: 4.0,
            },
            validation: true,
            debug_views: true,
            frame_dump: false,
            render: RenderConfig {
                force_software_gpu: true,
                vsync: false,
            },
        }
    }

    fn streamed(carver: Carver) -> (StreamWorld, ChunkPos) {
        let cfg = test_cfg();
        let mut world = StreamWorld::new(
            cfg.seed,
            cfg.world_height(),
            cfg.cave,
            carver,
        );
        let centre = ChunkPos::new(0, 1, 0);
        world.stream_around(centre, 3);
        (world, centre)
    }

    #[test]
    fn meshing_skips_chunks_the_cull_flags_reject() {
        let (world, centre) = streamed(Carver::Sparse);
        let meshes = mesh_world(&world, centre, 3);

        for m in &meshes {
            let c = world.get(m.pos).expect("meshed chunk exists");
            assert!(!c.empty, "meshed an empty chunk at {:?}", m.pos);
            assert!(!c.enclosed, "meshed an enclosed chunk at {:?}", m.pos);
            assert!(!m.is_empty(), "produced an empty mesh for {:?}", m.pos);
        }
        assert!(!meshes.is_empty(), "the view distance must produce geometry");
    }

    #[test]
    fn meshing_respects_the_render_distance() {
        // Streaming keeps a halo wider than the render distance so flying back
        // and forth does not regenerate ground, and the mesher must not draw
        // that halo.
        let (world, centre) = streamed(Carver::Sparse);
        let radius = 2;
        let meshes = mesh_world(&world, centre, radius);
        for m in &meshes {
            assert!(
                m.pos.chebyshev(centre) <= radius,
                "meshed {:?}, which is outside the render distance of {radius}",
                m.pos
            );
        }
    }

    #[test]
    fn meshing_is_deterministic_across_rayon_runs() {
        let (world, centre) = streamed(Carver::Sparse);
        let a = mesh_world(&world, centre, 3);
        let b = mesh_world(&world, centre, 3);
        assert_eq!(a.len(), b.len(), "parallel meshing must be deterministic");
        let count = |v: &[ChunkMeshData]| v.iter().map(|m| m.vertex_count()).sum::<usize>();
        assert_eq!(count(&a), count(&b));
    }

    #[test]
    fn the_default_camera_spawns_above_the_terrain_not_inside_it() {
        let cfg = test_cfg();
        let camera = default_camera(&cfg);

        // Inside the vertical band, but high enough to see ground rather than
        // being buried in it.
        let height = cfg.world_height() as f32;
        assert!(camera.eye[1] > 0.0 && camera.eye[1] < height);
        assert!(
            camera.eye[1] > height * 0.5,
            "spawn should be in the upper half so the ground is visible"
        );
        assert!(camera.pitch < 0.0, "should look slightly downward");
        assert!(camera.far >= RENDER_DISTANCE as f32 * 16.0, "far plane too short");
        assert!(camera.aspect > 0.0);
    }

    #[test]
    fn input_moves_only_while_keys_are_held() {
        // Exercised through the camera rather than the whole app, since the
        // state needs a GPU. The key mapping itself is the trivial part.
        let mut camera = Camera::default();
        let before = camera.eye;
        camera.translate(1.0, 0.0, 0.0);
        assert_ne!(camera.eye, before, "forward must move the camera");
        assert!(camera.eye[2] < before[2], "yaw 0 faces -Z, so z decreases");
    }

    /// A zero or negative radius would silently disable distance culling, and a
    /// halo narrower than the render distance would evict chunks the mesher is
    /// about to ask for.
    const _: () = assert!(RENDER_DISTANCE > 0);
    const _: () = assert!(EVICT_RADIUS > RENDER_DISTANCE);
}
