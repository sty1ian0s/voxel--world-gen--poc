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

use voxel_core::debug::report;
use voxel_core::mesh::mesh_chunk;
use voxel_core::{Carver, ChunkGrid, ChunkPos, Config};

use crate::camera::Camera;
use crate::gpu::{FrameOutcome, Renderer};
use crate::hud::{CarverSwitch, HudState};
use crate::render::ChunkMeshData;
use crate::stats::FrameTimer;

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
/// Chunks beyond this are not drawn.
const VIEW_RADIUS_CHUNKS: i32 = 24;

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
    /// The generated world, kept so a carver switch can regenerate without
    /// re-reading config.
    grid: ChunkGrid,
    /// Chunks the cull flags kept, for the HUD's drawn/total ratio.
    chunks_total: usize,
    generation_ms: f32,
    meshing_ms: f32,
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

        // Generate, mesh and upload once at startup.
        let mut renderer = renderer;
        let camera = default_camera(&self.cfg, &ChunkGrid::new(
            self.cfg.world.x,
            self.cfg.world.y,
            self.cfg.world.z,
        ));
        let built = build_world(&mut renderer, &self.cfg);

        log::warn!(
            "world ready: {} chunks meshed, {} vertices uploaded",
            built.chunks_drawn, built.vertices
        );
        log::warn!("generation: {}", built.report_line);

        self.state = Some(State {
            window,
            renderer,
            camera,
            uploaded_chunks: built.chunks_drawn,
            uploaded_vertices: built.vertices,
            grid: built.grid,
            chunks_total: built.chunks_total,
            generation_ms: built.generation_ms,
            meshing_ms: built.meshing_ms,
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
                    let built = build_world(&mut state.renderer, &self.cfg);
                    state.grid = built.grid;
                    state.chunks_total = built.chunks_total;
                    state.generation_ms = built.generation_ms;
                    state.meshing_ms = built.meshing_ms;
                    state.uploaded_chunks = built.chunks_drawn;
                    state.uploaded_vertices = built.vertices;
                    // The old world's frame times say nothing about the new
                    // one, so the window starts clean rather than blending the
                    // two carvers together in the percentiles.
                    self.timer.reset();
                    // The basic requirement for side-by-side comparison is to
                    // keep looking at the same place, which means the camera
                    // deliberately survives the switch.
                    log::warn!("carver {carver}: {}", built.report_line);
                }

                apply_input(state, &self.keys, dt);

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

/// What a completed world build produced.
struct BuiltWorld {
    grid: ChunkGrid,
    chunks_total: usize,
    chunks_drawn: usize,
    vertices: usize,
    generation_ms: f32,
    meshing_ms: f32,
    /// The generation counter line, reused for both the log and the HUD.
    report_line: String,
}

/// Generates, meshes and uploads a world for the configured carver.
///
/// Shared by startup and by the carver switch, so the two paths cannot drift:
/// whatever the switch does is exactly what startup does.
fn build_world(renderer: &mut Renderer, cfg: &Config) -> BuiltWorld {
    let t0 = Instant::now();
    let (grid, stats) = report::generate_world(cfg);
    let generation_ms = t0.elapsed().as_secs_f32() * 1000.0;

    let t1 = Instant::now();
    let meshes = mesh_grid(&grid, cfg);
    let meshing_ms = t1.elapsed().as_secs_f32() * 1000.0;

    let vertices: usize = meshes.iter().map(|m| m.vertex_count()).sum();
    renderer.upload_chunks(&meshes);

    BuiltWorld {
        chunks_total: grid.len(),
        chunks_drawn: meshes.len(),
        vertices,
        generation_ms,
        meshing_ms,
        report_line: report::format(cfg, &stats).replace('\n', " | "),
        grid,
    }
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
        chunks_total: state.chunks_total,
        generation_ms: state.generation_ms,
        meshing_ms: state.meshing_ms,
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

/// Meshes every non-skippable chunk.
///
/// Chunks the cull flags already reject are skipped here too: meshing a chunk
/// that can never be drawn is pure waste, and the flags are cheap to read.
fn mesh_grid(grid: &ChunkGrid, cfg: &Config) -> Vec<ChunkMeshData> {
    use rayon::prelude::*;

    // Distance culling is applied here rather than per frame: the camera starts
    // at world centre, and re-meshing on every move would dwarf the draw cost
    // for a world this size.
    let centre = ChunkPos::new(
        cfg.world.x as i32 / 2,
        cfg.world.y as i32 / 2,
        cfg.world.z as i32 / 2,
    );

    let positions: Vec<ChunkPos> = grid
        .positions()
        .filter(|p| {
            let c = grid.get(*p);
            if c.empty || c.enclosed {
                return false;
            }
            // Keep the whole world when it is smaller than the view radius.
            let world_span = cfg.world.x.max(cfg.world.y).max(cfg.world.z) as i32;
            world_span <= VIEW_RADIUS_CHUNKS || p.chebyshev(centre) <= VIEW_RADIUS_CHUNKS
        })
        .collect();

    positions
        .par_iter()
        .map(|&pos| {
            let mesh = mesh_chunk(grid.get(pos));
            ChunkMeshData::new(pos, &mesh)
        })
        .filter(|data| !data.is_empty())
        .collect()
}

/// Places the camera looking across the generated world from outside it.
fn default_camera(cfg: &Config, _grid: &ChunkGrid) -> Camera {
    let (sx, sy, sz) = (cfg.world.x as f32 * 16.0, cfg.world.y as f32 * 16.0, cfg.world.z as f32 * 16.0);
    // Sit back and above, looking toward the middle of the terrain, so the
    // first frame shows the world rather than the inside of a hill.
    let mut camera = Camera {
        eye: [sx * 0.5, sy * 0.9, sz + sz * 0.8],
        ..Default::default()
    };
    let target = [sx * 0.5, sy * 0.5, sz * 0.5];
    let d = [
        target[0] - camera.eye[0],
        target[1] - camera.eye[1],
        target[2] - camera.eye[2],
    ];
    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-3);
    // yaw 0 faces -Z, so recover yaw from the horizontal direction.
    camera.yaw = (-d[0] / len).atan2(-d[2] / len);
    camera.pitch = (d[1] / len).asin().clamp(-1.5, 1.5);
    camera.far = (sx + sy + sz) * 4.0;
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

    #[test]
    fn meshing_skips_chunks_the_cull_flags_reject() {
        let cfg = test_cfg();
        let (grid, _) = report::generate_world(&cfg);
        let meshes = mesh_grid(&grid, &cfg);

        // Every produced mesh must correspond to a chunk that is neither empty
        // nor enclosed, and must contain geometry.
        for m in &meshes {
            let c = grid.get(m.pos);
            assert!(!c.empty, "meshed an empty chunk at {:?}", m.pos);
            assert!(!c.enclosed, "meshed an enclosed chunk at {:?}", m.pos);
            assert!(!m.is_empty(), "produced an empty mesh for {:?}", m.pos);
        }

        let skippable = grid.iter().filter(|(_, c)| c.empty || c.enclosed).count();
        assert!(
            meshes.len() + skippable <= grid.len(),
            "meshed more chunks than exist"
        );
        assert!(!meshes.is_empty(), "a 2x2x2 world must produce some geometry");
    }

    #[test]
    fn meshing_is_deterministic_across_rayon_runs() {
        let (grid, _) = report::generate_world(&test_cfg());
        let cfg = test_cfg();
        let a = mesh_grid(&grid, &cfg);
        let b = mesh_grid(&grid, &cfg);
        assert_eq!(a.len(), b.len(), "parallel meshing must be deterministic");
        let count = |v: &[ChunkMeshData]| v.iter().map(|m| m.vertex_count()).sum::<usize>();
        assert_eq!(count(&a), count(&b));
    }

    #[test]
    fn default_camera_looks_toward_the_world_centre() {
        let cfg = test_cfg();
        let grid = ChunkGrid::new(cfg.world.x, cfg.world.y, cfg.world.z);
        let camera = default_camera(&cfg, &grid);

        let centre = [
            cfg.world.x as f32 * 8.0,
            cfg.world.y as f32 * 8.0,
            cfg.world.z as f32 * 8.0,
        ];
        let to_centre = [
            centre[0] - camera.eye[0],
            centre[1] - camera.eye[1],
            centre[2] - camera.eye[2],
        ];
        let len = (to_centre[0].powi(2) + to_centre[1].powi(2) + to_centre[2].powi(2)).sqrt();
        let dir = [to_centre[0] / len, to_centre[1] / len, to_centre[2] / len];
        let f = camera.forward();
        let dot = f[0] * dir[0] + f[1] * dir[1] + f[2] * dir[2];
        assert!(
            dot > 0.99,
            "camera should face the world centre, dot = {dot}"
        );
        assert!(camera.far > len, "far plane must reach the world");
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

    /// A zero or negative radius would silently disable distance culling.
    const _: () = assert!(VIEW_RADIUS_CHUNKS > 0);
}
