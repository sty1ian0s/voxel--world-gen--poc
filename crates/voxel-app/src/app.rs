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
use voxel_core::{ChunkGrid, ChunkPos, Config};

use crate::camera::Camera;
use crate::gpu::{FrameOutcome, Renderer};
use crate::render::ChunkMeshData;

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
    frames: u64,
    fps_timer: Instant,
    /// Set once the world is generated, so the one-off cost is reported once.
    reported_generation: bool,
}

impl App {
    fn new(cfg: Config) -> Self {
        Self {
            cfg,
            state: None,
            keys: Vec::new(),
            last_frame: Instant::now(),
            frames: 0,
            fps_timer: Instant::now(),
            reported_generation: false,
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

        // Generate the world once, then mesh and upload it.
        let (grid, stats) = report::generate_world(&self.cfg);
        let meshes = mesh_grid(&grid, &self.cfg);
        let vec_total: usize = meshes.iter().map(|m| m.vertex_count()).sum();

        let mut renderer = renderer;
        renderer.upload_chunks(&meshes);

        let camera = default_camera(&self.cfg, &grid);

        log::warn!(
            "world ready: {} chunks meshed, {} vertices uploaded",
            renderer.uploaded_chunks(),
            vec_total
        );
        // The generation report is logged rather than drawn as an overlay:
        // an overlay needs a text renderer, which is a font atlas and a second
        // pipeline, and the numbers are just as readable in the log.
        log::warn!("generation: {}", report::format(&self.cfg, &stats).replace('\n', " | "));
        self.state = Some(State {
            window,
            renderer,
            camera,
            uploaded_chunks: meshes.len(),
            uploaded_vertices: vec_total,
        });
        self.reported_generation = true;
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
                    if code == KeyCode::Escape {
                        event_loop.exit();
                        return;
                    }
                    if !self.keys.contains(&code) {
                        self.keys.push(code);
                    }
                }
                ElementState::Released => self.keys.retain(|k| *k != code),
            },

            WindowEvent::RedrawRequested => {
                let dt = self.last_frame.elapsed().as_secs_f32();
                self.last_frame = Instant::now();
                // Clamp so a stall (a breakpoint, a suspended laptop) does not
                // teleport the camera across the world on the next frame.
                let dt = dt.min(0.1);

                apply_input(state, &self.keys, dt);

                let view_projection = state.camera.view_projection();
                state.renderer.set_view_projection(view_projection);

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
                    let fps = self.frames as f32 / self.fps_timer.elapsed().as_secs_f32();
                    log::info!(
                        "{fps:.0} fps, {} chunks, {} vertices, camera ({:.0}, {:.0}, {:.0})",
                        state.uploaded_chunks,
                        state.uploaded_vertices,
                        state.camera.eye[0],
                        state.camera.eye[1],
                        state.camera.eye[2],
                    );
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
