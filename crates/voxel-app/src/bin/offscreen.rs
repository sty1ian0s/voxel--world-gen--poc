//! `offscreen` — renders one frame without a window and writes it as a PPM.
//!
//! This exists because the render path must still be verified on machines that
//! cannot open a window: no display, no compositor, or a sandbox that hides
//! `/dev/dri`. It drives the same [`voxel_app::gpu::Renderer`] pipeline, shader
//! and vertex buffers the windowed binary uses, only into a texture instead of
//! a swap chain, so a green run means the drawing code works.
//!
//! ```text
//! cargo run -p voxel-app --bin offscreen -- debug-out/frame.ppm
//! ```

use std::path::PathBuf;

use anyhow::{Context, Result};
use voxel_app::camera::Camera;
use voxel_app::gpu::Renderer;
use voxel_app::render::ChunkMeshData;
use voxel_core::debug::report;
use voxel_core::mesh::mesh_chunk;
use voxel_core::{Config, WorldSize};

const WIDTH: u32 = 640;
const HEIGHT: u32 = 400;

/// Sky, matching the windowed binary so the two images are comparable.
const SKY: (f32, f32, f32) = (0.53, 0.81, 0.92);

fn main() -> Result<()> {
    let cfg = Config::load().context("loading configuration")?;
    voxel_app::logging::init(cfg.env);

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mesh_view = args.iter().any(|a| a == "--mesh-view");
    let out = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .cloned()
        .unwrap_or_else(|| "debug-out/offscreen.ppm".to_string());

    let mut cfg = cfg;
    // Keep the default world small enough to render in a few seconds on a CPU
    // rasteriser, unless the caller asked for something specific.
    if cfg.world.chunk_count() > 512 {
        cfg.world = WorldSize { x: 4, y: 4, z: 4 };
        log::warn!("offscreen run: world clamped to {} for speed", cfg.world.chunk_count());
    }

    pollster::block_on(run(cfg, PathBuf::from(out), mesh_view))
}

async fn run(cfg: Config, out: PathBuf, mesh_view: bool) -> Result<()> {
    let mut renderer = Renderer::new_offscreen(WIDTH, HEIGHT, true)
        .await
        .context("creating the offscreen renderer")?;

    log::warn!(
        "offscreen renderer: {} ({:?}, {:?})",
        renderer.adapter_info.name,
        renderer.adapter_info.backend,
        renderer.adapter_info.device_type
    );

    let (grid, stats) = report::generate_world(&cfg);
    println!("{}", report::format(&cfg, &stats));

    // Mesh only the chunks the cull flags keep, exactly as the windowed app does.
    let meshes: Vec<ChunkMeshData> = grid
        .positions()
        .filter(|p| {
            let c = grid.get(*p);
            !c.empty && !c.enclosed
        })
        .map(|p| ChunkMeshData::new(p, &mesh_chunk(grid.get(p))))
        .filter(|m| !m.is_empty())
        .collect();

    let vertices: usize = meshes.iter().map(|m| m.vertex_count()).sum();
    println!(
        "meshes     {} chunks, {} vertices ({:.0} per chunk)",
        meshes.len(),
        vertices,
        vertices as f64 / meshes.len().max(1) as f64
    );
    anyhow::ensure!(!meshes.is_empty(), "no geometry to render; nothing to verify");

    renderer.upload_chunks(&meshes);

    // The mesh view draws the greedy quad outlines, so a captured frame shows
    // how much merging actually happened rather than only reporting a count.
    if mesh_view {
        let wire: Vec<ChunkMeshData> = meshes
            .iter()
            .map(|m| ChunkMeshData::new_wireframe(m.pos, &mesh_chunk(grid.get(m.pos))))
            .collect();
        let lines: usize = wire.iter().map(|m| m.vertex_count()).sum();
        println!("wireframe  {} chunks, {} line vertices", wire.len(), lines);
        renderer.upload_wireframe(&wire);
        renderer.wire_visible = true;
    }
    println!(
        "uploaded   {} chunks, {} vertices",
        renderer.uploaded_chunks(),
        renderer.uploaded_vertices()
    );

    // Frame the whole world from outside it, so the image shows terrain shape
    // rather than the inside of a hill.
    let camera = framing_camera(&cfg);
    renderer.set_view_projection(camera.view_projection());

    // Draw the overlay too, so the captured frame proves the HUD pipeline
    // works and not merely that the terrain does. The numbers are synthetic:
    // this binary has no frame loop to measure.
    let hud = sample_hud(&cfg, &stats, meshes.len(), vertices, grid.len());
    renderer
        .set_hud_lines(&hud)
        .context("building the HUD overlay")?;
    println!("hud        {} lines", hud.len());

    let (w, h, rgb) = renderer
        .render_to_rgb(wgpu::Color {
            r: SKY.0 as f64,
            g: SKY.1 as f64,
            b: SKY.2 as f64,
            a: 1.0,
        })
        .context("rendering an offscreen frame")?;

    // A frame that is entirely sky means nothing was drawn: the classic
    // symptom of a winding order that hides every front face.
    let sky = [(SKY.0 * 255.0) as u8, (SKY.1 * 255.0) as u8, (SKY.2 * 255.0) as u8];
    let non_sky = (0..(w * h) as usize)
        .filter(|&i| {
            let p = &rgb[i * 3..i * 3 + 3];
            // Tolerate the sRGB round trip rather than demanding an exact match.
            (p[0] as i32 - sky[0] as i32).abs() > 8
                || (p[1] as i32 - sky[1] as i32).abs() > 8
                || (p[2] as i32 - sky[2] as i32).abs() > 8
        })
        .count();
    let coverage = 100.0 * non_sky as f64 / (w * h) as f64;
    println!("coverage   {coverage:.1}% of {w}x{h} pixels are geometry");
    anyhow::ensure!(
        coverage > 1.0,
        "the frame is {coverage:.1}% geometry; nothing was drawn"
    );

    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    write_ppm(&out, w, h, &rgb)?;
    println!("wrote      {}", out.display());
    Ok(())
}

/// Camera placed back and above the world, looking at its centre.
fn framing_camera(cfg: &Config) -> Camera {
    let (sx, sy, sz) = (
        cfg.world.x as f32 * 16.0,
        cfg.world.y as f32 * 16.0,
        cfg.world.z as f32 * 16.0,
    );
    let mut camera = Camera {
        eye: [sx * 0.5, sy * 1.6, sz * 2.0],
        ..Default::default()
    };
    let target = [sx * 0.5, sy * 0.45, sz * 0.5];
    let d = [
        target[0] - camera.eye[0],
        target[1] - camera.eye[1],
        target[2] - camera.eye[2],
    ];
    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-3);
    // yaw 0 faces -Z, so recover the angles from the direction to the target.
    camera.yaw = (-d[0] / len).atan2(-d[2] / len);
    camera.pitch = (d[1] / len).asin().clamp(-1.5, 1.5);
    camera.far = (sx + sy + sz) * 6.0;
    camera.aspect = WIDTH as f32 / HEIGHT as f32;
    camera
}

/// Representative HUD lines, with plausible numbers.
///
/// A real frame loop supplies these from `FrameTimer`; here they are fixed so
/// the captured image is reproducible and can be diffed.
fn sample_hud(
    cfg: &Config,
    stats: &report::GenReport,
    chunks_drawn: usize,
    vertices: usize,
    chunks_total: usize,
) -> Vec<String> {
    use voxel_app::hud::HudState;
    use voxel_app::stats::FrameTimer;
    use std::time::Duration;

    // A synthetic window so the percentiles have something to show.
    let mut timer = FrameTimer::new();
    for i in 0..200 {
        // Mostly 12ms with an occasional hitch, which is what a real run looks
        // like and which makes the percentiles visibly different from p50.
        let ms = if i % 37 == 0 { 28.0 } else { 12.0 + (i % 5) as f32 * 0.1 };
        timer.record(Duration::from_secs_f32(ms / 1000.0));
    }

    HudState {
        fps: 0.0,
        frame: timer.stats(),
        carver: cfg.carver,
        chunks_drawn,
        vertices,
        chunks_total,
        generation_ms: stats.elapsed.as_secs_f32() * 1000.0,
        meshing_ms: 0.0,
        upload_ms: 0.0,
        chunks_streamed: chunks_drawn,
        render_distance: voxel_app::app::RENDER_DISTANCE,
        chunks_generated: chunks_drawn as u64,
        ms_per_chunk: if chunks_drawn > 0 {
            stats.elapsed.as_secs_f32() * 1000.0 / chunks_drawn as f32
        } else {
            0.0
        },
        // On, so a captured frame documents the timing and coverage readout.
        benchmark_mode: true,
        density_percent: voxel_core::CarveParity::measure(cfg.seed, &cfg.cave, 2).density_percent(),
        sparse_percent: voxel_core::CarveParity::measure(cfg.seed, &cfg.cave, 2).sparse_percent(),
        camera_pos: [16.0, 70.0, 60.0],
        yaw: 0.0,
        pitch: -0.6,
        world_size: (cfg.world.x, cfg.world.y, cfg.world.z),
        seed: cfg.seed,
        env: if cfg.env.is_dev() { "dev" } else { "prod" },
    }
    .lines()
}

fn write_ppm(path: &PathBuf, w: u32, h: u32, rgb: &[u8]) -> Result<()> {
    use std::io::Write;
    let mut f = std::io::BufWriter::new(
        std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?,
    );
    write!(f, "P6\n{w} {h}\n255\n")?;
    f.write_all(rgb)?;
    f.flush()?;
    Ok(())
}
