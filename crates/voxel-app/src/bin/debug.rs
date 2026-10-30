//! `debug` — headless generation inspector.
//!
//! Renders the configured world's generation, geometry and lighting as text and
//! image views, and prints the work counters behind the performance claims.
//! Needs no GPU and no window, so it runs over SSH and in CI.
//!
//! ```text
//! cargo run -p voxel-app --bin debug -- --help
//! cargo run -p voxel-app --bin debug -- --carver sparse --slice 64
//! cargo run -p voxel-app --bin debug -- --compare --out-debug debug-out
//! ```

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use voxel_core::config::{Carver, Config};
use voxel_core::ChunkGrid;
use voxel_app::logging;
use voxel_core::debug::{ascii, image, report};

fn main() -> anyhow::Result<()> {
    let cfg = Config::load().context("loading configuration")?;
    logging::init(cfg.env);

    let args: Vec<String> = std::env::args().skip(1).collect();
    let opts = match Options::parse(&args) {
        Ok(Some(o)) => o,
        Ok(None) => {
            print_help();
            return Ok(());
        }
        Err(e) => {
            eprintln!("error: {e}\n");
            print_help();
            std::process::exit(2);
        }
    };

    // A debug dump of a prod-sized world is slow and fills the disk, so the
    // configured size is used unless the caller overrides it explicitly.
    let mut cfg = cfg;
    if let Some((x, y, z)) = opts.world {
        cfg.world = voxel_core::WorldSize { x, y, z };
    }
    if let Some(c) = opts.carver {
        cfg.carver = c;
    }
    cfg.debug_views = true;

    log::info!("{}", cfg.summary());

    let (grid, stats) = report::generate_world(&cfg);
    println!("{}", report::format(&cfg, &stats));

    let carved = report::carved_voxel_count(&grid);
    println!("caves      {carved} carved voxels (air not open to the sky)");

    if opts.compare {
        println!("\n{}", compare(&cfg));
    }

    if opts.slice.is_none() && !opts.images {
        return Ok(());
    }

    // One representative level inside the terrain if the caller gave none:
    // the middle of the world vertically is nearly always underground.
    let level = opts.slice.unwrap_or((cfg.world.y * 16 / 2) as i32);

    if let Some(level) = opts.slice {
        println!("\n--- blocks, horizontal slice y={level} ---");
        print!("{}", ascii::blocks_slice(&grid, level).to_string_with_header());

        println!("\n--- cross-section, z={} ---", level / 2);
        print!(
            "{}",
            ascii::blocks_slice_xz(&grid, level / 2).to_string_with_header()
        );

        println!("\n--- carved air in slice y={level} ('o' = enclosed by solid) ---");
        print!("{}", ascii::carve_slice(&grid, level).to_string_with_header());

        println!("\n--- sunlight, horizontal slice y={level} ---");
        print!(
            "{}",
            ascii::sunlight_slice(&grid, level).to_string_with_header()
        );
    }

    if opts.images {
        let out = opts.out.clone();
        std::fs::create_dir_all(&out).with_context(|| format!("creating {}", out.display()))?;
        write_images(&out, &grid, level)?;
        println!("\nimages     wrote PPM/PGM cross-sections to {}", out.display());
    }

    Ok(())
}

fn write_images(out: &Path, grid: &ChunkGrid, level: i32) -> Result<()> {
    let blocks = out.join(format!("blocks-y{level}.ppm"));
    let cross = out.join(format!("cross-z{}.ppm", level / 2));
    let light = out.join(format!("sunlight-y{level}.pgm"));

    image::write_blocks_slice_ppm(&blocks, grid, level)?;
    image::write_cross_section_ppm(&cross, grid, level / 2)?;
    image::write_sunlight_slice_pgm(&light, grid, level)?;

    for p in [&blocks, &cross, &light] {
        log::debug!("wrote {}", p.display());
    }
    Ok(())
}

fn compare(cfg: &Config) -> String {
    let (dense, sparse) = report::compare_carvers(cfg);
    let mut s = String::from("--- carver comparison (same world, same Pass 1) ---\n");
    s.push_str(&format!(
        "density  {:>10} noise samples  {:>8.1}ms\n",
        dense.density_samples,
        dense.elapsed.as_secs_f64() * 1000.0
    ));
    s.push_str(&format!(
        "sparse   {:>10} SDF voxels     {:>8.1}ms\n",
        sparse.sparse_voxels_tested,
        sparse.elapsed.as_secs_f64() * 1000.0
    ));
    let ratio = dense.density_samples as f64 / sparse.sparse_voxels_tested.max(1) as f64;
    s.push_str(&format!("work ratio: {ratio:.1}x fewer carver operations\n"));
    s
}

#[derive(Debug, Default)]
struct Options {
    slice: Option<i32>,
    images: bool,
    compare: bool,
    out: PathBuf,
    world: Option<(u32, u32, u32)>,
    carver: Option<Carver>,
}

impl Options {
    /// Returns `Ok(None)` when help was requested.
    fn parse(args: &[String]) -> Result<Option<Self>> {
        let mut o = Options {
            out: PathBuf::from("debug-out"),
            ..Default::default()
        };
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "-h" | "--help" => return Ok(None),
                "--slice" => {
                    let v = args.get(i + 1).context("--slice needs a Y level")?;
                    o.slice = Some(v.parse().context("--slice must be an integer")?);
                    i += 2;
                }
                "--world" => {
                    let v = args.get(i + 1).context("--world needs X,Y,Z")?;
                    let parts: Vec<&str> = v.split(',').collect();
                    anyhow::ensure!(parts.len() == 3, "--world must be X,Y,Z");
                    o.world = Some((
                        parts[0].trim().parse().context("world X")?,
                        parts[1].trim().parse().context("world Y")?,
                        parts[2].trim().parse().context("world Z")?,
                    ));
                    i += 2;
                }
                "--carver" => {
                    let v = args.get(i + 1).context("--carver needs density|sparse")?;
                    o.carver = Some(v.parse().map_err(|e: voxel_core::ConfigError| {
                        anyhow::anyhow!(e.to_string())
                    })?);
                    i += 2;
                }
                "--out-debug" => {
                    o.out = PathBuf::from(args.get(i + 1).context("--out-debug needs a path")?);
                    i += 2;
                }
                "--images" => {
                    o.images = true;
                    i += 1;
                }
                "--compare" => {
                    o.compare = true;
                    i += 1;
                }
                other => anyhow::bail!("unknown argument {other:?}"),
            }
        }
        Ok(Some(o))
    }
}

fn print_help() {
    println!(
        "\
voxel debug — inspect world generation without a GPU

USAGE:
    cargo run -p voxel-app --bin debug -- [OPTIONS]

OPTIONS:
    --slice <Y>          Print ASCII views at world level Y
    --images             Write PPM/PGM cross-sections
    --out-debug <DIR>    Where to write images [default: debug-out]
    --world <X,Y,Z>      Override world size in chunks
    --carver <NAME>      density | sparse
    --compare            Generate with both carvers and report the work ratio
    -h, --help           Show this help

ENVIRONMENT:
    APP_ENV              dev (default) | prod, selects the .env overlay
    VOXEL_CARVER         density | sparse
    VOXEL_WORLD_X/Y/Z    World size in chunks
    RUST_LOG             Log filter, e.g. voxel_core=debug

EXAMPLES:
    # Read a slice by eye
    cargo run -p voxel-app --bin debug -- --slice 60

    # Same world, both carvers, with the work counters
    cargo run -p voxel-app --bin debug -- --world 4,4,4 --compare

    # Cross-sections for an image viewer
    cargo run -p voxel-app --bin debug -- --slice 60 --images
"
    );
}
