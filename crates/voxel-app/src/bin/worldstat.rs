//! `worldstat` — cave shape statistics over a large, streamed region.
//!
//! `cavestat` measures an 8x8x8 chunk world in RAM. That box is small enough to
//! fit and too small to be interesting: Minecraft's trunks run 85-112 blocks, so
//! a 128-voxel-wide box cannot contain one, and connectivity measured inside it
//! is connectivity inside a crop. This binary measures the same statistics over a
//! region the in-RAM path cannot hold at all, one chunk at a time.
//!
//! The numbers it prints are **not** the same scope as `cavestat`'s, and the
//! output says so for every row. See `debug::window`'s module docs for the exact
//! global-vs-windowed table.
//!
//! ```text
//! cargo run --release -p voxel-app --bin worldstat
//! cargo run --release -p voxel-app --bin worldstat -- --side 64 --rows 4
//! APP_ENV=prod cargo run --release -p voxel-app --bin worldstat -- --side 256
//! ```
//!
//! Memory is a function of `--side` and `--rows`, never of the region's area in
//! the way the in-RAM path is: run it at `--side 64` and `--side 256` and
//! `/usr/bin/time -v` reports the same peak RSS.

use std::time::Instant;

use voxel_core::config::{CaveParams, Carver, Config};
use voxel_core::debug::window::{Mask, WindowAccumulator, WindowStats};
use voxel_core::world::{CarverSpec, Region, WorldGen};

/// Chunks per side by default. Large enough to hold several MC-length trunks and
/// small enough to finish in seconds.
const DEFAULT_SIDE: i32 = 256;

/// Window height in chunk rows. Bounds memory; does not change which numbers are
/// exact.
const DEFAULT_ROWS: i32 = 8;

fn main() {
    let cfg = Config::load().unwrap_or_else(|e| {
        eprintln!("config: {e}");
        std::process::exit(2);
    });
    let opts = Options::parse(std::env::args().skip(1).collect());
    if opts.help {
        print_help();
        return;
    }

    let side = opts.side.unwrap_or(DEFAULT_SIDE);
    let rows = opts.rows.unwrap_or(DEFAULT_ROWS);
    let world_height = cfg.world_height();
    let seed = opts.seed.unwrap_or(cfg.seed);

    println!(
        "worldstat  {side}x{side} chunks ({:.0}M voxels), band {} chunks ({} voxels), seed {seed}",
        (side as f64 * 16.0).powi(2) * world_height as f64 / 1e6,
        world_height.div_ceil(16),
        world_height,
    );
    println!(
        "window     {rows} chunk rows; memory is a function of side and rows, not of the region's z extent"
    );
    println!("scope      every row below is labelled global or windowed\n");

    let carvers: Vec<(&str, Carver)> = match opts.carver {
        Some(c) => vec![(name_of(c), c)],
        // Every carver is behind the enum now, so the default sweep is all of
        // them rather than a hand-written pair.
        None => Carver::ALL.iter().map(|&c| (c.name(), c)).collect(),
    };

    println!(
        "{:<9} {:>7} {:>9} {:>10} {:>9} {:>8} {:>8} {:>9} {:>8}",
        "carver", "void%", "windowed", "global p50", "global p99", "trunk", "s/1k-ch", "gen_ms", "chunks"
    );
    println!(
        "{:<9} {:>7} {:>9} {:>10} {:>9} {:>8} {:>8} {:>9} {:>8}",
        "", "", "conn%", "size", "size", "", "", "", ""
    );
    println!("{}", "-".repeat(88));
    println!(
        "conn% is CHUNK-GRANULAR and over-merges: it merges any two cave chunks that share a\n\
         chunk, so it reads far higher than cavestat's voxel-accurate connectivity. The two are\n\
         NOT comparable. Use cavestat for MC-likeness; use this for carver-vs-carver shape at scale."
    );
    println!("{}", "-".repeat(88));

    for (name, carver) in carvers {
        let spec = CarverSpec {
            seed,
            carver,
            cave: cfg.cave,
            world_height,
        };
        let mut generator = WorldGen::new(spec);
        let mut acc = WindowAccumulator::new(rows, world_height, Mask::Underground);
        let region = Region::square(side);

        let start = Instant::now();
        acc.run(&mut generator, &region);
        let elapsed = start.elapsed();
        let stats = acc.into_stats();

        println!("{}", row(name, &stats, elapsed, region.columns()));
    }

    println!(
        "\nif the two sides below are equal, peak RSS is independent of the region's area"
    );
    println!("  /usr/bin/time -v target/release/worldstat --side 64   | grep 'Maximum resident'");
    println!("  /usr/bin/time -v target/release/worldstat --side 256  | grep 'Maximum resident'");
}

fn row(name: &str, s: &WindowStats, elapsed: std::time::Duration, columns: u64) -> String {
    let chunks = s.chunks.max(1);
    let per_1k = elapsed.as_secs_f64() * 1000.0 / chunks as f64 * 1000.0;
    format!(
        "{:<9} {:>6.2}% {:>8.1}% {:>10} {:>9} {:>8} {:>8.1} {:>9.0} {:>8}\n\
         {:<9} components(windowed) {} (>=16: {}) | footprint {} cols | depth/col {:.2} | mean col depth {:.2}\n\
         {:<9} solid(global) {} | cave(global) {} | columns {} | empty chunks {} | size p90 {} max {} | slabs {}",
        name,
        s.void_fraction() * 100.0,
        s.chunk_connected_share() * 100.0,
        s.size_p50,
        s.size_p99,
        s.trunk_length_in_window,
        per_1k,
        elapsed.as_secs_f64() * 1000.0,
        s.chunks,
        name,
        s.components_in_window,
        s.components_ge_16_in_window,
        s.footprint_columns_in_window,
        s.largest_component_in_window as f64 / s.footprint_columns_in_window.max(1) as f64,
        s.cave_voxels as f64 / columns.max(1) as f64,
        name,
        s.solid_voxels,
        s.cave_voxels,
        s.columns,
        s.empty_chunks,
        s.size_p90,
        s.size_max,
        s.slabs,
    )
}

fn name_of(carver: Carver) -> &'static str {
    // One spelling, defined on the enum, so this cannot drift from `.env`.
    carver.name()
}

#[derive(Debug, Default)]
struct Options {
    side: Option<i32>,
    rows: Option<i32>,
    seed: Option<u32>,
    carver: Option<Carver>,
    help: bool,
}

impl Options {
    fn parse(args: Vec<String>) -> Self {
        let mut o = Self::default();
        let mut it = args.into_iter();
        while let Some(a) = it.next() {
            match a.as_str() {
                "--side" | "-s" => o.side = it.next().and_then(|v| v.parse().ok()),
                "--rows" | "-r" => o.rows = it.next().and_then(|v| v.parse().ok()),
                "--seed" => o.seed = it.next().and_then(|v| v.parse().ok()),
                "--carver" => o.carver = it.next().and_then(|v| v.parse::<Carver>().ok()),
                "--help" | "-h" => o.help = true,
                other => {
                    eprintln!("unknown argument {other:?}");
                    o.help = true;
                }
            }
        }
        o
    }
}

fn print_help() {
    println!(
        "worldstat — streamed cave shape statistics\n\n\
         USAGE:\n  worldstat [--side N] [--rows N] [--seed N] [--carver density|sparse]\n\n\
         OPTIONS:\n\
         \x20 --side N    region width and depth in chunks (default {DEFAULT_SIDE})\n\
         \x20 --rows N    window height in chunk rows (default {DEFAULT_ROWS})\n\
         \x20 --seed N    world seed (default: VOXEL_WORLD_SEED)\n\
         \x20 --carver C  measure one carver instead of both\n\n\
         Memory is set by --side and --rows; the region's area does not add to it."
    );
}

/// Every cave parameter the metrics depend on, for a caller that wants to print
/// its own provenance line.
pub fn cave_params_of(cfg: &Config) -> CaveParams {
    cfg.cave
}
