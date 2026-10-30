//! `carver_bench` — resource usage of both carvers over one realistic world.
//!
//! A plain binary rather than a criterion harness, so the whole run is a single
//! process that `perf stat` can wrap for hardware counters:
//!
//! ```text
//! cargo build -p voxel-app --bin carver_bench
//! perf stat -e instructions,cycles,cache-misses,branch-misses \
//!     ./target/release/carver_bench
//! cargo run --release -p voxel-app --bin carver_bench -- 8
//! ```
//!
//! The two carvers run in two clearly separated phases, sequentially and never
//! concurrently, so each phase's peak RSS is that phase's own peak and a
//! resource comparison between them means something.

use std::fmt::Write as _;

use voxel_core::config::{CarveParity, CaveParams};
use voxel_core::debug::report;
use voxel_core::{Carver, Config, Env, RenderConfig, WorldSize};

/// See [`ticks_per_second`].
const CLOCK_TICKS_PER_SECOND: f64 = 100.0;

/// The density the shipped `.env` uses, and the one the parity test asserts on.
/// Not 0.02: that carved 0.2% of the rock, so both carvers did far less work
/// than in a world anyone actually runs.
const ORIGIN_DENSITY: f32 = 2.0;
const MIN_RADIUS: f32 = 1.5;
const MAX_RADIUS: f32 = 4.0;
const SEED: u32 = 1337;

/// Chunks per side of the world box. 4 gives 64 chunks.
const DEFAULT_SIDE: u32 = 4;

fn main() -> anyhow::Result<()> {
    let side = match std::env::args().nth(1) {
        None => DEFAULT_SIDE,
        Some(a) if a == "-h" || a == "--help" => {
            println!("carver_bench [SIDE]\n\nSIDE  chunks per side of the world box [default: {DEFAULT_SIDE}]");
            return Ok(());
        }
        Some(a) => a
            .parse()
            .map_err(|_| anyhow::anyhow!("SIDE must be a positive integer, got {a:?}"))?,
    };
    anyhow::ensure!(side > 0, "SIDE must be at least 1");

    let params = CaveParams {
        origin_density: ORIGIN_DENSITY,
        min_radius: MIN_RADIUS,
        max_radius: MAX_RADIUS,
    };
    let world = WorldSize {
        x: side,
        y: side,
        z: side,
    };
    let cfg = |carver| Config {
        env: Env::Dev,
        world,
        seed: SEED,
        carver,
        cave: params,
        validation: false,
        debug_views: false,
        frame_dump: false,
        render: RenderConfig {
            force_software_gpu: false,
            vsync: false,
        },
    };

    println!("world      {side}x{side}x{side} chunks = {} ({:.1}M voxels), seed {SEED}",
        world.chunk_count(),
        world.voxel_count() as f64 / 1e6);
    println!("cave       origin_density {ORIGIN_DENSITY} per 1000 voxels^3, radius {MIN_RADIUS}..{MAX_RADIUS}");
    println!("ordering   phase 1 (density) fully finishes and is measured before phase 2 (sparse) starts;");
    println!("           the two never run concurrently, so each phase's peak RSS is its own\n");

    // Phase 1, then phase 2. Not concurrent: the grid built by each `phase` is
    // dropped when it returns, before the next carver allocates.
    let density = phase(&cfg(Carver::Density));
    println!("phase 1/2  density  done");

    let sparse = phase(&cfg(Carver::Sparse));
    println!("phase 2/2  sparse   done");

    // Carve coverage comes from the shared helper rather than a private copy,
    // so this stays in step with the HUD and the parity test.
    let parity = CarveParity::measure(SEED, &params, side as i32);

    print_table(side, &density, &sparse, &parity);
    print_machine(side, &density, &sparse, &parity);
    Ok(())
}

/// Everything one carver's world generation cost.
struct Phase {
    report: report::GenReport,
    user_s: f64,
    sys_s: f64,
    peak_rss_kb: u64,
    carved: u64,
}

/// Runs one carver over the whole world and reads the process counters around it.
///
/// CPU time is a delta of `/proc/self/stat`, so a phase measures itself rather
/// than inheriting the startup cost, which is charged to whoever runs first.
fn phase(cfg: &Config) -> Phase {
    let before = cpu_seconds();
    let (grid, report) = report::generate_world(cfg);
    let after = cpu_seconds();

    Phase {
        report,
        user_s: after.0 - before.0,
        sys_s: after.1 - before.1,
        peak_rss_kb: peak_rss_kb(),
        carved: report::carved_voxel_count(&grid),
    }
}

fn print_table(side: u32, d: &Phase, s: &Phase, parity: &CarveParity) {
    println!();
    println!("carver    chunks  wall_s   user_s    sys_s   cpu_s  cpu_util  peak_rss_mb  solid_voxels  carved_voxels  carve_cov_pct");
    println!("--------  ------  -------  -------  -------  ------  --------  -----------  ------------  -------------  -------------");
    for (name, p) in [("density", d), ("sparse", s)] {
        let cpu = p.user_s + p.sys_s;
        let wall = p.report.elapsed.as_secs_f64();
        println!(
            "{name:<8}  {:>6}  {:>7.3}  {:>7.3}  {:>7.3}  {:>6.3}  {:>7.1}%  {:>11.1}  {:>12}  {:>13}  {:>13.1}",
            p.report.chunks,
            wall,
            p.user_s,
            p.sys_s,
            cpu,
            if wall > 0.0 { 100.0 * cpu / wall } else { 0.0 },
            p.peak_rss_kb as f64 / 1024.0,
            p.report.solid_voxels,
            p.carved,
            match name {
                "density" => parity.density_percent(),
                _ => parity.sparse_percent(),
            },
        );
    }

    println!();
    println!(
        "coverage   measured by voxel_core::config::CarveParity over a {side}x2x{side} chunk band:"
    );
    println!(
        "           {} solid voxels before carving, density {}% vs sparse {}% ({}pp apart).",
        parity.solid,
        parity.density_percent(),
        parity.sparse_percent(),
        parity.divergence()
    );
    println!("work       density {} noise samples | sparse {} SDF voxels over {} regions",
        d.report.density_samples, s.report.sparse_voxels_tested, s.report.sparse_regions_visited);
    println!("culling    density {} empty + {} enclosed | sparse {} empty + {} enclosed",
        d.report.empty_chunks, d.report.enclosed_chunks, s.report.empty_chunks, s.report.enclosed_chunks);
}

/// A block meant to be pasted into the writeup: `perf stat` prepends its own
/// counters to the same output, and these lines stay parsable next to them.
fn print_machine(side: u32, d: &Phase, s: &Phase, parity: &CarveParity) {
    let mut out = String::new();
    let _ = writeln!(out, "# units: time_s=seconds cpu_util_pct=percent_of_wall peak_rss_mb=mebibytes");
    let _ = writeln!(out, "# units: solid_voxels=count carved_voxels=count carve_cov_pct=percent_of_solid_rock");
    let _ = writeln!(out, "# units: density_samples=noise_samples sparse_voxels_tested=SDF_evaluations");
    let _ = writeln!(out, "# order: density ran to completion first, then sparse; sequential, not concurrent");
    let _ = writeln!(out, "world_side_chunks={side}");
    let _ = writeln!(out, "world_chunks={}", d.report.chunks);
    let _ = writeln!(out, "world_voxels={}", d.report.chunks * 4096);
    let _ = writeln!(out, "seed={SEED}");
    let _ = writeln!(out, "cave_origin_density={ORIGIN_DENSITY}");
    let _ = writeln!(out, "cave_min_radius={MIN_RADIUS}");
    let _ = writeln!(out, "cave_max_radius={MAX_RADIUS}");
    for (name, p) in [("density", d), ("sparse", s)] {
        let wall = p.report.elapsed.as_secs_f64();
        let cpu = p.user_s + p.sys_s;
        let _ = writeln!(out, "{name}_wall_s={wall:.6}");
        let _ = writeln!(out, "{name}_user_s={:.6}", p.user_s);
        let _ = writeln!(out, "{name}_sys_s={:.6}", p.sys_s);
        let _ = writeln!(out, "{name}_cpu_s={cpu:.6}");
        let _ = writeln!(
            out,
            "{name}_cpu_util_pct={:.1}",
            if wall > 0.0 { 100.0 * cpu / wall } else { 0.0 }
        );
        let _ = writeln!(out, "{name}_peak_rss_mb={:.3}", p.peak_rss_kb as f64 / 1024.0);
        let _ = writeln!(out, "{name}_solid_voxels={}", p.report.solid_voxels);
        let _ = writeln!(out, "{name}_carved_voxels={}", p.carved);
        let _ = writeln!(out, "{name}_density_samples={}", p.report.density_samples);
        let _ = writeln!(
            out,
            "{name}_sparse_voxels_tested={}",
            p.report.sparse_voxels_tested
        );
        let _ = writeln!(
            out,
            "{name}_sparse_regions_visited={}",
            p.report.sparse_regions_visited
        );
    }
    let _ = writeln!(out, "parity_solid_voxels={}", parity.solid);
    let _ = writeln!(out, "parity_density_percent={:.4}", parity.density_percent());
    let _ = writeln!(out, "parity_sparse_percent={:.4}", parity.sparse_percent());
    let _ = writeln!(out, "parity_divergence_pp={:.4}", parity.divergence());

    println!("\n--- machine readable ---");
    println!("# carver_bench begin");
    print!("{out}");
    println!("# carver_bench end");
}

/// `(utime, stime)` of this process, in seconds.
fn cpu_seconds() -> (f64, f64) {
    let tps = ticks_per_second();
    let Some(stat) = read("/proc/self/stat") else {
        return (0.0, 0.0);
    };
    // Field 2, `comm`, is parenthesised and may itself contain spaces, so the
    // only safe split is after its *last* ')'. Field 3 then starts the payload
    // and utime/stime are fields 14/15, i.e. indices 11/12 from here.
    let Some(rest) = stat.rsplit_once(')').map(|(_, r)| r) else {
        return (0.0, 0.0);
    };
    let f: Vec<&str> = rest.split_whitespace().collect();
    let ticks = |i: usize| -> f64 { f.get(i).and_then(|v| v.parse().ok()).unwrap_or(0.0) };
    (ticks(11) / tps, ticks(12) / tps)
}

/// Peak resident set size of the whole process so far, in kB.
///
/// `VmHWM` and not `VmRSS`: the high-water mark is the honest peak, where the
/// current size would report whatever happened to be resident at the end and
/// would miss a phase that allocated a grid and freed it.
fn peak_rss_kb() -> u64 {
    read("/proc/self/status")
        .and_then(|s| {
            s.lines()
                .find_map(|l| l.strip_prefix("VmHWM:"))?
                .split_whitespace()
                .next()?
                .parse()
                .ok()
        })
        .unwrap_or(0)
}

fn read(path: &str) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

/// Ticks per second for `/proc/self/stat`'s utime/stime.
///
/// shortcut: assumes `_SC_CLK_TCK` is 100, which every Linux on x86-64 and
/// aarch64 ships; read it from `sysconf(_SC_CLK_TCK)` if a target ever reports
/// CPU times that are off by a constant factor, or if this is ported to a libc
/// where the constant is configurable. We cannot call sysconf without pulling in
/// `libc`, and this binary must not add a dependency.
fn ticks_per_second() -> f64 {
    CLOCK_TICKS_PER_SECOND
}
