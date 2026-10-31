//! Per-chunk carve latency: p50/p90/p99/p99.9 for MC's approach against ours.
//!
//! `benches/carvers.rs` uses criterion, which is the right tool for comparing
//! means and for reporting a confidence interval on them, but it deliberately
//! reports a central estimate: it has no notion of the tail. The tail is the
//! only part that matters here. World generation happens inside a frame, so a
//! carver whose median is fine but whose p99.9 is 10x the median is a carver
//! that drops frames, and a mean cannot tell the two apart.
//!
//! So this uses a plain `main` (harness = false) and raw `Instant`, with enough
//! samples per carver for p99.9 to mean something: the 99.9th percentile of
//! 5000 samples is the 5th worst, which is thin but far better than the mean.
//!
//! The spread is expected to be much wider for a carver whose cost depends on
//! how many features happen to land near a chunk than for one whose cost is a
//! fixed function of the chunk.
//!
//! # What is compared, and why
//!
//! Minecraft's approach (`mc`) against our 2D strategy (`tube`), both running
//! on the current architecture. `mc` is the 3D mechanism reproduced from
//! minecraft.wiki; `tube` is the voxel-local 2D carver. Measuring them on the
//! same band, the same seed and the same fresh chunks is the whole point, so
//! nothing else is in the table.
//!
//! `tube` is measured **twice**: once as the default world (three noise cave
//! types) and once with the opt-in pillar type enabled, because
//! `tube::carve` uses `default_configs()` and therefore silently excludes
//! pillars. Reporting only the first would understate what the 2D carver does.
//!
//! The original `density`/`sparse` pair is deliberately NOT here. Those are the
//! two Pass-2 algorithms from `PROJECT.md` section 1, which is a different
//! comparison (O(N^3) vs O(S)) and has its own benchmark in `benches/carvers.rs`.

use std::hint::black_box;
use std::time::{Duration, Instant};

use voxel_core::domain::ChunkPos;
use voxel_core::terrain::{self, capsule, heightmap::HeightMap, mc, tube, tube::CaveType};
use voxel_core::world::Chunk;

/// The shipped `.env` value and the one the parity test asserts on. At 0.02 the
/// sparse carver does almost no work, so its tail would be flattering fiction.
const SEED: u32 = 1337;
/// Band width used for the coverage note, matching the parity test's own band so
/// the two reports cannot disagree.
/// Samples per carver. p99.9 of fewer samples is the worst sample, not a
/// percentile.
const SAMPLES: usize = 5000;
/// Chunk x/z positions cycled over, so no single lucky chunk is measured
/// repeatedly. The sparse carver's cost varies with how many origins land near
/// a chunk, which is exactly the variance this benchmark is looking for.
const POSITIONS: [(i32, i32); 8] = [
    (0, 0),
    (1, 0),
    (2, 1),
    (3, 2),
    (4, 4),
    (5, 3),
    (6, 7),
    (7, 5),
];
/// `y` band the cave carvers work in.
///
/// This was a hardcoded `1` (world y 16..31) and became wrong when cave
/// placement went absolute: the band is now y 40..88, so `y = 1` timed the
/// carvers on chunks containing no caves at all. Every carver skips voxels that
/// are already air, so those samples measured the noise and arithmetic but
/// never the carve writes — a latency table for work that was not done.
///
/// Derived from the same source as the carvers and the coverage helper, so it
/// cannot drift again.
fn cave_band_y() -> i32 {
    let band = terrain::measured_band_chunks();
    band.start + (band.end - band.start) / 2
}

/// Nearest-rank percentile of a sorted slice.
///
/// `p` is in `[0, 1]`; `p50` is the lower median for an even count, which is
/// the convention criterion also labels `median`.
fn percentile(sorted: &[Duration], p: f64) -> Duration {
    assert!(!sorted.is_empty(), "percentile of an empty sample set");
    let rank = (p * sorted.len() as f64).ceil() as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

struct Latency {
    name: &'static str,
    samples: Vec<Duration>,
}

impl Latency {
    fn ms(d: Duration) -> f64 {
        d.as_secs_f64() * 1e3
    }

    fn us(d: Duration) -> f64 {
        d.as_secs_f64() * 1e6
    }

    fn sorted(&self) -> Vec<Duration> {
        let mut v = self.samples.clone();
        v.sort_unstable();
        v
    }

    fn mean(samples: &[Duration]) -> Duration {
        let total: Duration = samples.iter().sum();
        total / samples.len().max(1) as u32
    }
}

/// Times one carver over `SAMPLES` fresh chunks.
///
/// `carve` is timed on a freshly surface-generated chunk and only the carve
/// call is inside the timed region: surface generation is shared by both
/// carvers, so counting it would compress the ratio this benchmark exists to
/// show. Re-carving a chunk would undercount, because both carvers skip voxels
/// that are already air.
fn measure(name: &'static str, seed: u32, carve: impl Fn(&mut Chunk)) -> Latency {
    let hm = HeightMap::fitted_to_world(seed, 64);
    let cy = cave_band_y();
    let mut samples = Vec::with_capacity(SAMPLES);

    for i in 0..SAMPLES {
        let (cx, cz) = POSITIONS[i % POSITIONS.len()];
        let pos = ChunkPos::new(cx, cy, cz);

        // Untimed setup: a chunk with its surface already generated, whatever
        // the previous sample left behind.
        let mut chunk = Chunk::at(pos);
        terrain::generate_surface(&mut chunk, pos, &hm);

        let start = Instant::now();
        carve(black_box(&mut chunk));
        samples.push(start.elapsed());
    }

    Latency { name, samples }
}

/// Prints the percentile table for every carver measured.
///
/// Generic over the carver list rather than a `density`/`sparse` pair: the two
/// carvers that mattered when this benchmark was written are not the two that
/// matter now, and a hardcoded pair silently excludes every carver added later.
/// `mc`, `capsule` and `tube` had no measured percentiles at all until this
/// became a slice.
fn report(latencies: &[Latency], baseline: &str) {
    let sorted: Vec<Vec<Duration>> = latencies.iter().map(|l| l.sorted()).collect();

    println!();
    println!(
        "per-chunk carve latency, microseconds ({} samples per carver, seed {SEED})",
        SAMPLES
    );
    println!(
        "setup (surface generation) excluded; chunk positions cycled over {} x/z slots in the cave band",
        POSITIONS.len()
    );
    println!();

    // Header
    print!("  {:<10}", "metric");
    for l in latencies {
        print!(" {:>11}", l.name);
    }
    print!(" {:>11}", format!("vs {baseline}"));
    println!();
    println!("  {}", "-".repeat(12 + 12 * (latencies.len() + 1)));

    let Some(base_idx) = latencies.iter().position(|l| l.name == baseline) else {
        println!("  (baseline {baseline} was not measured)");
        return;
    };

    let row = |label: &str, pick: &dyn Fn(&[Duration]) -> Duration| {
        print!("  {label:<10}");
        for s in &sorted {
            print!(" {:>11.2}", Latency::us(pick(s)));
        }
        let b = pick(&sorted[base_idx]).as_secs_f64();
        for (i, s) in sorted.iter().enumerate() {
            if i == base_idx {
                print!(" {:>10.2}x", 1.0);
            } else {
                let r = pick(s).as_secs_f64() / b.max(f64::MIN_POSITIVE);
                print!(" {:>10.2}x", r);
            }
        }
        println!();
    };

    row("min", &|s| s[0]);
    for (label, p) in [("p50", 0.50), ("p90", 0.90), ("p99", 0.99), ("p99.9", 0.999)] {
        row(label, &|s| percentile(s, p));
    }
    row("mean", &|s| Latency::mean(s));
    row("max", &|s| s[s.len() - 1]);
    row("count", &|s| Duration::from_nanos(s.len() as u64));
    println!();

    // Tail against the median rather than against another carver: this is the
    // number a frame budget cares about, because one slow chunk stalls a frame
    // regardless of how fast the other carvers are.
    for l in latencies {
        let v = l.sorted();
        let p50 = percentile(&v, 0.50).as_secs_f64();
        println!(
            "  {:<10} tail/median: p90 {:.2}x  p99 {:.2}x  p99.9 {:.2}x   worst chunk {:.2} ms",
            l.name,
            percentile(&v, 0.90).as_secs_f64() / p50,
            percentile(&v, 0.99).as_secs_f64() / p50,
            percentile(&v, 0.999).as_secs_f64() / p50,
            Latency::ms(v[v.len() - 1])
        );
    }
    println!();
}

fn main() {
    let seed = SEED;

    // Every cave carver, not just the two the O(N^3)-vs-O(S) headline needed.
    // `tube` and its per-type rows are what the 2D strategy is judged on, and
    // `mc` is the baseline it is judged against, so all of them need percentiles.
    let all = vec![
        measure("mc", seed, |c| {
            black_box(mc::carve(c, seed));
        }),
        measure("capsule", seed, |c| {
            black_box(capsule::carve(c, seed));
        }),
        // Our 2D strategy in full: the three noise types PLUS the opt-in
        // pillar type, so the measurement covers everything the 2D carver can
        // do rather than only the default world. `tube::carve` would silently
        // measure three types, because `default_configs()` disables anything
        // not in `CaveType::ALL` - which is how the pillar work first went
        // unmeasured.
        measure("tube", seed, |c| {
            black_box(tube::carve(c, seed));
        }),
        measure("tube+pillar", seed, |c| {
            black_box(tube::carve_types(c, seed, &CaveType::ALL_TYPES.map(|t| t.cutoffed())));
        }),
    ];

    report(&all, "mc");

    // Self-check: a harness that measured nothing would print a plausible table
    // of zeros. Cheap, and it has to hold for the numbers to mean anything.
    for l in &all {
        assert_eq!(l.samples.len(), SAMPLES, "{} lost samples", l.name);
        assert!(
            Latency::mean(&l.samples) > Duration::ZERO,
            "{} measured a zero mean",
            l.name
        );
    }
    // The band must actually contain caves, or every timing above times work
    // the carvers skipped. The absence of this check is what let a hardcoded
    // `y = 1` measure empty chunks unnoticed.
    //
    // Asserted over the whole sampled position set, not one chunk: a single
    // chunk can legitimately be cave-free (noodle is 0.9% void, and pillars
    // sparser still), so a per-chunk check would be flaky rather than strict.
    // `blocks()` is the raw `u8` view, so AIR is the literal 0.
    let hm = HeightMap::fitted_to_world(seed, 64);
    let mut carved_total = 0usize;
    for &(cx, cz) in &POSITIONS {
        let pos = ChunkPos::new(cx, cave_band_y(), cz);
        let mut probe = Chunk::at(pos);
        terrain::generate_surface(&mut probe, pos, &hm);
        let before = probe.blocks().iter().filter(|&&b| b != 0).count();
        tube::carve(&mut probe, seed);
        let after = probe.blocks().iter().filter(|&&b| b != 0).count();
        carved_total += before.saturating_sub(after);
    }
    assert!(
        carved_total > 0,
        "the benchmarked band carves nothing across all {} sampled chunks; \
         the cave band and the measured band have drifted apart",
        POSITIONS.len()
    );
}
