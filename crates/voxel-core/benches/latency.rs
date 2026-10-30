//! Per-chunk carve latency distribution: p50/p90/p99/p99.9 for both carvers.
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
//! The spread is expected to be much wider for `sparse` than for `density`.
//! Density samples the same 4096 voxels every chunk, so its per-chunk cost is
//! near-constant; sparse enumerates 27 regions and stamps whatever origins land
//! near the chunk, which is a Poisson-ish count that occasionally spikes. That
//! difference is invisible in a criterion median.
//!
//! Both carvers are measured on the same world settings as the shipped `.env`
//! and the parity test, and on fresh surface-generated chunks with the setup
//! excluded from the timed region.

use std::hint::black_box;
use std::time::{Duration, Instant};

use voxel_core::config::{CarveParity, CaveParams};
use voxel_core::domain::ChunkPos;
use voxel_core::terrain::{self, density, heightmap::HeightMap, sparse};
use voxel_core::world::Chunk;

/// The shipped `.env` value and the one the parity test asserts on. At 0.02 the
/// sparse carver does almost no work, so its tail would be flattering fiction.
const ORIGIN_DENSITY: f32 = 2.0;
const SEED: u32 = 1337;
/// Band width used for the coverage note, matching the parity test's own band so
/// the two reports cannot disagree.
const PARITY_SIDE: i32 = 8;
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
/// `y` band the cave carvers work in; above it is sky and below it is bedrock
/// by world height, so a chunk outside this band measures nothing.
const CY: i32 = 1;

fn params() -> CaveParams {
    CaveParams {
        origin_density: ORIGIN_DENSITY,
        min_radius: 1.5,
        max_radius: 4.0,
    }
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
    let mut samples = Vec::with_capacity(SAMPLES);

    for i in 0..SAMPLES {
        let (cx, cz) = POSITIONS[i % POSITIONS.len()];
        let pos = ChunkPos::new(cx, CY, cz);

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

fn report(density: &Latency, sparse: &Latency) {
    let dp = density.sorted();
    let sp = sparse.sorted();

    println!();
    println!("per-chunk carve latency, microseconds ({} samples per carver, seed {SEED})", dp.len());
    println!(
        "cave params: origin_density={ORIGIN_DENSITY} min_radius=1.5 max_radius=4.0"
    );
    // Coverage comes from the shared helper over the parity band, NOT from this
    // benchmark's own samples. The samples here cycle 8 x/z slots at one `y` to
    // keep the per-chunk timings comparable, and those 8 chunks are not a
    // representative sample of the world: measuring coverage over them reads
    // ~16% for sparse against the helper's ~20%. A second, disagreeing coverage
    // figure in the header would be worse than none.
    let parity = CarveParity::measure(SEED, &params(), PARITY_SIDE);
    println!(
        "coverage: both carvers are tuned to the same coverage, so the latencies compare like with like"
    );
    println!(
        "  over the {}x2x{} parity band: density {:.1}% of rock, sparse {:.1}% ({:.1}pp apart)",
        PARITY_SIDE,
        PARITY_SIDE,
        parity.density_percent(),
        parity.sparse_percent(),
        parity.divergence()
    );
    println!(
        "setup (surface generation) excluded from every timing; chunk positions cycled over {} x/z slots",
        POSITIONS.len()
    );
    println!();

    println!("  {:<12} {:>12} {:>12} {:>10}", "metric", "density", "sparse", "sparse/dens");
    println!("  {}", "-".repeat(50));

    let ratio = |a: Duration, b: Duration| a.as_secs_f64() / b.as_secs_f64().max(f64::MIN_POSITIVE);

    println!(
        "  {:<12} {:>12.2} {:>12.2} {:>10.2}x",
        "min",
        Latency::us(dp[0]),
        Latency::us(sp[0]),
        ratio(sp[0], dp[0])
    );
    for (label, p) in [("p50", 0.50), ("p90", 0.90), ("p99", 0.99), ("p99.9", 0.999)] {
        let (a, b) = (percentile(&dp, p), percentile(&sp, p));
        println!(
            "  {label:<12} {:>12.2} {:>12.2} {:>10.2}x",
            Latency::us(a),
            Latency::us(b),
            ratio(b, a)
        );
    }
    let (dmean, smean) = (Latency::mean(&dp), Latency::mean(&sp));
    println!(
        "  {:<12} {:>12.2} {:>12.2} {:>10.2}x",
        "mean",
        Latency::us(dmean),
        Latency::us(smean),
        ratio(smean, dmean)
    );
    let (dmax, smax) = (dp[dp.len() - 1], sp[sp.len() - 1]);
    println!(
        "  {:<12} {:>12.2} {:>12.2} {:>10.2}x",
        "max",
        Latency::us(dmax),
        Latency::us(smax),
        ratio(smax, dmax)
    );
    println!("  {:<12} {:>12} {:>12}", "count", density.samples.len(), sparse.samples.len());
    println!();

    // Tail ratio against the median rather than against the other carver: this
    // is the number a frame budget cares about, because a single slow chunk
    // stalls the frame regardless of how fast the other carver is.
    for l in [density, sparse] {
        let v = l.sorted();
        let p50 = percentile(&v, 0.50).as_secs_f64();
        println!(
            "  {:<12} tail/median: p90 {:.2}x  p99 {:.2}x  p99.9 {:.2}x   worst chunk {:.2} ms",
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
    let p = params();

    let d = measure("density", seed, |c| {
        black_box(density::carve(c, seed));
    });
    let s = measure("sparse", seed, |c| {
        black_box(sparse::carve(c, seed, &p));
    });

    report(&d, &s);

    // Self-check: a harness that measured nothing would print a plausible
    // table of zeros. Cheap, and it has to hold for the numbers to mean
    // anything.
    assert_eq!(d.samples.len(), SAMPLES);
    assert_eq!(s.samples.len(), SAMPLES);
    // Sparse must actually have carved: a table of timings from a carver that
    // did nothing would look excellent and mean nothing.
    let parity = CarveParity::measure(seed, &p, PARITY_SIDE);
    assert!(
        parity.sparse_carved > 0 && parity.density_carved > 0,
        "a carver carved nothing: {}",
        parity.summary()
    );
}
