//! The two cave carvers must produce comparable worlds.
//!
//! An integration test rather than a unit test, because it is about the
//! relationship between two subsystems, and because it is the guarantee that
//! makes the benchmark meaningful: comparing `O(N^3)` against `O(S)` says
//! nothing if one of them carves a fifth as much rock.
//!
//! The failure this exists to prevent already happened once. The sparse carver
//! had a cap of 64 origins per region, which silently saturated every density
//! setting above about 0.245, so it carved 3% of the rock against the density
//! carver's 19% and the underground simply looked like it was missing. No unit
//! test caught it, because each carver was self-consistent. Only comparing them
//! does.

use voxel_core::config::{CarveParity, CaveParams};
use voxel_core::domain::{CHUNK_VOLUME, ChunkPos, VoxelIndex};
use voxel_core::terrain::{self, density, heightmap::HeightMap, sparse};
use voxel_core::world::Chunk;

/// The sparse density the defaults use, matching the density carver's coverage.
const SPARSE_DENSITY: f32 = 2.0;
/// Cave radius range, shared by both so only the strategy differs.
const MIN_RADIUS: f32 = 1.5;
const MAX_RADIUS: f32 = 4.0;

/// Chunks sampled, in a 3D band so this measures the underground rather than
/// one lucky slice.
fn sample_positions(side: i32) -> Vec<ChunkPos> {
    let mut out = Vec::new();
    for cy in 1..3 {
        for cx in 0..side {
            for cz in 0..side {
                out.push(ChunkPos::new(cx, cy, cz));
            }
        }
    }
    out
}

struct Measured {
    solid: usize,
    carved: usize,
    chunks_with_caves: usize,
    chunks: usize,
}

impl Measured {
    fn carve_percent(&self) -> f64 {
        100.0 * self.carved as f64 / self.solid.max(1) as f64
    }

    fn cave_chunk_percent(&self) -> f64 {
        100.0 * self.chunks_with_caves as f64 / self.chunks.max(1) as f64
    }
}

fn air(chunk: &Chunk) -> usize {
    (0..CHUNK_VOLUME as u16)
        .filter(|&i| {
            chunk
                .block(unsafe { VoxelIndex::from_raw_unchecked(i) })
                .is_air()
        })
        .count()
}

/// Measures the sparse carver over a band of chunks.
fn measure_sparse(seed: u32, params: &CaveParams, side: i32) -> Measured {
    let hm = HeightMap::fitted_to_world(seed, 64);
    let mut m = Measured {
        solid: 0,
        carved: 0,
        chunks_with_caves: 0,
        chunks: 0,
    };

    for pos in sample_positions(side) {
        let mut chunk = Chunk::at(pos);
        terrain::generate_surface(&mut chunk, pos, &hm);
        let solid_before = CHUNK_VOLUME - air(&chunk);
        let air_before = air(&chunk);
        sparse::carve(&mut chunk, seed, params);

        let carved = air(&chunk) - air_before;
        m.solid += solid_before;
        m.carved += carved;
        m.chunks += 1;
        if carved > 0 {
            m.chunks_with_caves += 1;
        }
    }
    m
}

/// Measures the density carver over the same band.
fn measure_density(seed: u32, threshold: f32, side: i32) -> Measured {
    let hm = HeightMap::fitted_to_world(seed, 64);
    let mut m = Measured {
        solid: 0,
        carved: 0,
        chunks_with_caves: 0,
        chunks: 0,
    };

    for pos in sample_positions(side) {
        let mut chunk = Chunk::at(pos);
        terrain::generate_surface(&mut chunk, pos, &hm);
        let solid_before = CHUNK_VOLUME - air(&chunk);
        let air_before = air(&chunk);
        density::carve_with(&mut chunk, seed, threshold);

        let carved = air(&chunk) - air_before;
        m.solid += solid_before;
        m.carved += carved;
        m.chunks += 1;
        if carved > 0 {
            m.chunks_with_caves += 1;
        }
    }
    m
}

fn default_params() -> CaveParams {
    CaveParams {
        origin_density: SPARSE_DENSITY,
        min_radius: MIN_RADIUS,
        max_radius: MAX_RADIUS,
    }
}

#[test]
fn both_carvers_actually_carve() {
    // Guards the harness itself: if either carver silently did nothing, the
    // parity assertions below would be comparing two zeros and passing.
    let seed = 1337;
    let side = 6;
    let d = measure_density(seed, density::CARVE_THRESHOLD, side);
    let s = measure_sparse(seed, &default_params(), side);

    assert!(d.solid > 0 && s.solid > 0, "no solid rock to carve");
    assert!(d.carved > 0, "density carver carved nothing");
    assert!(s.carved > 0, "sparse carver carved nothing");
}

#[test]
fn carve_fractions_are_comparable() {
    // The headline guarantee. The approaches are allowed to differ, but not by
    // so much that one is carving a different kind of world: 5 points of
    // coverage is well inside "these look like similar cave systems" and well
    // outside the 16-point gap the origin cap used to produce.
    let seed = 1337;
    let side = 8;
    let d = measure_density(seed, density::CARVE_THRESHOLD, side);
    let s = measure_sparse(seed, &default_params(), side);

    let (dp, sp) = (d.carve_percent(), s.carve_percent());
    assert!(
        (dp - sp).abs() <= 5.0,
        "carve fractions diverge: density {dp:.1}% vs sparse {sp:.1}%"
    );
    // Both should sit in a range that reads as a cave system rather than as a
    // solid block or an empty shell.
    for (name, pct) in [("density", dp), ("sparse", sp)] {
        assert!(
            (10.0..=35.0).contains(&pct),
            "{name} carves {pct:.1}%, outside the intended 10-35% band"
        );
    }
}

#[test]
fn caves_appear_in_a_comparable_number_of_chunks() {
    // Coverage alone could match while one carver concentrated everything into
    // a few chunks. This checks the caves are spread through the world, which
    // is what "the underground is missing" actually means in practice.
    let seed = 1337;
    let side = 8;
    let d = measure_density(seed, density::CARVE_THRESHOLD, side);
    let s = measure_sparse(seed, &default_params(), side);

    let (dp, sp) = (d.cave_chunk_percent(), s.cave_chunk_percent());
    assert!(
        (dp - sp).abs() <= 20.0,
        "cave-bearing chunks diverge: density {dp:.0}% vs sparse {sp:.0}%"
    );
    assert!(
        sp >= 60.0,
        "sparse caves appear in only {sp:.0}% of chunks; the underground would look empty"
    );
}

#[test]
fn dense_density_is_not_silently_capped() {
    // Directly targets the old bug. With the origin cap at 64, everything above
    // roughly 0.245 produced an identical world, so raising the knob did
    // nothing. Two clearly different densities must now differ.
    let seed = 1337;
    let side = 6;
    let sparse = measure_sparse(
        seed,
        &CaveParams {
            origin_density: 0.5,
            ..default_params()
        },
        side,
    );
    let dense = measure_sparse(seed, &default_params(), side);

    assert!(
        dense.carved > sparse.carved,
        "density 2.0 carved {} voxels but 0.5 carved {}; the knob is capped again",
        dense.carved,
        sparse.carved
    );
}

#[test]
fn the_parity_helper_agrees_with_a_direct_measurement() {
    // `CarveParity` is what the benchmark and the HUD report; if it disagreed
    // with a direct measurement the reported numbers would be fiction.
    let seed = 1337;
    let side = 4;
    let params = default_params();
    let direct = measure_sparse(seed, &params, side);
    let via_helper = CarveParity::measure(seed, &params, side);

    assert_eq!(via_helper.sparse_carved, direct.carved as u64);
    assert_eq!(via_helper.solid, direct.solid as u64);
    assert!(
        (via_helper.sparse_percent() - direct.carve_percent()).abs() < 0.01,
        "helper reports {:.2}% against a measured {:.2}%",
        via_helper.sparse_percent(),
        direct.carve_percent()
    );
}

#[test]
fn a_deliberately_wrong_density_is_detected() {
    // The parity check must actually fail when the carvers disagree, or it is
    // not a check. A near-zero density stands in for the origin-cap bug.
    let seed = 1337;
    let side = 6;
    let d = measure_density(seed, density::CARVE_THRESHOLD, side);
    let s = measure_sparse(
        seed,
        &CaveParams {
            origin_density: 0.02,
            ..default_params()
        },
        side,
    );

    let gap = (d.carve_percent() - s.carve_percent()).abs();
    assert!(
        gap > 5.0,
        "a sparse density of 0.02 should diverge sharply from density, gap was {gap:.1}"
    );
}
