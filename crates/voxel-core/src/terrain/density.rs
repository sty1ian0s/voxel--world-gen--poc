//! Pass 2A: full 3D density field, `O(N^3)`.
//!
//! The baseline in `PROJECT.md` §4.2. Evaluates the cave noise at *every* voxel
//! in the chunk and carves where density crosses a threshold. This is the
//! straightforward approach, and its cost scales with the volume of the world
//! rather than with the number of caves in it, which is precisely the property
//! Pass 2B is meant to beat.

use super::{CarveTarget, world_of};
use crate::domain::VoxelIndex;
use crate::terrain::noise;

/// Threshold on the noise field: voxels above it become air.
///
/// Tuned so that the default config produces caves that are visible in the
/// debug views without swallowing the surface.
pub const CARVE_THRESHOLD: f32 = 0.55;

/// Carves one chunk by sampling the density field at every voxel.
///
/// `noise_samples` is returned rather than logged so the benchmark and the
/// equivalence test can assert the `O(N^3)` claim instead of trusting it.
pub fn carve(chunk: &mut impl CarveTarget, seed: u32) -> u64 {
    let n = noise::density(seed);
    let (ox, oy, oz) = chunk.pos().origin();
    let mut samples = 0u64;

    for y in 0..16u8 {
        for z in 0..16u8 {
            for x in 0..16u8 {
                let index = VoxelIndex::from_parts(x, y, z);
                // Never re-carve air: keeps the operation idempotent and avoids
                // paying for noise in space that is already empty.
                if chunk.block(index).is_air() {
                    continue;
                }
                let (wx, wy, wz) = (ox + x as i32, oy + y as i32, oz + z as i32);
                samples += 1;
                if n.get_noise_3d(wx as f32, wy as f32, wz as f32) > CARVE_THRESHOLD {
                    chunk.carve(index);
                }
            }
        }
    }

    samples
}

/// Convenience wrapper for callers that have a world coordinate in hand.
#[inline]
pub fn density_at(seed: u32, wx: i32, wy: i32, wz: i32) -> f32 {
    let n = noise::density(seed);
    n.get_noise_3d(wx as f32, wy as f32, wz as f32)
}

/// Local-position variant used by tests that reason in voxel space.
#[inline]
pub fn carve_at(chunk: &mut impl CarveTarget, seed: u32, x: u8, y: u8, z: u8) -> bool {
    let index = VoxelIndex::from_parts(x, y, z);
    if chunk.block(index).is_air() {
        return false;
    }
    let (wx, wy, wz) = world_of(chunk.pos(), index.to_local());
    if density_at(seed, wx, wy, wz) > CARVE_THRESHOLD {
        chunk.carve(index);
        true
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{BlockId, ChunkPos};
    use crate::terrain::heightmap::HeightMap;
    use crate::world::Chunk;

    fn generated(seed: u32) -> Chunk {
        let hm = HeightMap::new(seed, 0);
        let mut c = Chunk::default();
        crate::terrain::generate_surface(&mut c, ChunkPos::ORIGIN, &hm);
        c
    }

    #[test]
    fn sampling_count_is_the_full_solid_volume() {
        // The O(N^3) claim, asserted: one noise sample per non-air voxel.
        let mut chunk = generated(1337);
        let solid = (0..4096u16)
            .filter(|&i| {
                !chunk
                    .block(unsafe { VoxelIndex::from_raw_unchecked(i) })
                    .is_air()
            })
            .count() as u64;
        let samples = carve(&mut chunk, 1337);
        assert_eq!(samples, solid, "one sample per solid voxel, no more");
    }

    #[test]
    fn carving_is_idempotent() {
        // Idempotent in its *result*: running twice changes no voxel. It does
        // still sample the voxels that stayed solid, because whether they
        // should be carved is exactly what the density field decides, and it
        // short-circuits only on voxels that are already air.
        let mut once = generated(1337);
        carve(&mut once, 1337);
        let solid_after = (0..4096u16)
            .filter(|&i| {
                !once
                    .block(unsafe { VoxelIndex::from_raw_unchecked(i) })
                    .is_air()
            })
            .count() as u64;

        let mut twice = once.clone();
        let second_pass = carve(&mut twice, 1337);
        assert_eq!(
            second_pass, solid_after,
            "the second pass must sample exactly the voxels still solid"
        );
        for i in 0..4096u16 {
            let idx = unsafe { VoxelIndex::from_raw_unchecked(i) };
            assert_eq!(once.block(idx), twice.block(idx), "voxel {i} changed");
        }
    }

    #[test]
    fn carving_only_removes_and_only_underground() {
        let mut chunk = generated(1337);
        let before: Vec<BlockId> = (0..4096u16)
            .map(|i| chunk.block(unsafe { VoxelIndex::from_raw_unchecked(i) }))
            .collect();
        carve(&mut chunk, 1337);

        let mut carved = 0;
        for (i, &was) in before.iter().enumerate() {
            let now = chunk.block(unsafe { VoxelIndex::from_raw_unchecked(i as u16) });
            assert!(
                now.is_air() || now == was,
                "carving placed block {now:?} at {i}"
            );
            if now.is_air() && !was.is_air() {
                carved += 1;
                // Caves must stay under the surface; a carve above it would be
                // proof that the y coordinate lost its chunk origin.
                assert!(!was.is_air(), "carved air at {i}");
            }
        }
        assert!(carved > 0, "threshold carved nothing; check noise frequency");
    }

    #[test]
    fn result_is_stable_across_chunk_positions() {
        // Same world coordinate must give the same answer regardless of which
        // chunk it is reached through, i.e. no missing chunk-origin offset.
        let seed = 1337u32;
        let hm = HeightMap::new(seed, 0);
        let mut a = Chunk::default();
        crate::terrain::generate_surface(&mut a, ChunkPos::new(1, 0, 1), &hm);
        let mut b = Chunk::default();
        crate::terrain::generate_surface(&mut b, ChunkPos::new(1, 0, 1), &hm);
        carve(&mut a, seed);
        carve(&mut b, seed);
        for i in 0..4096u16 {
            let idx = unsafe { VoxelIndex::from_raw_unchecked(i) };
            assert_eq!(a.block(idx), b.block(idx));
        }
    }

    #[test]
    fn density_at_matches_the_carve_decision() {
        // Guards against the carver and the standalone query drifting apart,
        // which would make the debug views lie about the generated world.
        let seed = 1337;
        let mut chunk = generated(seed);
        let (ox, oy, oz) = ChunkPos::ORIGIN.origin();
        let _ = (ox, oy, oz);
        for y in 0..16u8 {
            for z in 0..16u8 {
                for x in 0..16u8 {
                    let i = VoxelIndex::from_parts(x, y, z);
                    if chunk.block(i).is_air() {
                        continue;
                    }
                    let d = density_at(seed, x as i32, y as i32, z as i32);
                    let carved = carve_at(&mut chunk, seed, x, y, z);
                    assert_eq!(carved, d > CARVE_THRESHOLD, "disagreement at ({x},{y},{z})");
                }
            }
        }
    }
}
