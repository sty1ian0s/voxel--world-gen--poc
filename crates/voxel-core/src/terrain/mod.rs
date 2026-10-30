//! Terrain generation pipeline (`PROJECT.md` §4.2).
//!
//! * **Pass 1** — [`heightmap`]: 2D surface height per column. Cheap, shared by
//!   both carvers, so the two cave paradigms are compared on equal footing.
//! * **Pass 2A** — [`density`]: full 3D noise per voxel, `O(N^3)`.
//! * **Pass 2B** — [`sparse`]: deterministic origin lookup plus bounded SDF
//!   stamps, `O(S)`.
//!
//! Both Pass 2 implementations consume the same Pass 1 output and the same
//! [`CarveTarget`], so the benchmark measures only the cave-carving strategy.

pub mod density;
pub mod heightmap;
pub mod noise;
pub mod sparse;

use crate::domain::{BlockId, ChunkPos, LocalPos, VoxelIndex};
use crate::world::Chunk;

/// What a carver is allowed to write to. Keeps the two Pass 2 implementations
/// honest: a carver can only ever turn solid voxels into air within the chunk
/// it was given, so neither can affect a neighbour and the "bounded padding"
/// claim in `PROJECT.md` §4.2 stays testable.
pub trait CarveTarget {
    /// Chunk being carved, for stamp placement in world space.
    fn pos(&self) -> ChunkPos;

    /// Reads the current block, which is how a carver avoids re-carving air.
    fn block(&self, index: VoxelIndex) -> BlockId;

    /// Turns a voxel to air. Carvers never place blocks.
    fn carve(&mut self, index: VoxelIndex);
}

impl CarveTarget for Chunk {
    #[inline]
    fn pos(&self) -> ChunkPos {
        // Read from the chunk itself, so a carver cannot be handed a position
        // that disagrees with the chunk it is writing into.
        self.pos()
    }

    #[inline]
    fn block(&self, index: VoxelIndex) -> BlockId {
        Chunk::block(self, index)
    }

    #[inline]
    fn carve(&mut self, index: VoxelIndex) {
        self.set_block(index, BlockId::AIR);
    }
}

/// Fills one chunk with the Pass 1 surface, then runs the chosen Pass 2 carver.
///
/// Shared by both strategies so the only difference measured is the carver.
pub fn generate_surface(chunk: &mut Chunk, pos: ChunkPos, surface: &heightmap::HeightMap) {
    let (ox, oy, oz) = pos.origin();
    for y in 0..16u8 {
        for z in 0..16u8 {
            for x in 0..16u8 {
                let wy = oy + y as i32;
                let surface_y = surface.height(ox + x as i32, oz + z as i32);
                let block = if wy > surface_y {
                    // Above the surface: air, or water up to sea level.
                    if wy <= surface.sea_level() {
                        BlockId::WATER
                    } else {
                        BlockId::AIR
                    }
                } else if wy == surface_y {
                    BlockId::GRASS
                } else if wy >= surface_y - 3 {
                    BlockId::DIRT
                } else {
                    BlockId::STONE
                };
                let index = VoxelIndex::from_parts(x, y, z);
                chunk.set(index, crate::domain::PackedVoxel::of(block));
            }
        }
    }
}

/// Recomputes the cull flags after generation.
#[inline]
pub fn finish(chunk: &mut Chunk) {
    chunk.refresh_flags();
}

/// World-space voxel coordinate of a local position, for carvers that work in
/// world space (the sparse stamper does).
#[inline]
pub fn world_of(pos: ChunkPos, local: LocalPos) -> (i32, i32, i32) {
    let (ox, oy, oz) = pos.origin();
    (
        ox + local.x() as i32,
        oy + local.y() as i32,
        oz + local.z() as i32,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surface_is_stratified_grass_dirt_stone() {
        let hm = heightmap::HeightMap::new(1337, 0);
        let mut chunk = crate::world::Chunk::default();
        generate_surface(&mut chunk, ChunkPos::ORIGIN, &hm);

        for z in 0..16u8 {
            for x in 0..16u8 {
                let h = hm.height(x as i32, z as i32);
                if !(0..16).contains(&h) {
                    continue;
                }
                let at = |y: u8| chunk.block(VoxelIndex::from_parts(x, y, z));
                assert_eq!(at(h as u8), BlockId::GRASS, "surface at ({x},{h},{z})");
                if h >= 1 {
                    assert_eq!(at(h as u8 - 1), BlockId::DIRT);
                }
                if h >= 5 {
                    assert_eq!(at(h as u8 - 4), BlockId::STONE, "deep stone");
                }
            }
        }
    }

    #[test]
    fn water_fills_up_to_sea_level_only() {
        let hm = heightmap::HeightMap::with_sea_level(1337, 8);
        let mut chunk = crate::world::Chunk::default();
        generate_surface(&mut chunk, ChunkPos::ORIGIN, &hm);

        for z in 0..16u8 {
            for x in 0..16u8 {
                let h = hm.height(x as i32, z as i32);
                for y in 0..16u8 {
                    let wy = y as i32;
                    if wy > h && wy <= 8 {
                        assert_eq!(
                            chunk.block(VoxelIndex::from_parts(x, y, z)),
                            BlockId::WATER,
                            "({x},{y},{z}) should be water, surface {h}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn carve_target_only_ever_removes_blocks() {
        let hm = heightmap::HeightMap::new(7, 0);
        let mut chunk = crate::world::Chunk::default();
        generate_surface(&mut chunk, ChunkPos::ORIGIN, &hm);
        let before: Vec<BlockId> = (0..4096)
            .map(|i| chunk.block(unsafe { VoxelIndex::from_raw_unchecked(i as u16) }))
            .collect();

        // Carve every voxel, then confirm nothing became solid.
        for i in 0..4096u16 {
            chunk.carve(unsafe { VoxelIndex::from_raw_unchecked(i) });
        }
        for (i, &was) in before.iter().enumerate() {
            let now = chunk.block(unsafe { VoxelIndex::from_raw_unchecked(i as u16) });
            assert!(
                now.is_air() || now == was,
                "carving created block {now:?} at {i}"
            );
        }
    }

    #[test]
    fn world_of_round_trips_across_negative_chunks() {
        for &c in &[-3i32, -1, 0, 1, 5] {
            let pos = ChunkPos::new(c, c, c);
            for &l in &[0u8, 15] {
                let local = LocalPos::new(l, l, l);
                let (wx, wy, wz) = world_of(pos, local);
                let (rp, rl) = ChunkPos::split_world(wx, wy, wz);
                assert_eq!(rp, pos, "chunk round trip");
                assert_eq!(rl, local, "local round trip");
            }
        }
    }
}
