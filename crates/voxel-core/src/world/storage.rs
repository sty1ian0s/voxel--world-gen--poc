//! Chunk and grid storage.
//!
//! Layout note, since this is where the "SoA vs AoSoA" question actually bites.
//!
//! A pure SoA split (`blocks: [u8; 4096]`, `sun: [u8; 4096]`, `block: [u8; 4096]`)
//! optimises any loop that touches exactly one attribute: the light pass reads
//! 4 KiB of sunlight instead of streaming the full 8 KiB. But the terrain
//! carvers and the greedy mesher touch block id *and* light for the same voxel,
//! and under SoA those two reads land 4 KiB apart, so each voxel access can
//! become two cache lines instead of one.
//!
//! Interleaving the attributes (AoSoA, packing everything back into one `u16`
//! per voxel) fixes that locality and is what the GPU wants anyway, but it
//! doubles the bytes streamed by the light pass, which is the one stage that
//! genuinely touches a single attribute.
//!
//! Which wins is workload-dependent and is measured, not assumed — see
//! `benches/carvers.rs`. The storage here is the SoA split, because it is the
//! representation the doc specifies (`PROJECT.md` §2) and the one that keeps
//! the light pass cheap; `to_packed` converts to the interleaved `u16` form
//! once, at the GPU boundary, so neither consumer pays for the other's layout.

use crate::domain::{
    BlockId, CHUNK_VOLUME, ChunkPos, LightLevel, LocalPos, PackedVoxel, VoxelIndex,
};

/// One `16^3` chunk, split by attribute, 64-byte aligned.
///
/// Each attribute array is 4 KiB and 64-byte aligned, so a walk along `x` is a
/// straight sequential scan of one array with no false sharing between
/// attributes, and the whole struct is still small enough to stay hot while a
/// chunk is being generated.
///
/// Alignment matters because the chunk is staged to the GPU as a buffer;
/// deriving `Align(64)` means no realigning copy at upload time.
#[derive(Clone)]
#[repr(align(64))]
pub struct VoxelGroup {
    /// Block ids, indexed by the chunk's linear [`VoxelIndex`].
    blocks: [u8; CHUNK_VOLUME],
    /// Sunlight, same indexing.
    sun: [u8; CHUNK_VOLUME],
    /// Block light, same indexing.
    block: [u8; CHUNK_VOLUME],
}

impl Default for VoxelGroup {
    fn default() -> Self {
        Self {
            blocks: [0; CHUNK_VOLUME],
            sun: [0; CHUNK_VOLUME],
            block: [0; CHUNK_VOLUME],
        }
    }
}

impl VoxelGroup {
    /// Row-major slot, matching [`VoxelIndex`]'s bit layout so that walking `x`
    /// steps by 1 and the two representations never disagree.
    #[inline]
    const fn slot(pos: LocalPos) -> usize {
        VoxelIndex::from_local(pos).as_usize()
    }

    #[inline]
    pub fn block(&self, pos: LocalPos) -> BlockId {
        BlockId::new(self.blocks[Self::slot(pos)])
    }

    #[inline]
    pub fn set_block(&mut self, pos: LocalPos, block: BlockId) {
        self.blocks[Self::slot(pos)] = block.raw();
    }

    #[inline]
    pub fn sun(&self, pos: LocalPos) -> LightLevel {
        LightLevel::new(self.sun[Self::slot(pos)])
    }

    #[inline]
    pub fn set_sun(&mut self, pos: LocalPos, light: LightLevel) {
        self.sun[Self::slot(pos)] = light.raw();
    }

    #[inline]
    pub fn block_light(&self, pos: LocalPos) -> LightLevel {
        LightLevel::new(self.block[Self::slot(pos)])
    }

    #[inline]
    pub fn set_block_light(&mut self, pos: LocalPos, light: LightLevel) {
        self.block[Self::slot(pos)] = light.raw();
    }

    /// Reassembles the interleaved form the mesher and the GPU consume.
    #[inline]
    pub fn get(&self, pos: LocalPos) -> PackedVoxel {
        PackedVoxel::new(self.block(pos), self.sun(pos), self.block_light(pos))
    }

    #[inline]
    pub fn set(&mut self, pos: LocalPos, v: PackedVoxel) {
        let s = Self::slot(pos);
        self.blocks[s] = v.block().raw();
        self.sun[s] = v.sunlight().raw();
        self.block[s] = v.block_light().raw();
    }

    /// Interleaves the whole chunk into packed `u16`s, for GPU upload and for
    /// comparisons. This is the only place the two layouts meet.
    pub fn write_packed(&self, out: &mut [u16]) {
        debug_assert!(out.len() >= CHUNK_VOLUME);
        for (i, o) in out.iter_mut().take(CHUNK_VOLUME).enumerate() {
            *o = PackedVoxel::new(
                BlockId::new(self.blocks[i]),
                LightLevel::new(self.sun[i]),
                LightLevel::new(self.block[i]),
            )
            .raw();
        }
    }
}

/// One chunk: attribute-split voxels plus the cached cull flags.
///
/// The flags exist so the renderer can skip work without touching voxel data
/// at all (`PROJECT.md` §4.3).
#[derive(Clone, Default)]
pub struct Chunk {
    voxels: VoxelGroup,
    /// Set while the chunk has not been generated yet.
    pub dirty: bool,
    /// Cached "is every voxel air".
    pub empty: bool,
    /// Cached "is every voxel solid and opaque".
    pub enclosed: bool,
}

impl Chunk {
    #[inline]
    pub fn get(&self, index: VoxelIndex) -> PackedVoxel {
        self.voxels.get(index.to_local())
    }

    #[inline]
    pub fn set(&mut self, index: VoxelIndex, v: PackedVoxel) {
        self.voxels.set(index.to_local(), v);
    }

    #[inline]
    pub fn block(&self, index: VoxelIndex) -> BlockId {
        self.voxels.block(index.to_local())
    }

    #[inline]
    pub fn set_block(&mut self, index: VoxelIndex, block: BlockId) {
        self.voxels.set_block(index.to_local(), block);
    }

    #[inline]
    pub fn sun(&self, index: VoxelIndex) -> LightLevel {
        self.voxels.sun(index.to_local())
    }

    #[inline]
    pub fn set_sun(&mut self, index: VoxelIndex, light: LightLevel) {
        self.voxels.set_sun(index.to_local(), light);
    }

    #[inline]
    pub fn block_light(&self, index: VoxelIndex) -> LightLevel {
        self.voxels.block_light(index.to_local())
    }

    #[inline]
    pub fn set_block_light(&mut self, index: VoxelIndex, light: LightLevel) {
        self.voxels.set_block_light(index.to_local(), light);
    }

    /// World-coordinate convenience used at boundaries, not in hot loops.
    #[inline]
    pub fn block_at(&self, local: LocalPos) -> BlockId {
        self.voxels.block(local)
    }

    pub fn fill(&mut self, block: BlockId) {
        self.voxels.blocks.fill(block.raw());
    }

    /// Recomputes the cached skip flags. Called once after generation or
    /// editing, never during rendering.
    pub fn refresh_flags(&mut self) {
        let mut empty = true;
        let mut enclosed = true;
        for &b in self.voxels.blocks.iter() {
            let b = BlockId::new(b);
            if !b.is_air() {
                empty = false;
            }
            if !b.is_solid() {
                enclosed = false;
            }
            if !empty && !enclosed {
                break;
            }
        }
        // A chunk of pure air is vacuously "solid", which would wrongly cull
        // the sky; `empty` wins by construction.
        self.empty = empty;
        self.enclosed = enclosed && !empty;
        self.dirty = false;
    }

    /// Interleaves to the packed `u16` form used for GPU upload.
    pub fn to_packed(&self) -> Vec<u16> {
        let mut out = vec![0u16; CHUNK_VOLUME];
        self.voxels.write_packed(&mut out);
        out
    }

    /// Direct read of the block array, for bulk passes that do not need light.
    #[inline]
    pub fn blocks(&self) -> &[u8; CHUNK_VOLUME] {
        &self.voxels.blocks
    }
}

/// A dense rectangle of chunks.
///
/// The world in this POC is a fixed box, so a dense grid beats a hash map: one
/// multiply-add per lookup, no hashing, no collisions, and `chunks()` iterates
/// in `(y, z, x)` order which keeps the parallel carver's writes sequential.
pub struct ChunkGrid {
    chunks: Vec<Chunk>,
    size_x: u32,
    size_y: u32,
    size_z: u32,
}

impl ChunkGrid {
    pub fn new(size_x: u32, size_y: u32, size_z: u32) -> Self {
        assert!(
            size_x > 0 && size_y > 0 && size_z > 0,
            "chunk grid dimensions must be non-zero"
        );
        let count = size_x as usize * size_y as usize * size_z as usize;
        Self {
            chunks: (0..count).map(|_| Chunk::default()).collect(),
            size_x,
            size_y,
            size_z,
        }
    }

    #[inline]
    pub const fn size(&self) -> (u32, u32, u32) {
        (self.size_x, self.size_y, self.size_z)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// # Panics
    /// If `pos` is outside the grid. This is a boundary: callers convert a
    /// world position into a `ChunkPos` and must have checked the world bounds
    /// already, which `World` does once on entry.
    #[inline]
    pub fn index(&self, pos: ChunkPos) -> usize {
        debug_assert!(
            pos.x >= 0
                && pos.y >= 0
                && pos.z >= 0
                && pos.x < self.size_x as i32
                && pos.y < self.size_y as i32
                && pos.z < self.size_z as i32,
            "chunk {pos:?} outside grid {:?}",
            self.size()
        );
        (pos.y as usize * self.size_z as usize + pos.z as usize) * self.size_x as usize
            + pos.x as usize
    }

    #[inline]
    pub fn contains(&self, pos: ChunkPos) -> bool {
        pos.x >= 0
            && pos.y >= 0
            && pos.z >= 0
            && pos.x < self.size_x as i32
            && pos.y < self.size_y as i32
            && pos.z < self.size_z as i32
    }

    #[inline]
    pub fn get(&self, pos: ChunkPos) -> &Chunk {
        &self.chunks[self.index(pos)]
    }

    #[inline]
    pub fn get_mut(&mut self, pos: ChunkPos) -> &mut Chunk {
        let i = self.index(pos);
        &mut self.chunks[i]
    }

    /// All chunk positions, in storage order.
    pub fn positions(&self) -> impl Iterator<Item = ChunkPos> + '_ {
        let (sx, sy, sz) = (self.size_x as i32, self.size_y as i32, self.size_z as i32);
        (0..sy).flat_map(move |y| {
            (0..sz).flat_map(move |z| (0..sx).map(move |x| ChunkPos::new(x, y, z)))
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = (ChunkPos, &Chunk)> + '_ {
        self.positions().map(move |p| (p, self.get(p)))
    }

    /// Mutable access in storage order, for the parallel carver.
    pub fn par_chunks_mut(&mut self) -> impl rayon::iter::IndexedParallelIterator<Item = &mut Chunk> {
        use rayon::prelude::*;
        self.chunks.par_iter_mut()
    }

    pub fn chunks(&self) -> &[Chunk] {
        &self.chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_chunk_is_three_aligned_attribute_arrays() {
        // 3 * 4096 bytes = 12 KiB of voxel storage, and the alignment the GPU
        // upload relies on to avoid a realigning copy.
        assert_eq!(std::mem::size_of::<VoxelGroup>(), 3 * CHUNK_VOLUME);
        assert_eq!(std::mem::align_of::<VoxelGroup>(), 64);
    }

    #[test]
    fn voxels_survive_a_full_chunk_round_trip() {
        let mut chunk = Chunk::default();
        let mut expected = Vec::with_capacity(CHUNK_VOLUME);
        for y in 0..16u8 {
            for z in 0..16u8 {
                for x in 0..16u8 {
                    let v = PackedVoxel::new(
                        BlockId::new((x ^ z ^ y) | 1),
                        LightLevel::new(y % 16),
                        LightLevel::new(z % 16),
                    );
                    chunk.set(VoxelIndex::from_parts(x, y, z), v);
                    expected.push(v);
                }
            }
        }
        for y in 0..16u8 {
            for z in 0..16u8 {
                for x in 0..16u8 {
                    let i = VoxelIndex::from_parts(x, y, z);
                    let got = chunk.get(i);
                    assert_eq!(got, expected[i.as_usize()], "at ({x},{y},{z})");
                }
            }
        }
    }

    #[test]
    fn attribute_arrays_do_not_alias_across_the_x_axis() {
        // A stride bug in the slot index would make adjacent x values collide.
        let mut chunk = Chunk::default();
        for x in 0..16u8 {
            chunk.set_block(VoxelIndex::from_parts(x, 3, 7), BlockId::new(x + 1));
        }
        for x in 0..16u8 {
            assert_eq!(
                chunk.block(VoxelIndex::from_parts(x, 3, 7)).raw(),
                x + 1,
                "x={x} was aliased"
            );
        }
    }

    #[test]
    fn interleaving_matches_per_voxel_reads() {
        let mut chunk = Chunk::default();
        for y in 0..16u8 {
            for z in 0..16u8 {
                for x in 0..16u8 {
                    chunk.set(
                        VoxelIndex::from_parts(x, y, z),
                        PackedVoxel::new(BlockId::new(y + 1), LightLevel::new(x % 16), LightLevel::new(z % 16)),
                    );
                }
            }
        }
        let flat = chunk.to_packed();
        assert_eq!(flat.len(), CHUNK_VOLUME);
        for y in 0..16u8 {
            for z in 0..16u8 {
                for x in 0..16u8 {
                    let i = VoxelIndex::from_parts(x, y, z);
                    assert_eq!(
                        flat[i.as_usize()],
                        chunk.get(i).raw(),
                        "mismatch at ({x},{y},{z})"
                    );
                }
            }
        }
    }

    #[test]
    fn flags_distinguish_air_from_enclosed_solid() {
        let mut empty = Chunk::default();
        empty.refresh_flags();
        assert!(empty.empty, "all-air chunk must be cullable");
        assert!(!empty.enclosed, "all-air chunk must not be culled as solid");

        let mut solid = Chunk::default();
        solid.fill(BlockId::STONE);
        solid.refresh_flags();
        assert!(!solid.empty);
        assert!(solid.enclosed, "solid chunk must be cullable as enclosed");

        let mut mixed = Chunk::default();
        mixed.fill(BlockId::STONE);
        mixed.set_block(VoxelIndex::from_parts(1, 1, 1), BlockId::AIR);
        mixed.refresh_flags();
        assert!(!mixed.empty);
        assert!(!mixed.enclosed, "a single air voxel must defeat enclosure");

        // Water is solid ground for culling but not opaque, so it must not count
        // as enclosed either.
        let mut wet = Chunk::default();
        wet.fill(BlockId::WATER);
        wet.refresh_flags();
        assert!(!wet.enclosed);
    }

    #[test]
    fn grid_index_is_injective_and_ordered_y_z_x() {
        let grid = ChunkGrid::new(3, 2, 4);
        let mut seen = std::collections::HashSet::new();
        let mut ordered = Vec::new();
        for (i, pos) in grid.positions().enumerate() {
            assert!(grid.contains(pos));
            assert_eq!(grid.index(pos), i, "positions() must follow storage order");
            assert!(seen.insert(grid.index(pos)));
            ordered.push(pos);
        }
        assert_eq!(ordered.len(), 24);
        assert_eq!(ordered[0], ChunkPos::new(0, 0, 0));
        assert_eq!(ordered[1], ChunkPos::new(1, 0, 0));
        assert_eq!(ordered[3], ChunkPos::new(0, 0, 1));
        assert_eq!(ordered[12], ChunkPos::new(0, 1, 0));

        assert!(!grid.contains(ChunkPos::new(-1, 0, 0)));
        assert!(!grid.contains(ChunkPos::new(3, 0, 0)));
        assert!(!grid.contains(ChunkPos::new(0, 2, 0)));
    }

    #[test]
    fn parallel_iteration_reaches_every_chunk() {
        use rayon::prelude::*;
        let mut grid = ChunkGrid::new(4, 2, 4);
        grid.par_chunks_mut().enumerate().for_each(|(i, c)| {
            c.fill(BlockId::new((i % 200) as u8 + 1));
        });
        assert_eq!(grid.len(), 32);
        // `fill` does not refresh flags, so emptiness is asserted on voxel data.
        let populated = grid
            .iter()
            .filter(|(_, c)| !c.block(VoxelIndex::from_parts(0, 0, 0)).is_air())
            .count();
        assert_eq!(populated, grid.len(), "every chunk must have been written");
    }
}
