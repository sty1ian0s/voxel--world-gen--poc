//! Chunk coordinates.
//!
//! Chunk size is `16 = 2^4` (`PROJECT.md` §2), so a chunk coordinate is a
//! world coordinate shifted right by [`CHUNK_SHIFT`]. `i32` is kept rather than
//! bit-packing the position into a `u64` spatial hash: at `16^3` voxels per
//! chunk a hash key saves nothing measurable next to the 16 KiB of voxel data
//! it addresses, and `i32` triplets stay inspectable in a debugger, which is
//! what the `u64`-key alternative would have cost us.
//!
//! `O(1)` neighbour lookup comes from the index helpers below, not from the
//! key representation.

use super::LocalPos;

/// `log2(CHUNK_SIZE)`.
pub const CHUNK_SHIFT: u32 = 4;
/// Voxels per chunk axis.
pub const CHUNK_SIZE: usize = 1 << CHUNK_SHIFT;

/// A chunk's position in the chunk grid. `Ord` is derived so chunk maps can be
/// iterated in a stable order — reproducibility matters for the benchmarks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct ChunkPos {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl ChunkPos {
    pub const ORIGIN: Self = Self { x: 0, y: 0, z: 0 };

    #[inline]
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    /// The chunk containing world voxel `(wx, wy, wz)`.
    ///
    /// Arithmetic shift, not division: `/` rounds toward zero, which would make
    /// chunk `-1` hold voxels `-15..=0` instead of `-16..=-1`.
    #[inline]
    pub const fn from_world(wx: i32, wy: i32, wz: i32) -> Self {
        Self {
            x: wx >> CHUNK_SHIFT,
            y: wy >> CHUNK_SHIFT,
            z: wz >> CHUNK_SHIFT,
        }
    }

    /// Origin voxel of this chunk.
    #[inline]
    pub const fn origin(self) -> (i32, i32, i32) {
        (
            self.x << CHUNK_SHIFT,
            self.y << CHUNK_SHIFT,
            self.z << CHUNK_SHIFT,
        )
    }

    /// Chunk containing the voxel, plus the voxel's position inside it.
    #[inline]
    pub const fn split_world(wx: i32, wy: i32, wz: i32) -> (Self, LocalPos) {
        (
            Self::from_world(wx, wy, wz),
            LocalPos::from_world(wx, wy, wz),
        )
    }

    #[inline]
    pub const fn offset(self, dx: i32, dy: i32, dz: i32) -> Self {
        Self::new(self.x + dx, self.y + dy, self.z + dz)
    }

    /// Distance in chunks, used for level-of-detail and culling radii.
    #[inline]
    pub fn chebyshev(self, other: Self) -> i32 {
        (self.x - other.x)
            .abs()
            .max((self.y - other.y).abs())
            .max((self.z - other.z).abs())
    }

    /// Flat index for a dense chunk grid, row-major with `y` slowest.
    #[inline]
    pub const fn index(self, size_x: u32, size_z: u32) -> usize {
        (self.y as usize * size_z as usize + self.z as usize) * size_x as usize + self.x as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negative_world_coords_floor_toward_negative_infinity() {
        // The bug this guards: `/ 16` would put -1 in chunk 0, overlapping
        // chunk 1's voxels and corrupting every cave near the origin.
        assert_eq!(ChunkPos::from_world(-1, -1, -1), ChunkPos::new(-1, -1, -1));
        assert_eq!(ChunkPos::from_world(-16, -16, -16), ChunkPos::new(-1, -1, -1));
        assert_eq!(ChunkPos::from_world(-17, -17, -17), ChunkPos::new(-2, -2, -2));
        assert_eq!(ChunkPos::from_world(0, 0, 0), ChunkPos::ORIGIN);
        assert_eq!(ChunkPos::from_world(15, 15, 15), ChunkPos::ORIGIN);
        assert_eq!(ChunkPos::from_world(16, 16, 16), ChunkPos::new(1, 1, 1));
    }

    #[test]
    fn split_world_round_trips_to_the_original_voxel() {
        for &w in &[-33i32, -16, -1, 0, 1, 15, 16, 4095] {
            let (pos, local) = ChunkPos::split_world(w, w * 2, w);
            let (ox, oy, oz) = pos.origin();
            assert_eq!(ox + local.x() as i32, w, "x round trip at {w}");
            assert_eq!(oy + local.y() as i32, w * 2, "y round trip at {w}");
            assert_eq!(oz + local.z() as i32, w, "z round trip at {w}");
        }
    }

    #[test]
    fn index_is_injective_over_a_dense_grid() {
        let (sx, sz) = (4u32, 4u32);
        let mut seen = std::collections::HashSet::new();
        for y in 0..3 {
            for z in 0..sz as i32 {
                for x in 0..sx as i32 {
                    assert!(
                        seen.insert(ChunkPos::new(x, y, z).index(sx, sz)),
                        "collision at ({x},{y},{z})"
                    );
                }
            }
        }
        assert_eq!(seen.len(), 3 * 4 * 4);
    }
}
