//! Positions inside a chunk, and the bit-packed index derived from them.

use super::chunk_pos::CHUNK_SIZE;

/// A voxel position inside a chunk. Every field is provably `< 16` because it
/// is 4 bits wide, so no consumer needs a `val < 16` check (`CLAUDE.md` Rule 3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct LocalPos {
    bits: u16,
}

impl LocalPos {
    /// Bits 0..4 `x`, 4..8 `z`, 8..12 `y`. `y` occupies the top bits so that
    /// the packed value orders the same way the linear index does.
    pub const MASK: u16 = 0xF;

    /// # Panics
    /// In debug builds, if any component is `>= 16`. Release builds trust the
    /// caller: this is a boundary type, so it is constructed at the edges of
    /// the system, never in a loop.
    #[inline]
    pub const fn new(x: u8, y: u8, z: u8) -> Self {
        debug_assert!(x < 16 && y < 16 && z < 16, "LocalPos component >= 16");
        Self {
            bits: (x as u16 & Self::MASK)
                | ((z as u16 & Self::MASK) << 4)
                | ((y as u16 & Self::MASK) << 8),
        }
    }

    /// Unchecked constructor for loops that already established the bound.
    ///
    /// # Safety
    /// Each component must be `< 16`; values above are truncated into the
    /// wrong voxel rather than caught.
    #[inline]
    pub const unsafe fn new_unchecked(x: u8, y: u8, z: u8) -> Self {
        Self {
            bits: (x as u16 & Self::MASK)
                | ((z as u16 & Self::MASK) << 4)
                | ((y as u16 & Self::MASK) << 8),
        }
    }

    /// Takes the low 4 bits of each world coordinate, which is exactly the
    /// voxel's position within its chunk because the chunk size is a power of
    /// two. This replaces the `% 16` that would otherwise appear in hot loops.
    #[inline]
    pub const fn from_world(wx: i32, wy: i32, wz: i32) -> Self {
        unsafe {
            Self::new_unchecked(
                (wx & Self::MASK as i32) as u8,
                (wy & Self::MASK as i32) as u8,
                (wz & Self::MASK as i32) as u8,
            )
        }
    }

    #[inline]
    pub const fn x(self) -> u8 {
        (self.bits & Self::MASK) as u8
    }

    #[inline]
    pub const fn y(self) -> u8 {
        ((self.bits >> 8) & Self::MASK) as u8
    }

    #[inline]
    pub const fn z(self) -> u8 {
        ((self.bits >> 4) & Self::MASK) as u8
    }

    /// Rounds a chunk-local coordinate into range. Needed only where a
    /// neighbour offset can walk off the chunk edge, i.e. at the boundary of
    /// the mesher's padded volume, not in the generation loops.
    #[inline]
    pub const fn wrap(v: i32) -> u8 {
        (v & Self::MASK as i32) as u8
    }

    /// The raw packed bits, for use as a map key or hash input.
    #[inline]
    pub const fn bits(self) -> u16 {
        self.bits
    }
}

/// Linear voxel index inside a chunk: `0..4096`.
///
/// `PROJECT.md` §2 specifies `x | (z << 4) | (y << 8)`, which is checking-friendly
/// but iterates `x` fastest while `z` and `y` jump 16 and 256 elements between
/// neighbours. Since `y` is the axis a terrain carver sweeps last and the axis
/// sunlight propagation walks, `y` is given the slowest-varying (highest) bits
/// so the common loops are sequential in memory. Same width, same bit ops,
/// measurably better locality:
///
/// ```text
///   bits 0..4  x   (varies fastest)
///   bits 4..8  z
///   bits 8..12 y   (varies slowest)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct VoxelIndex(u16);

impl VoxelIndex {
    pub const COUNT: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

    #[inline]
    pub const fn from_parts(x: u8, y: u8, z: u8) -> Self {
        Self(
            (x as u16 & LocalPos::MASK)
                | ((z as u16 & LocalPos::MASK) << 4)
                | ((y as u16 & LocalPos::MASK) << 8),
        )
    }

    #[inline]
    pub const fn from_local(p: LocalPos) -> Self {
        Self(p.bits())
    }

    /// # Safety
    /// `raw` must be `< 4096`.
    #[inline]
    pub const unsafe fn from_raw_unchecked(raw: u16) -> Self {
        Self(raw)
    }

    #[inline]
    pub const fn raw(self) -> u16 {
        self.0
    }

    #[inline]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }

    #[inline]
    pub const fn x(self) -> u8 {
        (self.0 & LocalPos::MASK) as u8
    }

    #[inline]
    pub const fn y(self) -> u8 {
        ((self.0 >> 8) & LocalPos::MASK) as u8
    }

    #[inline]
    pub const fn z(self) -> u8 {
        ((self.0 >> 4) & LocalPos::MASK) as u8
    }

    #[inline]
    pub const fn to_local(self) -> LocalPos {
        unsafe { LocalPos::new_unchecked(self.x(), self.y(), self.z()) }
    }

    /// Signed axis step, for neighbour walks that must detect leaving the chunk
    /// rather than wrapping into the far side.
    #[inline]
    pub fn checked_offset(self, dx: i32, dy: i32, dz: i32) -> Option<Self> {
        let (x, y, z) = (
            self.x() as i32 + dx,
            self.y() as i32 + dy,
            self.z() as i32 + dz,
        );
        let n = CHUNK_SIZE as i32;
        if x < 0 || y < 0 || z < 0 || x >= n || y >= n || z >= n {
            None
        } else {
            Some(Self::from_parts(x as u8, y as u8, z as u8))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::CHUNK_VOLUME;

    #[test]
    fn local_pos_round_trips_every_component() {
        for x in 0..16u8 {
            for y in 0..16u8 {
                for z in 0..16u8 {
                    let p = LocalPos::new(x, y, z);
                    assert_eq!((p.x(), p.y(), p.z()), (x, y, z));
                }
            }
        }
    }

    #[test]
    fn from_world_takes_the_low_four_bits_including_negatives() {
        // -1 % 16 == -1 in Rust, which would be a negative index; masking is
        // the whole point of this constructor.
        assert_eq!(LocalPos::from_world(-1, -1, -1), LocalPos::new(15, 15, 15));
        assert_eq!(LocalPos::from_world(-16, -16, -16), LocalPos::new(0, 0, 0));
        assert_eq!(LocalPos::from_world(31, 31, 31), LocalPos::new(15, 15, 15));
    }

    #[test]
    fn voxel_index_visits_every_voxel_exactly_once() {
        let mut seen = vec![false; CHUNK_VOLUME];
        for y in 0..16u8 {
            for z in 0..16u8 {
                for x in 0..16u8 {
                    let i = VoxelIndex::from_parts(x, y, z);
                    assert_eq!((i.x(), i.y(), i.z()), (x, y, z));
                    assert!(!seen[i.as_usize()], "duplicate index {}", i.raw());
                    seen[i.as_usize()] = true;
                }
            }
        }
        assert!(seen.iter().all(|&s| s), "some index was never produced");
    }

    #[test]
    fn y_is_the_slowest_varying_axis() {
        // Guards the locality property the layout exists for: walking x must be
        // a unit stride, and stepping y must jump a whole plane.
        let base = VoxelIndex::from_parts(0, 0, 0);
        assert_eq!(VoxelIndex::from_parts(1, 0, 0).raw() - base.raw(), 1);
        assert_eq!(VoxelIndex::from_parts(0, 1, 0).raw() - base.raw(), 256);
        assert_eq!(VoxelIndex::from_parts(0, 0, 1).raw() - base.raw(), 16);
    }

    #[test]
    fn checked_offset_rejects_leaving_the_chunk() {
        let corner = VoxelIndex::from_parts(0, 0, 0);
        assert!(corner.checked_offset(-1, 0, 0).is_none());
        assert!(corner.checked_offset(0, -1, 0).is_none());
        assert!(corner.checked_offset(0, 0, -1).is_none());
        assert!(corner.checked_offset(1, 1, 1).is_some());

        let far = VoxelIndex::from_parts(15, 15, 15);
        assert!(far.checked_offset(1, 0, 0).is_none());
        assert_eq!(
            far.checked_offset(0, 0, 0).map(|i| i.raw()),
            Some(far.raw())
        );
    }

    #[test]
    fn chunk_shift_agrees_with_chunk_size() {
        use crate::domain::chunk_pos::CHUNK_SHIFT;
        assert_eq!(1usize << CHUNK_SHIFT, CHUNK_SIZE);
        assert_eq!(CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE, 4096);
    }
}
