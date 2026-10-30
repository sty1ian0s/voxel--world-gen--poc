//! Bit-packed voxel definition.
//!
//! `PROJECT.md` §2: one `u16` holding block id, sunlight level and block-light
//! level. Keeping light inside the voxel word means the mesher, the light
//! propagation and the carvers all read one contiguous 8 KiB array per chunk
//! instead of chasing a parallel array per attribute — and 8 KiB is small
//! enough to stay resident while a chunk is being worked on.

/// Block id, 8 bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct BlockId(u8);

impl BlockId {
    pub const AIR: Self = Self(0);
    pub const STONE: Self = Self(1);
    pub const DIRT: Self = Self(2);
    pub const GRASS: Self = Self(3);
    pub const SAND: Self = Self(4);
    pub const WATER: Self = Self(5);

    /// # Panics
    /// Never: the mask makes every `u8` representable.
    #[inline]
    pub const fn new(id: u8) -> Self {
        Self(id)
    }

    #[inline]
    pub const fn raw(self) -> u8 {
        self.0
    }

    /// Opaque blocks hide the face behind them and block light.
    #[inline]
    pub const fn is_opaque(self) -> bool {
        matches!(self, Self::STONE | Self::DIRT | Self::GRASS | Self::SAND)
    }

    /// Air is the only thing the mesher draws faces against.
    #[inline]
    pub const fn is_air(self) -> bool {
        matches!(self, Self::AIR)
    }

    /// Water attenuates light instead of stopping it, and is drawn but is not
    /// a valid surface to stand on for culling purposes.
    #[inline]
    pub const fn is_liquid(self) -> bool {
        matches!(self, Self::WATER)
    }

    /// Only opaque solids can occlude a whole chunk (`PROJECT.md` §4.3).
    #[inline]
    pub const fn is_solid(self) -> bool {
        self.is_opaque()
    }
}

impl From<u8> for BlockId {
    fn from(v: u8) -> Self {
        Self::new(v)
    }
}

/// Light level, 4 bits, `0..=15`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct LightLevel(u8);

impl LightLevel {
    pub const DARK: Self = Self(0);
    pub const MAX: Self = Self(15);

    #[inline]
    pub const fn new(v: u8) -> Self {
        Self(if v > 15 { 15 } else { v })
    }

    #[inline]
    pub const fn raw(self) -> u8 {
        self.0
    }

    #[inline]
    pub const fn is_dark(self) -> bool {
        self.0 == 0
    }

    /// One step of falloff, saturating at dark. Saturating is what stops a
    /// propagation loop from underflowing into 255 and lighting the world.
    #[inline]
    pub const fn dimmed(self) -> Self {
        Self(self.0.saturating_sub(1))
    }

    #[inline]
    pub const fn max(self, other: Self) -> Self {
        Self(if self.0 > other.0 { self.0 } else { other.0 })
    }
}

/// One voxel: `[ block id : 8 | block light : 4 | sunlight : 4 ]`.
///
/// Stored as a plain `u16` so a chunk is `[u16; 4096]`, which is exactly what
/// the GPU upload and the mesher want. Every accessor is a shift and a mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct PackedVoxel(u16);

impl PackedVoxel {
    pub const AIR: Self = Self(0);

    #[inline]
    pub const fn new(block: BlockId, sunlight: LightLevel, block_light: LightLevel) -> Self {
        Self(
            block.raw() as u16
                | ((block_light.raw() as u16 & 0xF) << 8)
                | ((sunlight.raw() as u16 & 0xF) << 12),
        )
    }

    /// Solely the block id, both light channels dark.
    #[inline]
    pub const fn of(block: BlockId) -> Self {
        Self::new(block, LightLevel::DARK, LightLevel::DARK)
    }

    #[inline]
    pub const fn raw(self) -> u16 {
        self.0
    }

    #[inline]
    pub const fn block(self) -> BlockId {
        BlockId::new((self.0 & 0xFF) as u8)
    }

    #[inline]
    pub const fn sunlight(self) -> LightLevel {
        LightLevel::new(((self.0 >> 12) & 0xF) as u8)
    }

    #[inline]
    pub const fn block_light(self) -> LightLevel {
        LightLevel::new(((self.0 >> 8) & 0xF) as u8)
    }

    #[inline]
    pub const fn with_block(self, block: BlockId) -> Self {
        Self::new(block, self.sunlight(), self.block_light())
    }

    #[inline]
    pub const fn with_sunlight(self, light: LightLevel) -> Self {
        Self::new(self.block(), light, self.block_light())
    }

    #[inline]
    pub const fn with_block_light(self, light: LightLevel) -> Self {
        Self::new(self.block(), self.sunlight(), light)
    }

    #[inline]
    pub const fn is_air(self) -> bool {
        self.block().is_air()
    }

    #[inline]
    pub const fn is_opaque(self) -> bool {
        self.block().is_opaque()
    }

    /// The value the mesher uses to decide whether two neighbours share a
    /// face: the same block and the same light on both sides.
    #[inline]
    pub const fn is_face_mergeable_with(self, other: Self) -> bool {
        self.0 == other.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fields_round_trip_at_their_extremes() {
        for &b in &[0u8, 1, 5, 127, 255] {
            for &s in &[0u8, 1, 15] {
                for &bl in &[0u8, 7, 15] {
                    let v = PackedVoxel::new(BlockId::new(b), LightLevel::new(s), LightLevel::new(bl));
                    assert_eq!(v.block().raw(), b, "block");
                    assert_eq!(v.sunlight().raw(), s, "sunlight");
                    assert_eq!(v.block_light().raw(), bl, "block light");
                }
            }
        }
    }

    #[test]
    fn light_channels_do_not_bleed_into_each_other_or_the_block_id() {
        // Sunlight sits in the top nibble and block light in bits 8..12; an
        // off-by-one shift here silently corrupts shading rather than crashing.
        let v = PackedVoxel::new(BlockId::STONE, LightLevel::MAX, LightLevel::new(1));
        assert_eq!(v.raw(), 0b1111_0001_0000_0001);
        assert_eq!(v.block(), BlockId::STONE);
        assert_eq!(v.sunlight(), LightLevel::MAX);
        assert_eq!(v.block_light().raw(), 1);

        // Changing one channel leaves the others untouched.
        let s = v.with_sunlight(LightLevel::new(3));
        assert_eq!(s.block_light(), v.block_light());
        assert_eq!(s.block(), v.block());
        let d = v.with_block_light(LightLevel::new(9));
        assert_eq!(d.sunlight(), v.sunlight());
        assert_eq!(d.block(), v.block());
    }

    #[test]
    fn air_is_all_zero() {
        assert_eq!(PackedVoxel::AIR.raw(), 0);
        assert!(PackedVoxel::AIR.is_air());
        assert!(PackedVoxel::AIR.sunlight().is_dark());
    }

    #[test]
    fn light_level_saturates_instead_of_underflowing() {
        assert_eq!(LightLevel::DARK.dimmed(), LightLevel::DARK);
        assert_eq!(LightLevel::new(3).dimmed().raw(), 2);
        assert_eq!(LightLevel::MAX.dimmed().raw(), 14);
        // Out-of-range input clamps rather than wrapping into a bright value.
        assert_eq!(LightLevel::new(200).raw(), 15);
    }

    #[test]
    fn face_merge_requires_identical_block_and_light() {
        let a = PackedVoxel::new(BlockId::STONE, LightLevel::new(5), LightLevel::DARK);
        let same = PackedVoxel::new(BlockId::STONE, LightLevel::new(5), LightLevel::DARK);
        // Same block, different light: merging these would smear the gradient.
        let lit = PackedVoxel::new(BlockId::STONE, LightLevel::new(6), LightLevel::DARK);
        assert!(a.is_face_mergeable_with(same));
        assert!(!a.is_face_mergeable_with(lit));
        assert!(!a.is_face_mergeable_with(PackedVoxel::of(BlockId::DIRT)));
    }

    #[test]
    fn block_classification_matches_physics_expectations() {
        assert!(BlockId::STONE.is_opaque() && BlockId::STONE.is_solid());
        assert!(!BlockId::WATER.is_opaque() && BlockId::WATER.is_liquid());
        assert!(BlockId::AIR.is_air() && !BlockId::AIR.is_opaque());
        logger_asserts();
    }

    /// Compile-time proof that a chunk's voxel array is `u16` and therefore
    /// uploadable as bytes without conversion.
    fn logger_asserts() {
        use bytemuck::Pod;
        fn assert_pod<T: Pod>() {}
        assert_pod::<u16>();
        assert_eq!(std::mem::size_of::<PackedVoxel>(), 2);
        assert_eq!(std::mem::size_of::<[PackedVoxel; 4096]>(), 8192);
    }
}
