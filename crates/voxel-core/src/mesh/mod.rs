//! Greedy meshing (`PROJECT.md` §4.4).
//!
//! Converts a chunk into as few quads as possible by merging coplanar faces
//! that share a block id and light level. The merge is done with bitwise scans
//! over a mask per slice rather than by growing rectangles two at a time, which
//! is both simpler and branch-free in the inner loop.
//!
//! Two implementations live here on purpose:
//!
//! * [`greedy`] — the real mesher.
//! * [`naive`] — one quad per visible face, slow but obviously correct.
//!
//! The naive mesher exists to be compared against. A greedy mesher that drops
//! or double-covers a face still produces a plausible-looking mesh, and the
//! only reliable way to catch that is to check that both meshers cover exactly
//! the same set of faces, which the tests do.

pub mod greedy;
pub mod naive;

pub use greedy::{Mesh, Quad, mesh_chunk};
pub use naive::mesh_chunk_naive;

use crate::domain::BlockId;

/// The six face directions, with the axis they face along.
///
/// `pub` because the culling and GPU stages both need to know a quad's normal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Direction {
    PosX,
    NegX,
    PosY,
    NegY,
    PosZ,
    NegZ,
}

impl Direction {
    pub const ALL: [Direction; 6] = [
        Direction::PosX,
        Direction::NegX,
        Direction::PosY,
        Direction::NegY,
        Direction::PosZ,
        Direction::NegZ,
    ];

    /// Outward normal, as a unit vector.
    pub const fn normal(self) -> [f32; 3] {
        match self {
            Direction::PosX => [1.0, 0.0, 0.0],
            Direction::NegX => [-1.0, 0.0, 0.0],
            Direction::PosY => [0.0, 1.0, 0.0],
            Direction::NegY => [0.0, -1.0, 0.0],
            Direction::PosZ => [0.0, 0.0, 1.0],
            Direction::NegZ => [0.0, 0.0, -1.0],
        }
    }

    /// Axis index the face is perpendicular to: `0 = x`, `1 = y`, `2 = z`.
    pub const fn axis(self) -> usize {
        match self {
            Direction::PosX | Direction::NegX => 0,
            Direction::PosY | Direction::NegY => 1,
            Direction::PosZ | Direction::NegZ => 2,
        }
    }

    /// True if the face points along the positive axis direction.
    pub const fn is_positive(self) -> bool {
        matches!(self, Direction::PosX | Direction::PosY | Direction::PosZ)
    }

    /// The opposite face.
    pub const fn opposite(self) -> Self {
        match self {
            Direction::PosX => Direction::NegX,
            Direction::NegX => Direction::PosX,
            Direction::PosY => Direction::NegY,
            Direction::NegY => Direction::PosY,
            Direction::PosZ => Direction::NegZ,
            Direction::NegZ => Direction::PosZ,
        }
    }

    /// Unit step toward the neighbour across this face.
    pub const fn step(self) -> (i32, i32, i32) {
        match self {
            Direction::PosX => (1, 0, 0),
            Direction::NegX => (-1, 0, 0),
            Direction::PosY => (0, 1, 0),
            Direction::NegY => (0, -1, 0),
            Direction::PosZ => (0, 0, 1),
            Direction::NegZ => (0, 0, -1),
        }
    }
}

/// Decides whether a face between two voxels is drawn.
///
/// A face is drawn when exactly one side is opaque: solid-to-air is visible,
/// solid-to-solid is hidden, and air-to-air is nothing at all. Liquids are
/// treated as non-opaque, so a face is drawn where water meets air and also
/// where water meets stone, which is what makes an ocean surface render.
#[inline]
pub fn should_draw_face(here: BlockId, neighbour: BlockId) -> bool {
    if here.is_air() && neighbour.is_air() {
        return false;
    }
    if here.is_opaque() && neighbour.is_opaque() {
        return false;
    }
    // Two identical non-opaque blocks (water against water) hide their shared
    // face, so an ocean interior is not meshed.
    if here == neighbour && !here.is_opaque() {
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solid_air_boundary_draws_one_face() {
        assert!(should_draw_face(BlockId::STONE, BlockId::AIR));
        assert!(should_draw_face(BlockId::AIR, BlockId::STONE));
    }

    #[test]
    fn solid_solid_and_air_air_draw_nothing() {
        assert!(!should_draw_face(BlockId::STONE, BlockId::DIRT));
        assert!(!should_draw_face(BlockId::AIR, BlockId::AIR));
    }

    #[test]
    fn water_surfaces_are_drawn_but_water_interiors_are_not() {
        assert!(should_draw_face(BlockId::WATER, BlockId::AIR), "ocean surface");
        assert!(should_draw_face(BlockId::WATER, BlockId::STONE), "sea floor");
        assert!(
            !should_draw_face(BlockId::WATER, BlockId::WATER),
            "water interior must not be meshed"
        );
    }

    #[test]
    fn directions_are_consistent_with_their_normals() {
        for d in Direction::ALL {
            let n = d.normal();
            let s = d.step();
            let step = [s.0, s.1, s.2];
            assert_eq!(n[d.axis()] as i32, step[d.axis()]);
            assert_eq!(d.opposite().opposite(), d);
            let o = d.opposite().normal();
            assert_eq!(n[0] + o[0], 0.0);
            assert_eq!(n[1] + o[1], 0.0);
            assert_eq!(n[2] + o[2], 0.0);
            // The normal must be a unit vector along exactly one axis.
            assert_eq!(n.iter().filter(|&&v| v != 0.0).count(), 1);
        }
    }
}
