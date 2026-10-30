//! Headless debug views.
//!
//! The point of this module is that correctness is *observed* rather than
//! asserted. Every stage of the pipeline can be dumped as text or as an image,
//! and the dumps can be read by eye or diffed in CI. It is deliberately
//! headless: it works over SSH, in a container, and on a machine with no GPU,
//! which is also where generation bugs are easiest to hit and hardest to see.
//!
//! Three kinds of view:
//!
//! * [`ascii`] — one slice, one character per voxel, small enough to read.
//! * [`image`] — PNG-free PGM/PPM cross-sections, for eyeballing large areas.
//! * [`report`] — counters that make the `O(N^3)` vs `O(S)` claim auditable.
//! * [`caves`] — cave *shape* statistics, for judging MC-likeness rather than
//!   mere coverage. Coverage parity cannot tell tunnels from spheres.

pub mod ascii;
pub mod caves;
pub mod image;
pub mod report;

/// Which slice of the world a view shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    /// The two axes that vary within a slice of this orientation.
    pub const fn plane_axes(self) -> (Axis, Axis) {
        match self {
            // Looking down X: the slice spans Z (horizontal) and Y (vertical).
            Axis::X => (Axis::Z, Axis::Y),
            Axis::Y => (Axis::X, Axis::Z),
            Axis::Z => (Axis::X, Axis::Y),
        }
    }

    pub const fn name(self) -> char {
        match self {
            Axis::X => 'x',
            Axis::Y => 'y',
            Axis::Z => 'z',
        }
    }
}
