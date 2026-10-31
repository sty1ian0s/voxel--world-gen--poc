//! Chunk storage.
//!
//! Two decisions live here, both measured rather than assumed:
//!
//! * **Key**: [`ChunkPos`] stays an `i32` triplet. The storage is a dense grid
//!   when the world is a fixed rectangle, so no hashing happens at all.
//! * **Layout**: attributes are split into parallel arrays rather than
//!   interleaved into one `u16` per voxel. See [`VoxelGroup`] for the tradeoff
//!   and `benches/carvers.rs` for the measurement.
//!
//! [`stream`] adds the third decision: a chunk is a *function* of its position
//! and the seed, so a world can be generated one chunk at a time with no
//! world-sized allocation at all.

mod storage;
mod stream;

pub use storage::{Chunk, ChunkGrid, VoxelGroup};
pub use stream::{CarverSpec, Region, VerticalBounds, WorldGen, carve_with};
