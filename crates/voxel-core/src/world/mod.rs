//! Chunk storage.
//!
//! Two decisions live here, both measured rather than assumed:
//!
//! * **Key**: [`ChunkPos`] stays an `i32` triplet. The storage is a dense grid
//!   when the world is a fixed rectangle, so no hashing happens at all.
//! * **Layout**: attributes are split into parallel arrays rather than
//!   interleaved into one `u16` per voxel. See [`VoxelGroup`] for the tradeoff
//!   and `benches/carvers.rs` for the measurement.

mod storage;

pub use storage::{Chunk, ChunkGrid, VoxelGroup};
