//! Core domain types.
//!
//! "Parse, don't validate": every type here is constructed once from raw input
//! and from then on its invariants hold by construction, so inner loops index
//! and compare without a single bounds or range check.

mod chunk_pos;
mod local_pos;
mod voxel;

pub use chunk_pos::{CHUNK_SHIFT, CHUNK_SIZE, ChunkPos};
pub use local_pos::{LocalPos, VoxelIndex};
pub use voxel::{BlockId, LightLevel, PackedVoxel};

/// Voxels in a chunk, `16^3`.
pub const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
