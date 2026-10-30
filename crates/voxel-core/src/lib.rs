//! Data-oriented voxel world generation core.
//!
//! Layout of the crate mirrors the pipeline in `PROJECT.md` §4:
//! [`config`] -> [`chunk`] storage -> terrain passes -> [`mesh`] -> culling.
//! Every stage is plain data in, plain data out, so each one is independently
//! testable and benchmarkable without a GPU or a window.

pub mod config;
pub mod domain;
pub mod world;

pub use config::{Carver, CaveParams, Config, ConfigError, Env, RenderConfig, WorldSize};
pub use domain::{
    BlockId, CHUNK_SHIFT, CHUNK_SIZE, CHUNK_VOLUME, ChunkPos, LightLevel, LocalPos, PackedVoxel,
    VoxelIndex,
};
pub use world::{Chunk, ChunkGrid, VoxelGroup};
