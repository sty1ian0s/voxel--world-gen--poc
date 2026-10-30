//! Data-oriented voxel world generation core.
//!
//! Layout of the crate mirrors the pipeline in `PROJECT.md` §4:
//! [`config`] -> [`chunk`] storage -> terrain passes -> [`mesh`] -> culling.
//! Every stage is plain data in, plain data out, so each one is independently
//! testable and benchmarkable without a GPU or a window.

pub mod config;
pub mod cull;
pub mod debug;
pub mod domain;
pub mod light;
pub mod mesh;
pub mod terrain;
pub mod world;

pub use config::{
    CarveParity, Carver, CaveParams, Config, ConfigError, Env, RenderConfig, WorldSize,
};
pub use domain::{
    BlockId, CHUNK_SHIFT, CHUNK_SIZE, CHUNK_VOLUME, ChunkPos, LightLevel, LocalPos, PackedVoxel,
    VoxelIndex,
};
pub use terrain::density::CARVE_THRESHOLD;
pub use terrain::heightmap::HeightMap;
pub use terrain::sparse::{CaveOrigin, REGION_CHUNKS, REGION_SIZE, StampStats};
pub use cull::{CullStats, Frustum, Plane, VisibleChunk, select_visible};
pub use light::{Channel, relight_chunk, relight_grid};
pub use mesh::{Direction, Mesh, Quad, mesh_chunk};
pub use world::{Chunk, ChunkGrid, VoxelGroup};
