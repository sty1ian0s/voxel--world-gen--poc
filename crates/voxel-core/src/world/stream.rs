//! Streaming generation: a chunk is a pure function of `(seed, pos, carver)`.
//!
//! The batch path ([`crate::debug::report::generate_world`]) allocates a
//! [`ChunkGrid`](crate::world::ChunkGrid) sized to the world and fills it. That
//! is fine for an 8x8x8 box and impossible for a 256x256 region: 12 KiB per
//! chunk means a 65 536-chunk region is 768 MiB of voxels that only need to be
//! looked at once. This module keeps one chunk alive at a time.
//!
//! Nothing had to change to allow it. Every carver takes
//! `&mut impl CarveTarget` and writes only inside the chunk it was handed, and
//! `HeightMap::height` is a pure function of world `(x, z)` — there is no global
//! state anywhere in Pass 1 or Pass 2, which is what the parity test against the
//! batch path pins down.
//!
//! ## Its one limitation, asserted rather than hidden
//!
//! The batch path sizes the surface band to the *world box*
//! (`HeightMap::fitted_to_world(seed, cfg.world_height())`), so a 256-tall world
//! is impossible to reproduce chunk-by-chunk: the band depends on the world
//! height, not on the seed alone. The vertical extent is therefore a parameter
//! here, and `streaming_matches_the_batch_path_byte_for_byte` shows the chunks
//! agreeing at the configured world height. Streaming a world of a *different*
//! height is a different world, and correctly so.
//!
//! ## Carver dispatch
//!
//! [`crate::config::Carver`] names two of the four carvers in the codebase;
//! [`CarverSpec`] adds the two that are not behind the enum yet, without
//! touching it. Dispatch is one `match` in [`WorldGen::generate`], which is cold
//! relative to the carving it invokes, so there is no function pointer to chase
//! per voxel.

use crate::config::{CaveParams, Carver};
use crate::domain::ChunkPos;
use crate::terrain::heightmap::HeightMap;
use crate::terrain::{self, density, sparse};
use crate::world::Chunk;

/// Everything that changes what a chunk contains.
///
/// Grouped into one `Copy` struct because the parity test has to prove that two
/// construction paths with the same parameters agree: comparing a tuple of five
/// fields in two places is a test that drifts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CarverSpec {
    pub seed: u32,
    pub carver: Carver,
    pub cave: CaveParams,
    /// World height in voxels; the surface band is fitted to it.
    pub world_height: u32,
}

impl CarverSpec {
    /// The band `HeightMap::fitted_to_world` should be built from.
    ///
    /// `HeightMap::fitted_to_world(seed, w)` sets the sea level to `w / 2` and
    /// then reads a height in that band, so a world shorter than
    /// `2 * sea_level` has *no* column that reaches the top of a chunk. Every
    /// chunk would then hold at least one fully solid voxel column, so
    /// [`Chunk::enclosed`] could never be set and the renderer would lose the
    /// culling of solid rock that `PROJECT.md` §4.3 asks for.
    ///
    /// That is the configured world's problem, not the streamer's, so it is
    /// worked around the same way [`HeightMap::height_clamped`] works around it:
    /// floor the band at 32 voxels.
    ///
    /// shortcut: a world shorter than 32 voxels cannot contain a column with sky
    /// above it, so it has no cullable solid chunks. Upgrade when the vertical
    /// extent stops being a fixed band.
    pub fn band_height(&self) -> u32 {
        self.world_height.max(32)
    }
}

/// The vertical extent of a streamed world, in chunks.
///
/// The world is unbounded on `x` and `z` and bounded on `y`: an unbounded column
/// would spend its time on chunks that are always solid below the terrain band
/// and always air above it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerticalBounds {
    pub min_y: i32,
    pub max_y: i32,
}

impl VerticalBounds {
    /// Chunks covering `world_height` voxels from `y = 0`.
    ///
    /// The batch path always starts at `y = 0`, so streaming has to as well or
    /// the two disagree about what voxel `y` a chunk holds.
    pub fn for_height(world_height: u32) -> Self {
        let chunks = world_height.div_ceil(16).max(1) as i32;
        Self {
            min_y: 0,
            max_y: chunks,
        }
    }

    #[inline]
    pub fn contains(&self, y: i32) -> bool {
        y >= self.min_y && y < self.max_y
    }

    #[inline]
    pub fn len(&self) -> i32 {
        self.max_y - self.min_y
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() <= 0
    }
}

/// A rectangle of chunks, the unit of work for the streaming metrics.
///
/// Deliberately not a general "shape": a rectangle is what makes the statistics
/// comparable between two region sizes, because the only thing that differs
/// between a 64x64 and a 256x256 run is the area.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Region {
    pub min: ChunkPos,
    pub size_x: i32,
    pub size_z: i32,
}

impl Region {
    /// A region `side` chunks square, starting at the world origin.
    ///
    /// Starts at 0 because the batch path it is compared against is also
    /// `0..size`, and because negative chunk coordinates are then covered by a
    /// separate, smaller correctness test rather than by every measurement.
    pub fn square(side: i32) -> Self {
        assert!(side > 0, "region side must be positive, got {side}");
        Self {
            min: ChunkPos::new(0, 0, 0),
            size_x: side,
            size_z: side,
        }
    }

    pub fn at(min: ChunkPos, size_x: i32, size_z: i32) -> Self {
        assert!(size_x > 0 && size_z > 0, "region must have area");
        Self { min, size_x, size_z }
    }

    #[inline]
    pub fn columns(&self) -> u64 {
        self.size_x as u64 * self.size_z as u64
    }

    pub fn contains_column(&self, x: i32, z: i32) -> bool {
        x >= self.min.x
            && x < self.min.x + self.size_x
            && z >= self.min.z
            && z < self.min.z + self.size_z
    }

    /// The region's columns in `(z, x)` order, `x` varying fastest.
    ///
    /// That order is what makes [`crate::debug::window`]'s ring a single
    /// contiguous slice with no per-column arithmetic.
    pub fn columns_in_order(&self) -> impl Iterator<Item = (i32, i32)> + '_ {
        (self.min.z..self.min.z + self.size_z)
            .flat_map(move |z| (self.min.x..self.min.x + self.size_x).map(move |x| (x, z)))
    }

    /// Every chunk position in the region, `y` slowest.
    pub fn positions(&self, bounds: VerticalBounds) -> impl Iterator<Item = ChunkPos> + '_ {
        (bounds.min_y..bounds.max_y)
            .flat_map(move |y| self.columns_in_order().map(move |(x, z)| ChunkPos::new(x, y, z)))
    }
}

/// A chunk-at-a-time generator over an unbounded `(x, z)` plane.
pub struct WorldGen {
    heightmap: HeightMap,
    spec: CarverSpec,
    bounds: VerticalBounds,
    /// Reused across chunks, so streaming a large region allocates once rather
    /// than once per chunk. 12 KiB, held for the lifetime of the generator.
    scratch: Chunk,
}

impl WorldGen {
    pub fn new(spec: CarverSpec) -> Self {
        let bounds = VerticalBounds::for_height(spec.world_height);
        Self {
            heightmap: HeightMap::fitted_to_world(spec.seed, spec.band_height()),
            spec,
            bounds,
            scratch: Chunk::at(ChunkPos::ORIGIN),
        }
    }

    pub fn spec(&self) -> CarverSpec {
        self.spec
    }

    pub fn bounds(&self) -> VerticalBounds {
        self.bounds
    }

    pub fn heightmap(&self) -> &HeightMap {
        &self.heightmap
    }

    /// Generates `pos` into the reusable buffer and hands it to `f`.
    ///
    /// A visitor rather than a `-> Chunk` because returning a chunk would put a
    /// 12 KiB copy on every call, which is exactly the allocation this module
    /// exists to avoid. `f` cannot keep the chunk; if a caller needs to, it
    /// clones.
    pub fn with_chunk<R>(&mut self, pos: ChunkPos, f: impl FnOnce(&Chunk) -> R) -> R {
        self.fill(pos);
        f(&self.scratch)
    }

    /// Generates `pos` into an owned chunk, for callers that must keep it.
    ///
    /// Same code path as [`Self::with_chunk`]; only the destination differs, so
    /// the two cannot disagree.
    pub fn chunk_at(&mut self, pos: ChunkPos) -> Chunk {
        self.fill(pos);
        self.scratch.clone()
    }

    /// The whole pipeline for one chunk, in the order the batch path uses.
    fn fill(&mut self, pos: ChunkPos) {
        // Rebuilding rather than overwriting: the surface pass leaves sky voxels
        // untouched, so regenerating in place would keep the previous chunk's
        // terrain and light there. `Chunk::at` is exactly the state the batch
        // path starts from, and `Chunk` is 12 KiB of plain bytes, so there is
        // nothing cheaper to do that means the same thing.
        let mut chunk = Chunk::at(pos);
        terrain::generate_surface(&mut chunk, pos, &self.heightmap);
        match self.spec.carver {
            Carver::Density => {
                density::carve(&mut chunk, self.spec.seed);
            }
            Carver::Sparse => {
                sparse::carve(&mut chunk, self.spec.seed, &self.spec.cave);
            }
        }
        // Last, because carving changes which voxels can see the sky.
        terrain::finish(&mut chunk);
        self.scratch = chunk;
    }

    /// Generates one chunk outside the vertical band, as pure air with its
    /// flags already refreshed.
    ///
    /// Not a special case bolted on: a chunk above the world is air by the same
    /// rule the surface pass uses (`wy > surface_y`), so generating it as air is
    /// what the batch path would produce if it had the height for it. It exists
    /// so a region visitor can walk a full column without branching.
    pub fn fill_out_of_band(&mut self) {
        let pos = self.scratch.pos();
        self.scratch = Chunk::at(pos);
        self.scratch.refresh_flags();
    }

    /// Generates `pos` if it is inside the vertical band, and reports whether it
    /// did. Out-of-band positions leave the buffer holding an all-air chunk.
    pub fn generate(&mut self, pos: ChunkPos) -> bool {
        if self.bounds.contains(pos.y) {
            self.fill(pos);
            true
        } else {
            self.fill_out_of_band();
            false
        }
    }

    /// Visits every chunk of `region`, one at a time, in `(y, z, x)` order.
    ///
    /// `visit` is called for out-of-band chunks too, with `false`; they are air,
    /// and a caller accumulating world statistics needs them to compute a void
    /// fraction over the declared band rather than over the band that happens to
    /// intersect terrain.
    pub fn visit_region(&mut self, region: &Region, mut visit: impl FnMut(ChunkPos, &Chunk)) {
        for y in self.bounds.min_y..self.bounds.max_y {
            for (x, z) in region.columns_in_order() {
                let pos = ChunkPos::new(x, y, z);
                self.fill(pos);
                visit(pos, &self.scratch);
            }
        }
    }

    /// Counts the chunks [`Self::visit_region`] would visit.
    pub fn region_chunks(&self, region: &Region) -> u64 {
        region.columns() * self.bounds.len().max(0) as u64
    }
}
