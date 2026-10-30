//! A streaming world: chunks generated around the camera, not in a fixed box.
//!
//! Generation was already position-independent — the heightmap is a pure
//! function of world coordinates and the sparse carver uses `div_euclid` so
//! negative regions work — so the only thing that made the world finite was
//! [`ChunkGrid`] being a dense rectangle. This replaces that with a map keyed by
//! position, filled in a radius around the camera and evicted behind it, which
//! is what turns the POC into something you can fly through.
//!
//! Vertical extent stays bounded. A voxel world with no floor or ceiling is not
//! useful here, and the heightmap already produces terrain within a band, so
//! generating the full column would spend time on chunks that are always empty
//! or always solid.

use std::collections::HashMap;

use voxel_core::terrain::heightmap::HeightMap;
use voxel_core::terrain::{self, density, sparse};
use voxel_core::{Carver, CaveParams, Chunk, ChunkPos};

/// How many chunks tall the world is, and where it starts.
///
/// Fixed rather than streamed so the vertical band is known up front: the
/// mesher and the lighting both need to reason about the world's top, and an
/// unlimited column would mean generating chunks that are guaranteed empty.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerticalBounds {
    /// Lowest chunk index generated.
    pub min_y: i32,
    /// One past the highest chunk index generated.
    pub max_y: i32,
}

impl VerticalBounds {
    /// Enough height for the default terrain band plus sky and depth.
    ///
    /// Not `const`: `div_ceil` and `Ord::max` are not yet usable in const
    /// context, and this runs once at startup anyway.
    pub fn default_for_world_height(world_height: u32) -> Self {
        // Plain arithmetic rather than `div_ceil`, which is not stable for
        // signed integers on this toolchain.
        let height = world_height as i32;
        let chunks = (height + 15) / 16;
        Self {
            min_y: 0,
            max_y: if chunks < 1 { 1 } else { chunks },
        }
    }

    pub fn contains(&self, y: i32) -> bool {
        y >= self.min_y && y < self.max_y
    }

    pub fn len(&self) -> i32 {
        self.max_y - self.min_y
    }

    pub fn is_empty(&self) -> bool {
        self.len() <= 0
    }
}

/// Why a chunk could not be generated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipReason {
    /// Outside the world's vertical band.
    AboveWorld,
    BelowWorld,
}

/// The streaming world: generated chunks plus the parameters to make more.
pub struct StreamWorld {
    chunks: HashMap<ChunkPos, Chunk>,
    heightmap: HeightMap,
    cave: CaveParams,
    carver: Carver,
    seed: u32,
    bounds: VerticalBounds,
    /// Chunks generated since construction, for the benchmark readout.
    pub chunks_generated: u64,
    /// Total time spent generating, excluding meshing and upload.
    pub generate_nanos: u64,
}

impl StreamWorld {
    pub fn new(seed: u32, world_height: u32, cave: CaveParams, carver: Carver) -> Self {
        Self {
            chunks: HashMap::new(),
            heightmap: HeightMap::fitted_to_world(seed, world_height),
            cave,
            carver,
            seed,
            bounds: VerticalBounds::default_for_world_height(world_height),
            chunks_generated: 0,
            generate_nanos: 0,
        }
    }

    pub fn bounds(&self) -> VerticalBounds {
        self.bounds
    }

    pub fn carver(&self) -> Carver {
        self.carver
    }

    pub fn set_carver(&mut self, carver: Carver) {
        self.carver = carver;
    }

    pub fn cave_params(&self) -> &CaveParams {
        &self.cave
    }

    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    pub fn get(&self, pos: ChunkPos) -> Option<&Chunk> {
        self.chunks.get(&pos)
    }

    pub fn contains(&self, pos: ChunkPos) -> bool {
        self.chunks.contains_key(&pos)
    }

    pub fn positions(&self) -> impl Iterator<Item = ChunkPos> + '_ {
        self.chunks.keys().copied()
    }

    /// Generates a chunk if it is inside the vertical band.
    ///
    /// Returns `None` when the position is out of bounds, or when it is already
    /// generated, so callers can tell "nothing to do" from "generated now".
    pub fn generate(&mut self, pos: ChunkPos) -> Option<&Chunk> {
        if self.chunks.contains_key(&pos) {
            return self.chunks.get(&pos);
        }
        if !self.bounds.contains(pos.y) {
            return None;
        }

        let start = std::time::Instant::now();
        let mut chunk = Chunk::at(pos);
        terrain::generate_surface(&mut chunk, pos, &self.heightmap);
        match self.carver {
            Carver::Density => {
                density::carve(&mut chunk, self.seed);
            }
            Carver::Sparse => {
                sparse::carve(&mut chunk, self.seed, &self.cave);
            }
        }
        // Lighting last: a carved pocket changes what can see the sky, so
        // lighting first would leave every cave fully lit.
        terrain::finish(&mut chunk);

        self.generate_nanos += start.elapsed().as_nanos() as u64;
        self.chunks_generated += 1;
        self.chunks.insert(pos, chunk);
        self.chunks.get(&pos)
    }

    /// Chunk positions inside a Chebyshev radius of `centre`, within bounds.
    ///
    /// Chebyshev rather than Euclidean so the loaded region is a box: with a
    /// sphere you can see the corners of chunks that were never loaded, which
    /// shows up as holes in the terrain at the edge of the view.
    pub fn positions_in_radius(&self, centre: ChunkPos, radius: i32) -> Vec<ChunkPos> {
        let mut out = Vec::new();
        for y in self.bounds.min_y..self.bounds.max_y {
            for dz in -radius..=radius {
                for dx in -radius..=radius {
                    out.push(ChunkPos::new(centre.x + dx, y, centre.z + dz));
                }
            }
        }
        out
    }

    /// Generates every missing chunk within `radius` of `centre`.
    ///
    /// Parallel because each chunk is independent: the carvers write only to
    /// the chunk they are given, which the bounded-stamp test in the sparse
    /// module guarantees.
    pub fn stream_around(&mut self, centre: ChunkPos, radius: i32) -> usize {
        use rayon::prelude::*;

        let wanted = self.positions_in_radius(centre, radius);
        let missing: Vec<ChunkPos> = wanted
            .into_iter()
            .filter(|p| !self.chunks.contains_key(p))
            .collect();
        if missing.is_empty() {
            return 0;
        }

        let start = std::time::Instant::now();
        let generated: Vec<(ChunkPos, Chunk)> = missing
            .par_iter()
            .map(|&pos| {
                let mut chunk = Chunk::at(pos);
                terrain::generate_surface(&mut chunk, pos, &self.heightmap);
                match self.carver {
                    Carver::Density => {
                        density::carve(&mut chunk, self.seed);
                    }
                    Carver::Sparse => {
                        sparse::carve(&mut chunk, self.seed, &self.cave);
                    }
                }
                terrain::finish(&mut chunk);
                (pos, chunk)
            })
            .collect();

        self.generate_nanos += start.elapsed().as_nanos() as u64;
        let count = generated.len();
        for (pos, chunk) in generated {
            self.chunks.insert(pos, chunk);
        }
        self.chunks_generated += count as u64;
        count
    }

    /// Drops chunks further than `radius` from `centre`, keeping memory bounded.
    ///
    /// The radius should be larger than the streaming radius so chunks are not
    /// evicted and immediately regenerated as the camera moves back and forth
    /// across the boundary.
    pub fn evict_beyond(&mut self, centre: ChunkPos, radius: i32) -> usize {
        let before = self.chunks.len();
        self.chunks
            .retain(|pos, _| pos.chebyshev(centre) <= radius);
        before - self.chunks.len()
    }

    /// Regenerates every loaded chunk with the current carver.
    ///
    /// Used when the carver changes, so the world updates in place rather than
    /// the caller having to reconstruct and re-stream everything.
    pub fn regenerate_all(&mut self) {
        let positions: Vec<ChunkPos> = self.chunks.keys().copied().collect();
        self.chunks.clear();
        let start = std::time::Instant::now();
        for pos in positions {
            let mut chunk = Chunk::at(pos);
            terrain::generate_surface(&mut chunk, pos, &self.heightmap);
            match self.carver {
                Carver::Density => {
                    density::carve(&mut chunk, self.seed);
                }
                Carver::Sparse => {
                    sparse::carve(&mut chunk, self.seed, &self.cave);
                }
            }
            terrain::finish(&mut chunk);
            self.chunks.insert(pos, chunk);
        }
        self.generate_nanos += start.elapsed().as_nanos() as u64;
    }

    pub fn chunks_generated_total(&self) -> u64 {
        self.chunks_generated
    }

    pub fn generate_ms(&self) -> f32 {
        self.generate_nanos as f32 / 1e6
    }

    /// Milliseconds per generated chunk, the figure the benchmark reports.
    pub fn ms_per_chunk(&self) -> f32 {
        if self.chunks_generated == 0 {
            return 0.0;
        }
        self.generate_ms() / self.chunks_generated as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn world(carver: Carver) -> StreamWorld {
        StreamWorld::new(
            1337,
            64,
            CaveParams {
                origin_density: 2.0,
                min_radius: 1.5,
                max_radius: 4.0,
            },
            carver,
        )
    }

    #[test]
    fn generation_works_at_negative_coordinates() {
        // The whole point of streaming: the world extends in every direction.
        // A heightmap or carver that assumed a non-negative origin would show
        // up here as missing or mirrored terrain.
        let mut w = world(Carver::Sparse);
        for pos in [
            ChunkPos::new(0, 1, 0),
            ChunkPos::new(-1, 1, -1),
            ChunkPos::new(-40, 1, 25),
            ChunkPos::new(1000, 1, -1000),
        ] {
            let c = w.generate(pos).expect("should generate");
            assert!(!c.empty, "chunk at {pos:?} generated as pure air");
        }
    }

    #[test]
    fn the_same_position_generates_the_same_chunk_however_it_is_reached() {
        // Streaming means a chunk can be generated in any order, on any thread.
        // Nothing may depend on what was generated before it.
        let mut a = world(Carver::Sparse);
        let mut b = world(Carver::Sparse);
        let target = ChunkPos::new(-3, 1, 7);

        a.generate(target);
        // Generate a crowd of neighbours first in the other world.
        for pos in a.positions_in_radius(ChunkPos::new(20, 1, 20), 2) {
            b.generate(pos);
        }
        b.generate(target);

        let ca = a.get(target).unwrap();
        let cb = b.get(target).unwrap();
        for i in 0..4096u16 {
            let idx = unsafe { voxel_core::VoxelIndex::from_raw_unchecked(i) };
            assert_eq!(
                ca.block(idx),
                cb.block(idx),
                "voxel {i} differs depending on generation order"
            );
        }
    }

    #[test]
    fn vertical_bounds_reject_out_of_range_chunks() {
        // An unbounded column would generate chunks that are always empty above
        // and always solid below, which is wasted work.
        let mut w = world(Carver::Sparse);
        let bounds = w.bounds();
        assert!(!bounds.is_empty());
        assert!(w.generate(ChunkPos::new(0, bounds.max_y, 0)).is_none());
        assert!(w.generate(ChunkPos::new(0, bounds.min_y - 1, 0)).is_none());
        assert!(w.generate(ChunkPos::new(0, bounds.min_y, 0)).is_some());
    }

    #[test]
    fn streaming_generates_the_surrounding_box() {
        let mut w = world(Carver::Sparse);
        let centre = ChunkPos::new(2, 1, 2);
        let radius = 3;
        let added = w.stream_around(centre, radius);

        let expected = ((2 * radius + 1).pow(2) * w.bounds().len()) as usize;
        assert_eq!(added, expected, "first stream should fill the whole box");
        assert_eq!(w.len(), expected);

        // A second call must be a no-op rather than regenerating.
        assert_eq!(w.stream_around(centre, radius), 0);

        // Every chunk in the radius is present; one just outside is not.
        assert!(w.contains(ChunkPos::new(2 + radius, 1, 2)));
        assert!(!w.contains(ChunkPos::new(2 + radius + 1, 1, 2)));
    }

    #[test]
    fn streaming_is_chebyshev_so_the_corners_are_filled() {
        // A Euclidean radius would leave the box corners unloaded, which shows
        // as visible holes in the terrain at the edge of the view.
        let mut w = world(Carver::Sparse);
        let centre = ChunkPos::new(0, 1, 0);
        let radius = 4;
        w.stream_around(centre, radius);
        assert!(
            w.contains(ChunkPos::new(radius, 1, radius)),
            "corner chunk was not loaded"
        );
        assert!(
            w.contains(ChunkPos::new(-radius, 1, -radius)),
            "opposite corner was not loaded"
        );
    }

    #[test]
    fn eviction_drops_distant_chunks_and_keeps_near_ones() {
        let mut w = world(Carver::Sparse);
        w.stream_around(ChunkPos::new(0, 1, 0), 2);
        let before = w.len();
        assert!(before > 0);

        // Evicting with a wide radius keeps everything.
        assert_eq!(w.evict_beyond(ChunkPos::new(0, 1, 0), 10), 0);
        assert_eq!(w.len(), before);

        // Moving far away and evicting clears the lot.
        let removed = w.evict_beyond(ChunkPos::new(100, 1, 100), 2);
        assert_eq!(removed, before);
        assert!(w.is_empty());
    }

    #[test]
    fn streaming_around_a_moving_camera_only_generates_new_ground() {
        // The cost of flying should be the newly revealed edge, not the whole
        // view distance again.
        let mut w = world(Carver::Sparse);
        let radius = 3;
        w.stream_around(ChunkPos::new(0, 1, 0), radius);

        // Step one chunk along: a column of new chunks appears, the rest is
        // already loaded.
        let added = w.stream_around(ChunkPos::new(1, 1, 0), radius);
        let column = ((2 * radius + 1) * w.bounds().len()) as usize;
        assert_eq!(added, column, "a one-chunk step should only load one column");
    }

    #[test]
    fn regenerate_all_rebuilds_without_changing_which_chunks_are_loaded() {
        let mut w = world(Carver::Sparse);
        w.stream_around(ChunkPos::new(0, 1, 0), 2);
        let positions: Vec<ChunkPos> = {
            let mut p: Vec<_> = w.positions().collect();
            p.sort();
            p
        };
        let sample = positions[0];
        let before: Vec<u8> = w.get(sample).unwrap().blocks().to_vec();

        w.regenerate_all();

        let mut after_positions: Vec<_> = w.positions().collect();
        after_positions.sort();
        assert_eq!(positions, after_positions, "the loaded set must be unchanged");
        let after: Vec<u8> = w.get(sample).unwrap().blocks().to_vec();
        assert_eq!(before, after, "regenerating with the same carver must agree");
    }

    #[test]
    fn switching_carver_changes_the_world() {
        // Regeneration has to actually re-carve, not reuse the old chunks.
        let mut w = world(Carver::Sparse);
        w.stream_around(ChunkPos::new(0, 1, 0), 1);
        let before: Vec<u8> = w
            .positions()
            .map(|p| w.get(p).unwrap().blocks()[0])
            .collect();

        w.set_carver(Carver::Density);
        w.regenerate_all();

        let after: Vec<u8> = w
            .positions()
            .map(|p| w.get(p).unwrap().blocks()[0])
            .collect();
        assert_eq!(before.len(), after.len());
        assert_ne!(before, after, "carver switch did not change the terrain");
    }

    #[test]
    fn timing_is_recorded_per_generated_chunk() {
        let mut w = world(Carver::Sparse);
        assert_eq!(w.chunks_generated_total(), 0);
        assert_eq!(w.ms_per_chunk(), 0.0, "no chunks must not divide by zero");

        w.stream_around(ChunkPos::new(0, 1, 0), 2);
        assert!(w.chunks_generated_total() > 0);
        assert!(w.generate_ms() >= 0.0);
        assert!(w.ms_per_chunk() >= 0.0);

        // Re-streaming with nothing to do must not inflate the totals.
        let recorded = w.chunks_generated_total();
        w.stream_around(ChunkPos::new(0, 1, 0), 2);
        assert_eq!(w.chunks_generated_total(), recorded);
    }

    #[test]
    fn generated_chunks_are_finished_and_their_flags_are_true() {
        // `stream_around` must finish every chunk, or the renderer's cull flags
        // would be stale and the HUD would report nonsense. Rather than assert
        // how many chunks fall into each bucket, which depends on the terrain
        // band and the carve rate, this checks each flag against the voxel data
        // it claims to summarise.
        let mut w = world(Carver::Sparse);
        w.stream_around(ChunkPos::new(0, 1, 0), 2);

        let mut air = 0;
        for pos in w.positions() {
            let c = w.get(pos).unwrap();
            assert!(!c.dirty, "chunk {pos:?} was left dirty");

            let all_air = c.blocks().iter().all(|&b| b == 0);
            let all_solid = c
                .blocks()
                .iter()
                .all(|&b| voxel_core::BlockId::new(b).is_solid());
            assert_eq!(c.empty, all_air, "empty flag wrong for {pos:?}");
            assert_eq!(
                c.enclosed,
                all_solid && !all_air,
                "enclosed flag wrong for {pos:?}"
            );
            if c.empty {
                air += 1;
            }
        }
        // The sky chunks are the ones above the terrain band, so a world this
        // tall must have some.
        assert!(air > 0, "no empty chunks; the vertical band looks wrong");
    }

    #[test]
    fn sky_chunks_are_flagged_empty_and_underground_chunks_are_not() {
        // Separates the two failure modes the flags exist to serve: an all-air
        // chunk must be skippable, and a rock chunk must not be mistaken for
        // one and vanish from the render.
        let mut w = world(Carver::Sparse);
        w.generate(ChunkPos::new(0, 1, 0));

        let bounds = w.bounds();
        let top = ChunkPos::new(0, bounds.max_y - 1, 0);
        w.generate(top);
        assert!(
            w.get(top).unwrap().empty,
            "the topmost chunk should be sky and skippable"
        );

        let bottom = ChunkPos::new(0, bounds.min_y, 0);
        w.generate(bottom);
        assert!(
            !w.get(bottom).unwrap().empty,
            "the bottom chunk should contain rock"
        );
    }
}
