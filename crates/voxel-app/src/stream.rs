//! A streaming world: chunks generated around the camera, not in a fixed box.
//!
//! This is a **residency cache**, not a generator. Everything about producing a
//! chunk — the surface pass, the carver, lighting — lives in
//! [`voxel_core::world::WorldGen`], which is the implementation
//! `streaming_matches_the_batch_path_byte_for_byte` proves byte-identical to the
//! in-RAM path. What stays here is the app-level question the core has no
//! business answering: *which* chunks are resident right now, and how much
//! generation that has cost.
//!
//! That split matters. Before it, this file carried a second copy of the
//! generation pipeline, so it could drift from the core one — and the byte
//! identity guarantee only covered the copy nobody rendered. Now there is one
//! pipeline and this file cannot generate a chunk differently from the one the
//! parity test checks, because it does not generate chunks at all.

use std::collections::HashMap;
use std::time::Instant;

use voxel_core::world::{CarverSpec, VerticalBounds, WorldGen};
use voxel_core::{Carver, CaveParams, Chunk, ChunkPos};

/// Re-exported so callers keep one import for the vertical extent.
///
/// The definition lives in the core now: the streamer and the app must agree on
/// what the window is, and two definitions of a band is how they stop agreeing.
pub use voxel_core::world::VerticalBounds as Bounds;

/// The streaming world: which chunks are resident, plus the counters.
pub struct StreamWorld {
    /// Generated chunks, keyed by position. The only world-sized structure in
    /// the app, and the thing eviction exists to bound.
    chunks: HashMap<ChunkPos, Chunk>,
    /// The one generator. Owns the band, the seed and the carver, so the app
    /// cannot describe a world the core would generate differently.
    emit: WorldGen,
    /// The cave parameters the generator was built with, kept so
    /// [`Self::cave_params`] can hand out a borrow.
    cave: CaveParams,
    /// Chunks generated since construction, for the benchmark readout.
    pub chunks_generated: u64,
    /// Total time spent generating, excluding meshing and upload.
    pub generate_nanos: u64,
}

impl StreamWorld {
    pub fn new(seed: u32, world_height: u32, cave: CaveParams, carver: Carver) -> Self {
        Self {
            chunks: HashMap::new(),
            cave,
            emit: WorldGen::new(CarverSpec {
                seed,
                carver,
                cave,
                world_height,
            }),
            chunks_generated: 0,
            generate_nanos: 0,
        }
    }

    pub fn bounds(&self) -> VerticalBounds {
        self.emit.bounds()
    }

    pub fn carver(&self) -> Carver {
        self.emit.spec().carver
    }

    /// Changes the carver for subsequent generation.
    ///
    /// Does not touch resident chunks; [`Self::regenerate_all`] is what makes the
    /// change visible on screen.
    pub fn set_carver(&mut self, carver: Carver) {
        let mut spec = self.emit.spec();
        spec.carver = carver;
        self.emit = WorldGen::new(spec);
    }

    pub fn cave_params(&self) -> &CaveParams {
        // `CarverSpec` is `Copy`, so `spec()` is a temporary; the cave params
        // live in a field of their own so this can hand out a borrow.
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
        if !self.bounds().contains(pos.y) {
            return None;
        }

        let start = Instant::now();
        let chunk = self.emit.chunk_at(pos);
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
        let bounds = self.bounds();
        let mut out = Vec::with_capacity(
            ((2 * radius + 1).max(0) as usize).pow(2) * bounds.len().max(0) as usize,
        );
        for y in bounds.min_y..bounds.max_y {
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
    /// Parallel because each chunk is independent: a carver writes only to the
    /// chunk it is given, which the bounded-stamp test in the sparse module
    /// guarantees. Each worker builds its own [`WorldGen`], since the generator
    /// reuses one scratch chunk and therefore cannot be shared.
    pub fn stream_around(&mut self, centre: ChunkPos, radius: i32) -> usize {
        use rayon::prelude::*;

        let missing: Vec<ChunkPos> = self
            .positions_in_radius(centre, radius)
            .into_iter()
            .filter(|p| !self.chunks.contains_key(p))
            .collect();
        if missing.is_empty() {
            return 0;
        }

        let start = Instant::now();
        let spec = self.emit.spec();
        // `WorldGen` owns a 12 KiB scratch chunk and is built once per worker.
        // It is boxed so that 12 KiB lives on the heap rather than being
        // constructed in the worker closure's stack frame: with several rayon
        // workers alive at once the unboxed version overflowed the stack in
        // debug builds, which is a real failure a release build would have
        // hidden behind a smaller frame.
        let generated: Vec<(ChunkPos, Chunk)> = missing
            .par_iter()
            .map_init(
                || Box::new(WorldGen::new(spec)),
                |generator, &pos| (pos, generator.chunk_at(pos)),
            )
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
        let start = Instant::now();
        for pos in positions {
            let chunk = self.emit.chunk_at(pos);
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

    /// A world whose window actually contains sky above the terrain band.
    ///
    /// Was 64 voxels, which under the absolute surface band (`heightmap::BASE_Y`
    /// 64 +- 24, i.e. absolute 40..88) is a window entirely at or below the
    /// surface: every chunk is rock and there is no sky to cull. 128 voxels
    /// spans the band with room above, which is what these tests need to say
    /// anything about sky versus underground.
    fn world(carver: Carver) -> StreamWorld {
        StreamWorld::new(
            1337,
            128,
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
        // The sky chunks are the ones above the terrain band. The band is
        // ABSOLUTE now (`heightmap::BASE_Y` 64 +- 24), so whether a window has
        // sky is a property of the window, not of the seed: a 128-voxel window
        // reaches above the band and must have empty chunks, while a 64-voxel
        // one tops out at the band's mean and must have none.
        assert!(air > 0, "no empty chunks; the vertical band looks wrong");

        // The same seed in a shorter window is the same world clipped, so it
        // has no sky: its ceiling is below the band's mean surface height.
        let mut low = StreamWorld::new(
            1337,
            64,
            voxel_core::CaveParams {
                origin_density: 2.0,
                min_radius: 1.5,
                max_radius: 4.0,
            },
            Carver::Sparse,
        );
        assert_eq!(low.bounds().len(), 4, "64 voxels is 4 chunks");
        low.stream_around(ChunkPos::new(0, 0, 0), 1);
        let low_air = low
            .positions()
            .filter(|&p| low.get(p).expect("generated").empty)
            .count();
        assert_eq!(
            low_air, 0,
            "a 64-voxel window tops out at the band's mean, so it has no sky"
        );
    }

    /// The clipping property: a world height selects a window over one fixed
    /// world, it does not rescale the world.
    ///
    /// Two windows of the same seed must agree byte-for-byte wherever they
    /// overlap. This is the property that makes a "world height" a view rather
    /// than a parameter, and it is what the absolute surface band buys.
    #[test]
    fn a_taller_window_clips_the_same_world_rather_than_rescaling_it() {
        use voxel_core::world::{CarverSpec, WorldGen};
        use voxel_core::{CaveParams, ChunkPos};

        let spec = |world_height: u32| CarverSpec {
            seed: 1337,
            carver: Carver::Sparse,
            cave: CaveParams {
                origin_density: 2.0,
                min_radius: 1.5,
                max_radius: 4.0,
            },
            world_height,
        };

        let mut short = WorldGen::new(spec(64));
        let mut tall = WorldGen::new(spec(128));

        assert_eq!(short.bounds().len(), 4);
        assert_eq!(tall.bounds().len(), 8);

        let mut compared = 0;
        for cy in 0..4 {
            let pos = ChunkPos::new(0, cy, 0);
            assert_eq!(
                short.chunk_at(pos).to_packed(),
                tall.chunk_at(pos).to_packed(),
                "chunk {pos:?} differs between a 64- and a 128-voxel window"
            );
            compared += 1;
        }
        assert_eq!(compared, 4, "the overlap must cover the whole short window");
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
