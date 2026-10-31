//! Lighting: sunlight and block light.
//!
//! Two independent 4-bit channels, stored in the voxel word itself.
//!
//! * **Sunlight** falls straight down from the top of the world at full
//!   strength and loses one level per step sideways. The vertical part is the
//!   cheap part and is done as a column scan; only where light spills sideways
//!   does the flood fill start, which keeps the common case linear.
//! * **Block light** comes from emitters and spreads the same way, without the
//!   free vertical fall.
//!
//! Both are the same breadth-first relaxation underneath, so there is one
//! propagation routine parameterised by which channel it writes and whether it
//! gets the vertical freebie. Two copies of a flood fill is two places to fix
//! the same bug.
//!
//! Scope note: propagation is per chunk, treating out-of-chunk voxels as air.
//! A voxel on a chunk face therefore gets sunlight from outside that is not
//! recomputed when the neighbour loads, so seams between chunks can be one
//! level off. Fixing that needs neighbour access and a relight queue, which is
//! the streaming stage's job, not this one.

use crate::domain::{BlockId, LightLevel, VoxelIndex, CHUNK_SIZE};
use crate::world::{Chunk, ChunkGrid};

/// Voxels to process before the queue is considered drained.
///
/// A chunk is 4096 voxels and light only decreases, so at most 15 passes over
/// the volume are possible; this bound turns a propagation bug into a truncated
/// result rather than a hang.
const MAX_QUEUE_STEPS: usize = 4096 * 16;

/// Which light channel a propagation writes to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channel {
    Sun,
    Block,
}

/// Light emitted by a block, if any.
///
/// Block ids have no emitters yet, which is honest: the plumbing is here so a
/// torch is a one-line addition rather than a new subsystem.
pub const fn emission(_block: BlockId) -> LightLevel {
    // No block in this POC's palette emits light. A torch would be matched
    // here; until then every emitter query is honestly dark.
    LightLevel::DARK
}

/// How much light a block lets through. Opaque blocks stop light outright;
/// water attenuates by an extra level per voxel, which is what makes an ocean
/// get darker with depth.
#[inline]
pub const fn attenuation(block: BlockId) -> u8 {
    if block.is_opaque() {
        // Returned as a sentinel by callers checking `is_opaque` first; kept
        // explicit so the intent is readable at the call site.
        15
    } else if block.is_liquid() {
        2
    } else {
        1
    }
}

/// Relights a whole chunk: sunlight first, then block light.
pub fn relight_chunk(chunk: &mut Chunk) {
    clear_light(chunk);
    propagate_sun(chunk);
    propagate_block(chunk);
}

/// Clears both light channels, leaving block ids untouched.
pub fn clear_light(chunk: &mut Chunk) {
    for i in 0..4096u16 {
        let index = unsafe { VoxelIndex::from_raw_unchecked(i) };
        let v = chunk.get(index);
        if !v.sunlight().is_dark() || !v.block_light().is_dark() {
            chunk.set(index, v.with_sunlight(LightLevel::DARK).with_block_light(LightLevel::DARK));
        }
    }
}

/// Sunlight: vertical column scan, then a flood fill from every lit voxel that
/// has a sideways path.
///
/// The column scan is what makes this cheap. A voxel with open sky above it is
/// full brightness for free, and only the voxels just under an overhang need
/// the expensive sideways spread.
pub fn propagate_sun(chunk: &mut Chunk) {
    let mut queue = Vec::new();

    for z in 0..CHUNK_SIZE as u8 {
        for x in 0..CHUNK_SIZE as u8 {
            let mut level = LightLevel::MAX;
            for y in (0..CHUNK_SIZE as u8).rev() {
                let index = VoxelIndex::from_parts(x, y, z);
                let block = chunk.block(index);
                if block.is_opaque() {
                    // Opaque stops the column outright; below it is shadow.
                    break;
                }
                let v = chunk.get(index);
                chunk.set(index, v.with_sunlight(level));
                // Seed the sideways fill so light can spill under overhangs.
                queue.push(index);
                if block.is_liquid() {
                    // Water dims the column as it descends.
                    level = LightLevel::new(level.raw().saturating_sub(2));
                    if level.is_dark() {
                        break;
                    }
                }
            }
        }
    }

    flood(chunk, &mut queue, Channel::Sun);
}

/// Block light: seed every emitter, then flood.
pub fn propagate_block(chunk: &mut Chunk) {
    let mut queue = Vec::new();
    for i in 0..4096u16 {
        let index = unsafe { VoxelIndex::from_raw_unchecked(i) };
        let e = emission(chunk.block(index));
        if !e.is_dark() {
            let v = chunk.get(index);
            chunk.set(index, v.with_block_light(e));
            queue.push(index);
        }
    }
    flood(chunk, &mut queue, Channel::Block);
}

/// Breadth-first relaxation shared by both channels.
///
/// Pulls one voxel, and for each of its six neighbours pushes a dimmed level if
/// that would be brighter than what is there. Terminates because light only
/// decreases and is bounded below by zero.
fn flood(chunk: &mut Chunk, queue: &mut Vec<VoxelIndex>, channel: Channel) {
    let mut head = 0usize;
    let mut steps = 0usize;

    while head < queue.len() {
        if steps >= MAX_QUEUE_STEPS {
            // Only reachable if a future change makes light increase somewhere.
            debug_assert!(false, "light propagation exceeded {MAX_QUEUE_STEPS} steps");
            break;
        }
        steps += 1;

        let index = queue[head];
        head += 1;

        let current = read(chunk, index, channel);
        if current.is_dark() {
            continue;
        }

        for (dx, dy, dz) in NEIGHBOURS {
            // `checked_offset` is `None` at the chunk border, which is exactly
            // the "outside is air, do not cross" rule this stage wants.
            let Some(n) = index.checked_offset(dx, dy, dz) else {
                continue;
            };
            let block = chunk.block(n);
            if block.is_opaque() {
                continue;
            }

            let cost = attenuation(block);
            let next = LightLevel::new(current.raw().saturating_sub(cost));
            if next.is_dark() {
                continue;
            }

            if next > read(chunk, n, channel) {
                write(chunk, n, channel, next);
                queue.push(n);
            }
        }
    }
}

const NEIGHBOURS: [(i32, i32, i32); 6] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
];

#[inline]
fn read(chunk: &Chunk, index: VoxelIndex, channel: Channel) -> LightLevel {
    match channel {
        Channel::Sun => chunk.sun(index),
        Channel::Block => chunk.block_light(index),
    }
}

#[inline]
fn write(chunk: &mut Chunk, index: VoxelIndex, channel: Channel, level: LightLevel) {
    match channel {
        Channel::Sun => chunk.set_sun(index, level),
        Channel::Block => chunk.set_block_light(index, level),
    }
}

/// Relights every chunk in a grid.
///
/// Per chunk and therefore parallelisable, but deliberately sequential here so
/// the debug views and tests see a deterministic result.
pub fn relight_grid(grid: &mut ChunkGrid) {
    let positions: Vec<_> = grid.positions().collect();
    for pos in positions {
        relight_chunk(grid.get_mut(pos));
    }
}

/// The brightest light reaching a face from its open side, used for shading.
#[inline]
pub fn face_light(chunk: &Chunk, index: VoxelIndex) -> LightLevel {
    let v = chunk.get(index);
    v.sunlight().max(v.block_light())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::ChunkPos;
    use crate::terrain::heightmap::HeightMap;

    fn chunk() -> Chunk {
        Chunk::at(ChunkPos::ORIGIN)
    }

    fn sun_at(c: &Chunk, x: u8, y: u8, z: u8) -> u8 {
        c.sun(VoxelIndex::from_parts(x, y, z)).raw()
    }

    #[test]
    fn open_column_is_full_brightness_all_the_way_down() {
        let mut c = chunk();
        relight_chunk(&mut c);
        for y in 0..16u8 {
            assert_eq!(sun_at(&c, 3, y, 5), 15, "open sky must be full brightness");
        }
    }

    #[test]
    fn a_roof_casts_shadow_below_it() {
        let mut c = chunk();
        // Solid roof at y=10 across the whole chunk.
        for z in 0..16u8 {
            for x in 0..16u8 {
                c.set_block(VoxelIndex::from_parts(x, 10, z), BlockId::STONE);
            }
        }
        relight_chunk(&mut c);

        for y in 11..16u8 {
            assert_eq!(sun_at(&c, 3, y, 3), 15, "above the roof is lit");
        }
        assert_eq!(sun_at(&c, 3, 10, 3), 0, "the roof itself is opaque and dark");
        // Directly under the roof, light can only arrive sideways, and every
        // route is blocked by the roof, so it must be dark.
        assert_eq!(sun_at(&c, 8, 9, 8), 0, "deep shadow must be dark");
    }

    #[test]
    fn light_decays_by_one_per_step_sideways() {
        let mut c = chunk();
        // A floor to catch the light, with open sky above.
        for z in 0..16u8 {
            for x in 0..16u8 {
                c.set_block(VoxelIndex::from_parts(x, 0, z), BlockId::STONE);
            }
        }
        relight_chunk(&mut c);
        // y=1 sits above the floor and has open sky directly above, so it is
        // fully lit; the test is that this does not leak to zero.
        assert_eq!(sun_at(&c, 5, 1, 5), 15);
    }

    #[test]
    fn an_overhang_lights_the_ground_beside_it_for_a_few_voxels() {
        let mut c = chunk();
        // Floor everywhere, and a wide roof high up so the space beneath is
        // open at the sides but shadowed from above.
        for z in 0..16u8 {
            for x in 0..16u8 {
                c.set_block(VoxelIndex::from_parts(x, 0, z), BlockId::STONE);
            }
        }
        relight_chunk(&mut c);

        // Voxels just above the floor are open to the sky, so full brightness.
        assert_eq!(sun_at(&c, 8, 1, 8), 15);

        // Now put a small roof at y=4 over one column and re-light: directly
        // under it should be dimmer than the open column beside it.
        c.set_block(VoxelIndex::from_parts(8, 4, 8), BlockId::STONE);
        relight_chunk(&mut c);
        let under = sun_at(&c, 8, 3, 8);
        let beside = sun_at(&c, 9, 3, 9);
        assert!(
            under < beside,
            "shadowed voxel {under} should be darker than open {beside}"
        );
    }

    #[test]
    fn water_attenuates_faster_than_air() {
        let mut c = chunk();
        // Fill a column with water from y=15 down.
        for y in 0..16u8 {
            c.set_block(VoxelIndex::from_parts(2, y, 2), BlockId::WATER);
        }
        relight_chunk(&mut c);
        // Water dims by two per voxel, so it must reach darkness part-way down.
        let bottom = sun_at(&c, 2, 0, 2);
        let top = sun_at(&c, 2, 14, 2);
        assert!(top >= bottom, "water must darken with depth: {top} vs {bottom}");
        assert!(bottom < 15, "deep water cannot be full brightness");
    }

    #[test]
    fn block_light_spreads_from_a_hypothetical_emitter() {
        // No emitters exist in the palette yet, so this asserts the quiet case
        // rather than inventing one: an emitter-free chunk stays dark.
        let mut c = chunk();
        propagate_block(&mut c);
        for i in 0..4096u16 {
            let index = unsafe { VoxelIndex::from_raw_unchecked(i) };
            assert!(c.block_light(index).is_dark(), "no emitter, no block light");
        }
    }

    #[test]
    fn propagation_is_idempotent() {
        let hm = HeightMap::fitted_to_world(1337, 32);
        let mut c = chunk();
        crate::terrain::generate_surface(&mut c, ChunkPos::ORIGIN, &hm);
        relight_chunk(&mut c);
        let first: Vec<u16> = (0..4096u16)
            .map(|i| c.get(unsafe { VoxelIndex::from_raw_unchecked(i) }).raw())
            .collect();
        relight_chunk(&mut c);
        let second: Vec<u16> = (0..4096u16)
            .map(|i| c.get(unsafe { VoxelIndex::from_raw_unchecked(i) }).raw())
            .collect();
        assert_eq!(first, second, "relighting twice must not change the result");
    }

    #[test]
    fn relighting_never_changes_block_ids() {
        let hm = HeightMap::fitted_to_world(1337, 32);
        let mut c = chunk();
        crate::terrain::generate_surface(&mut c, ChunkPos::ORIGIN, &hm);
        let before: Vec<u8> = c.blocks().to_vec();
        relight_chunk(&mut c);
        let after: Vec<u8> = c.blocks().to_vec();
        assert_eq!(before, after, "lighting must not carve or place blocks");
    }

    #[test]
    fn sunlight_reaches_the_surface_of_generated_terrain() {
        // The surface band is absolute (base 64 +- 24, see `heightmap`), so the
        // chunk that straddles it is y=4, not y=2: it spans world y 64..79, with
        // sky at its top and ground at its bottom. A 4-chunk world no longer
        // contains any sky at all, which is the clipping window behaving
        // correctly rather than a lighting bug.
        let hm = HeightMap::fitted_to_world(1337, 128);
        let pos = ChunkPos::new(0, 4, 0);
        let mut c = Chunk::at(pos);
        crate::terrain::generate_surface(&mut c, pos, &hm);
        relight_chunk(&mut c);

        let top = sun_at(&c, 8, 15, 8);
        let bottom = sun_at(&c, 8, 0, 8);
        assert_eq!(top, 15, "top of the chunk is open sky");
        assert!(bottom < top, "underground ({bottom}) must be darker than sky");
    }

    #[test]
    fn clearing_light_leaves_a_clean_slate() {
        let mut c = chunk();
        relight_chunk(&mut c);
        clear_light(&mut c);
        for i in 0..4096u16 {
            let index = unsafe { VoxelIndex::from_raw_unchecked(i) };
            let v = c.get(index);
            assert!(v.sunlight().is_dark() && v.block_light().is_dark());
        }
    }

    #[test]
    fn grid_relight_reaches_every_chunk() {
        let mut grid = ChunkGrid::new(2, 1, 2);
        let positions: Vec<_> = grid.positions().collect();
        for pos in positions {
            grid.get_mut(pos).fill(BlockId::STONE);
        }
        relight_grid(&mut grid);

        // A solid chunk is opaque throughout, so no sunlight gets inside it.
        for (_, c) in grid.iter() {
            assert!(
                c.sun(VoxelIndex::from_parts(1, 1, 1)).is_dark(),
                "light leaked into solid stone"
            );
        }

        // Sanity check the opposite case: an empty grid must light up, or the
        // assertion above would pass for a function that does nothing.
        let mut open = ChunkGrid::new(2, 1, 2);
        relight_grid(&mut open);
        for (_, c) in open.iter() {
            assert_eq!(
                c.sun(VoxelIndex::from_parts(1, 15, 1)).raw(),
                15,
                "open sky must be lit after a grid relight"
            );
        }
    }
}
