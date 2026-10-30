//! The naive mesher: one quad per visible face.
//!
//! Slow and obvious. It exists so the greedy mesher can be checked against
//! something that is hard to get wrong: a greedy merge that drops or
//! double-covers a face still looks plausible on screen, and comparing the two
//! meshers' covered face sets is the only dependable way to catch it.
//!
//! It is not dead code. It is used by the equivalence tests, and by the
//! benchmark as the baseline that greedy meshing is measured against.

use super::{Direction, Mesh, Quad, should_draw_face};
use crate::domain::{CHUNK_SIZE, VoxelIndex};
use crate::world::Chunk;

/// Meshes a chunk with one quad per visible face, merging nothing.
pub fn mesh_chunk_naive(chunk: &Chunk) -> Mesh {
    let mut mesh = Mesh::default();

    for dir in Direction::ALL {
        for y in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    let index = VoxelIndex::from_parts(x as u8, y as u8, z as u8);
                    let here = chunk.get(index);
                    let (dx, dy, dz) = dir.step();
                    let (nx, ny, nz) = (x as i32 + dx, y as i32 + dy, z as i32 + dz);
                    // Outside the chunk counts as air, matching the greedy
                    // mesher, so the two agree on boundary faces.
                    let neighbour = if nx < 0
                        || ny < 0
                        || nz < 0
                        || nx >= CHUNK_SIZE as i32
                        || ny >= CHUNK_SIZE as i32
                        || nz >= CHUNK_SIZE as i32
                    {
                        crate::domain::BlockId::AIR
                    } else {
                        chunk.block(VoxelIndex::from_parts(nx as u8, ny as u8, nz as u8))
                    };

                    // Only a non-air voxel owns its faces. Without this an air
                    // voxel adjacent to stone also reports the boundary as a
                    // face, so every solid surface is emitted twice: once by
                    // the solid voxel and once by the empty one beside it. The
                    // greedy mesher packs air as a zero mask so it can never
                    // make that mistake, which is why the two disagreed.
                    if here.is_air() || !should_draw_face(here.block(), neighbour) {
                        continue;
                    }

                    mesh.quads.push(Quad {
                        pos: [x as u8, y as u8, z as u8],
                        dir,
                        size: [1, 1],
                        block: here.block(),
                        light: here,
                    });
                    mesh.faces += 1;
                }
            }
        }
    }

    mesh
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{BlockId, ChunkPos};
    use crate::mesh::greedy::{covered_face_set, mesh_chunk};

    fn chunk_with(f: impl Fn(&mut Chunk)) -> Chunk {
        let mut c = Chunk::at(ChunkPos::ORIGIN);
        f(&mut c);
        c
    }

    #[test]
    fn naive_and_greedy_cover_the_same_faces() {
        // The core equivalence property. Both meshers must agree on exactly
        // which (voxel, direction) faces exist, whatever the merge does.
        let cases: Vec<(&str, Chunk)> = vec![
            ("empty", chunk_with(|_| {})),
            (
                "single",
                chunk_with(|c| c.set_block(VoxelIndex::from_parts(7, 7, 7), BlockId::STONE)),
            ),
            ("solid", chunk_with(|c| c.fill(BlockId::STONE))),
            (
                "corner block",
                chunk_with(|c| c.set_block(VoxelIndex::from_parts(0, 0, 0), BlockId::STONE)),
            ),
            (
                "far corner block",
                chunk_with(|c| c.set_block(VoxelIndex::from_parts(15, 15, 15), BlockId::GRASS)),
            ),
            (
                "checkerboard",
                chunk_with(|c| {
                    for y in 0..16u8 {
                        for z in 0..16u8 {
                            for x in 0..16u8 {
                                if (x + y + z) % 2 == 0 {
                                    c.set_block(
                                        VoxelIndex::from_parts(x, y, z),
                                        BlockId::STONE,
                                    );
                                }
                            }
                        }
                    }
                }),
            ),
        ];

        for (name, chunk) in cases {
            let greedy = mesh_chunk(&chunk);
            let naive = mesh_chunk_naive(&chunk);
            assert_eq!(
                covered_face_set(&greedy),
                covered_face_set(&naive),
                "{name}: greedy and naive disagree on which faces are visible"
            );
            assert_eq!(
                greedy.faces, naive.faces,
                "{name}: face totals disagree"
            );
        }
    }

    #[test]
    fn naive_matches_greedy_on_real_terrain() {
        use crate::terrain::heightmap::HeightMap;
        let hm = HeightMap::fitted_to_world(1337, 32);
        let pos = ChunkPos::new(0, 1, 0);
        let mut c = Chunk::at(pos);
        crate::terrain::generate_surface(&mut c, pos, &hm);

        let greedy = mesh_chunk(&c);
        let naive = mesh_chunk_naive(&c);
        assert_eq!(
            covered_face_set(&greedy),
            covered_face_set(&naive),
            "meshers disagree on generated terrain"
        );
        assert!(
            greedy.quads.len() < naive.quads.len(),
            "greedy produced {} quads, naive {}; greedy must merge",
            greedy.quads.len(),
            naive.quads.len()
        );
    }

    #[test]
    fn naive_quads_are_all_unit_sized() {
        let c = chunk_with(|c| c.fill(BlockId::STONE));
        let mesh = mesh_chunk_naive(&c);
        assert_eq!(mesh.quads.len(), 6 * 256);
        assert!(mesh.quads.iter().all(|q| q.size == [1, 1]));
        assert_eq!(mesh.faces, 6 * 256);
    }
}
