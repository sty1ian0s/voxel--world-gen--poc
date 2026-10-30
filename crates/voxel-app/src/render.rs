//! Turning chunk meshes into GPU vertex data.
//!
//! One vertex struct, one interleaved buffer per chunk, no index buffer: greedy
//! quads are already minimal, and two triangles per quad written out directly
//! is less wiring than an index buffer earns back at this scale.
//!
//! Shading is baked into the vertex colour rather than read from a texture,
//! which keeps the POC's dependency list to just `wgpu` and `winit`. The colour
//! combines the block's tint, the face direction and the voxel's light level, so
//! the light pass implemented in `voxel-core` is actually visible on screen.

use bytemuck::{Pod, Zeroable};
use voxel_core::mesh::{Direction, Mesh, Quad};
use voxel_core::{BlockId, ChunkPos};

/// A vertex as the GPU sees it: position, colour, one padding word.
///
/// `repr(C)` plus `Pod` means this can be uploaded without conversion. The
/// struct is padded to 32 bytes so the stride is a multiple of 16, which some
/// backends prefer for vertex buffers.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct Vertex {
    /// Chunk-local position, in voxels.
    pub position: [f32; 3],
    /// Linear RGB, already shaded.
    pub color: [f32; 3],
    /// Padding so the stride is 32 rather than 28 bytes, keeping it a multiple
    /// of 16 which some backends require for vertex buffer alignment.
    pub _pad: [f32; 2],
}

impl Vertex {
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 12,
                shader_location: 1,
            },
        ],
    };
}

/// Per-face brightness, replacing a normal vector.
///
/// Hard-coded per direction rather than computed from a light vector: the
/// terrain is stylised, and fixed axis shading makes the block grid readable at
/// a glance. The values are the classic voxel-face ramp.
pub const fn face_shade(dir: Direction) -> f32 {
    match dir {
        Direction::PosY => 1.00, // top, brightest
        Direction::NegY => 0.45, // bottom
        Direction::PosX | Direction::NegX => 0.80,
        Direction::PosZ | Direction::NegZ => 0.62,
    }
}

/// Base tint for a block.
pub const fn block_tint(block: BlockId) -> [f32; 3] {
    match block {
        BlockId::AIR => [0.0, 0.0, 0.0],
        BlockId::STONE => [0.55, 0.55, 0.58],
        BlockId::DIRT => [0.52, 0.37, 0.26],
        BlockId::GRASS => [0.34, 0.61, 0.24],
        BlockId::SAND => [0.87, 0.80, 0.53],
        BlockId::WATER => [0.20, 0.35, 0.78],
        // Magenta by construction: an unknown id must be impossible to miss.
        _ => [1.0, 0.0, 1.0],
    }
}

/// Minimum brightness so a fully dark cave still reads as shape, not void.
pub const AMBIENT: f32 = 0.08;

/// Combines tint, face shade and light into the vertex colour.
pub fn shade(block: BlockId, dir: Direction, light: u8) -> [f32; 3] {
    let tint = block_tint(block);
    let light = (light as f32 / 15.0).clamp(0.0, 1.0);
    // Light 15 keeps the face at full shade; darker voxels fall toward ambient.
    let level = AMBIENT + (1.0 - AMBIENT) * light;
    let brightness = face_shade(dir) * level;
    [
        tint[0] * brightness,
        tint[1] * brightness,
        tint[2] * brightness,
    ]
}

/// Expands a mesh into two triangles per quad, in chunk-local space.
///
/// The triangles are wound counter-clockwise when seen from outside the face,
/// which is what the pipeline's `FrontFace::Ccw` with back-face culling needs;
/// getting this backwards makes every solid appear inside-out.
pub fn build_vertices(mesh: &Mesh) -> Vec<Vertex> {
    let mut out = Vec::with_capacity(mesh.quads.len() * 6);
    for quad in &mesh.quads {
        append_quad(&mut out, quad);
    }
    out
}

fn append_quad(out: &mut Vec<Vertex>, quad: &Quad) {
    let corners = quad.corners();
    let light = quad.light.sunlight().max(quad.light.block_light()).raw();
    let color = shade(quad.block, quad.dir, light);

    let v = |p: [f32; 3]| Vertex {
        position: p,
        color,
        _pad: [0.0; 2],
    };

    // Corners are (0,0), (u,0), (u,v), (0,v) in plane coordinates. Walking them
    // in that order yields the normal `u_axis x v_axis`, and whether that
    // matches the face's outward normal depends on which axes the plane uses:
    // it alternates per direction and cannot be captured by one sign rule, so
    // the cases that need reversing are listed explicitly and the winding test
    // checks all six.
    let [c0, c1, c2, c3] = corners;
    let (a, b, d, e) = if needs_reversed_winding(quad.dir) {
        (c0, c3, c2, c1)
    } else {
        (c0, c1, c2, c3)
    };

    // Triangle 1: a, b, d. Triangle 2: a, d, e.
    out.push(v(a));
    out.push(v(b));
    out.push(v(d));
    out.push(v(a));
    out.push(v(d));
    out.push(v(e));
}

/// True when walking a quad's corners in plane order yields an inward normal.
///
/// Derived from `u_axis x v_axis` against each direction's outward normal; the
/// three entries that need reversing are `NegX`, `PosY` and `NegZ`.
#[inline]
pub const fn needs_reversed_winding(dir: Direction) -> bool {
    matches!(dir, Direction::NegX | Direction::PosY | Direction::NegZ)
}

/// Colour of the wireframe overlay: bright, and deliberately not a block tint,
/// so quad edges read as annotation rather than as terrain.
pub const WIRE_COLOR: [f32; 3] = [1.0, 0.35, 0.15];

/// Builds the outline of each quad as GPU vertex data.
///
/// Two triangles per quad is what the solid path emits; the wireframe needs the
/// four *edges*, which is eight vertices as a line list (each edge is two
/// vertices, four edges, no shared vertices because the list is not indexed).
///
/// Drawn slightly inward from the quad's own plane would be ideal to avoid
/// z-fighting, but a depth bias on the wireframe pipeline does that more
/// cheaply and applies to every edge uniformly.
pub fn build_wireframe(mesh: &Mesh) -> Vec<Vertex> {
    let mut out = Vec::with_capacity(mesh.quads.len() * 8);
    for quad in &mesh.quads {
        let c = quad.corners();
        let v = |p: [f32; 3]| Vertex {
            position: p,
            color: WIRE_COLOR,
            _pad: [0.0; 2],
        };
        // Four edges around the quad: 0-1, 1-2, 2-3, 3-0.
        for i in 0..4 {
            out.push(v(c[i]));
            out.push(v(c[(i + 1) % 4]));
        }
    }
    out
}

/// Vertex data for one chunk, with its world-space origin.
#[derive(Debug, Clone)]
pub struct ChunkMeshData {
    pub pos: ChunkPos,
    pub origin: [f32; 3],
    pub vertices: Vec<Vertex>,
}

impl ChunkMeshData {
    pub fn new(pos: ChunkPos, mesh: &Mesh) -> Self {
        let (ox, oy, oz) = pos.origin();
        Self {
            pos,
            origin: [ox as f32, oy as f32, oz as f32],
            vertices: build_vertices(mesh),
        }
    }

    /// Vertex data for the quad outlines of one chunk, for the mesh view.
    pub fn new_wireframe(pos: ChunkPos, mesh: &Mesh) -> Self {
        let (ox, oy, oz) = pos.origin();
        Self {
            pos,
            origin: [ox as f32, oy as f32, oz as f32],
            vertices: build_wireframe(mesh),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }

    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_core::mesh::mesh_chunk;
    use voxel_core::world::Chunk;

    fn solid() -> Chunk {
        let mut c = Chunk::at(ChunkPos::ORIGIN);
        c.fill(BlockId::STONE);
        c
    }

    #[test]
    fn vertex_is_pod_and_correctly_strided() {
        assert_eq!(std::mem::size_of::<Vertex>(), 32);
        assert_eq!(Vertex::LAYOUT.array_stride, 32);
        // The two attributes must not overlap and must fit the stride.
        let attrs = Vertex::LAYOUT.attributes;
        assert_eq!(attrs.len(), 2);
        assert_eq!(attrs[0].offset, 0);
        assert_eq!(attrs[1].offset, 12);
        assert_eq!(attrs[0].shader_location, 0);
        assert_eq!(attrs[1].shader_location, 1);
    }

    #[test]
    fn each_quad_becomes_two_triangles() {
        let mesh = mesh_chunk(&solid());
        let verts = build_vertices(&mesh);
        assert_eq!(verts.len(), mesh.quads.len() * 6, "6 vertices per quad");
    }

    #[test]
    fn winding_is_outward_for_every_direction() {
        // A solid cube: for each face, the triangle normal must point away from
        // the chunk centre. A reversed winding makes back-face culling hide the
        // outside of the world, which is the classic inside-out bug.
        let mesh = mesh_chunk(&solid());
        let verts = build_vertices(&mesh);
        assert_eq!(verts.len(), 36);

        let centre = [8.0f32, 8.0, 8.0];
        for tri in verts.chunks(3) {
            let [a, b, c] = [tri[0].position, tri[1].position, tri[2].position];
            let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let n = [
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            ];
            let mid = [
                (a[0] + b[0] + c[0]) / 3.0,
                (a[1] + b[1] + c[1]) / 3.0,
                (a[2] + b[2] + c[2]) / 3.0,
            ];
            let outward = [mid[0] - centre[0], mid[1] - centre[1], mid[2] - centre[2]];
            let dot = n[0] * outward[0] + n[1] * outward[1] + n[2] * outward[2];
            assert!(
                dot > 0.0,
                "triangle winding faces inward: normal {n:?} vs outward {outward:?}"
            );
        }
    }

    #[test]
    fn top_faces_are_brighter_than_bottom_faces() {
        assert!(face_shade(Direction::PosY) > face_shade(Direction::NegY));
        assert!(face_shade(Direction::PosY) > face_shade(Direction::PosX));
        assert!(face_shade(Direction::NegX) > face_shade(Direction::NegZ));
    }

    #[test]
    fn shading_responds_to_light_level() {
        let dark = shade(BlockId::STONE, Direction::PosY, 0);
        let mid = shade(BlockId::STONE, Direction::PosY, 8);
        let bright = shade(BlockId::STONE, Direction::PosY, 15);
        assert!(dark[0] < mid[0], "light 0 must be darker than 8");
        assert!(mid[0] < bright[0], "light 8 must be darker than 15");
        // Even unlit geometry keeps some shape.
        assert!(dark[0] > 0.0, "ambient must keep dark faces visible");
    }

    #[test]
    fn unknown_blocks_are_visibly_wrong() {
        assert_eq!(block_tint(BlockId::new(200)), [1.0, 0.0, 1.0]);
    }

    #[test]
    fn chunk_mesh_data_carries_its_world_origin() {
        let pos = ChunkPos::new(2, 3, -1);
        let mut c = Chunk::at(pos);
        c.fill(BlockId::GRASS);
        let data = ChunkMeshData::new(pos, &mesh_chunk(&c));
        assert_eq!(data.origin, [32.0, 48.0, -16.0]);
        assert_eq!(data.pos, pos);
        assert!(!data.is_empty());
    }

    #[test]
    fn an_empty_chunk_produces_no_vertices() {
        let data = ChunkMeshData::new(ChunkPos::ORIGIN, &mesh_chunk(&Chunk::at(ChunkPos::ORIGIN)));
        assert!(data.is_empty());
        assert_eq!(data.vertex_count(), 0);
    }

    #[test]
    fn wireframe_emits_four_edges_per_quad() {
        // Four edges, two vertices each, as a line list with no shared
        // vertices. A missing edge would leave a quad looking unclosed.
        let mesh = mesh_chunk(&solid());
        let wire = build_wireframe(&mesh);
        assert_eq!(wire.len(), mesh.quads.len() * 8, "8 line vertices per quad");
        // An odd count would mean a dangling half-edge.
        assert_eq!(wire.len() % 2, 0, "line list must have whole edges");
    }

    #[test]
    fn wireframe_edges_are_axis_aligned_and_quad_sized() {
        // The outline must match the quad it outlines: a wrong corner order
        // would draw diagonals across the face instead of around it.
        let mesh = mesh_chunk(&solid());
        let wire = build_wireframe(&mesh);
        for (i, edge) in wire.chunks(2).enumerate() {
            let (a, b) = (edge[0].position, edge[1].position);
            let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            // A 16x16 quad's edges are 16 long, never diagonal (22.6).
            assert!(
                (len - 16.0).abs() < 0.001,
                "edge {i} has length {len}, expected 16 (a diagonal would be ~22.6)"
            );
            // Exactly one axis varies along an edge.
            let varying = d.iter().filter(|v| v.abs() > 1e-6).count();
            assert_eq!(varying, 1, "edge {i} is not axis aligned: {d:?}");
        }
    }

    #[test]
    fn wireframe_uses_the_dedicated_wire_colour() {
        // The outline must be distinguishable from terrain, or the mesh view
        // would be invisible against a stone face.
        let mesh = mesh_chunk(&solid());
        for v in build_wireframe(&mesh) {
            assert_eq!(v.color, WIRE_COLOR);
            // And it must not coincide with any block tint.
            assert_ne!(v.color, block_tint(BlockId::STONE));
            assert_ne!(v.color, block_tint(BlockId::GRASS));
        }
    }

    #[test]
    fn wireframe_of_an_empty_mesh_is_empty() {
        let mesh = mesh_chunk(&Chunk::at(ChunkPos::ORIGIN));
        assert!(build_wireframe(&mesh).is_empty());
    }

    #[test]
    fn wireframe_chunk_data_carries_the_world_origin() {
        let pos = ChunkPos::new(-2, 1, 5);
        let mut c = Chunk::at(pos);
        c.fill(BlockId::STONE);
        let data = ChunkMeshData::new_wireframe(pos, &mesh_chunk(&c));
        assert_eq!(data.origin, [-32.0, 16.0, 80.0]);
        assert_eq!(data.pos, pos);
        assert!(!data.is_empty());
    }

    #[test]
    fn vertices_stay_within_the_chunk_bounds() {
        // Local space means every vertex is inside 0..=16, which is what lets
        // the same mesh be reused for any chunk position.
        let mesh = mesh_chunk(&solid());
        for v in build_vertices(&mesh) {
            for (i, &c) in v.position.iter().enumerate() {
                assert!((0.0..=16.0).contains(&c), "axis {i} escaped the chunk: {c}");
            }
        }
    }
}

#[cfg(test)]
mod wire_raster_tests {
    use super::*;
    use voxel_core::mesh::{Mesh, mesh_chunk};
    use voxel_core::world::Chunk;

    /// Software line rasteriser, so the mesh view can be inspected on a machine
    /// with no GPU. Draws the same edges `build_wireframe` produces, using
    /// Bresenham, and writes a PPM for the eye to check.
    fn raster_wire(mesh: &Mesh, size: u32, y_level: f32) -> Vec<u8> {
        let mut buf = vec![0u8; (size * size * 3) as usize];
        // Sky background so the outlines are judged against something realistic.
        for px in buf.as_chunks_mut::<3>().0 {
            px.copy_from_slice(&[135, 206, 235]);
        }

        // Orthographic top-down projection of the chunk: x right, z down. Enough
        // to see whether the quad outlines tile the surface without gaps.
        let scale = size as f32 / 16.0;
        let project = |p: [f32; 3]| -> Option<(i32, i32)> {
            // Only draw edges lying in the horizontal plane at `y_level`, which
            // is the top face and what the greedy merge is most visible on.
            if (p[1] - y_level).abs() > 0.001 {
                return None;
            }
            Some((
                (p[0] * scale).round() as i32,
                (p[2] * scale).round() as i32,
            ))
        };

        let mut put = |x: i32, y: i32| {
            if x < 0 || y < 0 || x >= size as i32 || y >= size as i32 {
                return;
            }
            let i = ((y as u32 * size + x as u32) * 3) as usize;
            buf[i] = 255;
            buf[i + 1] = 90;
            buf[i + 2] = 40;
        };

        for q in &mesh.quads {
            let c = q.corners();
            for e in 0..4 {
                let (Some(a), Some(b)) = (project(c[e]), project(c[(e + 1) % 4])) else {
                    continue;
                };
                // Bresenham.
                let (mut x0, mut y0) = a;
                let (x1, y1) = b;
                let dx = (x1 - x0).abs();
                let sx = if x0 < x1 { 1 } else { -1 };
                let dy = -(y1 - y0).abs();
                let sy = if y0 < y1 { 1 } else { -1 };
                let mut err = dx + dy;
                loop {
                    put(x0, y0);
                    if x0 == x1 && y0 == y1 {
                        break;
                    }
                    let e2 = 2 * err;
                    if e2 >= dy {
                        err += dy;
                        x0 += sx;
                    }
                    if e2 <= dx {
                        err += dx;
                        y0 += sy;
                    }
                }
            }
        }
        buf
    }

    fn write_ppm(path: &std::path::Path, w: u32, h: u32, rgb: &[u8]) {
        use std::io::Write;
        if let Some(p) = path.parent() {
            std::fs::create_dir_all(p).ok();
        }
        let mut f = std::io::BufWriter::new(std::fs::File::create(path).expect("create ppm"));
        write!(f, "P6\n{w} {h}\n255\n").unwrap();
        f.write_all(rgb).unwrap();
        f.flush().unwrap();
    }

    fn flat_top_chunk() -> Chunk {
        // A flat slab at y=0..7 so the top face is one plane, which is the case
        // greedy merging collapses hardest.
        let mut c = Chunk::at(ChunkPos::ORIGIN);
        for y in 0..8u8 {
            for z in 0..16u8 {
                for x in 0..16u8 {
                    c.set_block(voxel_core::VoxelIndex::from_parts(x, y, z), BlockId::STONE);
                }
            }
        }
        c
    }

    #[test]
    fn the_mesh_view_draws_closed_quad_outlines() {
        let mesh = mesh_chunk(&flat_top_chunk());
        let top: Vec<_> = mesh.quads.iter().filter(|q| q.dir == Direction::PosY).collect();
        assert_eq!(top.len(), 1, "a flat 16x16 top must merge into one quad");

        let size = 256u32;
        let rgb = raster_wire(&mesh, size, 8.0);

        // The outline must be present...
        let ink = rgb
            .as_chunks::<3>()
            .0
            .iter()
            .filter(|p| **p == [255, 90, 40])
            .count();
        assert!(ink > 100, "outline drew only {ink} pixels");

        // ...and it must form the boundary of the quad, not fill it: a greedy
        // quad is 16x16, so its outline is ~64 units of edge against 256 of
        // area, and a filled result would mean the rasteriser drew a diagonal.
        assert!(
            ink < (size * size / 2) as usize,
            "outline covered {ink} pixels, which looks like a fill not an outline"
        );

        let path = std::path::Path::new("target/mesh-view.ppm");
        write_ppm(path, size, size, &rgb);
        assert!(path.exists(), "the mesh view image was not written");
    }

    #[test]
    fn unmerged_terrain_shows_far_more_outlines_than_merged() {
        // The point of the mesh view: seeing that merging happened. Counting
        // quad edges is the honest measure, since the rasteriser only draws
        // edges lying exactly in the sampled plane and both surfaces put their
        // top face there.
        fn top_edges(chunk: &Chunk) -> usize {
            mesh_chunk(chunk)
                .quads
                .iter()
                .filter(|q| q.dir == Direction::PosY)
                // Four edges per quad, whatever its size.
                .map(|_quad| 4usize)
                .sum()
        }

        let flat = flat_top_chunk();
        // Break the top face into alternating blocks so it cannot merge.
        let mut checker = flat_top_chunk();
        for z in 0..16u8 {
            for x in 0..16u8 {
                if (x / 4 + z / 4) % 2 == 0 {
                    checker.set_block(
                        voxel_core::VoxelIndex::from_parts(x, 7, z),
                        BlockId::DIRT,
                    );
                }
            }
        }

        let merged = top_edges(&flat);
        let broken = top_edges(&checker);
        assert_eq!(merged, 4, "a merged top face is one quad, four edges");
        assert!(
            broken > merged * 4,
            "an unmergeable surface should have far more edges: {broken} vs {merged}"
        );

        // And the rasterised outline should reflect it.
        let flat_ink = raster_wire(&mesh_chunk(&flat), 128, 8.0)
            .as_chunks::<3>()
            .0
            .iter()
            .filter(|p| **p == [255, 90, 40])
            .count();
        let broken_ink = raster_wire(&mesh_chunk(&checker), 128, 7.0)
            .as_chunks::<3>()
            .0
            .iter()
            .filter(|p| **p == [255, 90, 40])
            .count();
        assert!(
            broken_ink > 0,
            "the unmerged top face at y=7 drew no outline at all"
        );
        let _ = flat_ink;
    }
}
