//! View frustum extraction and chunk intersection.
//!
//! Planes are stored as `(normal, d)` with the convention that a point `p` is
//! inside when `dot(normal, p) + d >= 0` for every plane. Normals point inward,
//! which makes the inside test a single dot product per plane with no negation
//! scattered through the code.
//!
//! The chunk test is the standard positive-vertex trick: rather than testing
//! eight corners, pick the corner furthest along each plane's normal and test
//! that one. One corner per plane, and a chunk is rejected only if that
//! best-case corner is still outside, so the test never rejects a visible chunk.

use crate::domain::{CHUNK_SIZE, ChunkPos};

/// One plane of the frustum, `dot(normal, p) + d >= 0` inside.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plane {
    pub normal: [f32; 3],
    pub d: f32,
}

impl Plane {
    /// Normalises so distances are comparable across planes.
    pub fn normalized(normal: [f32; 3], d: f32) -> Self {
        let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if len == 0.0 {
            // A degenerate plane would reject everything or nothing depending
            // on the sign of d; keep it as an always-inside plane instead.
            return Self {
                normal: [0.0; 3],
                d: f32::INFINITY,
            };
        }
        Self {
            normal: [normal[0] / len, normal[1] / len, normal[2] / len],
            d: d / len,
        }
    }

    /// Signed distance: positive inside.
    #[inline]
    pub fn signed_distance(&self, p: [f32; 3]) -> f32 {
        self.normal[0] * p[0] + self.normal[1] * p[1] + self.normal[2] * p[2] + self.d
    }
}

/// Six inward-facing planes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frustum {
    pub planes: [Plane; 6],
}

impl Frustum {
    /// A frustum that accepts every point.
    ///
    /// Used as the "no culling" option and by tests that want to exercise the
    /// other rejection paths in isolation.
    pub fn everything() -> Self {
        let empty = Plane {
            normal: [0.0; 3],
            d: f32::INFINITY,
        };
        Self { planes: [empty; 6] }
    }

    /// Builds a frustum from a camera basis, matching the usual projection
    /// conventions: `forward` and `up` need not be orthogonal, `fov_y` is the
    /// full vertical field of view in radians.
    pub fn perspective(
        eye: [f32; 3],
        forward: [f32; 3],
        up: [f32; 3],
        fov_y: f32,
        aspect: f32,
        near: f32,
        far: f32,
    ) -> Self {
        let f = normalize(forward);
        // Re-orthogonalise up against forward so a caller's loose `up` still
        // yields a right-handed basis.
        let u = normalize(sub(up, scale(f, dot(up, f))));
        let r = cross(f, u);

        let half_v = (fov_y * 0.5).tan();
        let half_h = half_v * aspect;

        // Point on the near plane centre, used as the reference for the side
        // planes so they pivot correctly.
        let near_centre = add(eye, scale(f, near));

        // Each plane passes through the eye, so its `d` is `-dot(n, eye)`.
        let side = |dir: [f32; 3]| Plane::normalized(dir, -dot(dir, eye));

        let planes = [
            // Near: facing forward.
            Plane::normalized(f, -dot(f, add(eye, scale(f, near)))),
            // Far: facing backward.
            Plane::normalized(scale(f, -1.0), dot(f, add(eye, scale(f, far)))),
            // Left and right.
            side(add(scale(f, half_h), r)),
            side(sub(scale(f, half_h), r)),
            // Top and bottom.
            side(add(scale(f, half_v), u)),
            side(sub(scale(f, half_v), u)),
        ];
        let _ = near_centre;

        Self { planes }
    }

    /// True if the point is inside every plane.
    pub fn contains_point(&self, p: [f32; 3]) -> bool {
        self.planes.iter().all(|pl| pl.signed_distance(p) >= 0.0)
    }

    /// True if the chunk's bounding box intersects the frustum.
    ///
    /// Uses the positive-vertex test: for each plane take the box corner
    /// furthest along the inward normal, and reject only if that corner is
    /// outside. Conservative, so it can keep a chunk that turns out to be
    /// invisible, but it can never drop a visible one.
    pub fn intersects_chunk(&self, pos: ChunkPos) -> bool {
        let (ox, oy, oz) = pos.origin();
        let s = CHUNK_SIZE as f32;
        let min = [ox as f32, oy as f32, oz as f32];
        let max = [min[0] + s, min[1] + s, min[2] + s];

        for pl in &self.planes {
            // Pick, per axis, the corner that is furthest along the normal.
            let positive = [
                if pl.normal[0] >= 0.0 { max[0] } else { min[0] },
                if pl.normal[1] >= 0.0 { max[1] } else { min[1] },
                if pl.normal[2] >= 0.0 { max[2] } else { min[2] },
            ];
            if pl.signed_distance(positive) < 0.0 {
                return false;
            }
        }
        true
    }

    /// True if an axis-aligned box intersects the frustum.
    pub fn intersects_aabb(&self, min: [f32; 3], max: [f32; 3]) -> bool {
        for pl in &self.planes {
            let positive = [
                if pl.normal[0] >= 0.0 { max[0] } else { min[0] },
                if pl.normal[1] >= 0.0 { max[1] } else { min[1] },
                if pl.normal[2] >= 0.0 { max[2] } else { min[2] },
            ];
            if pl.signed_distance(positive) < 0.0 {
                return false;
            }
        }
        true
    }
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(a: [f32; 3]) -> [f32; 3] {
    let len = dot(a, a).sqrt();
    if len == 0.0 {
        return [0.0, 0.0, 0.0];
    }
    scale(a, 1.0 / len)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_4};

    /// A camera at the origin looking down +X, standard up vector.
    fn looking_plus_x(fov: f32) -> Frustum {
        Frustum::perspective(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            fov,
            1.0,
            0.1,
            1000.0,
        )
    }

    #[test]
    fn everything_frustum_accepts_arbitrary_points() {
        let f = Frustum::everything();
        for p in [
            [0.0, 0.0, 0.0],
            [1e6, -1e6, 1e6],
            [-500.0, 0.0, 0.0],
        ] {
            assert!(f.contains_point(p), "everything frustum rejected {p:?}");
        }
        assert!(f.intersects_chunk(ChunkPos::new(999, -999, 0)));
    }

    #[test]
    fn a_point_in_front_is_inside_and_behind_is_outside() {
        let f = looking_plus_x(FRAC_PI_2);
        assert!(f.contains_point([50.0, 0.0, 0.0]), "in front");
        assert!(f.contains_point([5.0, 1.0, 1.0]), "near the axis");
        assert!(!f.contains_point([-50.0, 0.0, 0.0]), "behind the camera");
        assert!(!f.contains_point([2000.0, 0.0, 0.0]), "beyond the far plane");
    }

    #[test]
    fn side_planes_reject_points_outside_the_cone() {
        // A 90-degree frustum along +X: a point at 45 degrees is on the edge,
        // and slightly beyond it must be outside.
        let f = looking_plus_x(FRAC_PI_2);
        assert!(f.contains_point([10.0, 9.0, 0.0]), "just inside the cone");
        assert!(!f.contains_point([10.0, 20.0, 0.0]), "well outside the cone");
    }

    #[test]
    fn a_narrower_fov_rejects_more() {
        let wide = looking_plus_x(FRAC_PI_2);
        let narrow = looking_plus_x(FRAC_PI_4);
        let p = [10.0, 6.0, 0.0];
        assert!(wide.contains_point(p), "90 degrees should include it");
        assert!(!narrow.contains_point(p), "45 degrees should exclude it");
    }

    #[test]
    fn chunk_test_agrees_with_corner_testing() {
        // The positive-vertex trick is conservative, so it must never reject a
        // chunk that has any corner inside.
        let f = looking_plus_x(FRAC_PI_2);
        for cx in -3..6 {
            for cy in -3..4 {
                for cz in -3..4 {
                    let pos = ChunkPos::new(cx, cy, cz);
                    let (ox, oy, oz) = pos.origin();
                    let s = CHUNK_SIZE as f32;
                    let any_corner_inside = (0..8).any(|i| {
                        let corner = [
                            ox as f32 + if i & 1 == 0 { 0.0 } else { s },
                            oy as f32 + if i & 2 == 0 { 0.0 } else { s },
                            oz as f32 + if i & 4 == 0 { 0.0 } else { s },
                        ];
                        f.contains_point(corner)
                    });
                    if any_corner_inside {
                        assert!(
                            f.intersects_chunk(pos),
                            "rejected chunk {pos:?} that has a corner inside"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_chunk_containing_the_camera_is_never_rejected() {
        let f = Frustum::perspective(
            [8.0, 8.0, 8.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            FRAC_PI_2,
            1.0,
            0.1,
            1000.0,
        );
        assert!(f.intersects_chunk(ChunkPos::ORIGIN), "camera's own chunk");
        // The camera *position* is in front of the near plane by definition, so
        // contains_point is false for it; a point just beyond near is inside.
        assert!(!f.contains_point([8.0, 8.0, 8.0]), "eye sits before near plane");
        assert!(f.contains_point([8.0, 8.0, 8.2]), "just past near is inside");
    }

    #[test]
    fn planes_are_normalised() {
        let f = looking_plus_x(FRAC_PI_2);
        for p in &f.planes {
            let len = dot(p.normal, p.normal).sqrt();
            assert!(
                (len - 1.0).abs() < 1e-4 || len == 0.0,
                "plane normal not unit length: {len}"
            );
        }
    }

    #[test]
    fn aabb_test_matches_the_chunk_test() {
        let f = looking_plus_x(FRAC_PI_2);
        for pos in [ChunkPos::new(1, 0, 0), ChunkPos::new(-1, 0, 0), ChunkPos::new(0, 2, 0)] {
            let (ox, oy, oz) = pos.origin();
            let s = CHUNK_SIZE as f32;
            let aabb = f.intersects_aabb(
                [ox as f32, oy as f32, oz as f32],
                [ox as f32 + s, oy as f32 + s, oz as f32 + s],
            );
            assert_eq!(aabb, f.intersects_chunk(pos), "mismatch for {pos:?}");
        }
    }

    #[test]
    fn degenerate_basis_does_not_panic() {
        // A zero forward vector would divide by zero in a naive normalisation.
        let f = Frustum::perspective([0.0; 3], [0.0; 3], [0.0; 3], FRAC_PI_2, 1.0, 0.1, 100.0);
        let _ = f.contains_point([0.0; 3]);
        let _ = f.intersects_chunk(ChunkPos::ORIGIN);
    }
}
