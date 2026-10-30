//! Camera and its view/projection matrices.
//!
//! Matrices are built by hand in column-major order, which is what WGSL expects
//! from a `mat4x4<f32>` uniform. No matrix library: four matrices with fixed
//! conventions do not justify a dependency, and hand-rolling makes the handedness
//! and depth-range conventions explicit, which is where rendering bugs hide.

/// A perspective camera looking down `-Z` in view space, wgpu's depth range.
#[derive(Debug, Clone, Copy)]
pub struct Camera {
    pub eye: [f32; 3],
    /// Yaw around Y, radians. Zero looks toward `-Z`.
    pub yaw: f32,
    /// Pitch, radians, clamped to just under a right angle to avoid gimbal flip.
    pub pitch: f32,
    pub fov_y: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            eye: [0.0, 0.0, 0.0],
            yaw: 0.0,
            pitch: 0.0,
            fov_y: 70f32.to_radians(),
            aspect: 16.0 / 9.0,
            near: 0.1,
            far: 2048.0,
        }
    }
}

impl Camera {
    /// Unit forward vector derived from yaw and pitch.
    pub fn forward(&self) -> [f32; 3] {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        // Yaw 0 looks toward -Z, matching the view matrix below.
        [-sy * cp, sp, -cy * cp]
    }

    /// Unit right vector, always horizontal.
    pub fn right(&self) -> [f32; 3] {
        let (sy, cy) = self.yaw.sin_cos();
        [cy, 0.0, -sy]
    }

    /// Up vector, derived from the other two so it stays orthogonal.
    pub fn up(&self) -> [f32; 3] {
        let f = self.forward();
        let r = self.right();
        [
            r[1] * f[2] - r[2] * f[1],
            r[2] * f[0] - r[0] * f[2],
            r[0] * f[1] - r[1] * f[0],
        ]
    }

    /// Moves the camera in its own basis. `m` is `(forward, right, up)` amounts.
    pub fn translate(&mut self, forward: f32, right: f32, up: f32) {
        let f = self.forward();
        let r = self.right();
        for i in 0..3 {
            self.eye[i] += f[i] * forward + r[i] * right;
            // World up, not camera up: flying should not roll the view.
            self.eye[i] += if i == 1 { up } else { 0.0 };
        }
    }

    pub fn rotate(&mut self, dyaw: f32, dpitch: f32) {
        self.yaw += dyaw;
        // Stop just short of vertical: at exactly +-pi/2 the forward and up
        // vectors become parallel and the view matrix degenerates.
        let limit = std::f32::consts::FRAC_PI_2 - 0.001;
        self.pitch = (self.pitch + dpitch).clamp(-limit, limit);
    }

    /// View matrix: world space into camera space.
    ///
    /// Column-major, `[right; up; -forward]` with the eye translation folded in,
    /// which is the transpose of the usual row-major layout.
    pub fn view_matrix(&self) -> [[f32; 4]; 4] {
        let f = self.forward();
        let r = self.right();
        let u = self.up();
        let e = self.eye;

        // Camera space looks down -Z, so the third basis row is -forward.
        let z = [-f[0], -f[1], -f[2]];

        let dot = |a: [f32; 3]| -(a[0] * e[0] + a[1] * e[1] + a[2] * e[2]);
        let (tx, ty, tz) = (dot(r), dot(u), dot(z));

        [
            [r[0], u[0], z[0], 0.0],
            [r[1], u[1], z[1], 0.0],
            [r[2], u[2], z[2], 0.0],
            [tx, ty, tz, 1.0],
        ]
    }

    /// Perspective projection with a `0..1` depth range, which is what wgpu,
    /// D3D and Metal use; OpenGL's `-1..1` would put half the depth buffer out
    /// of range and break the depth test.
    pub fn projection_matrix(&self) -> [[f32; 4]; 4] {
        let f = 1.0 / (self.fov_y * 0.5).tan();
        let nf = 1.0 / (self.near - self.far);
        [
            [f / self.aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, self.far * nf, -1.0],
            [0.0, 0.0, self.far * self.near * nf, 0.0],
        ]
    }

    /// Combined view-projection, the order the vertex shader applies them.
    pub fn view_projection(&self) -> [[f32; 4]; 4] {
        mul(self.projection_matrix(), self.view_matrix())
    }
}

/// Column-major 4x4 multiply: `a * b`, so `a` is applied second.
pub fn mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0f32; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            out[col][row] = (0..4).map(|k| a[k][row] * b[col][k]).sum();
        }
    }
    out
}

/// Transforms a point, returning the clip-space position and its `w`.
pub fn transform_point(m: [[f32; 4]; 4], p: [f32; 3]) -> [f32; 4] {
    let mut out = [0.0f32; 4];
    for row in 0..4 {
        out[row] = m[0][row] * p[0] + m[1][row] * p[1] + m[2][row] * p[2] + m[3][row];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn forward_right_up_form_an_orthonormal_basis() {
        for &(yaw, pitch) in &[
            (0.0, 0.0),
            (1.0, 0.3),
            (-2.5, -0.7),
            (FRAC_PI_2, 0.0),
            (0.0, FRAC_PI_2 - 0.01),
        ] {
            let c = Camera {
                yaw,
                pitch,
                ..Default::default()
            };
            let (f, r, u) = (c.forward(), c.right(), c.up());
            let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
            let len = |a: [f32; 3]| dot(a, a).sqrt();

            for (name, v) in [("forward", f), ("right", r), ("up", u)] {
                assert!(
                    approx(len(v), 1.0, 1e-3),
                    "{name} not unit at yaw={yaw} pitch={pitch}: {}",
                    len(v)
                );
            }
            assert!(approx(dot(f, r), 0.0, 1e-3), "forward/right not orthogonal");
            assert!(approx(dot(f, u), 0.0, 1e-3), "forward/up not orthogonal");
            assert!(approx(dot(r, u), 0.0, 1e-3), "right/up not orthogonal");
        }
    }

    #[test]
    fn yaw_zero_looks_down_negative_z() {
        let c = Camera::default();
        let f = c.forward();
        assert!(approx(f[0], 0.0, 1e-5) && approx(f[1], 0.0, 1e-5));
        assert!(approx(f[2], -1.0, 1e-5), "expected -Z, got {f:?}");
    }

    #[test]
    fn a_point_in_front_maps_into_the_clip_volume() {
        let c = Camera {
            fov_y: FRAC_PI_2,
            aspect: 1.0,
            ..Default::default()
        };
        // Directly ahead by 10, which is inside near..far.
        let clip = transform_point(c.view_projection(), [0.0, 0.0, -10.0]);
        assert!(clip[3] > 0.0, "point in front must have positive w");
        let ndc = [clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3]];
        for (i, v) in ndc.iter().enumerate() {
            assert!(
                (-1.0..=1.0).contains(v),
                "ndc[{i}] = {v} outside the clip volume: {ndc:?}"
            );
        }
    }

    #[test]
    fn depth_range_is_zero_to_one_not_minus_one_to_one() {
        // wgpu uses a 0..1 depth range; an OpenGL-style projection would put
        // the near plane at z=-1 and everything would fail the depth test.
        let c = Camera::default();
        let proj = c.projection_matrix();
        let near_ndc = transform_point(proj, [0.0, 0.0, -c.near]);
        let far_ndc = transform_point(proj, [0.0, 0.0, -c.far]);
        assert!(
            approx(near_ndc[2] / near_ndc[3], 0.0, 1e-4),
            "near plane should map to z=0, got {}",
            near_ndc[2] / near_ndc[3]
        );
        assert!(
            approx(far_ndc[2] / far_ndc[3], 1.0, 1e-4),
            "far plane should map to z=1, got {}",
            far_ndc[2] / far_ndc[3]
        );
    }

    #[test]
    fn a_point_behind_the_camera_has_negative_w() {
        let c = Camera::default();
        let clip = transform_point(c.view_projection(), [0.0, 0.0, 10.0]);
        assert!(clip[3] < 0.0, "behind the camera must be clipped by w");
    }

    #[test]
    fn the_eye_maps_to_the_view_space_origin() {
        let c = Camera {
            eye: [7.0, -3.0, 11.0],
            yaw: 0.7,
            pitch: -0.2,
            ..Default::default()
        };
        let view = c.view_matrix();
        let p = transform_point(view, c.eye);
        assert!(approx(p[0], 0.0, 1e-3) && approx(p[1], 0.0, 1e-3) && approx(p[2], 0.0, 1e-3),
            "eye should be the view-space origin, got {p:?}");
    }

    #[test]
    fn rotating_never_reaches_singular_pitch() {
        let mut c = Camera::default();
        for _ in 0..100 {
            c.rotate(0.0, 0.5);
        }
        assert!(c.pitch < FRAC_PI_2, "pitch reached the singularity");
        let _ = c.up();
        let _ = c.view_matrix();
    }

    #[test]
    fn translating_forward_moves_along_the_view_direction() {
        let mut c = Camera::default();
        let before = c.eye;
        c.translate(5.0, 0.0, 0.0);
        let moved = [c.eye[0] - before[0], c.eye[1] - before[1], c.eye[2] - before[2]];
        let f = c.forward();
        for i in 0..3 {
            assert!(approx(moved[i], f[i] * 5.0, 1e-3), "axis {i} moved wrong");
        }
    }

    #[test]
    fn vertical_translation_is_always_world_up() {
        // Pitching must not turn "up" into a forward component, or flying
        // becomes unintuitive.
        let mut c = Camera {
            pitch: 1.0,
            ..Default::default()
        };
        let before = c.eye;
        c.translate(0.0, 0.0, 3.0);
        assert!(approx(c.eye[0], before[0], 1e-5), "x drifted");
        assert!(approx(c.eye[2], before[2], 1e-5), "z drifted");
        assert!(approx(c.eye[1] - before[1], 3.0, 1e-5), "y should rise by 3");
    }

    #[test]
    fn matrix_multiply_matches_manual_application() {
        // Applying view then projection must equal applying the combined matrix.
        let c = Camera::default();
        let p = [3.0, -4.0, -20.0];
        let via_view = transform_point(c.view_matrix(), p);
        let manual = transform_point(
            c.projection_matrix(),
            [via_view[0], via_view[1], via_view[2]],
        );
        let combined = transform_point(c.view_projection(), p);
        for i in 0..4 {
            assert!(
                approx(combined[i], manual[i], 1e-2),
                "component {i}: {} vs {}",
                combined[i],
                manual[i]
            );
        }
    }
}
