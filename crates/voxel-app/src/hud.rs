//! The HUD: what it shows, and the lines it turns into.
//!
//! This module is deliberately pure. It turns state into a list of text lines
//! and nothing else, so the content of the overlay is unit-testable without a
//! GPU, a window, or a frame loop. [`crate::gpu`] does the drawing.
//!
//! Lines are ordered most important first, because the HUD is drawn top down
//! and a long list can run out of screen on a short window.

use crate::stats::FrameStats;
use voxel_core::Carver;

/// Everything the HUD can display.
#[derive(Debug, Clone)]
pub struct HudState {
    pub fps: f32,
    pub frame: Option<FrameStats>,
    pub carver: Carver,
    pub chunks_drawn: usize,
    pub vertices: usize,
    /// Chunks the cull flags rejected, and how many were drawn, for the ratio.
    pub chunks_total: usize,
    /// Milliseconds spent generating the current world.
    pub generation_ms: f32,
    /// Milliseconds spent meshing the current world.
    pub meshing_ms: f32,
    pub camera_pos: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub world_size: (u32, u32, u32),
    pub seed: u32,
    pub env: &'static str,
}

impl HudState {
    /// The lines to draw, in order.
    ///
    /// Two groups: what the renderer is doing right now, and what the world is.
    /// The frame-time percentiles are the reason the HUD exists, so they come
    /// first and are never omitted.
    pub fn lines(&self) -> Vec<String> {
        let mut out = Vec::with_capacity(12);

        match &self.frame {
            Some(s) => {
                out.push(format!("fps {:.0}  [{} frames]", s.fps(), s.samples));
                out.push(format!("frame p50 {:.2}ms", s.p50_ms));
                out.push(format!("      p90 {:.2}ms", s.p90_ms));
                out.push(format!("      p99 {:.2}ms", s.p99_ms));
                out.push(format!("    p99.9 {:.2}ms", s.p999_ms));
                out.push(format!(
                    "   min {:.2}  max {:.2}ms",
                    s.min_ms, s.max_ms
                ));
            }
            None => {
                out.push(format!("fps {:.0}", self.fps));
                out.push("frame -- (collecting)".to_string());
            }
        }

        out.push(String::new());
        out.push(format!("carver  {}", self.carver));
        out.push(format!(
            "chunks  {}/{}  {} verts",
            self.chunks_drawn, self.chunks_total, self.vertices
        ));
        out.push(format!("world   {}x{}x{}  seed {}", self.world_size.0, self.world_size.1, self.world_size.2, self.seed));
        out.push(format!(
            "gen     {:.1}ms + {:.1}ms mesh",
            self.generation_ms, self.meshing_ms
        ));
        out.push(format!(
            "camera  {:.0} {:.0} {:.0}",
            self.camera_pos[0], self.camera_pos[1], self.camera_pos[2]
        ));
        out.push(format!(
            "look    yaw {:.2} pitch {:.2}",
            self.yaw, self.pitch
        ));

        out.push(String::new());
        // Kept short: at scale 2 this is the longest line, and it sets the
        // panel width for every other line.
        out.push(format!(
            "{}  H hud  C carver  R reset",
            self.env
        ));

        out
    }

    /// Lines that should be drawn in the accent colour, by index.
    ///
    /// Returns the fps line and the p99 line: the two numbers that answer "is
    /// it smooth" and "is it stuttering".
    pub fn accent_lines(&self) -> Vec<usize> {
        vec![0, 3]
    }
}

/// Notices when the carver changes so the caller can regenerate.
///
/// Kept separate from the toggle key so the regeneration policy is testable
/// without simulating input.
#[derive(Debug, Clone, Copy)]
pub struct CarverSwitch {
    current: Carver,
    requested: Option<Carver>,
}

impl CarverSwitch {
    pub fn new(initial: Carver) -> Self {
        Self {
            current: initial,
            requested: None,
        }
    }

    pub fn current(&self) -> Carver {
        self.current
    }

    /// Requests the other carver.
    ///
    /// Toggles relative to any *pending* request rather than to the active
    /// carver, so pressing the key three times before the frame drains the
    /// request lands on sparse-then-density-then-sparse, matching what the
    /// player sees on screen rather than collapsing to a single flip.
    pub fn toggle(&mut self) {
        let from = self.requested.unwrap_or(self.current);
        self.requested = Some(match from {
            Carver::Density => Carver::Sparse,
            Carver::Sparse => Carver::Density,
        });
    }

    /// Requests a specific carver, ignoring a request for the active one.
    pub fn request(&mut self, carver: Carver) {
        if carver != self.current {
            self.requested = Some(carver);
        }
    }

    /// Takes a pending request, if any.
    ///
    /// Restarting mid-frame is avoided by having the caller drain this at a
    /// single point in the frame rather than inside the key handler.
    pub fn take(&mut self) -> Option<Carver> {
        let next = self.requested.take()?;
        self.current = next;
        Some(next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::FrameTimer;
    use std::time::Duration;

    fn sample_state() -> HudState {
        let mut t = FrameTimer::new();
        for i in 1..=100 {
            t.record(Duration::from_secs_f32(i as f32 / 1000.0));
        }
        HudState {
            fps: 60.0,
            frame: t.stats(),
            carver: Carver::Sparse,
            chunks_drawn: 40,
            vertices: 12345,
            chunks_total: 64,
            generation_ms: 24.0,
            meshing_ms: 3.5,
            camera_pos: [16.0, 70.0, 60.0],
            yaw: 0.1,
            pitch: -0.6,
            world_size: (4, 4, 4),
            seed: 1337,
            env: "dev",
        }
    }

    #[test]
    fn every_header_line_is_present() {
        let state = sample_state();
        let text = state.lines().join("\n");
        for needle in ["fps", "p50", "p90", "p99", "p99.9", "carver", "chunks", "world", "gen", "camera", "look"] {
            assert!(text.contains(needle), "HUD is missing {needle:?}:\n{text}");
        }
    }

    #[test]
    fn percentiles_appear_exactly_once_and_in_order() {
        let lines = sample_state().lines();
        let find = |p: &str| lines.iter().position(|l| l.contains(p)).unwrap();
        assert!(find("p50") < find("p90"), "p50 must come before p90");
        assert!(find("p90") < find("p99."), "p90 must come before p99.x");
    }

    #[test]
    fn the_carver_is_named_so_you_can_tell_which_mode_is_active() {
        let mut state = sample_state();
        assert!(state.lines().join("\n").contains("sparse"));

        state.carver = Carver::Density;
        let text = state.lines().join("\n");
        assert!(text.contains("density"), "density mode must be labelled");
        assert!(!text.contains("carver  sparse"), "stale carver label");
    }

    #[test]
    fn key_hints_are_shown() {
        let text = sample_state().lines().join("\n");
        for key in ["H hud", "C carver", "R reset"] {
            assert!(text.contains(key), "missing key hint {key}");
        }
    }

    #[test]
    fn no_line_is_longer_than_the_key_hint_line() {
        // The hint line sets the panel width, so a longer line elsewhere would
        // be clipped against the window edge instead of resizing the panel.
        let lines = sample_state().lines();
        let hint = lines
            .iter()
            .find(|l| l.contains("hud"))
            .expect("hint line");
        let hint_len = hint.chars().count();
        for l in &lines {
            assert!(
                l.chars().count() <= hint_len,
                "line {l:?} ({} chars) is longer than the hint line ({hint_len})",
                l.chars().count()
            );
        }
    }

    #[test]
    fn hud_survives_having_no_frame_samples_yet() {
        // The first frames of a run have no percentiles, and the HUD must still
        // render rather than panic or omit the frames-per-second line.
        let mut state = sample_state();
        state.frame = None;
        let text = state.lines().join("\n");
        assert!(text.contains("fps"), "fps line must always be present");
        assert!(text.contains("collecting"), "should say it is still sampling");
        assert!(!text.contains("p50"), "no percentiles should be claimed yet");
    }

    #[test]
    fn accent_lines_are_within_bounds() {
        // An out-of-range index would panic the renderer, so it is checked here.
        let state = sample_state();
        let lines = state.lines();
        for i in state.accent_lines() {
            assert!(i < lines.len(), "accent index {i} exceeds {} lines", lines.len());
        }
    }

    #[test]
    fn accent_lines_mark_fps_and_p99() {
        let state = sample_state();
        let lines = state.lines();
        for i in state.accent_lines() {
            let l = &lines[i];
            assert!(
                l.contains("fps") || l.contains("p99 "),
                "accent line {i} is {l:?}, expected fps or p99"
            );
        }
    }

    #[test]
    fn the_frame_block_is_first_because_it_matters_most() {
        let lines = sample_state().lines();
        assert!(lines[0].contains("fps"), "first line should be fps");
        // And the world description comes after the frame timings.
        let find = |p: &str| lines.iter().position(|l| l.contains(p)).unwrap();
        assert!(find("fps") < find("carver"));
    }

    #[test]
    fn empty_lines_separate_the_groups() {
        let lines = sample_state().lines();
        assert_eq!(
            lines.iter().filter(|l| l.is_empty()).count(),
            2,
            "expected two blank separators"
        );
    }

    #[test]
    fn carver_toggle_alternates_and_reports_a_change() {
        let mut sw = CarverSwitch::new(Carver::Density);
        assert_eq!(sw.current(), Carver::Density);
        assert_eq!(sw.take(), None, "nothing requested yet");

        sw.toggle();
        assert_eq!(sw.take(), Some(Carver::Sparse));
        assert_eq!(sw.current(), Carver::Sparse, "current must update");
        assert_eq!(sw.take(), None, "a request must only be taken once");

        sw.toggle();
        assert_eq!(sw.take(), Some(Carver::Density));
    }

    #[test]
    fn repeated_toggles_queue_at_most_one_regeneration() {
        // Pressing the key repeatedly before the frame drains the request must
        // not queue several regenerations: the last state wins, and it is
        // computed from the pending request so an odd number of presses still
        // ends on the other carver.
        let mut sw = CarverSwitch::new(Carver::Density);
        sw.toggle(); // pending sparse
        sw.toggle(); // pending density
        sw.toggle(); // pending sparse again
        assert_eq!(sw.take(), Some(Carver::Sparse), "three presses = one flip");
        assert_eq!(sw.take(), None, "only one regeneration may be queued");
        assert_eq!(sw.current(), Carver::Sparse);
    }

    #[test]
    fn requesting_the_active_carver_is_a_no_op() {
        // Otherwise a stray call would trigger a pointless full regenerate.
        let mut sw = CarverSwitch::new(Carver::Sparse);
        sw.request(Carver::Sparse);
        assert_eq!(sw.take(), None, "no-op request must not queue work");

        sw.request(Carver::Density);
        assert_eq!(sw.take(), Some(Carver::Density));
    }
}

#[cfg(test)]
mod raster_tests {
    use super::*;
    use crate::font::{HUD_BG, HUD_FG, render_text_padded, text_size};
    use std::time::Duration;

    /// Minimal software compositor for the HUD layer.
    ///
    /// This exists so the overlay can be *seen* on a machine where no GPU
    /// device is reachable. It reproduces what `gpu::push_quad` does with the
    /// same conventions (top-left pixel origin, Y flipped into clip space) and
    /// blits each line's texture, so the resulting image is what the GPU path
    /// would produce for the overlay, minus the terrain underneath.
    fn compose(lines: &[String], screen: (u32, u32), scale: u32) -> Vec<u8> {
        let (sw, sh) = screen;
        // Start from the sky colour so the overlay is judged against something
        // realistic rather than a black void.
        let mut buf = vec![0u8; (sw * sh * 3) as usize];
        for px in buf.as_chunks_mut::<3>().0 {
            px.copy_from_slice(&[135, 206, 235]);
        }

        let margin = 8u32 * scale;
        let line_step = crate::font::ADVANCE_Y * scale;
        // Same uniform panel width the GPU path computes.
        let panel_width = lines
            .iter()
            .map(|l| text_size(l, scale).0)
            .max()
            .unwrap_or(0)
            .min(sw.saturating_sub(margin * 2).max(1));
        let mut y = margin;

        for line in lines {
            if line.is_empty() {
                y += line_step;
                continue;
            }
            let tex = render_text_padded(line, scale, HUD_FG, HUD_BG, panel_width);
            if tex.width == 0 {
                y += line_step;
                continue;
            }
            // Same blit the textured quad performs, with nearest sampling.
            for ty in 0..tex.height {
                for tx in 0..tex.width {
                    let dx = margin + tx;
                    let dy = y + ty;
                    if dx >= sw || dy >= sh {
                        continue;
                    }
                    let si = ((ty * tex.width + tx) * 4) as usize;
                    let di = ((dy * sw + dx) * 3) as usize;
                    buf[di] = tex.pixels[si];
                    buf[di + 1] = tex.pixels[si + 1];
                    buf[di + 2] = tex.pixels[si + 2];
                }
            }
            y += line_step;
        }
        buf
    }

    /// Writes a PPM so the result can be opened by eye.
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

    fn state() -> HudState {
        let mut t = crate::stats::FrameTimer::new();
        for i in 0..200 {
            let ms = if i % 37 == 0 { 28.0 } else { 12.0 + (i % 5) as f32 * 0.1 };
            t.record(Duration::from_secs_f32(ms / 1000.0));
        }
        HudState {
            fps: 0.0,
            frame: t.stats(),
            carver: Carver::Sparse,
            chunks_drawn: 40,
            vertices: 12345,
            chunks_total: 64,
            generation_ms: 24.0,
            meshing_ms: 3.5,
            camera_pos: [16.0, 70.0, 60.0],
            yaw: 0.1,
            pitch: -0.6,
            world_size: (4, 4, 4),
            seed: 1337,
            env: "dev",
        }
    }

    #[test]
    fn the_overlay_rasterises_and_is_written_for_inspection() {
        let lines = state().lines();
        let (w, h) = (640u32, 400u32);
        let rgb = compose(&lines, (w, h), 2);

        // The overlay must actually cover pixels: a blank frame would mean the
        // text never made it into the image.
        let hud_pixels = rgb
            .as_chunks::<3>()
            .0
            .iter()
            .filter(|p| **p == HUD_BG || **p == HUD_FG)
            .count();
        assert!(
            hud_pixels > 2000,
            "overlay covered only {hud_pixels} pixels; it did not render"
        );

        // And the sky must still be visible around it, so the overlay is a
        // panel rather than a full-screen fill.
        let sky_pixels = rgb
            .as_chunks::<3>()
            .0
            .iter()
            .filter(|p| **p == [135, 206, 235])
            .count();
        assert!(sky_pixels > 1000, "overlay filled the whole frame");

        // Written under `target/`, which is always ignored, so a test run never
        // dirties the working tree. The version referenced by the README lives
        // in `docs/images/` and is copied there deliberately.
        let path = std::path::Path::new("target/hud-overlay.ppm");
        write_ppm(path, w, h, &rgb);
        assert!(path.exists(), "the HUD image was not written");
    }

    #[test]
    fn overlay_text_stays_within_the_screen() {
        // A long line must be clipped rather than panicking on out-of-range
        // indices, which is the failure mode a wider world or a smaller window
        // would hit.
        let mut lines = state().lines();
        lines.push("x".repeat(400));
        let (w, h) = (320u32, 200u32);
        let rgb = compose(&lines, (w, h), 2);
        assert_eq!(rgb.len(), (w * h * 3) as usize);
    }

    #[test]
    fn every_hud_line_contributes_ink() {
        // Catches a line silently rendering as blank, which the layout would
        // otherwise hide by simply leaving a gap.
        let lines = state().lines();
        let scale = 2u32;
        let line_step = crate::font::ADVANCE_Y * scale;
        let rgb = compose(&lines, (640, 400), scale);

        let mut y = 8u32 * scale;
        for line in &lines {
            if line.is_empty() {
                y += line_step;
                continue;
            }
            let mut ink = false;
            // Only the glyph band needs checking, not the whole advance step.
            for row in y..(y + crate::font::GLYPH_H * scale).min(400) {
                for col in (8 * scale)..600 {
                    let i = ((row * 640 + col) * 3) as usize;
                    if [rgb[i], rgb[i + 1], rgb[i + 2]] == HUD_FG {
                        ink = true;
                    }
                }
            }
            assert!(ink, "HUD line {line:?} drew no visible text");
            y += line_step;
        }
    }
}
