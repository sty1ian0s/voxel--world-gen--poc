//! Frame-time statistics with percentiles.
//!
//! A mean frame rate hides exactly the thing that matters: the occasional long
//! frame that makes a world feel like it stutters. `p99` and `p99.9` are what
//! expose those, so this keeps a rolling window of recent frame times and
//! computes percentiles on demand rather than only reporting an average.
//!
//! Percentiles are computed by sorting a copy of the window. At a few hundred
//! samples that is faster than maintaining a running quantile estimate, and it
//! is exact rather than approximate, which matters when the numbers are being
//! used to judge a change.

use std::time::Duration;

/// How many recent frames the window holds.
///
/// 600 is ten seconds at 60 fps: long enough to be stable, short enough that a
/// change is visible within a couple of seconds.
pub const WINDOW: usize = 600;

/// Percentiles reported, as fractions.
pub const PERCENTILES: [(f64, &str); 4] = [(0.50, "p50"), (0.90, "p90"), (0.99, "p99"), (0.999, "p99.9")];

/// Summary of one window of frame times.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameStats {
    pub samples: usize,
    /// Fastest frame in the window.
    pub min_ms: f32,
    /// Median frame time.
    pub p50_ms: f32,
    pub p90_ms: f32,
    pub p99_ms: f32,
    pub p999_ms: f32,
    /// Slowest frame in the window.
    pub max_ms: f32,
    /// Mean frame time over the window.
    pub mean_ms: f32,
}

impl FrameStats {
    /// Frames per second implied by the median frame time.
    ///
    /// Median rather than mean: a single 200 ms hitch drags the mean fps down
    /// and makes a healthy run look bad.
    pub fn fps(&self) -> f32 {
        if self.p50_ms <= 0.0 {
            return 0.0;
        }
        1000.0 / self.p50_ms
    }

    /// A one-line summary for the log.
    pub fn summary(&self) -> String {
        format!(
            "{:.0} fps | p50 {:.2}ms p90 {:.2}ms p99 {:.2}ms p99.9 {:.2}ms | min {:.2} max {:.2} ({})",
            self.fps(),
            self.p50_ms,
            self.p90_ms,
            self.p99_ms,
            self.p999_ms,
            self.min_ms,
            self.max_ms,
            self.samples
        )
    }
}

/// A rolling window of frame times.
#[derive(Debug, Clone)]
pub struct FrameTimer {
    /// Ring buffer of frame times in milliseconds.
    samples: Vec<f32>,
    /// Where the next sample goes.
    next: usize,
    /// Number of samples written so far, saturating at `WINDOW`.
    filled: usize,
    /// Time accumulated since the last sample was taken.
    accumulator: Duration,
    /// Frame count within the current sample bucket.
    frames_in_bucket: u32,
    /// How many frames to average into one sample.
    frames_per_sample: u32,
    /// Scratch buffer reused when sorting, so sampling does not allocate.
    scratch: Vec<f32>,
}

impl Default for FrameTimer {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameTimer {
    pub fn new() -> Self {
        Self {
            samples: vec![0.0; WINDOW],
            next: 0,
            filled: 0,
            accumulator: Duration::ZERO,
            frames_in_bucket: 0,
            frames_per_sample: 1,
            scratch: Vec::with_capacity(WINDOW),
        }
    }

    /// Averages `n` frames into one sample.
    ///
    /// Useful when the frame time is too noisy to read individually, or when
    /// only a coarse summary is wanted.
    pub fn with_frames_per_sample(mut self, n: u32) -> Self {
        self.frames_per_sample = n.max(1);
        self
    }

    /// Records one frame's duration.
    pub fn record(&mut self, frame: Duration) {
        self.accumulator += frame;
        self.frames_in_bucket += 1;
        if self.frames_in_bucket < self.frames_per_sample {
            return;
        }

        let ms = self.accumulator.as_secs_f32() * 1000.0 / self.frames_in_bucket as f32;
        self.samples[self.next] = ms;
        self.next = (self.next + 1) % WINDOW;
        self.filled = (self.filled + 1).min(WINDOW);

        self.accumulator = Duration::ZERO;
        self.frames_in_bucket = 0;
    }

    /// Discards the window, for measuring a specific moment after a reset.
    ///
    /// The partially accumulated frame is dropped too, so a reset genuinely
    /// starts from nothing rather than carrying a stale bucket across.
    pub fn reset(&mut self) {
        self.next = 0;
        self.filled = 0;
        self.accumulator = Duration::ZERO;
        self.frames_in_bucket = 0;
    }

    pub fn len(&self) -> usize {
        self.filled
    }

    pub fn is_empty(&self) -> bool {
        self.filled == 0
    }

    /// Computes the summary. Returns `None` until at least one sample exists.
    pub fn stats(&mut self) -> Option<FrameStats> {
        if self.filled == 0 {
            return None;
        }

        self.scratch.clear();
        self.scratch.extend_from_slice(&self.samples[..self.filled]);
        // `total_cmp` rather than `partial_cmp().unwrap()`: a NaN frame time
        // would otherwise panic, and NaN should sort to an end deterministically.
        self.scratch.sort_unstable_by(f32::total_cmp);

        let n = self.scratch.len();
        let at = |q: f64| -> f32 {
            // Nearest-rank: the smallest value at or above the requested
            // fraction. For a window this size the difference from
            // interpolation is far below the noise floor.
            let idx = ((q * n as f64).ceil() as usize).clamp(1, n) - 1;
            self.scratch[idx]
        };

        let sum: f32 = self.scratch.iter().sum();
        Some(FrameStats {
            samples: n,
            min_ms: self.scratch[0],
            p50_ms: at(0.50),
            p90_ms: at(0.90),
            p99_ms: at(0.99),
            p999_ms: at(0.999),
            max_ms: self.scratch[n - 1],
            mean_ms: sum / n as f32,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(v: f32) -> Duration {
        Duration::from_secs_f32(v / 1000.0)
    }

    #[test]
    fn empty_timer_reports_nothing() {
        let mut t = FrameTimer::new();
        assert!(t.is_empty());
        assert!(t.stats().is_none(), "no samples must not fabricate stats");
        assert_eq!(t.len(), 0);
    }

    #[test]
    fn a_uniform_series_has_every_percentile_equal() {
        let mut t = FrameTimer::new();
        for _ in 0..100 {
            t.record(ms(16.0));
        }
        let s = t.stats().unwrap();
        assert_eq!(s.samples, 100);
        for v in [s.p50_ms, s.p90_ms, s.p99_ms, s.p999_ms, s.min_ms, s.max_ms] {
            assert!((v - 16.0).abs() < 0.01, "expected 16ms, got {v}");
        }
        assert!((s.fps() - 62.5).abs() < 0.5, "fps from median: {}", s.fps());
    }

    #[test]
    fn percentiles_are_ordered_and_bracket_the_extremes() {
        // Frame times 1..=100 ms, so the percentiles are predictable.
        let mut t = FrameTimer::new();
        for i in 1..=100 {
            t.record(ms(i as f32));
        }
        let s = t.stats().unwrap();
        assert!(
            s.min_ms <= s.p50_ms && s.p50_ms <= s.p90_ms && s.p90_ms <= s.p99_ms,
            "percentiles out of order: {s:?}"
        );
        assert!(s.p99_ms <= s.p999_ms && s.p999_ms <= s.max_ms, "{s:?}");
        assert!((s.min_ms - 1.0).abs() < 0.01);
        assert!((s.max_ms - 100.0).abs() < 0.01);
        // Median of 1..=100 is 50 or 51 depending on the rank rule.
        assert!((49.0..=51.0).contains(&s.p50_ms), "p50 = {}", s.p50_ms);
        // Nearest-rank p90 of 1..=100 is 90.
        assert!((89.0..=91.0).contains(&s.p90_ms), "p90 = {}", s.p90_ms);
    }

    #[test]
    fn a_single_hitch_shows_up_in_the_tail_but_not_the_median() {
        // The whole reason for tracking p99 rather than a mean: one bad frame
        // in a hundred should be visible in the tail and invisible in the
        // median, because that is how it feels to play.
        let mut t = FrameTimer::new();
        for _ in 0..99 {
            t.record(ms(10.0));
        }
        t.record(ms(500.0));

        let s = t.stats().unwrap();
        assert!((s.p50_ms - 10.0).abs() < 0.01, "median moved: {}", s.p50_ms);
        assert!(
            s.p999_ms > 100.0,
            "the hitch must be visible in p99.9, got {}",
            s.p999_ms
        );
        assert!(s.max_ms >= 499.0);
        // The mean is dragged up, which is precisely why it is not the headline.
        assert!(s.mean_ms > s.p50_ms, "mean should exceed median here");
    }

    #[test]
    fn the_window_is_bounded_and_old_samples_are_evicted() {
        let mut t = FrameTimer::new();
        // Fill past the window with a slow frame time.
        for _ in 0..WINDOW {
            t.record(ms(100.0));
        }
        assert_eq!(t.len(), WINDOW);

        // Now push fast frames through; once they fill the window the slow
        // ones must be gone entirely.
        for _ in 0..WINDOW {
            t.record(ms(5.0));
        }
        let s = t.stats().unwrap();
        assert_eq!(s.samples, WINDOW);
        assert!(
            (s.max_ms - 5.0).abs() < 0.01,
            "old samples were not evicted: max = {}",
            s.max_ms
        );
    }

    #[test]
    fn reset_clears_the_window() {
        let mut t = FrameTimer::new();
        for _ in 0..50 {
            t.record(ms(20.0));
        }
        assert_eq!(t.len(), 50);
        t.reset();
        assert!(t.is_empty());
        assert!(t.stats().is_none());

        // And a partial bucket does not leak across the reset.
        t.record(ms(20.0));
        t.reset();
        t.record(ms(7.0));
        let s = t.stats().unwrap();
        assert!((s.p50_ms - 7.0).abs() < 0.05, "stale bucket leaked: {s:?}");
    }

    #[test]
    fn frames_per_sample_averages_buckets() {
        let mut t = FrameTimer::new().with_frames_per_sample(4);
        // Four frames of 4ms each average to one 4ms sample.
        for _ in 0..4 {
            t.record(ms(4.0));
        }
        assert_eq!(t.len(), 1);
        let s = t.stats().unwrap();
        assert!((s.p50_ms - 4.0).abs() < 0.01, "bucket average: {s:?}");

        // Fewer than a full bucket produces no sample yet.
        t.record(ms(9.0));
        assert_eq!(t.len(), 1, "partial bucket must not emit a sample");
    }

    #[test]
    fn frames_per_sample_is_at_least_one() {
        // A zero would divide by zero when averaging.
        let t = FrameTimer::new().with_frames_per_sample(0);
        assert_eq!(t.frames_per_sample, 1);
    }

    #[test]
    fn absurd_frame_times_do_not_panic_the_sort() {
        // `Duration` cannot hold NaN, so an unrepresentable time is rejected
        // before it reaches the timer. What can arrive is a very large
        // duration, which must sort and report rather than overflow or panic;
        // `total_cmp` in the sort is what keeps this total rather than partial.
        let mut t = FrameTimer::new();
        t.record(ms(10.0));
        t.record(Duration::from_secs(9_000_000));
        t.record(ms(20.0));
        let s = t.stats().unwrap();
        assert_eq!(s.samples, 3);
        assert!(s.max_ms >= s.p50_ms, "ordering must remain total");
        assert!(s.fps() > 0.0);
    }

    #[test]
    fn summary_mentions_every_percentile() {
        let mut t = FrameTimer::new();
        for i in 1..=10 {
            t.record(ms(i as f32 * 2.0));
        }
        let text = t.stats().unwrap().summary();
        for needle in ["fps", "p50", "p90", "p99", "p99.9"] {
            assert!(text.contains(needle), "summary missing {needle:?}: {text}");
        }
    }

    #[test]
    fn fps_is_computed_from_the_median_not_the_mean() {
        let mut t = FrameTimer::new();
        for _ in 0..99 {
            t.record(ms(10.0));
        }
        t.record(ms(1000.0));
        let s = t.stats().unwrap();
        // Median 10ms is 100 fps; the mean would be far lower.
        assert!((s.fps() - 100.0).abs() < 1.0, "fps = {}", s.fps());
        assert!(s.mean_ms > 15.0, "mean should be dragged up: {}", s.mean_ms);
    }
}
