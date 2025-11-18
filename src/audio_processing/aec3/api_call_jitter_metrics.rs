use crate::audio_processing::aec3::aec3_common::NUM_BLOCKS_PER_SECOND;

#[derive(Debug, Clone)]
pub struct ApiCallJitterMetrics {
    render_jitter: Jitter,
    capture_jitter: Jitter,
    num_api_calls_in_a_row: i32,
    frames_since_last_report: i32,
    last_call_was_render: bool,
    proper_call_observed: bool,
}

#[derive(Debug, Clone)]
pub struct Jitter {
    min: i32,
    max: i32,
}

impl Jitter {
    pub fn new() -> Self {
        Self {
            min: i32::MAX,
            max: 0,
        }
    }

    pub fn update(&mut self, num_api_calls_in_a_row: i32) {
        if num_api_calls_in_a_row < self.min {
            self.min = num_api_calls_in_a_row;
        }
        if num_api_calls_in_a_row > self.max {
            self.max = num_api_calls_in_a_row;
        }
    }

    pub fn reset(&mut self) {
        self.min = i32::MAX;
        self.max = 0;
    }

    pub fn min(&self) -> i32 {
        self.min
    }

    pub fn max(&self) -> i32 {
        self.max
    }
}

impl ApiCallJitterMetrics {
    pub fn new() -> Self {
        let mut s = Self {
            render_jitter: Jitter::new(),
            capture_jitter: Jitter::new(),
            num_api_calls_in_a_row: 0,
            frames_since_last_report: 0,
            last_call_was_render: false,
            proper_call_observed: false,
        };
        s.reset();
        s
    }

    fn reset(&mut self) {
        self.render_jitter.reset();
        self.capture_jitter.reset();
        self.num_api_calls_in_a_row = 0;
        self.frames_since_last_report = 0;
        self.last_call_was_render = false;
        self.proper_call_observed = false;
    }

    pub fn report_render_call(&mut self) {
        if !self.last_call_was_render {
            if self.proper_call_observed {
                self.capture_jitter.update(self.num_api_calls_in_a_row);
            }
            self.num_api_calls_in_a_row = 0;
        }
        self.num_api_calls_in_a_row += 1;
        self.last_call_was_render = true;
    }

    pub fn report_capture_call(&mut self) {
        if self.last_call_was_render {
            if self.proper_call_observed {
                self.render_jitter.update(self.num_api_calls_in_a_row);
            }
            self.num_api_calls_in_a_row = 0;
            self.proper_call_observed = true;
        }
        self.num_api_calls_in_a_row += 1;
        self.last_call_was_render = false;

        if self.proper_call_observed {
            self.frames_since_last_report += 1;
            if Self::time_to_report_metrics(self.frames_since_last_report) {
                // In the original this reports histograms; here we just reset
                self.frames_since_last_report = 0;
                self.reset();
            }
        }
    }

    fn time_to_report_metrics(frames_since_last_report: i32) -> bool {
        frames_since_last_report == 10 * NUM_BLOCKS_PER_SECOND as i32
    }

    pub fn will_report_metrics_at_next_capture(&self) -> bool {
        Self::time_to_report_metrics(self.frames_since_last_report + 1)
    }

    // Accessors for tests
    pub fn render_jitter(&self) -> &Jitter {
        &self.render_jitter
    }

    pub fn capture_jitter(&self) -> &Jitter {
        &self.capture_jitter
    }
}

impl Default for ApiCallJitterMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_processing::aec3::aec3_common::NUM_BLOCKS_PER_SECOND;

    #[test]
    fn constant_jitter() {
        for jitter in 1..20 {
            let mut metrics = ApiCallJitterMetrics::new();
            for _ in 0..30 * NUM_BLOCKS_PER_SECOND {
                for _ in 0..jitter {
                    metrics.report_render_call();
                }
                for _ in 0..jitter {
                    metrics.report_capture_call();
                    if metrics.will_report_metrics_at_next_capture() {
                        assert_eq!(jitter as i32, metrics.render_jitter().min());
                        assert_eq!(jitter as i32, metrics.render_jitter().max());
                        assert_eq!(jitter as i32, metrics.capture_jitter().min());
                        assert_eq!(jitter as i32, metrics.capture_jitter().max());
                    }
                }
            }
        }
    }

    #[test]
    fn jitter_peak_render() {
        const K_MIN_JITTER: i32 = 2;
        const K_JITTER_PEAK: i32 = 10;
        const K_PEAK_INTERVAL: i32 = 100;

        let mut metrics = ApiCallJitterMetrics::new();
        let mut render_surplus: i32 = 0;

        for k in 0..30 * NUM_BLOCKS_PER_SECOND {
            let num_render_calls = if (k as i32) % K_PEAK_INTERVAL == 0 {
                K_JITTER_PEAK
            } else {
                K_MIN_JITTER
            };
            for _ in 0..num_render_calls {
                metrics.report_render_call();
                render_surplus += 1;
            }

            assert!(render_surplus >= K_MIN_JITTER);
            let num_capture_calls = if render_surplus == K_MIN_JITTER {
                K_MIN_JITTER
            } else {
                K_MIN_JITTER + 1
            };
            for _ in 0..num_capture_calls {
                metrics.report_capture_call();
                if metrics.will_report_metrics_at_next_capture() {
                    assert_eq!(K_MIN_JITTER, metrics.render_jitter().min());
                    assert_eq!(K_JITTER_PEAK, metrics.render_jitter().max());
                    assert_eq!(K_MIN_JITTER, metrics.capture_jitter().min());
                    assert_eq!(K_MIN_JITTER + 1, metrics.capture_jitter().max());
                }
                render_surplus -= 1;
            }
        }
    }

    #[test]
    fn jitter_peak_capture() {
        const K_MIN_JITTER: i32 = 2;
        const K_JITTER_PEAK: i32 = 10;
        const K_PEAK_INTERVAL: i32 = 100;

        let mut metrics = ApiCallJitterMetrics::new();
        let mut capture_surplus: i32 = K_MIN_JITTER;

        for k in 0..30 * NUM_BLOCKS_PER_SECOND {
            assert!(capture_surplus >= K_MIN_JITTER);
            let num_render_calls = if capture_surplus == K_MIN_JITTER {
                K_MIN_JITTER
            } else {
                K_MIN_JITTER + 1
            };
            for _ in 0..num_render_calls {
                metrics.report_render_call();
                capture_surplus -= 1;
            }

            let num_capture_calls = if (k as i32) % K_PEAK_INTERVAL == 0 {
                K_JITTER_PEAK
            } else {
                K_MIN_JITTER
            };
            for _ in 0..num_capture_calls {
                metrics.report_capture_call();
                if metrics.will_report_metrics_at_next_capture() {
                    assert_eq!(K_MIN_JITTER, metrics.render_jitter().min());
                    assert_eq!(K_MIN_JITTER + 1, metrics.render_jitter().max());
                    assert_eq!(K_MIN_JITTER, metrics.capture_jitter().min());
                    assert_eq!(K_JITTER_PEAK, metrics.capture_jitter().max());
                }
                capture_surplus += 1;
            }
        }
    }
}
