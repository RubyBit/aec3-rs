use crate::audio_processing::aec3::aec3_common::{
    BLOCK_SIZE, METRICS_REPORTING_INTERVAL_BLOCKS, NUM_BLOCKS_PER_SECOND,
};
use crate::audio_processing::aec3::clockdrift_detector::ClockDriftLevel;

const MAX_SKEW_SHIFT_COUNT: usize = 20;

/// Handles the reporting of metrics for the render delay controller path.
#[derive(Debug, Clone)]
pub struct RenderDelayControllerMetrics {
    delay_blocks: usize,
    reliable_delay_estimate_counter: usize,
    delay_change_counter: usize,
    call_counter: usize,
    skew_report_timer: usize,
    initial_call_counter: usize,
    metrics_reported: bool,
    initial_update: bool,
    skew_shift_count: usize,
}

impl RenderDelayControllerMetrics {
    pub fn new() -> Self {
        Self {
            delay_blocks: 0,
            reliable_delay_estimate_counter: 0,
            delay_change_counter: 0,
            call_counter: 0,
            skew_report_timer: 0,
            initial_call_counter: 0,
            metrics_reported: false,
            initial_update: true,
            skew_shift_count: 0,
        }
    }

    pub fn metrics_reported(&self) -> bool {
        self.metrics_reported
    }

    pub fn update(
        &mut self,
        delay_samples: Option<usize>,
        buffer_delay_blocks: usize,
        skew_shift_blocks: Option<i32>,
        clockdrift: ClockDriftLevel,
    ) {
        let _ = (buffer_delay_blocks, clockdrift);
        self.call_counter += 1;

        if !self.initial_update {
            let delay_blocks = if let Some(samples) = delay_samples {
                self.reliable_delay_estimate_counter += 1;
                samples / BLOCK_SIZE + 2
            } else {
                0
            };

            if delay_blocks != self.delay_blocks {
                self.delay_blocks = delay_blocks;
                self.delay_change_counter += 1;
            }

            if skew_shift_blocks.is_some() {
                self.skew_shift_count = self.skew_shift_count.min(MAX_SKEW_SHIFT_COUNT);
            }
        } else {
            self.initial_call_counter += 1;
            if self.initial_call_counter == 5 * NUM_BLOCKS_PER_SECOND {
                self.initial_update = false;
            }
        }

        if self.call_counter == METRICS_REPORTING_INTERVAL_BLOCKS {
            self.metrics_reported = true;
            self.call_counter = 0;
            self.reset_metrics();
        } else {
            self.metrics_reported = false;
        }

        if !self.initial_update {
            self.skew_report_timer += 1;
            if self.skew_report_timer == 60 * NUM_BLOCKS_PER_SECOND {
                self.skew_shift_count = 0;
                self.skew_report_timer = 0;
            }
        }
    }

    fn reset_metrics(&mut self) {
        self.delay_change_counter = 0;
        self.reliable_delay_estimate_counter = 0;
    }
}

impl Default for RenderDelayControllerMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_usage_reports_at_interval() {
        let mut metrics = RenderDelayControllerMetrics::new();

        for _ in 0..3 {
            for _ in 0..METRICS_REPORTING_INTERVAL_BLOCKS - 1 {
                metrics.update(None, 0, None, ClockDriftLevel::None);
                assert!(!metrics.metrics_reported());
            }

            metrics.update(None, 0, None, ClockDriftLevel::None);
            assert!(metrics.metrics_reported());
        }
    }
}
