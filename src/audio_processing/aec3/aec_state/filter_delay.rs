use crate::api::config::EchoCanceller3Config;
use crate::audio_processing::aec3::aec3_common::{BLOCK_SIZE, NUM_BLOCKS_PER_SECOND};
use crate::audio_processing::aec3::delay_estimate::DelayEstimate;

pub(super) struct FilterDelay {
    delay_headroom_samples: usize,
    external_delay_reported: bool,
    filter_delays_blocks: Vec<i32>,
    min_filter_delay: i32,
    external_delay: Option<DelayEstimate>,
}

impl FilterDelay {
    pub(super) fn new(config: &EchoCanceller3Config, num_capture_channels: usize) -> Self {
        Self {
            delay_headroom_samples: config.delay.delay_headroom_samples,
            external_delay_reported: false,
            filter_delays_blocks: vec![0; num_capture_channels],
            min_filter_delay: 0,
            external_delay: None,
        }
    }

    pub(super) fn external_delay_reported(&self) -> bool {
        self.external_delay_reported
    }

    pub(super) fn direct_path_filter_delays(&self) -> &[i32] {
        &self.filter_delays_blocks
    }

    pub(super) fn min_direct_path_filter_delay(&self) -> i32 {
        self.min_filter_delay
    }

    pub(super) fn update(
        &mut self,
        analyzer_filter_delay_estimates_blocks: &[i32],
        external_delay: Option<DelayEstimate>,
        blocks_with_proper_filter_adaptation: usize,
    ) {
        if let Some(delay) = external_delay {
            if self
                .external_delay
                .map(|stored| stored.delay != delay.delay)
                .unwrap_or(true)
            {
                self.external_delay = Some(delay);
                self.external_delay_reported = true;
            }
        }

        let delay_estimator_may_not_have_converged =
            blocks_with_proper_filter_adaptation < 2 * NUM_BLOCKS_PER_SECOND;
        if delay_estimator_may_not_have_converged && self.external_delay.is_some() {
            let delay_guess = (self.delay_headroom_samples / BLOCK_SIZE) as i32;
            self.filter_delays_blocks.fill(delay_guess);
        } else {
            assert_eq!(
                self.filter_delays_blocks.len(),
                analyzer_filter_delay_estimates_blocks.len()
            );
            self.filter_delays_blocks
                .copy_from_slice(analyzer_filter_delay_estimates_blocks);
        }

        self.min_filter_delay = *self.filter_delays_blocks.iter().min().unwrap_or(&0);
    }
}
