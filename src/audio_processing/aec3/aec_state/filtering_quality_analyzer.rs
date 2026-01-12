use crate::api::config::EchoCanceller3Config;
use crate::audio_processing::aec3::aec3_common::NUM_BLOCKS_PER_SECOND;
use crate::audio_processing::aec3::delay_estimate::DelayEstimate;

pub(super) struct FilteringQualityAnalyzer {
    use_linear_filter: bool,
    overall_usable_linear_estimates: bool,
    filter_update_blocks_since_reset: usize,
    filter_update_blocks_since_start: usize,
    convergence_seen: bool,
    usable_linear_filter_estimates: Vec<bool>,
}

impl FilteringQualityAnalyzer {
    pub(super) fn new(config: &EchoCanceller3Config, num_capture_channels: usize) -> Self {
        Self {
            use_linear_filter: config.filter.use_linear_filter,
            overall_usable_linear_estimates: false,
            filter_update_blocks_since_reset: 0,
            filter_update_blocks_since_start: 0,
            convergence_seen: false,
            usable_linear_filter_estimates: vec![false; num_capture_channels],
        }
    }

    pub(super) fn reset(&mut self) {
        self.usable_linear_filter_estimates.fill(false);
        self.overall_usable_linear_estimates = false;
        self.filter_update_blocks_since_reset = 0;
    }

    pub(super) fn update(
        &mut self,
        active_render: bool,
        transparent_mode: bool,
        saturated_capture: bool,
        external_delay: Option<DelayEstimate>,
        any_filter_converged: bool,
    ) {
        let filter_update = active_render && !saturated_capture;
        if filter_update {
            self.filter_update_blocks_since_reset += 1;
            self.filter_update_blocks_since_start += 1;
        }

        if any_filter_converged {
            self.convergence_seen = true;
        }

        let sufficient_data_to_converge_at_startup =
            self.filter_update_blocks_since_start as f32 > 0.4 * NUM_BLOCKS_PER_SECOND as f32;
        let sufficient_data_to_converge_at_reset = sufficient_data_to_converge_at_startup
            && self.filter_update_blocks_since_reset as f32 > 0.2 * NUM_BLOCKS_PER_SECOND as f32;

        self.overall_usable_linear_estimates =
            sufficient_data_to_converge_at_startup && sufficient_data_to_converge_at_reset;
        self.overall_usable_linear_estimates &= external_delay.is_some() || self.convergence_seen;
        self.overall_usable_linear_estimates &= !transparent_mode;

        if self.use_linear_filter {
            self.usable_linear_filter_estimates
                .fill(self.overall_usable_linear_estimates);
        }
    }

    pub(super) fn linear_filter_usable(&self) -> bool {
        self.overall_usable_linear_estimates
    }

    pub(super) fn usable_linear_filter_outputs(&self) -> &[bool] {
        &self.usable_linear_filter_estimates
    }
}
