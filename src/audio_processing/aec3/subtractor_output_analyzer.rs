use crate::audio_processing::aec3::aec3_common::BLOCK_SIZE;

use super::subtractor_output::SubtractorOutput;

/// Analyzes subtractor output statistics to determine convergence/divergence.
pub struct SubtractorOutputAnalyzer {
    filters_converged: Vec<bool>,
}

impl SubtractorOutputAnalyzer {
    pub fn new(num_capture_channels: usize) -> Self {
        Self {
            filters_converged: vec![false; num_capture_channels],
        }
    }

    pub fn update(
        &mut self,
        subtractor_output: &[SubtractorOutput],
        any_filter_converged: &mut bool,
        all_filters_diverged: &mut bool,
    ) {
        assert_eq!(subtractor_output.len(), self.filters_converged.len());

        *any_filter_converged = false;
        *all_filters_diverged = true;

        for (ch, output) in subtractor_output.iter().enumerate() {
            let y2 = output.y2;
            let e2_main = output.e2_main;
            let e2_shadow = output.e2_shadow;

            const CONVERGENCE_THRESHOLD: f32 = 50.0 * 50.0 * BLOCK_SIZE as f32;
            let main_filter_converged = e2_main < 0.5 * y2 && y2 > CONVERGENCE_THRESHOLD;
            let shadow_filter_converged = e2_shadow < 0.05 * y2 && y2 > CONVERGENCE_THRESHOLD;
            let min_e2 = e2_main.min(e2_shadow);
            let diverged = min_e2 > 1.5 * y2 && y2 > 30.0 * 30.0 * BLOCK_SIZE as f32;

            self.filters_converged[ch] = main_filter_converged || shadow_filter_converged;
            *any_filter_converged |= self.filters_converged[ch];
            *all_filters_diverged &= diverged;
        }
    }

    pub fn converged_filters(&self) -> &[bool] {
        &self.filters_converged
    }

    pub fn handle_echo_path_change(&mut self) {
        self.filters_converged.fill(false);
    }
}
