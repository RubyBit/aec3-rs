//! Wiener filter estimator for NS frequency-domain suppression.

use super::fast_math::sqrt_fast_approximation;
use super::ns_common::{
    FFT_SIZE_BY_2_PLUS_1, LONG_STARTUP_PHASE_BLOCKS, SHORT_STARTUP_PHASE_BLOCKS,
};
use super::suppression_params::SuppressionParams;

pub struct WienerFilter {
    suppression_params: SuppressionParams,
    spectrum_prev_process: [f32; FFT_SIZE_BY_2_PLUS_1],
    initial_spectral_estimate: [f32; FFT_SIZE_BY_2_PLUS_1],
    filter: [f32; FFT_SIZE_BY_2_PLUS_1],
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_processing::ns::ns_config::SuppressionLevel;

    #[test]
    fn update_clamps_filter_to_valid_range() {
        let params = SuppressionParams::new(SuppressionLevel::K12dB);
        let min_gain = params.minimum_attenuating_gain;
        let mut wf = WienerFilter::new(params);

        let noise = [1.0f32; FFT_SIZE_BY_2_PLUS_1];
        let prev_noise = [1.0f32; FFT_SIZE_BY_2_PLUS_1];
        let param_noise = [1.0f32; FFT_SIZE_BY_2_PLUS_1];
        let signal = [2.5f32; FFT_SIZE_BY_2_PLUS_1];

        wf.update(10, &noise, &prev_noise, &param_noise, &signal);

        for &g in wf.filter() {
            assert!(g >= min_gain);
            assert!(g <= 1.0);
        }
    }

    #[test]
    fn scaling_factor_is_unity_during_startup_or_when_disabled() {
        let p6 = SuppressionParams::new(SuppressionLevel::K6dB);
        let wf6 = WienerFilter::new(p6);
        assert_eq!(
            wf6.compute_overall_scaling_factor(400, 0.5, 100.0, 50.0),
            1.0
        );

        let p12 = SuppressionParams::new(SuppressionLevel::K12dB);
        let wf12 = WienerFilter::new(p12);
        assert_eq!(
            wf12.compute_overall_scaling_factor(100, 0.5, 100.0, 50.0),
            1.0
        );
    }
}

impl WienerFilter {
    pub fn new(suppression_params: SuppressionParams) -> Self {
        Self {
            suppression_params,
            spectrum_prev_process: [0.0; FFT_SIZE_BY_2_PLUS_1],
            initial_spectral_estimate: [0.0; FFT_SIZE_BY_2_PLUS_1],
            filter: [1.0; FFT_SIZE_BY_2_PLUS_1],
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        num_analyzed_frames: i32,
        noise_spectrum: &[f32; FFT_SIZE_BY_2_PLUS_1],
        prev_noise_spectrum: &[f32; FFT_SIZE_BY_2_PLUS_1],
        parametric_noise_spectrum: &[f32; FFT_SIZE_BY_2_PLUS_1],
        signal_spectrum: &[f32; FFT_SIZE_BY_2_PLUS_1],
    ) {
        for i in 0..FFT_SIZE_BY_2_PLUS_1 {
            let prev_tsa =
                self.spectrum_prev_process[i] / (prev_noise_spectrum[i] + 0.0001) * self.filter[i];

            let current_tsa = if signal_spectrum[i] > noise_spectrum[i] {
                signal_spectrum[i] / (noise_spectrum[i] + 0.0001) - 1.0
            } else {
                0.0
            };

            let snr_prior = 0.98 * prev_tsa + (1.0 - 0.98) * current_tsa;
            let updated = snr_prior / (self.suppression_params.over_subtraction_factor + snr_prior);
            self.filter[i] = updated.clamp(self.suppression_params.minimum_attenuating_gain, 1.0);
        }

        if num_analyzed_frames < SHORT_STARTUP_PHASE_BLOCKS {
            for i in 0..FFT_SIZE_BY_2_PLUS_1 {
                self.initial_spectral_estimate[i] += signal_spectrum[i];
                let mut filter_initial = self.initial_spectral_estimate[i]
                    - self.suppression_params.over_subtraction_factor
                        * parametric_noise_spectrum[i];
                filter_initial /= self.initial_spectral_estimate[i] + 0.0001;

                filter_initial =
                    filter_initial.clamp(self.suppression_params.minimum_attenuating_gain, 1.0);

                const ONE_BY_SHORT_STARTUP_PHASE_BLOCKS: f32 =
                    1.0 / SHORT_STARTUP_PHASE_BLOCKS as f32;

                filter_initial *= SHORT_STARTUP_PHASE_BLOCKS as f32 - num_analyzed_frames as f32;
                self.filter[i] *= num_analyzed_frames as f32;
                self.filter[i] += filter_initial;
                self.filter[i] *= ONE_BY_SHORT_STARTUP_PHASE_BLOCKS;
            }
        }

        self.spectrum_prev_process.copy_from_slice(signal_spectrum);
    }

    pub fn compute_overall_scaling_factor(
        &self,
        num_analyzed_frames: i32,
        prior_speech_probability: f32,
        energy_before_filtering: f32,
        energy_after_filtering: f32,
    ) -> f32 {
        if !self.suppression_params.use_attenuation_adjustment
            || num_analyzed_frames <= LONG_STARTUP_PHASE_BLOCKS
        {
            return 1.0;
        }

        let mut gain =
            sqrt_fast_approximation(energy_after_filtering / (energy_before_filtering + 1.0));

        const B_LIM: f32 = 0.5;

        let mut scale_factor1 = 1.0;
        if gain > B_LIM {
            scale_factor1 = 1.0 + 1.3 * (gain - B_LIM);
            if gain * scale_factor1 > 1.0 {
                scale_factor1 = 1.0 / gain;
            }
        }

        let mut scale_factor2 = 1.0;
        if gain < B_LIM {
            gain = gain.max(self.suppression_params.minimum_attenuating_gain);
            scale_factor2 = 1.0 - 0.3 * (B_LIM - gain);
        }

        prior_speech_probability * scale_factor1 + (1.0 - prior_speech_probability) * scale_factor2
    }

    pub fn filter(&self) -> &[f32; FFT_SIZE_BY_2_PLUS_1] {
        &self.filter
    }
}
