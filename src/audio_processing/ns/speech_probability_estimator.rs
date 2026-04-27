//! Speech probability estimation for NS.

use super::fast_math::exp_approximation_sign_flip_slice;
use super::ns_common::{FFT_SIZE_BY_2_PLUS_1, LONG_STARTUP_PHASE_BLOCKS};
use super::signal_model_estimator::SignalModelEstimator;

#[derive(Debug, Clone)]
pub struct SpeechProbabilityEstimator {
    signal_model_estimator: SignalModelEstimator,
    prior_speech_prob: f32,
    speech_probability: [f32; FFT_SIZE_BY_2_PLUS_1],
}

impl Default for SpeechProbabilityEstimator {
    fn default() -> Self {
        Self {
            signal_model_estimator: SignalModelEstimator::default(),
            prior_speech_prob: 0.5,
            speech_probability: [0.0; FFT_SIZE_BY_2_PLUS_1],
        }
    }
}

impl SpeechProbabilityEstimator {
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        num_analyzed_frames: i32,
        prior_snr: &[f32; FFT_SIZE_BY_2_PLUS_1],
        post_snr: &[f32; FFT_SIZE_BY_2_PLUS_1],
        conservative_noise_spectrum: &[f32; FFT_SIZE_BY_2_PLUS_1],
        signal_spectrum: &[f32; FFT_SIZE_BY_2_PLUS_1],
        signal_spectral_sum: f32,
        signal_energy: f32,
    ) {
        if num_analyzed_frames < LONG_STARTUP_PHASE_BLOCKS {
            self.signal_model_estimator
                .adjust_normalization(num_analyzed_frames, signal_energy);
        }
        self.signal_model_estimator.update(
            prior_snr,
            post_snr,
            conservative_noise_spectrum,
            signal_spectrum,
            signal_spectral_sum,
            signal_energy,
        );

        let model = self.signal_model_estimator.model();
        let prior_model = self.signal_model_estimator.prior_model();

        const WIDTH_PRIOR_0: f32 = 4.0;
        const WIDTH_PRIOR_1: f32 = 2.0 * WIDTH_PRIOR_0;

        let mut width_prior = if model.lrt < prior_model.lrt {
            WIDTH_PRIOR_1
        } else {
            WIDTH_PRIOR_0
        };
        let indicator0 = 0.5 * ((width_prior * (model.lrt - prior_model.lrt)).tanh() + 1.0);

        width_prior = if model.spectral_flatness > prior_model.flatness_threshold {
            WIDTH_PRIOR_1
        } else {
            WIDTH_PRIOR_0
        };
        let indicator1 = 0.5
            * ((width_prior * (prior_model.flatness_threshold - model.spectral_flatness)).tanh()
                + 1.0);

        width_prior = if model.spectral_diff < prior_model.template_diff_threshold {
            WIDTH_PRIOR_1
        } else {
            WIDTH_PRIOR_0
        };
        let indicator2 = 0.5
            * ((width_prior * (model.spectral_diff - prior_model.template_diff_threshold)).tanh()
                + 1.0);

        let ind_prior = prior_model.lrt_weighting * indicator0
            + prior_model.flatness_weighting * indicator1
            + prior_model.difference_weighting * indicator2;

        self.prior_speech_prob += 0.1 * (ind_prior - self.prior_speech_prob);
        self.prior_speech_prob = self.prior_speech_prob.clamp(0.01, 1.0);

        let gain_prior = (1.0 - self.prior_speech_prob) / (self.prior_speech_prob + 0.0001);

        let mut inv_lrt = [0.0f32; FFT_SIZE_BY_2_PLUS_1];
        exp_approximation_sign_flip_slice(&model.avg_log_lrt, &mut inv_lrt);

        for (i, inv) in inv_lrt.iter().enumerate() {
            self.speech_probability[i] = 1.0 / (1.0 + gain_prior * *inv);
        }
    }

    pub fn prior_probability(&self) -> f32 {
        self.prior_speech_prob
    }

    pub fn probability(&self) -> &[f32; FFT_SIZE_BY_2_PLUS_1] {
        &self.speech_probability
    }
}
