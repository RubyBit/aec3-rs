//! Noise spectrum estimator.

use super::fast_math::{exp_approximation, log_approximation, pow_approximation};
use super::ns_common::{FFT_SIZE_BY_2_PLUS_1, SHORT_STARTUP_PHASE_BLOCKS};
use super::quantile_noise_estimator::QuantileNoiseEstimator;
use super::suppression_params::SuppressionParams;

#[derive(Debug, Clone)]
pub struct NoiseEstimator {
    suppression_params: SuppressionParams,
    white_noise_level: f32,
    pink_noise_numerator: f32,
    pink_noise_exp: f32,
    prev_noise_spectrum: [f32; FFT_SIZE_BY_2_PLUS_1],
    conservative_noise_spectrum: [f32; FFT_SIZE_BY_2_PLUS_1],
    parametric_noise_spectrum: [f32; FFT_SIZE_BY_2_PLUS_1],
    noise_spectrum: [f32; FFT_SIZE_BY_2_PLUS_1],
    quantile_noise_estimator: QuantileNoiseEstimator,
}

impl NoiseEstimator {
    pub fn new(suppression_params: SuppressionParams) -> Self {
        Self {
            suppression_params,
            white_noise_level: 0.0,
            pink_noise_numerator: 0.0,
            pink_noise_exp: 0.0,
            prev_noise_spectrum: [0.0; FFT_SIZE_BY_2_PLUS_1],
            conservative_noise_spectrum: [0.0; FFT_SIZE_BY_2_PLUS_1],
            parametric_noise_spectrum: [0.0; FFT_SIZE_BY_2_PLUS_1],
            noise_spectrum: [0.0; FFT_SIZE_BY_2_PLUS_1],
            quantile_noise_estimator: QuantileNoiseEstimator::default(),
        }
    }

    pub fn prepare_analysis(&mut self) {
        self.prev_noise_spectrum
            .copy_from_slice(&self.noise_spectrum);
    }

    pub fn pre_update(
        &mut self,
        num_analyzed_frames: i32,
        signal_spectrum: &[f32; FFT_SIZE_BY_2_PLUS_1],
        signal_spectral_sum: f32,
    ) {
        self.quantile_noise_estimator
            .estimate(signal_spectrum, &mut self.noise_spectrum);

        if num_analyzed_frames < SHORT_STARTUP_PHASE_BLOCKS {
            const START_BAND: usize = 5;
            let mut sum_log_i_log_magn = 0.0;
            let mut sum_log_i = 0.0;
            let mut sum_log_i_square = 0.0;
            let mut sum_log_magn = 0.0;

            for (i, &spec) in signal_spectrum.iter().enumerate().skip(START_BAND) {
                let log_i = (i as f32).ln();
                sum_log_i += log_i;
                sum_log_i_square += log_i * log_i;
                let log_signal = log_approximation(spec);
                sum_log_magn += log_signal;
                sum_log_i_log_magn += log_i * log_signal;
            }

            const ONE_BY_FFT: f32 = 1.0 / FFT_SIZE_BY_2_PLUS_1 as f32;
            self.white_noise_level +=
                signal_spectral_sum * ONE_BY_FFT * self.suppression_params.over_subtraction_factor;

            let bins = (FFT_SIZE_BY_2_PLUS_1 - START_BAND) as f32;
            let denom = sum_log_i_square * bins - sum_log_i * sum_log_i;
            let mut pink_adj =
                (sum_log_i_square * sum_log_magn - sum_log_i * sum_log_i_log_magn) / denom;
            pink_adj = pink_adj.max(0.0);
            self.pink_noise_numerator += pink_adj;

            pink_adj = (sum_log_i * sum_log_magn - bins * sum_log_i_log_magn) / denom;
            pink_adj = pink_adj.clamp(0.0, 1.0);
            self.pink_noise_exp += pink_adj;

            let one_by_num_analyzed_frames_plus_1 = 1.0 / (num_analyzed_frames as f32 + 1.0);
            let mut parametric_exp = 0.0;
            let mut parametric_num = 0.0;
            if self.pink_noise_exp > 0.0 {
                parametric_num = exp_approximation(
                    self.pink_noise_numerator * one_by_num_analyzed_frames_plus_1,
                );
                parametric_num *= num_analyzed_frames as f32 + 1.0;
                parametric_exp = self.pink_noise_exp * one_by_num_analyzed_frames_plus_1;
            }

            for i in 0..FFT_SIZE_BY_2_PLUS_1 {
                self.parametric_noise_spectrum[i] = if self.pink_noise_exp == 0.0 {
                    self.white_noise_level
                } else {
                    let use_band = (i.max(START_BAND)) as f32;
                    let parametric_denom = pow_approximation(use_band, parametric_exp);
                    parametric_num / parametric_denom
                };
            }

            const ONE_BY_SHORT_STARTUP_PHASE_BLOCKS: f32 = 1.0 / SHORT_STARTUP_PHASE_BLOCKS as f32;
            for i in 0..FFT_SIZE_BY_2_PLUS_1 {
                self.noise_spectrum[i] *= num_analyzed_frames as f32;
                let tmp = self.parametric_noise_spectrum[i]
                    * (SHORT_STARTUP_PHASE_BLOCKS - num_analyzed_frames) as f32;
                self.noise_spectrum[i] += tmp * one_by_num_analyzed_frames_plus_1;
                self.noise_spectrum[i] *= ONE_BY_SHORT_STARTUP_PHASE_BLOCKS;
            }
        }
    }

    pub fn post_update(
        &mut self,
        speech_probability: &[f32],
        signal_spectrum: &[f32; FFT_SIZE_BY_2_PLUS_1],
    ) {
        const NOISE_UPDATE: f32 = 0.9;
        let mut gamma = NOISE_UPDATE;

        for i in 0..FFT_SIZE_BY_2_PLUS_1 {
            let prob_speech = speech_probability[i];
            let prob_non_speech = 1.0 - prob_speech;

            let noise_update_tmp = gamma * self.prev_noise_spectrum[i]
                + (1.0 - gamma)
                    * (prob_non_speech * signal_spectrum[i]
                        + prob_speech * self.prev_noise_spectrum[i]);

            let gamma_old = gamma;
            const PROB_RANGE: f32 = 0.2;
            gamma = if prob_speech > PROB_RANGE {
                0.99
            } else {
                NOISE_UPDATE
            };

            if prob_speech < PROB_RANGE {
                self.conservative_noise_spectrum[i] +=
                    0.05 * (signal_spectrum[i] - self.conservative_noise_spectrum[i]);
            }

            if gamma == gamma_old {
                self.noise_spectrum[i] = noise_update_tmp;
            } else {
                self.noise_spectrum[i] = gamma * self.prev_noise_spectrum[i]
                    + (1.0 - gamma)
                        * (prob_non_speech * signal_spectrum[i]
                            + prob_speech * self.prev_noise_spectrum[i]);
                self.noise_spectrum[i] = self.noise_spectrum[i].min(noise_update_tmp);
            }
        }
    }

    pub fn noise_spectrum(&self) -> &[f32; FFT_SIZE_BY_2_PLUS_1] {
        &self.noise_spectrum
    }

    pub fn prev_noise_spectrum(&self) -> &[f32; FFT_SIZE_BY_2_PLUS_1] {
        &self.prev_noise_spectrum
    }

    pub fn parametric_noise_spectrum(&self) -> &[f32; FFT_SIZE_BY_2_PLUS_1] {
        &self.parametric_noise_spectrum
    }

    pub fn conservative_noise_spectrum(&self) -> &[f32; FFT_SIZE_BY_2_PLUS_1] {
        &self.conservative_noise_spectrum
    }
}
