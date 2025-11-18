use crate::api::config::EchoCanceller3Config;
use crate::audio_processing::logging::apm_data_dumper::ApmDataDumper;

use super::aec3_common::FFT_LENGTH_BY_2_PLUS_1;
use super::reverb_decay_estimator::ReverbDecayEstimator;
use super::reverb_frequency_response::ReverbFrequencyResponse;

pub struct ReverbModelEstimator {
    reverb_decay_estimators: Vec<ReverbDecayEstimator>,
    reverb_frequency_responses: Vec<ReverbFrequencyResponse>,
}

impl ReverbModelEstimator {
    pub fn new(config: &EchoCanceller3Config, num_capture_channels: usize) -> Self {
        let mut decay_estimators = Vec::with_capacity(num_capture_channels);
        let mut frequency_responses = Vec::with_capacity(num_capture_channels);
        for _ in 0..num_capture_channels {
            decay_estimators.push(ReverbDecayEstimator::new(config));
            frequency_responses.push(ReverbFrequencyResponse::new());
        }
        Self {
            reverb_decay_estimators: decay_estimators,
            reverb_frequency_responses: frequency_responses,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        impulse_responses: &[Vec<f32>],
        frequency_responses: &[Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>],
        linear_filter_qualities: &[Option<f32>],
        filter_delays_blocks: &[i32],
        usable_linear_estimates: &[bool],
        stationary_block: bool,
    ) {
        let num_capture_channels = self.reverb_decay_estimators.len();
        assert_eq!(impulse_responses.len(), num_capture_channels);
        assert_eq!(frequency_responses.len(), num_capture_channels);
        assert_eq!(linear_filter_qualities.len(), num_capture_channels);
        assert_eq!(filter_delays_blocks.len(), num_capture_channels);
        assert_eq!(usable_linear_estimates.len(), num_capture_channels);

        for ch in 0..num_capture_channels {
            self.reverb_frequency_responses[ch].update(
                &frequency_responses[ch],
                filter_delays_blocks[ch],
                linear_filter_qualities[ch],
                stationary_block,
            );
            self.reverb_decay_estimators[ch].update(
                &impulse_responses[ch],
                linear_filter_qualities[ch],
                filter_delays_blocks[ch],
                usable_linear_estimates[ch],
                stationary_block,
            );
        }
    }

    pub fn reverb_decay(&self) -> f32 {
        self.reverb_decay_estimators[0].decay()
    }

    pub fn get_reverb_frequency_response(&self) -> &[f32; FFT_LENGTH_BY_2_PLUS_1] {
        self.reverb_frequency_responses[0].frequency_response()
    }

    pub fn dump(&self, dumper: &ApmDataDumper) {
        if let Some(estimator) = self.reverb_decay_estimators.first() {
            estimator.dump(dumper);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_processing::aec3::aec3_common::{
        Aec3Optimization, FFT_LENGTH, FFT_LENGTH_BY_2, FFT_LENGTH_BY_2_PLUS_1,
    };
    use crate::audio_processing::aec3::aec3_fft::Aec3Fft;
    use crate::audio_processing::aec3::fft_data::FftData;

    const FILTER_DELAY_BLOCKS: usize = 2;
    const TRUE_POWER_DECAY: f32 = 0.5;
    const NUM_ITERATIONS: usize = 3000;

    struct ReverbModelEstimatorHarness {
        config: EchoCanceller3Config,
        estimated_decay: f32,
        estimated_power_tail: f32,
        true_power_tail: f32,
        impulse_responses: Vec<Vec<f32>>,
        frequency_responses: Vec<Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>>,
        quality_linear: Vec<Option<f32>>,
    }

    impl ReverbModelEstimatorHarness {
        fn new(default_decay: f32, num_capture_channels: usize) -> Self {
            let mut config = EchoCanceller3Config::default();
            config.ep_strength.default_len = default_decay;
            config.filter.main.length_blocks = 40;

            let num_blocks = config.filter.main.length_blocks;
            let impulse_responses =
                vec![vec![0.0; num_blocks * FFT_LENGTH_BY_2]; num_capture_channels];
            let frequency_responses =
                vec![vec![[0.0; FFT_LENGTH_BY_2_PLUS_1]; num_blocks]; num_capture_channels];
            let mut harness = Self {
                config,
                estimated_decay: default_decay.abs(),
                estimated_power_tail: 0.0,
                true_power_tail: 0.0,
                impulse_responses,
                frequency_responses,
                quality_linear: vec![Some(1.0); num_capture_channels],
            };
            harness.create_impulse_response_with_decay();
            harness
        }

        fn create_impulse_response_with_decay(&mut self) {
            let decay_sample = TRUE_POWER_DECAY.powf(0.5 / FFT_LENGTH_BY_2 as f32);
            let filter_delay_coefficients = FILTER_DELAY_BLOCKS * FFT_LENGTH_BY_2;
            for h in &mut self.impulse_responses {
                h.fill(0.0);
                h[filter_delay_coefficients] = 1.0;
                for k in (filter_delay_coefficients + 1)..h.len() {
                    h[k] = h[k - 1] * decay_sample;
                }
            }

            let fft = Aec3Fft::new();
            let mut fft_data = FftData::default();
            let mut fft_buffer = [0.0f32; FFT_LENGTH];
            for (ch, h) in self.impulse_responses.iter().enumerate() {
                for (block_idx, spectrum) in self.frequency_responses[ch].iter_mut().enumerate() {
                    let start = block_idx * FFT_LENGTH_BY_2;
                    fft_buffer.fill(0.0);
                    fft_buffer[..FFT_LENGTH_BY_2]
                        .copy_from_slice(&h[start..start + FFT_LENGTH_BY_2]);
                    fft.fft(&mut fft_buffer, &mut fft_data);
                    fft_data.spectrum(Aec3Optimization::None, spectrum);
                }
            }
            self.true_power_tail = self
                .frequency_responses
                .first()
                .and_then(|channel| channel.last())
                .map(|tail| tail.iter().sum())
                .unwrap_or(0.0);
        }

        fn run_estimator(&mut self) {
            let num_capture_channels = self.impulse_responses.len();
            let mut estimator = ReverbModelEstimator::new(&self.config, num_capture_channels);
            let filter_delays: Vec<i32> = vec![FILTER_DELAY_BLOCKS as i32; num_capture_channels];
            let usable = vec![true; num_capture_channels];
            for _ in 0..NUM_ITERATIONS {
                estimator.update(
                    &self.impulse_responses,
                    &self.frequency_responses,
                    &self.quality_linear,
                    &filter_delays,
                    &usable,
                    false,
                );
            }
            self.estimated_decay = estimator.reverb_decay();
            self.estimated_power_tail = estimator.get_reverb_frequency_response().iter().sum();
        }

        fn estimated_decay(&self) -> f32 {
            self.estimated_decay
        }

        fn true_decay(&self) -> f32 {
            TRUE_POWER_DECAY
        }

        fn estimated_tail_db(&self) -> f32 {
            power_to_db(self.estimated_power_tail)
        }

        fn true_tail_db(&self) -> f32 {
            power_to_db(self.true_power_tail)
        }
    }

    fn power_to_db(power: f32) -> f32 {
        if power <= 0.0 {
            f32::NEG_INFINITY
        } else {
            10.0 * power.log10()
        }
    }

    #[test]
    fn reverb_model_estimator_not_changing_decay() {
        const DEFAULT_DECAY: f32 = 0.9;
        for &num_capture_channels in &[1, 2, 4, 8] {
            let mut harness = ReverbModelEstimatorHarness::new(DEFAULT_DECAY, num_capture_channels);
            harness.run_estimator();
            assert!(
                (harness.estimated_decay() - DEFAULT_DECAY).abs() < f32::EPSILON,
                "expected decay to remain at {DEFAULT_DECAY}, got {}",
                harness.estimated_decay()
            );
            let tail_diff = (harness.estimated_tail_db() - harness.true_tail_db()).abs();
            assert!(
                tail_diff < 5.0,
                "tail power deviated by {tail_diff} dB for {num_capture_channels} channels"
            );
        }
    }

    #[test]
    fn reverb_model_estimator_changing_decay() {
        const DEFAULT_DECAY: f32 = -0.9;
        for &num_capture_channels in &[1, 2, 4, 8] {
            let mut harness = ReverbModelEstimatorHarness::new(DEFAULT_DECAY, num_capture_channels);
            harness.run_estimator();
            let decay_error = (harness.estimated_decay() - harness.true_decay()).abs();
            assert!(
                decay_error < 0.1,
                "decay error {decay_error} exceeded tolerance for {num_capture_channels} channels"
            );
            let tail_diff = (harness.estimated_tail_db() - harness.true_tail_db()).abs();
            assert!(
                tail_diff < 5.0,
                "tail power deviated by {tail_diff} dB for {num_capture_channels} channels"
            );
        }
    }
}
