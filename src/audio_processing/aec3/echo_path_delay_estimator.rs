use crate::api::config::EchoCanceller3Config;
use crate::audio_processing::aec3::aec3_common::{
    BLOCK_SIZE, MATCHED_FILTER_ALIGNMENT_SHIFT_SIZE_SUB_BLOCKS,
    MATCHED_FILTER_WINDOW_SIZE_SUB_BLOCKS, NUM_BLOCKS_PER_SECOND, detect_optimization,
};
use crate::audio_processing::aec3::alignment_mixer::AlignmentMixer;
use crate::audio_processing::aec3::clockdrift_detector::{ClockDriftDetector, ClockDriftLevel};
use crate::audio_processing::aec3::decimator::Decimator;
use crate::audio_processing::aec3::delay_estimate::{DelayEstimate, DelayEstimateQuality};
use crate::audio_processing::aec3::downsampled_render_buffer::DownsampledRenderBuffer;
use crate::audio_processing::aec3::matched_filter::MatchedFilter;
use crate::audio_processing::aec3::matched_filter_lag_aggregator::MatchedFilterLagAggregator;
use crate::audio_processing::logging::apm_data_dumper::{ApmDataDumper, DiagnosticLevel};

/// Estimates the render/capture delay by correlating the decimated capture signal
/// with the downsampled render buffer produced by the render delay buffer.
pub struct EchoPathDelayEstimator {
    data_dumper: ApmDataDumper,
    down_sampling_factor: usize,
    sub_block_size: usize,
    capture_mixer: AlignmentMixer,
    capture_decimator: Decimator,
    matched_filter: MatchedFilter,
    matched_filter_lag_aggregator: MatchedFilterLagAggregator,
    old_aggregated_lag: Option<DelayEstimate>,
    consistent_estimate_counter: usize,
    clockdrift_detector: ClockDriftDetector,
}

impl EchoPathDelayEstimator {
    pub fn new(config: &EchoCanceller3Config, num_capture_channels: usize) -> Self {
        assert!(num_capture_channels > 0);
        let down_sampling_factor = config.delay.down_sampling_factor;
        assert!(down_sampling_factor > 0);
        assert_eq!(0, BLOCK_SIZE % down_sampling_factor);
        let sub_block_size = BLOCK_SIZE / down_sampling_factor;

        let data_dumper = ApmDataDumper::new_unique();
        let capture_mixer = AlignmentMixer::new(
            num_capture_channels,
            config.delay.capture_alignment_mixing.clone(),
        );
        let capture_decimator = Decimator::new(down_sampling_factor);
        let excitation_limit = if down_sampling_factor == 8 {
            config.render_levels.poor_excitation_render_limit_ds8
        } else {
            config.render_levels.poor_excitation_render_limit
        };
        let matched_filter = MatchedFilter::new(
            data_dumper.clone(),
            detect_optimization(),
            sub_block_size,
            MATCHED_FILTER_WINDOW_SIZE_SUB_BLOCKS,
            config.delay.num_filters,
            MATCHED_FILTER_ALIGNMENT_SHIFT_SIZE_SUB_BLOCKS,
            excitation_limit,
            config.delay.delay_estimate_smoothing,
            config.delay.delay_candidate_detection_threshold,
        );
        let matched_filter_lag_aggregator = MatchedFilterLagAggregator::new(
            data_dumper.clone(),
            matched_filter.max_filter_lag(),
            config.delay.delay_selection_thresholds.clone(),
        );

        Self {
            data_dumper,
            down_sampling_factor,
            sub_block_size,
            capture_mixer,
            capture_decimator,
            matched_filter,
            matched_filter_lag_aggregator,
            old_aggregated_lag: None,
            consistent_estimate_counter: 0,
            clockdrift_detector: ClockDriftDetector::new(),
        }
    }

    pub fn reset(&mut self, reset_delay_confidence: bool) {
        self.reset_internal(true, reset_delay_confidence);
    }

    pub fn estimate_delay(
        &mut self,
        render_buffer: &DownsampledRenderBuffer,
        capture: &[Vec<f32>],
    ) -> Option<DelayEstimate> {
        debug_assert!(!capture.is_empty());
        debug_assert_eq!(capture[0].len(), BLOCK_SIZE);

        let mut downmixed_capture = [0.0f32; BLOCK_SIZE];
        self.capture_mixer
            .produce_output(capture, &mut downmixed_capture);

        let mut downsampled_capture_storage = [0.0f32; BLOCK_SIZE];
        let downsampled_capture = &mut downsampled_capture_storage[..self.sub_block_size];
        self.capture_decimator
            .decimate(&downmixed_capture, downsampled_capture);
        self.data_dumper.dump_wav(
            DiagnosticLevel::Developer,
            "aec3_capture_decimator_output",
            downsampled_capture.len(),
            downsampled_capture,
            16_000 / self.down_sampling_factor,
            1,
        );

        self.matched_filter
            .update(render_buffer, downsampled_capture);

        let mut aggregated = self
            .matched_filter_lag_aggregator
            .aggregate(self.matched_filter.lag_estimates());

        if let Some(estimate) = aggregated {
            if matches!(estimate.quality, DelayEstimateQuality::Refined) {
                self.clockdrift_detector.update(estimate.delay as i32);
            }
        }

        let dumped_delay = aggregated
            .map(|estimate| (estimate.delay * self.down_sampling_factor) as i32)
            .unwrap_or(-1);
        self.data_dumper
            .dump_raw_i32(DiagnosticLevel::Production, "aec3_echo_path_delay_estimator_delay", dumped_delay);

        if let Some(estimate) = aggregated.as_mut() {
            estimate.delay *= self.down_sampling_factor;
        }

        let previous = self.old_aggregated_lag;
        if let (Some(prev), Some(current)) = (previous, aggregated) {
            if prev.delay == current.delay {
                self.consistent_estimate_counter += 1;
            } else {
                self.consistent_estimate_counter = 0;
            }
        } else {
            self.consistent_estimate_counter = 0;
        }

        self.old_aggregated_lag = aggregated;
        if self.consistent_estimate_counter > (NUM_BLOCKS_PER_SECOND / 2) {
            self.reset_internal(false, false);
        }

        aggregated
    }

    pub fn log_delay_estimation_properties(&self, sample_rate_hz: i32, shift: usize) {
        self.matched_filter
            .log_filter_properties(sample_rate_hz, shift, self.down_sampling_factor);
    }

    pub fn clockdrift(&self) -> ClockDriftLevel {
        self.clockdrift_detector.level()
    }

    fn reset_internal(&mut self, reset_lag_aggregator: bool, reset_delay_confidence: bool) {
        if reset_lag_aggregator {
            self.matched_filter_lag_aggregator
                .reset(reset_delay_confidence);
        }
        self.matched_filter.reset();
        self.old_aggregated_lag = None;
        self.consistent_estimate_counter = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::config::EchoCanceller3Config;
    use crate::audio_processing::aec3::aec3_common::{BLOCK_SIZE, num_bands_for_rate};
    use crate::audio_processing::aec3::render_delay_buffer::RenderDelayBuffer;
    use crate::test_support::echo_canceller_test_tools::{DelayBuffer, randomize_sample_vector};
    use crate::test_support::random::Random;

    const SAMPLE_RATE_HZ: i32 = 48_000;

    fn make_render_block(num_bands: usize, num_channels: usize) -> Vec<Vec<Vec<f32>>> {
        (0..num_bands)
            .map(|_| vec![vec![0.0f32; BLOCK_SIZE]; num_channels])
            .collect()
    }

    #[test]
    fn basic_api_calls() {
        let num_bands = num_bands_for_rate(SAMPLE_RATE_HZ);
        for &num_capture_channels in &[1usize, 2, 4] {
            for &num_render_channels in &[1usize, 2, 3, 6, 8] {
                let config = EchoCanceller3Config::default();
                let mut buffer =
                    RenderDelayBuffer::new(config.clone(), SAMPLE_RATE_HZ, num_render_channels);
                let mut estimator = EchoPathDelayEstimator::new(&config, num_capture_channels);
                let mut render_block = make_render_block(num_bands, num_render_channels);
                let capture_block = vec![vec![0.0f32; BLOCK_SIZE]; num_capture_channels];

                for _ in 0..100 {
                    buffer.insert(&render_block);
                    buffer.prepare_capture_processing();
                    estimator.estimate_delay(buffer.downsampled_render_buffer(), &capture_block);
                }

                // Prevent compiler from optimizing away mutable usage.
                render_block[0][0][0] = 0.0;
            }
        }
    }

    #[test]
    fn delay_estimation_matches_reference_behavior() {
        const NUM_RENDER_CHANNELS: usize = 1;
        const NUM_CAPTURE_CHANNELS: usize = 1;
        let num_bands = num_bands_for_rate(SAMPLE_RATE_HZ);
        let mut rng = Random::new(42);
        let down_sampling_factors = [2usize, 4, 8];
        let test_delays = [30usize, 64, 150, 200, 800, 4000];

        for &factor in &down_sampling_factors {
            let mut config = EchoCanceller3Config::default();
            config.delay.down_sampling_factor = factor;
            config.delay.num_filters = 10;

            for &delay_samples in &test_delays {
                let mut buffer =
                    RenderDelayBuffer::new(config.clone(), SAMPLE_RATE_HZ, NUM_RENDER_CHANNELS);
                let mut estimator = EchoPathDelayEstimator::new(&config, NUM_CAPTURE_CHANNELS);
                let mut render_block = make_render_block(num_bands, NUM_RENDER_CHANNELS);
                let mut capture_block = vec![vec![0.0f32; BLOCK_SIZE]; NUM_CAPTURE_CHANNELS];
                let mut delay_buffer = DelayBuffer::<f32>::new(delay_samples);
                let mut estimated_delay = None;
                let iterations = 500 + delay_samples / BLOCK_SIZE;

                for k in 0..iterations {
                    randomize_sample_vector(&mut rng, &mut render_block[0][0]);
                    delay_buffer.delay(&render_block[0][0], &mut capture_block[0]);
                    buffer.insert(&render_block);
                    if k == 0 {
                        buffer.reset();
                    }
                    buffer.prepare_capture_processing();
                    if let Some(estimate) =
                        estimator.estimate_delay(buffer.downsampled_render_buffer(), &capture_block)
                    {
                        estimated_delay = Some(estimate);
                    }
                }

                if let Some(estimate) = estimated_delay {
                    let expected_ds = delay_samples / factor;
                    let estimated_ds = estimate.delay / factor;
                    let diff = expected_ds as isize - estimated_ds as isize;
                    assert!(diff.abs() <= 1);
                } else {
                    panic!(
                        "No delay estimate produced for delay {} and factor {}",
                        delay_samples, factor
                    );
                }
            }
        }
    }

    #[test]
    fn rejects_low_level_render_signals() {
        const NUM_RENDER_CHANNELS: usize = 1;
        const NUM_CAPTURE_CHANNELS: usize = 1;
        let num_bands = num_bands_for_rate(SAMPLE_RATE_HZ);
        let mut rng = Random::new(42);
        let config = EchoCanceller3Config::default();
        let mut buffer =
            RenderDelayBuffer::new(config.clone(), SAMPLE_RATE_HZ, NUM_RENDER_CHANNELS);
        let mut estimator = EchoPathDelayEstimator::new(&config, NUM_CAPTURE_CHANNELS);
        let mut render_block = make_render_block(num_bands, NUM_RENDER_CHANNELS);
        let mut capture_block = vec![vec![0.0f32; BLOCK_SIZE]; NUM_CAPTURE_CHANNELS];

        for _ in 0..100 {
            randomize_sample_vector(&mut rng, &mut render_block[0][0]);
            for sample in &mut render_block[0][0] {
                *sample *= 100.0 / 32767.0;
            }
            capture_block[0].copy_from_slice(&render_block[0][0]);
            buffer.insert(&render_block);
            buffer.prepare_capture_processing();
            assert!(
                estimator
                    .estimate_delay(buffer.downsampled_render_buffer(), &capture_block)
                    .is_none()
            );
        }
    }
}
