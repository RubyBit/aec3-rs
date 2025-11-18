use crate::api::config::EchoCanceller3Config;
use crate::audio_processing::aec3::aec3_common::{
    BLOCK_SIZE, BLOCK_SIZE_LOG2, NUM_BLOCKS_PER_SECOND, valid_full_band_rate,
};
use crate::audio_processing::aec3::clockdrift_detector::ClockDriftLevel;
use crate::audio_processing::aec3::delay_estimate::{DelayEstimate, DelayEstimateQuality};
use crate::audio_processing::aec3::downsampled_render_buffer::DownsampledRenderBuffer;
use crate::audio_processing::aec3::echo_path_delay_estimator::EchoPathDelayEstimator;
use crate::audio_processing::aec3::render_delay_controller_metrics::RenderDelayControllerMetrics;
use crate::audio_processing::logging::apm_data_dumper::ApmDataDumper;

/// Aligns the render buffer content with the capture signal by selecting a delay that
/// accounts for buffer headroom and hysteresis.
pub struct RenderDelayController {
    data_dumper: ApmDataDumper,
    hysteresis_limit_blocks: usize,
    delay_headroom_samples: usize,
    delay: Option<DelayEstimate>,
    delay_estimator: EchoPathDelayEstimator,
    metrics: RenderDelayControllerMetrics,
    delay_samples: Option<DelayEstimate>,
    capture_call_counter: usize,
    delay_change_counter: usize,
    last_delay_estimate_quality: DelayEstimateQuality,
}

impl RenderDelayController {
    pub fn new(
        config: &EchoCanceller3Config,
        sample_rate_hz: i32,
        num_capture_channels: usize,
    ) -> Self {
        assert!(valid_full_band_rate(sample_rate_hz));
        let data_dumper = ApmDataDumper::new_unique();
        let delay_estimator = EchoPathDelayEstimator::new(config, num_capture_channels);
        delay_estimator.log_delay_estimation_properties(sample_rate_hz, 0);

        Self {
            data_dumper,
            hysteresis_limit_blocks: config.delay.hysteresis_limit_blocks,
            delay_headroom_samples: config.delay.delay_headroom_samples,
            delay: None,
            delay_estimator,
            metrics: RenderDelayControllerMetrics::new(),
            delay_samples: None,
            capture_call_counter: 0,
            delay_change_counter: 0,
            last_delay_estimate_quality: DelayEstimateQuality::Coarse,
        }
    }

    pub fn reset(&mut self, reset_delay_confidence: bool) {
        self.delay = None;
        self.delay_samples = None;
        self.delay_estimator.reset(reset_delay_confidence);
        self.delay_change_counter = 0;
        if reset_delay_confidence {
            self.last_delay_estimate_quality = DelayEstimateQuality::Coarse;
        }
    }

    pub fn log_render_call(&mut self) {}

    pub fn get_delay(
        &mut self,
        render_buffer: &DownsampledRenderBuffer,
        render_delay_buffer_delay: usize,
        capture: &[Vec<f32>],
    ) -> Option<DelayEstimate> {
        let _ = render_delay_buffer_delay;
        debug_assert!(!capture.is_empty());
        debug_assert_eq!(capture[0].len(), BLOCK_SIZE);

        self.capture_call_counter += 1;

        let delay_samples = self.delay_estimator.estimate_delay(render_buffer, capture);

        if let Some(estimate) = delay_samples {
            let delay_changed = self
                .delay_samples
                .map(|stored| stored.delay != estimate.delay)
                .unwrap_or(true);
            if delay_changed {
                self.delay_change_counter = 0;
            }

            if let Some(mut stored) = self.delay_samples {
                if stored.delay == estimate.delay {
                    stored.blocks_since_last_change += 1;
                } else {
                    stored.blocks_since_last_change = 0;
                }
                stored.blocks_since_last_update = 0;
                stored.delay = estimate.delay;
                stored.quality = estimate.quality;
                self.delay_samples = Some(stored);
            } else {
                self.delay_samples = Some(estimate);
            }
        } else if let Some(mut stored) = self.delay_samples {
            stored.blocks_since_last_change += 1;
            stored.blocks_since_last_update += 1;
            self.delay_samples = Some(stored);
        }

        if self.delay_change_counter < 2 * NUM_BLOCKS_PER_SECOND {
            self.delay_change_counter += 1;
        }

        if let Some(delay_samples) = self.delay_samples {
            let use_hysteresis = self.last_delay_estimate_quality == DelayEstimateQuality::Refined
                && delay_samples.quality == DelayEstimateQuality::Refined;
            let hysteresis = if use_hysteresis {
                self.hysteresis_limit_blocks
            } else {
                0
            };
            let new_delay = compute_buffer_delay(
                self.delay.as_ref(),
                hysteresis,
                self.delay_headroom_samples,
                delay_samples,
            );
            self.delay = Some(new_delay);
            self.last_delay_estimate_quality = delay_samples.quality;
        }

        let delay_samples_value = self.delay_samples.map(|d| d.delay);
        let buffer_delay_blocks = self.delay.map(|d| d.delay).unwrap_or(0);
        self.metrics.update(
            delay_samples_value,
            buffer_delay_blocks,
            None,
            self.delay_estimator.clockdrift(),
        );

        self.data_dumper.dump_raw_usize(
            "aec3_render_delay_controller_delay",
            delay_samples_value.unwrap_or(0),
        );
        self.data_dumper.dump_raw_usize(
            "aec3_render_delay_controller_buffer_delay",
            buffer_delay_blocks,
        );

        self.delay
    }

    pub fn has_clockdrift(&self) -> bool {
        self.delay_estimator.clockdrift() != ClockDriftLevel::None
    }
}

fn compute_buffer_delay(
    current_delay: Option<&DelayEstimate>,
    hysteresis_limit_blocks: usize,
    delay_headroom_samples: usize,
    mut estimated_delay: DelayEstimate,
) -> DelayEstimate {
    let delay_with_headroom = estimated_delay.delay.saturating_sub(delay_headroom_samples);
    let mut new_delay_blocks = delay_with_headroom >> BLOCK_SIZE_LOG2;

    if let Some(current) = current_delay {
        if new_delay_blocks > current.delay
            && new_delay_blocks <= current.delay + hysteresis_limit_blocks
        {
            new_delay_blocks = current.delay;
        }
    }

    estimated_delay.delay = new_delay_blocks;
    estimated_delay
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_processing::aec3::aec3_common::num_bands_for_rate;
    use crate::audio_processing::aec3::render_delay_buffer::RenderDelayBuffer;
    use crate::test_support::echo_canceller_test_tools::{DelayBuffer, randomize_sample_vector};
    use crate::test_support::random::Random;
    use std::cmp::max;

    fn make_render_block(num_bands: usize, num_channels: usize) -> Vec<Vec<Vec<f32>>> {
        (0..num_bands)
            .map(|_| vec![vec![0.0f32; BLOCK_SIZE]; num_channels])
            .collect()
    }

    fn zero_capture(num_channels: usize) -> Vec<Vec<f32>> {
        vec![vec![0.0f32; BLOCK_SIZE]; num_channels]
    }

    #[test]
    #[ignore = "Disabled in the reference implementation"]
    fn no_render_signal_yields_zero_delay() {
        for &num_render_channels in &[1usize, 2, 8] {
            let capture_block = zero_capture(1);
            let mut config = EchoCanceller3Config::default();
            for num_filters in 4..=10 {
                for &down_sampling_factor in &[2usize, 4, 8] {
                    config.delay.down_sampling_factor = down_sampling_factor;
                    config.delay.num_filters = num_filters;
                    for &rate in &[16_000i32, 32_000, 48_000] {
                        let render_delay_buffer =
                            RenderDelayBuffer::new(config.clone(), rate, num_render_channels);
                        let mut controller = RenderDelayController::new(&config, rate, 1);
                        for _ in 0..100 {
                            let delay = controller.get_delay(
                                render_delay_buffer.downsampled_render_buffer(),
                                render_delay_buffer.delay(),
                                &capture_block,
                            );
                            assert!(delay.is_some());
                            assert_eq!(0, delay.unwrap().delay);
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "Disabled in the reference implementation"]
    fn basic_api_calls() {
        for &num_capture_channels in &[1usize, 2, 4] {
            for &num_render_channels in &[1usize, 2, 8] {
                let capture_block = zero_capture(num_capture_channels);
                for num_filters in 4..=10 {
                    for &down_sampling_factor in &[2usize, 4, 8] {
                        let mut buffer_config = EchoCanceller3Config::default();
                        buffer_config.delay.down_sampling_factor = down_sampling_factor;
                        buffer_config.delay.num_filters = num_filters;
                        buffer_config.delay.capture_alignment_mixing.downmix = false;
                        buffer_config
                            .delay
                            .capture_alignment_mixing
                            .adaptive_selection = false;
                        for &rate in &[16_000i32, 32_000, 48_000] {
                            let num_bands = num_bands_for_rate(rate);
                            let render_block = make_render_block(num_bands, num_render_channels);
                            let mut render_delay_buffer = RenderDelayBuffer::new(
                                buffer_config.clone(),
                                rate,
                                num_render_channels,
                            );
                            let mut controller = RenderDelayController::new(
                                &EchoCanceller3Config::default(),
                                rate,
                                num_capture_channels,
                            );
                            let mut delay = None;
                            for _ in 0..10 {
                                render_delay_buffer.insert(&render_block);
                                render_delay_buffer.prepare_capture_processing();
                                delay = controller.get_delay(
                                    render_delay_buffer.downsampled_render_buffer(),
                                    render_delay_buffer.delay(),
                                    &capture_block,
                                );
                            }
                            assert!(delay.is_some());
                            assert_eq!(0, delay.unwrap().delay);
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "Disabled in the reference implementation"]
    fn aligns_for_positive_delays() {
        let mut rng = Random::new(42);
        for &num_capture_channels in &[1usize, 2, 4] {
            let mut capture_block = zero_capture(num_capture_channels);
            for num_filters in 4..=10 {
                for &down_sampling_factor in &[2usize, 4, 8] {
                    let mut config = EchoCanceller3Config::default();
                    config.delay.down_sampling_factor = down_sampling_factor;
                    config.delay.num_filters = num_filters;
                    config.delay.capture_alignment_mixing.downmix = false;
                    config.delay.capture_alignment_mixing.adaptive_selection = false;
                    for &num_render_channels in &[1usize, 2, 8] {
                        for &rate in &[16_000i32, 32_000, 48_000] {
                            let num_bands = num_bands_for_rate(rate);
                            let mut render_block =
                                make_render_block(num_bands, num_render_channels);
                            for &delay_samples in &[15usize, 50, 150, 200, 800, 4000] {
                                let mut render_delay_buffer = RenderDelayBuffer::new(
                                    config.clone(),
                                    rate,
                                    num_render_channels,
                                );
                                let mut controller =
                                    RenderDelayController::new(&config, rate, num_capture_channels);
                                let mut signal_delay_buffer =
                                    DelayBuffer::<f32>::new(delay_samples);
                                let mut delay_blocks = None;
                                let iterations = 400 + delay_samples / BLOCK_SIZE;
                                for _ in 0..iterations {
                                    for band in 0..render_block.len() {
                                        for channel in 0..render_block[band].len() {
                                            randomize_sample_vector(
                                                &mut rng,
                                                &mut render_block[band][channel],
                                            );
                                        }
                                    }
                                    {
                                        let capture_ch0 = capture_block
                                            .get_mut(0)
                                            .expect("at least one capture channel");
                                        signal_delay_buffer.delay(&render_block[0][0], capture_ch0);
                                    }
                                    render_delay_buffer.insert(&render_block);
                                    render_delay_buffer.prepare_capture_processing();
                                    delay_blocks = controller.get_delay(
                                        render_delay_buffer.downsampled_render_buffer(),
                                        render_delay_buffer.delay(),
                                        &capture_block,
                                    );
                                }
                                assert!(delay_blocks.is_some());
                                let expected_blocks =
                                    max(0, delay_samples as isize / BLOCK_SIZE as isize - 1)
                                        as usize;
                                assert_eq!(expected_blocks, delay_blocks.unwrap().delay);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "Disabled in the reference implementation"]
    fn no_alignment_for_non_causal_delays() {
        let mut rng = Random::new(42);
        for &num_capture_channels in &[1usize, 2, 4] {
            for &num_render_channels in &[1usize, 2, 8] {
                for num_filters in 4..=10 {
                    for &down_sampling_factor in &[2usize, 4, 8] {
                        let mut config = EchoCanceller3Config::default();
                        config.delay.down_sampling_factor = down_sampling_factor;
                        config.delay.num_filters = num_filters;
                        config.delay.capture_alignment_mixing.downmix = false;
                        config.delay.capture_alignment_mixing.adaptive_selection = false;
                        for &rate in &[16_000i32, 32_000, 48_000] {
                            let num_bands = num_bands_for_rate(rate);
                            let mut render_block =
                                make_render_block(num_bands, num_render_channels);
                            let mut capture_block = zero_capture(num_capture_channels);
                            for &delay_samples in &[-15i32, -50, -150, -200] {
                                let mut render_delay_buffer = RenderDelayBuffer::new(
                                    config.clone(),
                                    rate,
                                    num_render_channels,
                                );
                                let mut controller =
                                    RenderDelayController::new(&config, rate, num_capture_channels);
                                let mut signal_delay_buffer =
                                    DelayBuffer::<f32>::new(delay_samples.abs() as usize);
                                let iterations =
                                    (400 - (delay_samples / BLOCK_SIZE as i32)) as usize;
                                let mut delay_blocks = None;
                                for _ in 0..iterations {
                                    {
                                        let capture_ch0 = capture_block
                                            .get_mut(0)
                                            .expect("at least one capture channel");
                                        randomize_sample_vector(&mut rng, capture_ch0);
                                    }
                                    signal_delay_buffer
                                        .delay(&capture_block[0], &mut render_block[0][0]);
                                    render_delay_buffer.insert(&render_block);
                                    render_delay_buffer.prepare_capture_processing();
                                    delay_blocks = controller.get_delay(
                                        render_delay_buffer.downsampled_render_buffer(),
                                        render_delay_buffer.delay(),
                                        &capture_block,
                                    );
                                }
                                assert!(delay_blocks.is_none());
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "Disabled in the reference implementation"]
    fn aligns_with_jitter() {
        const MAX_TEST_JITTER_BLOCKS: usize = 26;
        let mut rng = Random::new(42);
        for &num_capture_channels in &[1usize, 2, 4] {
            for &num_render_channels in &[1usize, 2, 8] {
                let mut capture_block = zero_capture(num_capture_channels);
                for num_filters in 4..=10 {
                    for &down_sampling_factor in &[2usize, 4, 8] {
                        let mut config = EchoCanceller3Config::default();
                        config.delay.down_sampling_factor = down_sampling_factor;
                        config.delay.num_filters = num_filters;
                        config.delay.capture_alignment_mixing.downmix = false;
                        config.delay.capture_alignment_mixing.adaptive_selection = false;
                        for &rate in &[16_000i32, 32_000, 48_000] {
                            let num_bands = num_bands_for_rate(rate);
                            let mut render_block =
                                make_render_block(num_bands, num_render_channels);
                            for &delay_samples in &[15usize, 50, 300, 800] {
                                let mut render_delay_buffer = RenderDelayBuffer::new(
                                    config.clone(),
                                    rate,
                                    num_render_channels,
                                );
                                let mut controller =
                                    RenderDelayController::new(&config, rate, num_capture_channels);
                                let mut signal_delay_buffer =
                                    DelayBuffer::<f32>::new(delay_samples);
                                let mut delay_blocks = None;
                                let loops = (1000 + delay_samples / BLOCK_SIZE)
                                    / MAX_TEST_JITTER_BLOCKS
                                    + 1;
                                for _ in 0..loops {
                                    let mut capture_block_buffer = Vec::new();
                                    for _ in 0..(MAX_TEST_JITTER_BLOCKS - 1) {
                                        randomize_sample_vector(&mut rng, &mut render_block[0][0]);
                                        {
                                            let capture_ch0 = capture_block
                                                .get_mut(0)
                                                .expect("at least one capture channel");
                                            signal_delay_buffer
                                                .delay(&render_block[0][0], capture_ch0);
                                        }
                                        capture_block_buffer.push(capture_block.clone());
                                        render_delay_buffer.insert(&render_block);
                                    }
                                    for block in capture_block_buffer.iter() {
                                        render_delay_buffer.prepare_capture_processing();
                                        delay_blocks = controller.get_delay(
                                            render_delay_buffer.downsampled_render_buffer(),
                                            render_delay_buffer.delay(),
                                            block,
                                        );
                                    }
                                }
                                assert!(delay_blocks.is_some());
                                let mut expected =
                                    max(0, delay_samples as isize / BLOCK_SIZE as isize - 1)
                                        as usize;
                                if expected < 2 {
                                    expected = 0;
                                }
                                assert_eq!(expected, delay_blocks.unwrap().delay);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[should_panic]
    fn rejects_wrong_capture_block_size() {
        let config = EchoCanceller3Config::default();
        let rate = 48_000;
        let render_delay_buffer = RenderDelayBuffer::new(config.clone(), rate, 1);
        let mut controller = RenderDelayController::new(&config, rate, 1);
        let capture = vec![vec![0.0f32; BLOCK_SIZE - 1]];
        controller.get_delay(
            render_delay_buffer.downsampled_render_buffer(),
            render_delay_buffer.delay(),
            &capture,
        );
    }
}
