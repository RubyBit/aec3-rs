use crate::api::config::EchoCanceller3Config;
use crate::api::control::Metrics;
use crate::audio_processing::aec3::aec3_common::{
    BLOCK_SIZE, num_bands_for_rate, valid_full_band_rate,
};
use crate::audio_processing::aec3::block_processor_metrics::BlockProcessorMetrics;
use crate::audio_processing::aec3::delay_estimate::DelayEstimate;
use crate::audio_processing::aec3::echo_path_variability::{DelayAdjustment, EchoPathVariability};
use crate::audio_processing::aec3::echo_remover::EchoRemover;
use crate::audio_processing::aec3::render_delay_buffer::{BufferingEvent, RenderDelayBuffer};
use crate::audio_processing::aec3::render_delay_controller::RenderDelayController;
use crate::audio_processing::logging::apm_data_dumper::ApmDataDumper;

/// High-level coordinator for the AEC3 pipeline that processes blocks of capture/render audio.
pub struct BlockProcessor {
    #[allow(dead_code)]
    data_dumper: ApmDataDumper,
    config: EchoCanceller3Config,
    capture_properly_started: bool,
    render_properly_started: bool,
    sample_rate_hz: i32,
    render_buffer: RenderDelayBuffer,
    delay_controller: Option<RenderDelayController>,
    echo_remover: EchoRemover,
    metrics: BlockProcessorMetrics,
    render_event: BufferingEvent,
    capture_call_counter: usize,
    estimated_delay: Option<DelayEstimate>,
}

impl BlockProcessor {
    /// Creates a block processor with internally constructed submodules.
    pub fn new(
        config: EchoCanceller3Config,
        sample_rate_hz: i32,
        num_render_channels: usize,
        num_capture_channels: usize,
    ) -> Self {
        let render_buffer =
            RenderDelayBuffer::new(config.clone(), sample_rate_hz, num_render_channels);
        let delay_controller = (!config.delay.use_external_delay_estimator)
            .then(|| RenderDelayController::new(&config, sample_rate_hz, num_capture_channels));
        let echo_remover = EchoRemover::new(
            config.clone(),
            sample_rate_hz,
            num_render_channels,
            num_capture_channels,
        );
        Self::with_components(
            config,
            sample_rate_hz,
            render_buffer,
            delay_controller,
            echo_remover,
        )
    }

    /// Creates a block processor with injected submodules (primarily for testing).
    pub fn with_components(
        config: EchoCanceller3Config,
        sample_rate_hz: i32,
        render_buffer: RenderDelayBuffer,
        delay_controller: Option<RenderDelayController>,
        echo_remover: EchoRemover,
    ) -> Self {
        assert!(valid_full_band_rate(sample_rate_hz));
        Self {
            data_dumper: ApmDataDumper::new_unique(),
            config,
            capture_properly_started: false,
            render_properly_started: false,
            sample_rate_hz,
            render_buffer,
            delay_controller,
            echo_remover,
            metrics: BlockProcessorMetrics::new(),
            render_event: BufferingEvent::None,
            capture_call_counter: 0,
            estimated_delay: None,
        }
    }

    /// Processes a capture block and, if available, produces the linear echo estimate.
    pub fn process_capture(
        &mut self,
        echo_path_gain_change: bool,
        capture_signal_saturation: bool,
        linear_output: Option<&mut Vec<Vec<Vec<f32>>>>,
        capture_block: &mut Vec<Vec<Vec<f32>>>,
    ) {
        let expected_bands = num_bands_for_rate(self.sample_rate_hz);
        assert_eq!(expected_bands, capture_block.len());
        assert!(expected_bands > 0);
        assert!(!capture_block[0].is_empty());
        assert_eq!(BLOCK_SIZE, capture_block[0][0].len());

        self.capture_call_counter += 1;

        if !self.render_properly_started {
            self.render_buffer.handle_skipped_capture_processing();
            return;
        }
        if !self.capture_properly_started {
            self.capture_properly_started = true;
            self.render_buffer.reset();
            if let Some(controller) = &mut self.delay_controller {
                controller.reset(true);
            }
        }

        let mut echo_path_variability =
            EchoPathVariability::new(echo_path_gain_change, DelayAdjustment::None, false);

        if matches!(self.render_event, BufferingEvent::RenderOverrun) {
            echo_path_variability.delay_change = DelayAdjustment::BufferFlush;
            if let Some(controller) = &mut self.delay_controller {
                controller.reset(true);
            }
        }
        self.render_event = BufferingEvent::None;

        let buffer_event = self.render_buffer.prepare_capture_processing();
        if matches!(buffer_event, BufferingEvent::RenderUnderrun) {
            if let Some(controller) = &mut self.delay_controller {
                controller.reset(false);
            }
        }

        let has_delay_estimator = !self.config.delay.use_external_delay_estimator;
        if has_delay_estimator {
            if let Some(controller) = &mut self.delay_controller {
                self.estimated_delay = controller.get_delay(
                    self.render_buffer.downsampled_render_buffer(),
                    self.render_buffer.delay(),
                    &capture_block[0],
                );
                if let Some(delay) = self.estimated_delay {
                    let delay_changed = self.render_buffer.align_from_delay(delay.delay);
                    if delay_changed {
                        echo_path_variability.delay_change = DelayAdjustment::NewDetectedDelay;
                    }
                }
                echo_path_variability.clock_drift = controller.has_clockdrift();
            }
        } else {
            self.render_buffer.align_from_external_delay();
            self.estimated_delay = None;
        }

        if has_delay_estimator || self.render_buffer.has_received_buffer_delay() {
            let render_view = self.render_buffer.render_buffer();
            let mut linear_output = linear_output;
            let linear_view = linear_output.as_mut().map(|output| &mut **output);
            self.echo_remover.process_capture(
                echo_path_variability,
                capture_signal_saturation,
                self.estimated_delay,
                &render_view,
                linear_view,
                capture_block,
            );
        }

        self.metrics.update_capture(false);
    }

    /// Buffers a render block ahead of capture processing.
    pub fn buffer_render(&mut self, render_block: &[Vec<Vec<f32>>]) {
        let expected_bands = num_bands_for_rate(self.sample_rate_hz);
        assert_eq!(expected_bands, render_block.len());
        assert!(expected_bands > 0);
        assert!(!render_block[0].is_empty());
        assert_eq!(BLOCK_SIZE, render_block[0][0].len());

        self.render_event = self.render_buffer.insert(render_block);
        self.metrics
            .update_render(!matches!(self.render_event, BufferingEvent::None));
        self.render_properly_started = true;
        if let Some(controller) = &mut self.delay_controller {
            controller.log_render_call();
        }
    }

    /// Propagates leakage detection to the echo remover.
    pub fn update_echo_leakage_status(&mut self, leakage_detected: bool) {
        self.echo_remover
            .update_echo_leakage_status(leakage_detected);
    }

    /// Applies an externally supplied buffer delay hint (in milliseconds).
    pub fn set_audio_buffer_delay(&mut self, delay_ms: i32) {
        self.render_buffer.set_audio_buffer_delay(delay_ms);
    }

    /// Returns the current echo control metrics, including delay information.
    pub fn get_metrics(&self) -> Metrics {
        let mut metrics = self.echo_remover.metrics();
        const BLOCK_SIZE_MS: i32 = 4;
        if self.capture_properly_started {
            let delay_blocks = self.render_buffer.delay() as i32;
            metrics.delay_ms = delay_blocks * BLOCK_SIZE_MS;
        } else {
            metrics.delay_ms = 0;
        }
        metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_processing::aec3::aec3_common::{NUM_BLOCKS_PER_SECOND, num_bands_for_rate};

    fn make_block(num_bands: usize, num_channels: usize, value: f32) -> Vec<Vec<Vec<f32>>> {
        (0..num_bands)
            .map(|_| {
                (0..num_channels)
                    .map(|_| vec![value; BLOCK_SIZE])
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn run_basic_setup_and_api_call_test(sample_rate_hz: i32, num_iterations: usize) {
        let config = EchoCanceller3Config::default();
        let num_bands = num_bands_for_rate(sample_rate_hz);
        let render_channels = 1usize;
        let capture_channels = 1usize;
        let mut processor =
            BlockProcessor::new(config, sample_rate_hz, render_channels, capture_channels);
        let render_block = make_block(num_bands, render_channels, 1_000.0);
        let mut capture_block = make_block(num_bands, capture_channels, 1_000.0);
        for _ in 0..num_iterations {
            processor.buffer_render(&render_block);
            processor.process_capture(false, false, None, &mut capture_block);
            processor.update_echo_leakage_status(false);
        }
    }

    #[test]
    fn basic_setup_and_api_calls() {
        for rate in [16_000, 32_000, 48_000] {
            run_basic_setup_and_api_call_test(rate, 1);
        }
    }

    #[test]
    fn test_longer_call() {
        run_basic_setup_and_api_call_test(16_000, 20 * NUM_BLOCKS_PER_SECOND);
    }

    #[test]
    #[should_panic]
    fn verify_capture_block_size_check() {
        let config = EchoCanceller3Config::default();
        let mut processor = BlockProcessor::new(config, 16_000, 1, 1);
        let mut block = vec![vec![vec![0.0f32; BLOCK_SIZE - 1]]];
        processor.process_capture(false, false, None, &mut block);
    }

    #[test]
    #[should_panic]
    fn verify_render_num_bands_check() {
        let config = EchoCanceller3Config::default();
        let mut processor = BlockProcessor::new(config, 16_000, 1, 1);
        let block = make_block(2, 1, 0.0);
        processor.buffer_render(&block);
    }

    #[test]
    #[should_panic]
    fn verify_capture_num_bands_check() {
        let config = EchoCanceller3Config::default();
        let mut processor = BlockProcessor::new(config, 16_000, 1, 1);
        let render_block = make_block(1, 1, 0.0);
        let mut capture_block = make_block(2, 1, 0.0);
        processor.buffer_render(&render_block);
        processor.process_capture(false, false, None, &mut capture_block);
    }
}
