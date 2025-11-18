use crate::api::config::EchoCanceller3Config;
use crate::audio_processing::aec3::aec3_common::{
    Aec3Optimization, BLOCK_SIZE, FFT_LENGTH_BY_2, detect_optimization,
    get_down_sampled_buffer_size, get_render_delay_buffer_size, num_bands_for_rate,
};
use crate::audio_processing::aec3::aec3_fft::Aec3Fft;
use crate::audio_processing::aec3::alignment_mixer::AlignmentMixer;
use crate::audio_processing::aec3::block_buffer::BlockBuffer;
use crate::audio_processing::aec3::decimator::Decimator;
use crate::audio_processing::aec3::downsampled_render_buffer::DownsampledRenderBuffer;
use crate::audio_processing::aec3::fft_buffer::FftBuffer;
use crate::audio_processing::aec3::render_buffer::RenderBuffer;
use crate::audio_processing::aec3::spectrum_buffer::SpectrumBuffer;
use crate::audio_processing::logging::apm_data_dumper::ApmDataDumper;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BufferingEvent {
    None,
    RenderUnderrun,
    RenderOverrun,
    ApiCallSkew,
}

pub struct RenderDelayBuffer {
    data_dumper: ApmDataDumper,
    optimization: Aec3Optimization,
    config: EchoCanceller3Config,
    render_linear_amplitude_gain: f32,
    down_sampling_factor: usize,
    sub_block_size: usize,
    blocks: BlockBuffer,
    spectra: SpectrumBuffer,
    ffts: FftBuffer,
    delay: Option<usize>,
    low_rate: DownsampledRenderBuffer,
    render_mixer: AlignmentMixer,
    render_decimator: Decimator,
    fft: Aec3Fft,
    render_ds: Vec<f32>,
    buffer_headroom: usize,
    last_call_was_render: bool,
    num_api_calls_in_a_row: i32,
    max_observed_jitter: i32,
    capture_call_counter: i64,
    render_call_counter: i64,
    render_activity_pending: bool,
    render_activity_latched: bool,
    render_activity_counter: usize,
    external_audio_buffer_delay: Option<i32>,
    external_audio_buffer_delay_verified_after_reset: bool,
    min_latency_blocks: usize,
    excess_render_detection_counter: usize,
}

impl RenderDelayBuffer {
    pub fn new(
        mut config: EchoCanceller3Config,
        sample_rate_hz: i32,
        num_render_channels: usize,
    ) -> Self {
        let requested_down_sampling_factor = config.delay.down_sampling_factor.max(1);
        config.validate();
        config.delay.down_sampling_factor = requested_down_sampling_factor;
        let down_sampling_factor = requested_down_sampling_factor;
        let sub_block_size = BLOCK_SIZE / down_sampling_factor;
        let buffer_size = get_render_delay_buffer_size(
            down_sampling_factor,
            config.delay.num_filters,
            config.filter.main.length_blocks,
        );
        let num_bands = num_bands_for_rate(sample_rate_hz);
        assert!(num_bands > 0, "Unsupported sample rate {sample_rate_hz}");

        let blocks = BlockBuffer::new(buffer_size, num_bands, num_render_channels, BLOCK_SIZE);
        let spectra = SpectrumBuffer::new(buffer_size, num_render_channels);
        let ffts = FftBuffer::new(buffer_size, num_render_channels);
        let low_rate = DownsampledRenderBuffer::new(get_down_sampled_buffer_size(
            down_sampling_factor,
            config.delay.num_filters,
        ));

        let mut instance = Self {
            data_dumper: ApmDataDumper::new_unique(),
            optimization: detect_optimization(),
            render_linear_amplitude_gain: 10f32
                .powf(config.render_levels.render_power_gain_db / 20.0),
            down_sampling_factor,
            sub_block_size,
            blocks,
            spectra,
            ffts,
            delay: Some(config.delay.default_delay),
            low_rate,
            render_mixer: AlignmentMixer::new(
                num_render_channels,
                config.delay.render_alignment_mixing.clone(),
            ),
            render_decimator: Decimator::new(down_sampling_factor),
            fft: Aec3Fft::new(),
            render_ds: vec![0.0; sub_block_size],
            buffer_headroom: config.filter.main.length_blocks,
            last_call_was_render: false,
            num_api_calls_in_a_row: 1,
            max_observed_jitter: 1,
            capture_call_counter: 0,
            render_call_counter: 0,
            render_activity_pending: false,
            render_activity_latched: false,
            render_activity_counter: 0,
            external_audio_buffer_delay: None,
            external_audio_buffer_delay_verified_after_reset: false,
            min_latency_blocks: 0,
            excess_render_detection_counter: 0,
            config,
        };

        instance.reset();
        instance
    }

    pub fn reset(&mut self) {
        self.last_call_was_render = false;
        self.num_api_calls_in_a_row = 1;
        self.min_latency_blocks = 0;
        self.excess_render_detection_counter = 0;

        self.low_rate.read = self
            .low_rate
            .offset_index(self.low_rate.write, self.sub_block_size as isize);

        if let Some(external_delay) = self.external_audio_buffer_delay {
            let headroom = 2;
            let mut audio_buffer_delay = if external_delay <= headroom {
                1
            } else {
                (external_delay - headroom) as usize
            };
            audio_buffer_delay = audio_buffer_delay.min(self.max_delay());
            self.apply_total_delay(audio_buffer_delay as isize);
            self.delay = Some(self.compute_delay().max(0) as usize);
            self.external_audio_buffer_delay_verified_after_reset = false;
        } else {
            self.apply_total_delay(self.config.delay.default_delay as isize);
            self.delay = None;
        }
    }

    pub fn insert(&mut self, block: &[Vec<Vec<f32>>]) -> BufferingEvent {
        self.render_call_counter += 1;
        if self.delay.is_some() {
            if !self.last_call_was_render {
                self.last_call_was_render = true;
                self.num_api_calls_in_a_row = 1;
            } else {
                self.num_api_calls_in_a_row += 1;
                self.max_observed_jitter =
                    self.max_observed_jitter.max(self.num_api_calls_in_a_row);
            }
        }

        let previous_write = self.blocks.write;
        self.increment_write_indices();

        let mut event = BufferingEvent::None;
        if self.render_overrun() {
            event = BufferingEvent::RenderOverrun;
        }

        if !self.render_activity_pending {
            if self.detect_active_render(&block[0][0]) {
                self.render_activity_counter += 1;
                if self.render_activity_counter >= 20 {
                    self.render_activity_pending = true;
                }
            }
        }

        self.insert_block(block, previous_write);

        if matches!(event, BufferingEvent::RenderOverrun) {
            self.reset();
        }

        event
    }

    pub fn prepare_capture_processing(&mut self) -> BufferingEvent {
        let mut event = BufferingEvent::None;
        self.capture_call_counter += 1;

        if self.delay.is_some() {
            if self.last_call_was_render {
                self.last_call_was_render = false;
                self.num_api_calls_in_a_row = 1;
            } else {
                self.num_api_calls_in_a_row += 1;
                self.max_observed_jitter =
                    self.max_observed_jitter.max(self.num_api_calls_in_a_row);
            }
        }

        if self.detect_excess_render_blocks() {
            self.reset();
            event = BufferingEvent::RenderOverrun;
        } else if self.render_underrun() {
            self.increment_read_indices();
            if let Some(delay) = self.delay {
                if delay > 0 {
                    self.delay = Some(delay - 1);
                }
            }
            event = BufferingEvent::RenderUnderrun;
        } else {
            self.increment_low_rate_read_indices();
            self.increment_read_indices();
        }

        self.render_activity_latched = self.render_activity_pending;
        if self.render_activity_pending {
            self.render_activity_counter = 0;
            self.render_activity_pending = false;
        }

        event
    }

    pub fn align_from_delay(&mut self, delay: usize) -> bool {
        assert!(!self.config.delay.use_external_delay_estimator);
        if !self.external_audio_buffer_delay_verified_after_reset
            && self.external_audio_buffer_delay.is_some()
            && self.delay.is_some()
        {
            self.external_audio_buffer_delay_verified_after_reset = true;
        }
        if self.delay == Some(delay) {
            return false;
        }
        self.delay = Some(delay);
        let mut total_delay = self.map_delay_to_total_delay(delay);
        total_delay = total_delay.clamp(0, self.max_delay() as i32);
        self.apply_total_delay(total_delay as isize);
        true
    }

    pub fn align_from_external_delay(&mut self) {
        assert!(self.config.delay.use_external_delay_estimator);
        if let Some(external) = self.external_audio_buffer_delay {
            let delay = self.render_call_counter - self.capture_call_counter + external as i64;
            self.apply_total_delay(delay as isize);
        }
    }

    pub fn delay(&self) -> usize {
        self.compute_delay().max(0) as usize
    }

    pub fn max_delay(&self) -> usize {
        self.blocks.size() - 1 - self.buffer_headroom
    }

    pub fn render_buffer(&self) -> RenderBuffer<'_> {
        RenderBuffer::new(
            &self.blocks,
            &self.spectra,
            &self.ffts,
            self.render_activity_latched,
        )
    }

    pub fn downsampled_render_buffer(&self) -> &DownsampledRenderBuffer {
        &self.low_rate
    }

    pub fn set_audio_buffer_delay(&mut self, delay_ms: i32) {
        if self.external_audio_buffer_delay.is_none() {
            self.external_audio_buffer_delay_verified_after_reset = false;
        }
        self.external_audio_buffer_delay = Some(delay_ms / 4);
    }

    pub fn has_received_buffer_delay(&self) -> bool {
        self.external_audio_buffer_delay.is_some()
    }

    fn map_delay_to_total_delay(&self, external_delay_blocks: usize) -> i32 {
        self.buffer_latency() + external_delay_blocks as i32
    }

    fn compute_delay(&self) -> i32 {
        let latency_blocks = self.buffer_latency();
        let size = self.spectra.size();
        let internal_delay = if self.spectra.read >= self.spectra.write {
            self.spectra.read - self.spectra.write
        } else {
            size + self.spectra.read - self.spectra.write
        };
        internal_delay as i32 - latency_blocks
    }

    fn apply_total_delay(&mut self, delay: isize) {
        self.blocks.read = self.blocks.offset_index(self.blocks.write, -delay);
        self.spectra.read = self.spectra.offset_index(self.spectra.write, delay);
        self.ffts.read = self.ffts.offset_index(self.ffts.write, delay);
    }

    fn insert_block(&mut self, block: &[Vec<Vec<f32>>], previous_write: usize) {
        let num_bands = self.blocks.buffer[self.blocks.write].len();
        assert_eq!(block.len(), num_bands);
        for band in 0..num_bands {
            let num_channels = self.blocks.buffer[self.blocks.write][band].len();
            assert_eq!(block[band].len(), num_channels);
            for ch in 0..num_channels {
                assert_eq!(block[band][ch].len(), BLOCK_SIZE);
                self.blocks.buffer[self.blocks.write][band][ch].copy_from_slice(&block[band][ch]);
                if (self.render_linear_amplitude_gain - 1.0).abs() > f32::EPSILON {
                    for sample in &mut self.blocks.buffer[self.blocks.write][band][ch] {
                        *sample *= self.render_linear_amplitude_gain;
                    }
                }
            }
        }

        let mut downmixed = [0.0f32; BLOCK_SIZE];
        self.render_mixer
            .produce_output(&self.blocks.buffer[self.blocks.write][0], &mut downmixed);
        self.render_decimator
            .decimate(&downmixed, &mut self.render_ds);
        self.data_dumper.dump_wav(
            "aec3_render_decimator_output",
            self.render_ds.len(),
            &self.render_ds,
            16_000 / self.down_sampling_factor,
            1,
        );
        assert!(self.low_rate.write + self.render_ds.len() <= self.low_rate.buffer.len());
        for (i, &sample) in self.render_ds.iter().rev().enumerate() {
            self.low_rate.buffer[self.low_rate.write + i] = sample;
        }

        let num_channels = self.blocks.buffer[self.blocks.write][0].len();
        for ch in 0..num_channels {
            let current = &self.blocks.buffer[self.blocks.write][0][ch];
            let previous = &self.blocks.buffer[previous_write][0][ch];
            self.fft.padded_fft(
                current,
                previous,
                &mut self.ffts.buffer[self.ffts.write][ch],
            );
            self.ffts.buffer[self.ffts.write][ch].spectrum(
                self.optimization,
                &mut self.spectra.buffer[self.spectra.write][ch],
            );
        }
    }

    fn detect_active_render(&self, x: &[f32]) -> bool {
        let energy: f32 = x.iter().map(|v| v * v).sum();
        let limit = self.config.render_levels.active_render_limit;
        energy > (limit * limit) * FFT_LENGTH_BY_2 as f32
    }

    fn detect_excess_render_blocks(&mut self) -> bool {
        let latency_blocks = self.buffer_latency() as usize;
        self.min_latency_blocks = self.min_latency_blocks.min(latency_blocks);
        self.excess_render_detection_counter += 1;
        let mut excess_detected = false;
        if self.excess_render_detection_counter
            >= self
                .config
                .buffering
                .excess_render_detection_interval_blocks
        {
            excess_detected =
                self.min_latency_blocks > self.config.buffering.max_allowed_excess_render_blocks;
            self.min_latency_blocks = latency_blocks;
            self.excess_render_detection_counter = 0;
        }
        self.data_dumper
            .dump_raw_f32("aec3_latency_blocks", latency_blocks as f32);
        self.data_dumper
            .dump_raw_f32("aec3_min_latency_blocks", self.min_latency_blocks as f32);
        self.data_dumper.dump_raw_f32(
            "aec3_excess_render_detected",
            if excess_detected { 1.0 } else { 0.0 },
        );
        excess_detected
    }

    fn buffer_latency(&self) -> i32 {
        let size = self.low_rate.buffer.len() as isize;
        let mut latency_samples =
            (size + self.low_rate.read as isize - self.low_rate.write as isize) % size;
        if latency_samples < 0 {
            latency_samples += size;
        }
        (latency_samples as usize / self.sub_block_size) as i32
    }

    fn increment_write_indices(&mut self) {
        self.low_rate
            .update_write_index(-(self.sub_block_size as isize));
        self.blocks.inc_write_index();
        self.spectra.dec_write_index();
        self.ffts.dec_write_index();
    }

    fn increment_low_rate_read_indices(&mut self) {
        self.low_rate
            .update_read_index(-(self.sub_block_size as isize));
    }

    fn increment_read_indices(&mut self) {
        if self.blocks.read != self.blocks.write {
            self.blocks.inc_read_index();
            self.spectra.dec_read_index();
            self.ffts.dec_read_index();
        }
    }

    fn render_overrun(&self) -> bool {
        self.low_rate.read == self.low_rate.write || self.blocks.read == self.blocks.write
    }

    fn render_underrun(&self) -> bool {
        self.low_rate.read == self.low_rate.write
    }
}

#[cfg(test)]
mod tests {
    use super::super::aec3_common::BLOCK_SIZE;
    use super::num_bands_for_rate;
    use super::*;

    fn make_block(num_bands: usize, num_channels: usize) -> Vec<Vec<Vec<f32>>> {
        (0..num_bands)
            .map(|_| {
                (0..num_channels)
                    .map(|_| vec![0.0f32; BLOCK_SIZE])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    }

    #[test]
    fn buffer_overflow_detection() {
        let config = EchoCanceller3Config::default();
        for num_channels in [1usize, 2, 4] {
            for rate in [16_000i32, 32_000, 48_000] {
                let num_bands = num_bands_for_rate(rate);
                let mut buffer = RenderDelayBuffer::new(config.clone(), rate, num_channels);
                let block = make_block(num_bands, num_channels);
                let mut overrun = false;
                for _ in 0..10 {
                    buffer.insert(&block);
                }
                for _ in 0..1000 {
                    if matches!(buffer.insert(&block), BufferingEvent::RenderOverrun) {
                        overrun = true;
                        break;
                    }
                }
                assert!(
                    overrun,
                    "Expected overrun for rate {rate} channels {num_channels}"
                );
            }
        }
    }

    #[test]
    fn available_block_after_insert() {
        let config = EchoCanceller3Config::default();
        let mut buffer = RenderDelayBuffer::new(config, 48_000, 1);
        let block = make_block(3, 1);
        buffer.insert(&block);
        let event = buffer.prepare_capture_processing();
        assert!(!matches!(event, BufferingEvent::RenderUnderrun));
    }

    #[test]
    fn align_from_delay_updates_delay() {
        let config = EchoCanceller3Config::default();
        let mut buffer = RenderDelayBuffer::new(config, 16_000, 1);
        buffer.reset();
        for delay in 0..20 {
            assert!(buffer.align_from_delay(delay));
            assert_eq!(buffer.delay(), delay);
        }
    }

    #[test]
    #[should_panic]
    fn insert_rejects_wrong_band_count() {
        let config = EchoCanceller3Config::default();
        let mut buffer = RenderDelayBuffer::new(config, 48_000, 1);
        let block = make_block(1, 1);
        buffer.insert(&block);
    }
}
