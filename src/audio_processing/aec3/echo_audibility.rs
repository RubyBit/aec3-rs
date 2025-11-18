use super::aec3_common::FFT_LENGTH_BY_2_PLUS_1;
use super::block_buffer::BlockBuffer;
use super::render_buffer::RenderBuffer;
use super::spectrum_buffer::SpectrumBuffer;
use super::stationarity_estimator::StationarityEstimator;

/// Estimates the audibility of the echo based on render stationarity.
pub struct EchoAudibility {
    render_spectrum_write_prev: Option<usize>,
    render_block_write_prev: usize,
    non_zero_render_seen: bool,
    use_render_stationarity_at_init: bool,
    render_stationarity: StationarityEstimator,
}

impl EchoAudibility {
    pub fn new(use_render_stationarity_at_init: bool) -> Self {
        let mut instance = Self {
            render_spectrum_write_prev: None,
            render_block_write_prev: 0,
            non_zero_render_seen: false,
            use_render_stationarity_at_init,
            render_stationarity: StationarityEstimator::new(),
        };
        instance.reset();
        instance
    }

    pub fn reset(&mut self) {
        self.render_stationarity.reset();
        self.non_zero_render_seen = false;
        self.render_spectrum_write_prev = None;
    }

    pub fn update(
        &mut self,
        render_buffer: &RenderBuffer,
        average_reverb: &[f32],
        min_channel_delay_blocks: i32,
        external_delay_seen: bool,
    ) {
        self.update_render_noise_estimator(
            render_buffer.spectrum_buffer(),
            render_buffer.block_buffer(),
            external_delay_seen,
        );

        if external_delay_seen || self.use_render_stationarity_at_init {
            self.update_render_stationarity_flags(
                render_buffer,
                average_reverb,
                min_channel_delay_blocks,
            );
        }
    }

    pub fn get_residual_echo_scaling(
        &self,
        filter_has_had_time_to_converge: bool,
        residual_scaling: &mut [f32],
    ) {
        for (band, value) in residual_scaling.iter_mut().enumerate() {
            if self.render_stationarity.is_band_stationary(band)
                && (filter_has_had_time_to_converge || self.use_render_stationarity_at_init)
            {
                *value = 0.0;
            } else {
                *value = 1.0;
            }
        }
    }

    pub fn is_block_stationary(&self) -> bool {
        self.render_stationarity.is_block_stationary()
    }

    fn update_render_stationarity_flags(
        &mut self,
        render_buffer: &RenderBuffer,
        average_reverb: &[f32],
        min_channel_delay_blocks: i32,
    ) {
        assert_eq!(average_reverb.len(), FFT_LENGTH_BY_2_PLUS_1);
        let spectrum_buffer = render_buffer.spectrum_buffer();
        let idx_at_delay =
            spectrum_buffer.offset_index(spectrum_buffer.read, min_channel_delay_blocks as isize);
        let mut lookahead = render_buffer.headroom() as i32 - min_channel_delay_blocks + 1;
        if lookahead < 0 {
            lookahead = 0;
        }
        self.render_stationarity.update_stationarity_flags(
            spectrum_buffer,
            average_reverb,
            idx_at_delay,
            lookahead,
        );
    }

    fn update_render_noise_estimator(
        &mut self,
        spectrum_buffer: &SpectrumBuffer,
        block_buffer: &BlockBuffer,
        external_delay_seen: bool,
    ) {
        if self.render_spectrum_write_prev.is_none() {
            self.render_spectrum_write_prev = Some(spectrum_buffer.write);
            self.render_block_write_prev = block_buffer.write;
            return;
        }

        if !self.non_zero_render_seen && !external_delay_seen {
            self.non_zero_render_seen = !self.is_render_too_low(block_buffer);
        }

        if self.non_zero_render_seen {
            let mut idx = self.render_spectrum_write_prev.unwrap();
            let current = spectrum_buffer.write;
            while idx != current {
                self.render_stationarity
                    .update_noise_estimator(&spectrum_buffer.buffer[idx]);
                idx = spectrum_buffer.dec_index(idx);
            }
        }

        self.render_spectrum_write_prev = Some(spectrum_buffer.write);
    }

    fn is_render_too_low(&mut self, block_buffer: &BlockBuffer) -> bool {
        let num_render_channels = block_buffer.buffer[0][0].len();
        let current = block_buffer.write;
        let mut too_low = false;
        if current == self.render_block_write_prev {
            too_low = true;
        } else {
            let mut idx = self.render_block_write_prev;
            loop {
                let mut max_abs_over_channels = 0.0f32;
                for ch in 0..num_render_channels {
                    let block = &block_buffer.buffer[idx][0][ch];
                    let mut min_sample = f32::INFINITY;
                    let mut max_sample = f32::NEG_INFINITY;
                    for &sample in block {
                        if sample < min_sample {
                            min_sample = sample;
                        }
                        if sample > max_sample {
                            max_sample = sample;
                        }
                    }
                    let max_abs_channel = min_sample.abs().max(max_sample.abs());
                    if max_abs_channel > max_abs_over_channels {
                        max_abs_over_channels = max_abs_channel;
                    }
                }
                if max_abs_over_channels < 10.0 {
                    too_low = true;
                    break;
                }
                idx = block_buffer.inc_index(idx);
                if idx == current {
                    break;
                }
            }
        }

        self.render_block_write_prev = current;
        too_low
    }
}

impl Default for EchoAudibility {
    fn default() -> Self {
        Self::new(false)
    }
}
