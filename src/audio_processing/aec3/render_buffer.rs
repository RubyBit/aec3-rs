use super::aec3_common::FFT_LENGTH_BY_2_PLUS_1;
use super::block_buffer::BlockBuffer;
use super::fft_buffer::FftBuffer;
use super::fft_data::FftData;
use super::spectrum_buffer::SpectrumBuffer;

/// Provides a coherent view over the block, spectrum, and FFT buffers.
pub struct RenderBuffer<'a> {
    block_buffer: &'a BlockBuffer,
    spectrum_buffer: &'a SpectrumBuffer,
    fft_buffer: &'a FftBuffer,
    render_activity: bool,
}

impl<'a> RenderBuffer<'a> {
    pub fn new(
        block_buffer: &'a BlockBuffer,
        spectrum_buffer: &'a SpectrumBuffer,
        fft_buffer: &'a FftBuffer,
        render_activity: bool,
    ) -> Self {
        assert_eq!(block_buffer.size(), fft_buffer.size());
        assert_eq!(spectrum_buffer.size(), fft_buffer.size());
        assert_eq!(spectrum_buffer.read, fft_buffer.read);
        assert_eq!(spectrum_buffer.write, fft_buffer.write);
        Self {
            block_buffer,
            spectrum_buffer,
            fft_buffer,
            render_activity,
        }
    }

    pub fn block(&self, buffer_offset_blocks: isize) -> &Vec<Vec<Vec<f32>>> {
        let position = self
            .block_buffer
            .offset_index(self.block_buffer.read, buffer_offset_blocks);
        &self.block_buffer.buffer[position]
    }

    pub fn spectrum(&self, buffer_offset_ffts: isize) -> &Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]> {
        let position = self
            .spectrum_buffer
            .offset_index(self.spectrum_buffer.read, buffer_offset_ffts);
        &self.spectrum_buffer.buffer[position]
    }

    pub fn fft_buffer(&self) -> &[Vec<FftData>] {
        &self.fft_buffer.buffer
    }

    pub fn position(&self) -> usize {
        assert_eq!(self.spectrum_buffer.read, self.fft_buffer.read);
        assert_eq!(self.spectrum_buffer.write, self.fft_buffer.write);
        self.fft_buffer.read
    }

    pub fn spectral_sum(&self, num_spectra: usize, x2: &mut [f32; FFT_LENGTH_BY_2_PLUS_1]) {
        x2.fill(0.0);
        let mut position = self.spectrum_buffer.read;
        for _ in 0..num_spectra {
            for channel_spectrum in &self.spectrum_buffer.buffer[position] {
                for (dst, &value) in x2.iter_mut().zip(channel_spectrum.iter()) {
                    *dst += value;
                }
            }
            position = self.spectrum_buffer.inc_index(position);
        }
    }

    pub fn spectral_sums(
        &self,
        num_spectra_shorter: usize,
        num_spectra_longer: usize,
        x2_shorter: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
        x2_longer: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
    ) {
        assert!(num_spectra_shorter <= num_spectra_longer);
        x2_shorter.fill(0.0);
        let mut position = self.spectrum_buffer.read;
        let mut processed = 0usize;
        while processed < num_spectra_shorter {
            for channel_spectrum in &self.spectrum_buffer.buffer[position] {
                for (dst, &value) in x2_shorter.iter_mut().zip(channel_spectrum.iter()) {
                    *dst += value;
                }
            }
            position = self.spectrum_buffer.inc_index(position);
            processed += 1;
        }
        x2_longer.copy_from_slice(x2_shorter);
        while processed < num_spectra_longer {
            for channel_spectrum in &self.spectrum_buffer.buffer[position] {
                for (dst, &value) in x2_longer.iter_mut().zip(channel_spectrum.iter()) {
                    *dst += value;
                }
            }
            position = self.spectrum_buffer.inc_index(position);
            processed += 1;
        }
    }

    pub fn render_activity(&self) -> bool {
        self.render_activity
    }

    pub fn set_render_activity(&mut self, activity: bool) {
        self.render_activity = activity;
    }

    pub fn headroom(&self) -> usize {
        let write = self.fft_buffer.write;
        let read = self.fft_buffer.read;
        if write < read {
            read - write
        } else {
            self.fft_buffer.size() - write + read
        }
    }

    pub fn spectrum_buffer(&self) -> &SpectrumBuffer {
        self.spectrum_buffer
    }

    pub fn block_buffer(&self) -> &BlockBuffer {
        self.block_buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_processing::aec3::aec3_common::{BLOCK_SIZE, FFT_LENGTH_BY_2_PLUS_1};

    #[test]
    fn spectral_sum_accumulates_channels() {
        let block_buffer = BlockBuffer::new(4, 1, 2, BLOCK_SIZE);
        let mut spectrum_buffer = SpectrumBuffer::new(4, 2);
        let fft_buffer = FftBuffer::new(4, 2);

        for bin in 0..FFT_LENGTH_BY_2_PLUS_1 {
            spectrum_buffer.buffer[0][0][bin] = bin as f32;
            spectrum_buffer.buffer[0][1][bin] = 2.0 * bin as f32;
            spectrum_buffer.buffer[1][0][bin] = 3.0 * bin as f32;
            spectrum_buffer.buffer[1][1][bin] = 4.0 * bin as f32;
        }

        let render_buffer = RenderBuffer::new(&block_buffer, &spectrum_buffer, &fft_buffer, false);
        let mut sum = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        render_buffer.spectral_sum(1, &mut sum);
        for bin in 0..FFT_LENGTH_BY_2_PLUS_1 {
            assert_eq!(3.0 * bin as f32, sum[bin]);
        }

        render_buffer.spectral_sum(2, &mut sum);
        for bin in 0..FFT_LENGTH_BY_2_PLUS_1 {
            assert_eq!(10.0 * bin as f32, sum[bin]);
        }
    }

    #[test]
    fn spectral_sums_produces_consistent_results() {
        let block_buffer = BlockBuffer::new(4, 1, 1, BLOCK_SIZE);
        let mut spectrum_buffer = SpectrumBuffer::new(4, 1);
        let fft_buffer = FftBuffer::new(4, 1);

        for bin in 0..FFT_LENGTH_BY_2_PLUS_1 {
            spectrum_buffer.buffer[0][0][bin] = bin as f32;
            spectrum_buffer.buffer[1][0][bin] = 2.0 * bin as f32;
            spectrum_buffer.buffer[2][0][bin] = 3.0 * bin as f32;
        }

        let render_buffer = RenderBuffer::new(&block_buffer, &spectrum_buffer, &fft_buffer, false);
        let mut short = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let mut long = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        render_buffer.spectral_sums(2, 3, &mut short, &mut long);
        for bin in 0..FFT_LENGTH_BY_2_PLUS_1 {
            assert_eq!(3.0 * bin as f32, short[bin]);
            assert_eq!(6.0 * bin as f32, long[bin]);
        }
    }
}
