use std::collections::VecDeque;

use crate::audio_processing::audio_buffer::AudioBuffer;

/// Applies a fixed sample delay to an AudioBuffer partitioned into frequency
/// bands (split-band representation). The delay is applied per-band for the
/// first channel (channel 0), matching the C++ BlockDelayBuffer behavior used
/// in tests.
pub struct BlockDelayBuffer {
    frame_length: usize,
    delay: usize,
    bufs: Vec<VecDeque<f32>>,
}

impl BlockDelayBuffer {
    pub fn new(num_bands: usize, frame_length: usize, delay_samples: usize) -> Self {
        let mut bufs = Vec::with_capacity(num_bands);
        for _ in 0..num_bands {
            // Initialize deque with `delay_samples` zeros so the first outputs are zeros
            let mut dq = VecDeque::with_capacity(delay_samples + frame_length);
            for _ in 0..delay_samples {
                dq.push_back(0.0);
            }
            bufs.push(dq);
        }
        Self {
            frame_length,
            delay: delay_samples,
            bufs,
        }
    }

    /// Delays the samples in-place for channel 0 of `frame`.
    pub fn delay_signal(&mut self, frame: &mut AudioBuffer) {
        // If no delay requested, do nothing.
        if self.delay == 0 {
            return;
        }

        assert_eq!(frame.num_frames_per_band(), self.frame_length);

        let num_bands = frame.num_bands();
        assert_eq!(num_bands, self.bufs.len());

        // Operate on channel 0 (matching reference tests which use split_bands(0)).
        for band in 0..num_bands {
            let data = frame.split_band_mut(0, band);
            // For each sample: pop oldest (delayed) sample and push current input
            for i in 0..self.frame_length {
                let input = data[i];
                let out = self.bufs[band].pop_front().unwrap_or(0.0);
                self.bufs[band].push_back(input);
                data[i] = out;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_processing::audio_buffer::AudioBuffer;

    fn sample_value(sample_index: usize) -> f32 {
        (sample_index % 32768) as f32
    }

    #[test]
    fn correct_delay_applied() {
        for &delay in &[0usize, 1, 27, 160, 4321, 7021] {
            for &rate in &[16_000usize, 32_000, 48_000] {
                let mut audio_buffer = AudioBuffer::from_sample_rates(rate, 1, rate, 1, rate);
                if rate > 16_000 {
                    audio_buffer.split_into_frequency_bands();
                }

                let num_bands = audio_buffer.num_bands();
                let subband_frame_length = audio_buffer.num_frames_per_band();

                let mut delay_buffer =
                    BlockDelayBuffer::new(num_bands, subband_frame_length, delay);

                const NUM_FRAMES_TO_PROCESS: usize = 20;
                for frame_index in 0..NUM_FRAMES_TO_PROCESS {
                    let first_sample_index = frame_index * subband_frame_length;
                    // Populate input frame per-band for channel 0
                    for band in 0..num_bands {
                        let dest = audio_buffer.split_band_mut(0, band);
                        for i in 0..subband_frame_length {
                            dest[i] = sample_value(first_sample_index + i);
                        }
                    }

                    delay_buffer.delay_signal(&mut audio_buffer);

                    for k in 0..num_bands {
                        let dest = audio_buffer.split_band(0, k);
                        let mut sample_index = first_sample_index;
                        for i in 0..subband_frame_length {
                            if sample_index < delay {
                                assert_eq!(0.0f32, dest[i]);
                            } else {
                                assert_eq!(sample_value(sample_index - delay), dest[i]);
                            }
                            sample_index += 1;
                        }
                    }
                }
            }
        }
    }
}
