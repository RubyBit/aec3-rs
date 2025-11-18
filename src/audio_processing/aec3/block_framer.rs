use crate::audio_processing::aec3::aec3_common::{BLOCK_SIZE, SUB_FRAME_LENGTH};

/// Reconstructs 80-sample subframes from 64-sample multiband blocks.
pub struct BlockFramer {
    num_bands: usize,
    num_channels: usize,
    buffer: Vec<Vec<Vec<f32>>>,
}

impl BlockFramer {
    pub fn new(num_bands: usize, num_channels: usize) -> Self {
        assert!(num_bands > 0, "number of bands must be positive");
        assert!(num_channels > 0, "number of channels must be positive");

        let buffer = (0..num_bands)
            .map(|_| {
                (0..num_channels)
                    .map(|_| vec![0.0f32; BLOCK_SIZE])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        Self {
            num_bands,
            num_channels,
            buffer,
        }
    }

    pub fn insert_block(&mut self, block: &[Vec<Vec<f32>>]) {
        assert_eq!(self.num_bands, block.len());
        for band in 0..self.num_bands {
            assert_eq!(self.num_channels, block[band].len());
            for channel in 0..self.num_channels {
                assert_eq!(BLOCK_SIZE, block[band][channel].len());
                let buffered = &mut self.buffer[band][channel];
                assert!(buffered.is_empty());
                buffered.extend_from_slice(&block[band][channel]);
            }
        }
    }

    pub fn insert_block_and_extract_sub_frame(
        &mut self,
        block: &[Vec<Vec<f32>>],
        sub_frame: &mut [Vec<Vec<f32>>],
    ) {
        assert_eq!(self.num_bands, block.len());
        assert_eq!(self.num_bands, sub_frame.len());
        for band in 0..self.num_bands {
            assert_eq!(self.num_channels, block[band].len());
            assert_eq!(self.num_channels, sub_frame[band].len());
            for channel in 0..self.num_channels {
                let buffered = &mut self.buffer[band][channel];
                assert!(buffered.len() + BLOCK_SIZE >= SUB_FRAME_LENGTH);
                assert!(buffered.len() <= BLOCK_SIZE);
                assert_eq!(BLOCK_SIZE, block[band][channel].len());
                assert_eq!(SUB_FRAME_LENGTH, sub_frame[band][channel].len());

                let samples_to_frame = SUB_FRAME_LENGTH - buffered.len();
                assert!(samples_to_frame <= BLOCK_SIZE);

                sub_frame[band][channel][..buffered.len()].copy_from_slice(buffered);
                sub_frame[band][channel][buffered.len()..SUB_FRAME_LENGTH]
                    .copy_from_slice(&block[band][channel][..samples_to_frame]);

                buffered.clear();
                buffered.extend_from_slice(&block[band][channel][samples_to_frame..]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_processing::aec3::aec3_common::{
        BLOCK_SIZE, SUB_FRAME_LENGTH, num_bands_for_rate,
    };

    const SAMPLE_RATES: [i32; 3] = [16_000, 32_000, 48_000];

    fn compute_sample_value(
        chunk_counter: usize,
        chunk_size: usize,
        band: usize,
        channel: usize,
        sample_index: usize,
        offset: i32,
    ) -> f32 {
        let value = chunk_counter * chunk_size + sample_index + channel;
        (100 + value as i32 + offset) as f32 + 5000.0 * band as f32
    }

    fn make_tensor(num_bands: usize, num_channels: usize, length: usize) -> Vec<Vec<Vec<f32>>> {
        (0..num_bands)
            .map(|_| {
                (0..num_channels)
                    .map(|_| vec![0.0f32; length])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    }

    fn fill_block(block_counter: usize, block: &mut [Vec<Vec<f32>>]) {
        for (band_idx, band) in block.iter_mut().enumerate() {
            for (channel_idx, channel) in band.iter_mut().enumerate() {
                for (sample_idx, sample) in channel.iter_mut().enumerate() {
                    *sample = compute_sample_value(
                        block_counter,
                        BLOCK_SIZE,
                        band_idx,
                        channel_idx,
                        sample_idx,
                        0,
                    );
                }
            }
        }
    }

    fn verify_sub_frame(sub_frame_counter: usize, offset: i32, sub_frame: &[Vec<Vec<f32>>]) {
        for (band_idx, band) in sub_frame.iter().enumerate() {
            for (channel_idx, channel) in band.iter().enumerate() {
                for (sample_idx, &sample) in channel.iter().enumerate() {
                    let reference = compute_sample_value(
                        sub_frame_counter,
                        SUB_FRAME_LENGTH,
                        band_idx,
                        channel_idx,
                        sample_idx,
                        offset,
                    );
                    assert!(
                        (reference - sample).abs() < f32::EPSILON,
                        "Mismatch at band {band_idx}, channel {channel_idx}, sample {sample_idx}: expected {reference}, got {sample}"
                    );
                }
            }
        }
    }

    fn run_framer_test(sample_rate_hz: i32, num_channels: usize) {
        const NUM_SUB_FRAMES: usize = 10;
        let num_bands = num_bands_for_rate(sample_rate_hz);
        let mut block = make_tensor(num_bands, num_channels, BLOCK_SIZE);
        let mut sub_frame = make_tensor(num_bands, num_channels, SUB_FRAME_LENGTH);
        let mut framer = BlockFramer::new(num_bands, num_channels);

        let mut block_counter = 0usize;
        for sub_frame_idx in 0..NUM_SUB_FRAMES {
            fill_block(block_counter, &mut block);
            block_counter += 1;
            framer.insert_block_and_extract_sub_frame(&block, &mut sub_frame);
            if sub_frame_idx > 1 {
                verify_sub_frame(sub_frame_idx, -64, &sub_frame);
            }

            if (sub_frame_idx + 1) % 4 == 0 {
                fill_block(block_counter, &mut block);
                block_counter += 1;
                framer.insert_block(&block);
            }
        }
    }

    #[test]
    fn block_framer_produces_expected_frames() {
        for &rate in &SAMPLE_RATES {
            for &channels in &[1usize, 2, 4, 8] {
                run_framer_test(rate, channels);
            }
        }
    }

    #[test]
    #[should_panic]
    fn insert_block_and_extract_panics_on_wrong_block_length() {
        let mut framer = BlockFramer::new(1, 1);
        let block = make_tensor(1, 1, BLOCK_SIZE - 1);
        let mut sub_frame = make_tensor(1, 1, SUB_FRAME_LENGTH);
        framer.insert_block_and_extract_sub_frame(&block, &mut sub_frame);
    }

    #[test]
    #[should_panic]
    fn insert_block_and_extract_panics_on_wrong_sub_frame_length() {
        let mut framer = BlockFramer::new(1, 1);
        let block = make_tensor(1, 1, BLOCK_SIZE);
        let mut sub_frame = make_tensor(1, 1, SUB_FRAME_LENGTH - 1);
        framer.insert_block_and_extract_sub_frame(&block, &mut sub_frame);
    }

    #[test]
    #[should_panic]
    fn insert_block_panics_when_buffer_not_empty() {
        let mut framer = BlockFramer::new(1, 1);
        let block = make_tensor(1, 1, BLOCK_SIZE);
        let mut sub_frame = make_tensor(1, 1, SUB_FRAME_LENGTH);
        // Need four extract calls to drain the initial buffer.
        for _ in 0..4 {
            framer.insert_block_and_extract_sub_frame(&block, &mut sub_frame);
        }
        // At this point the buffer is empty and InsertBlock is allowed once.
        framer.insert_block(&block);
        // Calling insert_block again without another four extracts should panic.
        framer.insert_block(&block);
    }

    #[test]
    #[should_panic]
    fn zero_bands_not_allowed() {
        BlockFramer::new(0, 1);
    }

    #[test]
    #[should_panic]
    fn zero_channels_not_allowed() {
        BlockFramer::new(1, 0);
    }
}
