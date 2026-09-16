use crate::audio_processing::aec3::aec3_common::{BLOCK_SIZE, SUB_FRAME_LENGTH};
use crate::audio_processing::aec3::block::Block;

/// Converts multiband subframes consisting of 2x80 samples into 64-sample blocks.
pub struct FrameBlocker {
    num_bands: usize,
    num_channels: usize,
    buffer: Vec<Vec<Vec<f32>>>,
}

impl FrameBlocker {
    pub fn new(num_bands: usize, num_channels: usize) -> Self {
        assert!(num_bands > 0, "number of bands must be positive");
        assert!(num_channels > 0, "number of channels must be positive");

        let mut buffer = Vec::with_capacity(num_bands);
        for _ in 0..num_bands {
            let mut bands = Vec::with_capacity(num_channels);
            for _ in 0..num_channels {
                bands.push(Vec::with_capacity(BLOCK_SIZE));
            }
            buffer.push(bands);
        }

        Self {
            num_bands,
            num_channels,
            buffer,
        }
    }

    /// Inserts an 80-sample subframe and extracts the corresponding 64-sample block.
    pub fn insert_sub_frame_and_extract_block(
        &mut self,
        sub_frame: &[Vec<Vec<f32>>],
        block: &mut Block,
    ) {
        assert_eq!(self.num_bands, sub_frame.len());
        assert_eq!(self.num_bands, block.num_bands());
        assert_eq!(self.num_channels, block.num_channels());

        for band in 0..self.num_bands {
            assert_eq!(self.num_channels, sub_frame[band].len());
            for channel in 0..self.num_channels {
                let buffered = &mut self.buffer[band][channel];
                assert!(buffered.len() <= BLOCK_SIZE - (SUB_FRAME_LENGTH - BLOCK_SIZE));
                assert_eq!(SUB_FRAME_LENGTH, sub_frame[band][channel].len());

                let samples_to_block = BLOCK_SIZE - buffered.len();
                assert!(samples_to_block <= SUB_FRAME_LENGTH);

                let view = block.view_mut(band, channel);
                view[..buffered.len()].copy_from_slice(buffered);
                view[buffered.len()..BLOCK_SIZE]
                    .copy_from_slice(&sub_frame[band][channel][..samples_to_block]);

                buffered.clear();
                buffered.extend_from_slice(&sub_frame[band][channel][samples_to_block..]);
            }
        }
    }

    pub fn is_block_available(&self) -> bool {
        self.buffer[0][0].len() == BLOCK_SIZE
    }

    pub fn extract_block(&mut self, block: &mut Block) {
        assert!(self.is_block_available());
        assert_eq!(self.num_bands, block.num_bands());
        assert_eq!(self.num_channels, block.num_channels());
        for band in 0..self.num_bands {
            for channel in 0..self.num_channels {
                assert_eq!(BLOCK_SIZE, self.buffer[band][channel].len());
                block
                    .view_mut(band, channel)
                    .copy_from_slice(&self.buffer[band][channel]);
                self.buffer[band][channel].clear();
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
    use crate::audio_processing::aec3::block_framer::BlockFramer;

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
        let value = value as i32 + offset;
        if value > 0 {
            5000.0 * band as f32 + value as f32
        } else {
            0.0
        }
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

    fn fill_sub_frame(sub_frame_counter: usize, offset: i32, sub_frame: &mut [Vec<Vec<f32>>]) {
        for (band_idx, band) in sub_frame.iter_mut().enumerate() {
            for (channel_idx, channel) in band.iter_mut().enumerate() {
                for (sample_idx, sample) in channel.iter_mut().enumerate() {
                    *sample = compute_sample_value(
                        sub_frame_counter,
                        SUB_FRAME_LENGTH,
                        band_idx,
                        channel_idx,
                        sample_idx,
                        offset,
                    );
                }
            }
        }
    }

    fn verify_block(block_counter: usize, offset: i32, block: &Block) {
        for band_idx in 0..block.num_bands() {
            for channel_idx in 0..block.num_channels() {
                for (sample_idx, &sample) in block.view(band_idx, channel_idx).iter().enumerate() {
                    let reference = compute_sample_value(
                        block_counter,
                        BLOCK_SIZE,
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

    fn run_blocker_test(sample_rate_hz: i32, num_channels: usize) {
        const NUM_SUB_FRAMES: usize = 20;
        let num_bands = num_bands_for_rate(sample_rate_hz);
        let mut block = Block::new(num_bands, num_channels);
        let mut sub_frame = make_tensor(num_bands, num_channels, SUB_FRAME_LENGTH);
        let mut blocker = FrameBlocker::new(num_bands, num_channels);

        let mut block_counter = 0usize;
        for sub_frame_idx in 0..NUM_SUB_FRAMES {
            fill_sub_frame(sub_frame_idx, 0, &mut sub_frame);
            blocker.insert_sub_frame_and_extract_block(&sub_frame, &mut block);
            verify_block(block_counter, 0, &block);
            block_counter += 1;

            if (sub_frame_idx + 1) % 4 == 0 {
                assert!(blocker.is_block_available());
            } else {
                assert!(!blocker.is_block_available());
            }

            if blocker.is_block_available() {
                blocker.extract_block(&mut block);
                verify_block(block_counter, 0, &block);
                block_counter += 1;
            }
        }
    }

    fn run_blocker_and_framer_test(sample_rate_hz: i32, num_channels: usize) {
        const NUM_SUB_FRAMES: usize = 20;
        let num_bands = num_bands_for_rate(sample_rate_hz);
        let mut block = Block::new(num_bands, num_channels);
        let mut input_sub_frame = make_tensor(num_bands, num_channels, SUB_FRAME_LENGTH);
        let mut output_sub_frame = make_tensor(num_bands, num_channels, SUB_FRAME_LENGTH);
        let mut blocker = FrameBlocker::new(num_bands, num_channels);
        let mut framer = BlockFramer::new(num_bands, num_channels);

        for sub_frame_idx in 0..NUM_SUB_FRAMES {
            fill_sub_frame(sub_frame_idx, 0, &mut input_sub_frame);

            blocker.insert_sub_frame_and_extract_block(&input_sub_frame, &mut block);
            framer.insert_block_and_extract_sub_frame(&block, &mut output_sub_frame);

            if (sub_frame_idx + 1) % 4 == 0 {
                assert!(blocker.is_block_available());
            } else {
                assert!(!blocker.is_block_available());
            }

            if blocker.is_block_available() {
                blocker.extract_block(&mut block);
                framer.insert_block(&block);
            }

            if sub_frame_idx > 1 {
                verify_sub_frame(sub_frame_idx, -64, &output_sub_frame);
            }
        }
    }

    #[test]
    fn frame_blocker_produces_expected_blocks() {
        for &rate in &SAMPLE_RATES {
            for &channels in &[1usize, 2, 4, 8] {
                run_blocker_test(rate, channels);
            }
        }
    }

    #[test]
    fn frame_blocker_and_block_framer_are_inverse() {
        for &rate in &SAMPLE_RATES {
            for &channels in &[1usize, 2, 4, 8] {
                run_blocker_and_framer_test(rate, channels);
            }
        }
    }

    #[test]
    #[should_panic]
    fn frame_blocker_requires_nonzero_bands() {
        FrameBlocker::new(0, 1);
    }

    #[test]
    #[should_panic]
    fn frame_blocker_requires_nonzero_channels() {
        FrameBlocker::new(1, 0);
    }

    #[test]
    #[should_panic]
    fn extract_block_panics_if_not_available() {
        let mut blocker = FrameBlocker::new(1, 1);
        let mut block = Block::new(1, 1);
        blocker.extract_block(&mut block);
    }
}
