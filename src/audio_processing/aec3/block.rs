//! Contiguous multiband, multichannel block of audio.
//!
//! Ported from `reference_aec_cpp/modules/audio_processing/aec3/block.h`.

use crate::audio_processing::aec3::aec3_common::BLOCK_SIZE;

/// One or more channels of 4 milliseconds of audio, split into one or more
/// frequency bands each sampled at 16 kHz.
///
/// The samples live in a single allocation laid out as `[band][channel][sample]`,
/// so a block is one allocation rather than one per channel per band.
#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    num_bands: usize,
    num_channels: usize,
    data: Vec<f32>,
}

impl Block {
    pub fn new(num_bands: usize, num_channels: usize) -> Self {
        Self::with_value(num_bands, num_channels, 0.0)
    }

    pub fn with_value(num_bands: usize, num_channels: usize, value: f32) -> Self {
        Self {
            num_bands,
            num_channels,
            data: vec![value; num_bands * num_channels * BLOCK_SIZE],
        }
    }

    pub fn num_bands(&self) -> usize {
        self.num_bands
    }

    pub fn num_channels(&self) -> usize {
        self.num_channels
    }

    /// Changes the channel count and zeroes every sample.
    pub fn set_num_channels(&mut self, num_channels: usize) {
        self.num_channels = num_channels;
        self.data
            .resize(self.num_bands * num_channels * BLOCK_SIZE, 0.0);
        self.data.fill(0.0);
    }

    pub fn view(&self, band: usize, channel: usize) -> &[f32; BLOCK_SIZE] {
        let start = self.index(band, channel);
        self.data[start..start + BLOCK_SIZE]
            .try_into()
            .expect("block slice has BLOCK_SIZE samples")
    }

    pub fn view_mut(&mut self, band: usize, channel: usize) -> &mut [f32; BLOCK_SIZE] {
        let start = self.index(band, channel);
        (&mut self.data[start..start + BLOCK_SIZE])
            .try_into()
            .expect("block slice has BLOCK_SIZE samples")
    }

    /// All channels of one band, as a contiguous slice of `num_channels`
    /// consecutive `BLOCK_SIZE` runs.
    pub fn band(&self, band: usize) -> &[f32] {
        let start = self.index(band, 0);
        &self.data[start..start + self.num_channels * BLOCK_SIZE]
    }

    /// Iterates the channels of one band.
    pub fn band_channels(&self, band: usize) -> impl Iterator<Item = &[f32; BLOCK_SIZE]> {
        (0..self.num_channels).map(move |channel| self.view(band, channel))
    }

    pub fn fill(&mut self, value: f32) {
        self.data.fill(value);
    }

    pub fn swap(&mut self, other: &mut Block) {
        std::mem::swap(self, other);
    }

    /// Builds a block from `[band][channel][sample]` nested vectors.
    #[cfg(test)]
    pub fn from_nested(nested: &[Vec<Vec<f32>>]) -> Self {
        let num_bands = nested.len();
        let num_channels = nested[0].len();
        let mut block = Self::new(num_bands, num_channels);
        for (band, channels) in nested.iter().enumerate() {
            for (channel, samples) in channels.iter().enumerate() {
                block.view_mut(band, channel).copy_from_slice(samples);
            }
        }
        block
    }

    fn index(&self, band: usize, channel: usize) -> usize {
        debug_assert!(band < self.num_bands);
        debug_assert!(channel < self.num_channels);
        (band * self.num_channels + channel) * BLOCK_SIZE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn views_are_disjoint_and_indexed_by_band_then_channel() {
        let mut block = Block::new(2, 3);
        assert_eq!(2, block.num_bands());
        assert_eq!(3, block.num_channels());

        for band in 0..2 {
            for channel in 0..3 {
                block
                    .view_mut(band, channel)
                    .fill((band * 3 + channel) as f32);
            }
        }
        for band in 0..2 {
            for channel in 0..3 {
                assert!(
                    block
                        .view(band, channel)
                        .iter()
                        .all(|s| *s == (band * 3 + channel) as f32)
                );
            }
        }
    }

    #[test]
    fn band_spans_every_channel() {
        let mut block = Block::new(2, 2);
        block.view_mut(1, 1).fill(7.0);
        let band = block.band(1);
        assert_eq!(2 * BLOCK_SIZE, band.len());
        assert_eq!(0.0, band[0]);
        assert_eq!(7.0, band[BLOCK_SIZE]);
    }

    #[test]
    fn set_num_channels_resizes_and_clears() {
        let mut block = Block::with_value(2, 1, 5.0);
        block.set_num_channels(3);
        assert_eq!(3, block.num_channels());
        for band in 0..2 {
            for channel in 0..3 {
                assert!(block.view(band, channel).iter().all(|s| *s == 0.0));
            }
        }
    }
}
