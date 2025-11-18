use crate::api::config::AlignmentMixing;

use super::aec3_common::{BLOCK_SIZE, NUM_BLOCKS_PER_SECOND};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MixingVariant {
    Downmix,
    Adaptive,
    Fixed,
}

fn choose_mixing_variant(num_channels: usize, config: &AlignmentMixing) -> MixingVariant {
    assert!(num_channels > 0);
    if num_channels == 1 {
        return MixingVariant::Fixed;
    }
    if config.downmix {
        return MixingVariant::Downmix;
    }
    if config.adaptive_selection {
        return MixingVariant::Adaptive;
    }
    MixingVariant::Fixed
}

pub struct AlignmentMixer {
    num_channels: usize,
    one_by_num_channels: f32,
    excitation_energy_threshold: f32,
    prefer_first_two_channels: bool,
    selection_variant: MixingVariant,
    strong_block_counters: [usize; 2],
    cumulative_energies: Vec<f32>,
    selected_channel: usize,
    block_counter: usize,
}

impl AlignmentMixer {
    pub fn new(num_channels: usize, config: AlignmentMixing) -> Self {
        assert!(num_channels > 0);
        let selection_variant = choose_mixing_variant(num_channels, &config);
        let cumulative_energies = if matches!(selection_variant, MixingVariant::Adaptive) {
            vec![0.0; num_channels]
        } else {
            Vec::new()
        };
        Self {
            num_channels,
            one_by_num_channels: 1.0f32 / num_channels as f32,
            excitation_energy_threshold: BLOCK_SIZE as f32 * config.activity_power_threshold,
            prefer_first_two_channels: config.prefer_first_two_channels,
            selection_variant,
            strong_block_counters: [0; 2],
            cumulative_energies,
            selected_channel: 0,
            block_counter: 0,
        }
    }

    pub fn produce_output(&mut self, x: &[Vec<f32>], y: &mut [f32; BLOCK_SIZE]) {
        assert_eq!(x.len(), self.num_channels);
        for channel in x {
            assert_eq!(channel.len(), BLOCK_SIZE);
        }
        match self.selection_variant {
            MixingVariant::Downmix => self.downmix(x, y),
            MixingVariant::Adaptive => {
                let ch = self.select_channel(x);
                y.copy_from_slice(&x[ch]);
            }
            MixingVariant::Fixed => y.copy_from_slice(&x[0]),
        }
    }

    fn downmix(&self, x: &[Vec<f32>], y: &mut [f32; BLOCK_SIZE]) {
        y.fill(0.0);
        for channel in x {
            for (dst, &sample) in y.iter_mut().zip(channel.iter()) {
                *dst += sample;
            }
        }
        for sample in y.iter_mut() {
            *sample *= self.one_by_num_channels;
        }
    }

    fn select_channel(&mut self, x: &[Vec<f32>]) -> usize {
        let blocks_to_choose = (0.5 * NUM_BLOCKS_PER_SECOND as f32) as usize;
        let good_signal_in_lr = self.prefer_first_two_channels
            && (self.strong_block_counters[0] > blocks_to_choose
                || self.strong_block_counters[1] > blocks_to_choose);
        let num_channels_to_analyze = if good_signal_in_lr {
            2
        } else {
            self.num_channels
        };
        const NUM_BLOCKS_BEFORE_SMOOTHING: usize = 60 * NUM_BLOCKS_PER_SECOND;
        const SMOOTHING: f32 = 1.0 / (10 * NUM_BLOCKS_PER_SECOND) as f32;
        self.block_counter += 1;

        for ch in 0..num_channels_to_analyze {
            let x2_sum: f32 = x[ch].iter().map(|sample| sample * sample).sum();
            if ch < 2 && x2_sum > self.excitation_energy_threshold {
                self.strong_block_counters[ch] += 1;
            }

            if self.block_counter <= NUM_BLOCKS_BEFORE_SMOOTHING {
                self.cumulative_energies[ch] += x2_sum;
            } else {
                self.cumulative_energies[ch] += SMOOTHING * (x2_sum - self.cumulative_energies[ch]);
            }
        }

        if self.block_counter == NUM_BLOCKS_BEFORE_SMOOTHING {
            let factor = 1.0f32 / NUM_BLOCKS_BEFORE_SMOOTHING as f32;
            for ch in 0..num_channels_to_analyze {
                self.cumulative_energies[ch] *= factor;
            }
        }

        let mut strongest = 0usize;
        for ch in 1..num_channels_to_analyze {
            if self.cumulative_energies[ch] > self.cumulative_energies[strongest] {
                strongest = ch;
            }
        }

        if (good_signal_in_lr && self.selected_channel > 1)
            || self.cumulative_energies[strongest]
                > 2.0 * self.cumulative_energies[self.selected_channel]
        {
            self.selected_channel = strongest;
        }

        self.selected_channel.min(self.num_channels - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> AlignmentMixing {
        AlignmentMixing {
            downmix: false,
            adaptive_selection: true,
            activity_power_threshold: 0.01,
            prefer_first_two_channels: true,
        }
    }

    #[test]
    fn single_channel_is_passthrough() {
        let config = default_config();
        let mut mixer = AlignmentMixer::new(1, config);
        let input = vec![vec![1.0f32; BLOCK_SIZE]];
        let mut output = [0.0f32; BLOCK_SIZE];
        mixer.produce_output(&input, &mut output);
        for (a, b) in input[0].iter().zip(output.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn downmix_averages_channels() {
        let mut config = default_config();
        config.downmix = true;
        config.adaptive_selection = false;
        let mut mixer = AlignmentMixer::new(2, config);
        let mut ch0 = vec![0.0f32; BLOCK_SIZE];
        let mut ch1 = vec![0.0f32; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            ch0[i] = i as f32;
            ch1[i] = (BLOCK_SIZE - i) as f32;
        }
        let input = vec![ch0.clone(), ch1.clone()];
        let mut output = [0.0f32; BLOCK_SIZE];
        mixer.produce_output(&input, &mut output);
        for i in 0..BLOCK_SIZE {
            let expected = 0.5 * (ch0[i] + ch1[i]);
            assert!((output[i] - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn adaptive_selection_picks_strong_channel() {
        let config = default_config();
        let mut mixer = AlignmentMixer::new(2, config);
        let strong = vec![2.0f32; BLOCK_SIZE];
        let weak = vec![0.1f32; BLOCK_SIZE];
        let mut output = [0.0f32; BLOCK_SIZE];
        for _ in 0..(NUM_BLOCKS_PER_SECOND * 2) {
            let blocks = vec![strong.clone(), weak.clone()];
            mixer.produce_output(&blocks, &mut output);
        }
        for (out, expected) in output.iter().zip(strong.iter()) {
            assert!((out - expected).abs() < 1e-6);
        }
    }
}
