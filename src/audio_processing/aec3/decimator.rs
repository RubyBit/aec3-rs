use num_complex::Complex32;

use crate::audio_processing::utility::cascaded_biquad_filter::{BiQuadParam, CascadedBiQuadFilter};

use super::aec3_common::BLOCK_SIZE;

pub struct Decimator {
    down_sampling_factor: usize,
    anti_aliasing_filter: CascadedBiQuadFilter,
    noise_reduction_filter: CascadedBiQuadFilter,
}

impl Decimator {
    pub fn new(down_sampling_factor: usize) -> Self {
        assert!(matches!(down_sampling_factor, 2 | 4 | 8));
        let anti_alias_params = match down_sampling_factor {
            2 => low_pass_filter_ds2(),
            4 => low_pass_filter_ds4(),
            8 => band_pass_filter_ds8(),
            _ => unreachable!(),
        };
        let noise_reduction_params = if down_sampling_factor == 8 {
            pass_through_filter()
        } else {
            high_pass_filter()
        };
        Self {
            down_sampling_factor,
            anti_aliasing_filter: CascadedBiQuadFilter::from_params(&anti_alias_params),
            noise_reduction_filter: CascadedBiQuadFilter::from_params(&noise_reduction_params),
        }
    }

    pub fn decimate(&mut self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), BLOCK_SIZE);
        assert_eq!(output.len(), BLOCK_SIZE / self.down_sampling_factor);
        let mut filtered = [0.0f32; BLOCK_SIZE];
        self.anti_aliasing_filter.process(input, &mut filtered);
        self.noise_reduction_filter.process_in_place(&mut filtered);
        let mut k = 0usize;
        for sample in output.iter_mut() {
            *sample = filtered[k];
            k += self.down_sampling_factor;
        }
    }
}

fn low_pass_filter_ds2() -> Vec<BiQuadParam> {
    vec![
        BiQuadParam::new(
            Complex32::new(-1.0, 0.0),
            Complex32::new(0.13833231, 0.40743176),
            0.22711796393486466,
        ),
        BiQuadParam::new(
            Complex32::new(-1.0, 0.0),
            Complex32::new(0.13833231, 0.40743176),
            0.22711796393486466,
        ),
        BiQuadParam::new(
            Complex32::new(-1.0, 0.0),
            Complex32::new(0.13833231, 0.40743176),
            0.22711796393486466,
        ),
    ]
}

fn low_pass_filter_ds4() -> Vec<BiQuadParam> {
    vec![
        BiQuadParam::new(
            Complex32::new(-0.08873842, 0.99605496),
            Complex32::new(0.75916227, 0.23841065),
            0.26250696827,
        ),
        BiQuadParam::new(
            Complex32::new(0.62273832, 0.78243018),
            Complex32::new(0.74892112, 0.5410152),
            0.26250696827,
        ),
        BiQuadParam::new(
            Complex32::new(0.71107693, 0.70311421),
            Complex32::new(0.74895534, 0.63924616),
            0.26250696827,
        ),
    ]
}

fn band_pass_filter_ds8() -> Vec<BiQuadParam> {
    let param = BiQuadParam {
        zero: Complex32::new(1.0, 0.0),
        pole: Complex32::new(0.7601815, 0.46423542),
        gain: 0.10330478266505948,
        mirror_zero_along_i_axis: true,
    };
    vec![param, param, param, param, param]
}

fn high_pass_filter() -> Vec<BiQuadParam> {
    vec![BiQuadParam::new(
        Complex32::new(1.0, 0.0),
        Complex32::new(0.72712179, 0.21296904),
        0.7570763753338849,
    )]
}

fn pass_through_filter() -> Vec<BiQuadParam> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::super::aec3_common::BLOCK_SIZE;
    use super::*;

    #[test]
    fn rejects_invalid_down_sampling_factor() {
        let result = std::panic::catch_unwind(|| Decimator::new(3));
        assert!(result.is_err());
    }

    #[test]
    fn decimates_to_expected_length() {
        for factor in [2usize, 4, 8] {
            let mut decimator = Decimator::new(factor);
            let input = vec![1.0f32; BLOCK_SIZE];
            let mut output = vec![0.0f32; BLOCK_SIZE / factor];
            decimator.decimate(&input, &mut output);
            assert_eq!(output.len(), BLOCK_SIZE / factor);
        }
    }

    #[test]
    fn reduces_aliasing_for_high_frequencies() {
        const NUM_STARTUP_BLOCKS: usize = 50;
        const NUM_BLOCKS: usize = 200;
        let frequency_scale = 3.0 / 8.0;
        for factor in [2usize, 4, 8] {
            let sub_block = BLOCK_SIZE / factor;
            let mut decimator = Decimator::new(factor);
            let mut input = vec![0.0f32; BLOCK_SIZE * NUM_BLOCKS];
            for (k, sample) in input.iter_mut().enumerate() {
                *sample = (2.0 * std::f32::consts::PI * frequency_scale * k as f32).sin();
            }
            let mut output = vec![0.0f32; sub_block * NUM_BLOCKS];
            let mut tmp = vec![0.0f32; sub_block];
            for block in 0..NUM_BLOCKS {
                decimator.decimate(
                    &input[block * BLOCK_SIZE..(block + 1) * BLOCK_SIZE],
                    &mut tmp,
                );
                output[block * sub_block..(block + 1) * sub_block].copy_from_slice(&tmp);
            }
            let input_tail = &input[NUM_STARTUP_BLOCKS * BLOCK_SIZE..];
            let output_tail = &output[NUM_STARTUP_BLOCKS * sub_block..];
            let input_power =
                input_tail.iter().map(|v| v * v).sum::<f32>() / input_tail.len() as f32;
            let output_power =
                output_tail.iter().map(|v| v * v).sum::<f32>() / output_tail.len() as f32;
            assert!(output_power <= 0.01 * input_power);
        }
    }
}
