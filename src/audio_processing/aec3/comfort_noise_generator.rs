use crate::audio_processing::aec3::aec3_common::{
    Aec3Optimization, FFT_LENGTH_BY_2, FFT_LENGTH_BY_2_PLUS_1,
};
use crate::audio_processing::aec3::fft_data::FftData;
use crate::audio_processing::aec3::vector_math::VectorMath;

const K_SQRT2_SIN: [f32; 32] = [
    0.0000000f32,
    0.2758994f32,
    0.5411961f32,
    0.7856950f32,
    1.0000000f32,
    1.1758756f32,
    1.3065630f32,
    1.3870398f32,
    1.4142136f32,
    1.3870398f32,
    1.3065630f32,
    1.1758756f32,
    1.0000000f32,
    0.7856950f32,
    0.5411961f32,
    0.2758994f32,
    0.0000000f32,
    -0.2758994f32,
    -0.5411961f32,
    -0.7856950f32,
    -1.0000000f32,
    -1.1758756f32,
    -1.3065630f32,
    -1.3870398f32,
    -1.4142136f32,
    -1.3870398f32,
    -1.3065630f32,
    -1.1758756f32,
    -1.0000000f32,
    -0.7856950f32,
    -0.5411961f32,
    -0.2758994f32,
];

fn generate_comfort_noise(
    optimization: Aec3Optimization,
    n2: &[f32; FFT_LENGTH_BY_2_PLUS_1],
    seed: &mut u32,
    lower_band_noise: &mut FftData,
    upper_band_noise: &mut FftData,
) {
    // Compute sqrt spectrum.
    let mut n = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
    n.copy_from_slice(n2);
    VectorMath::new(optimization).sqrt(&mut n);

    // Compute high band noise level (average over upper half)
    let k_half = FFT_LENGTH_BY_2_PLUS_1 / 2;
    let k_one_by_num_bands = 1.0f32 / (k_half as f32 + 1.0f32);
    let mut high_sum = 0.0f32;
    for k in k_half..FFT_LENGTH_BY_2_PLUS_1 {
        high_sum += n[k];
    }
    let high_band_noise_level = high_sum * k_one_by_num_bands;

    // Zero DC and Nyquist
    lower_band_noise.re[0] = 0.0;
    lower_band_noise.re[FFT_LENGTH_BY_2] = 0.0;
    upper_band_noise.re[0] = 0.0;
    upper_band_noise.re[FFT_LENGTH_BY_2] = 0.0;
    lower_band_noise.im[0] = 0.0;
    lower_band_noise.im[FFT_LENGTH_BY_2] = 0.0;
    upper_band_noise.im[0] = 0.0;
    upper_band_noise.im[FFT_LENGTH_BY_2] = 0.0;

    for k in 1..FFT_LENGTH_BY_2 {
        // Generate a random 31-bit integer, matching reference implementation.
        *seed = (*seed).wrapping_mul(69069).wrapping_add(1) & (0x8000_0000 - 1);
        let i = (*seed >> 26) as usize & 31;
        let x = K_SQRT2_SIN[i];
        let y = K_SQRT2_SIN[(i + 8) & 31];

        lower_band_noise.re[k] = n[k] * x;
        lower_band_noise.im[k] = n[k] * y;

        upper_band_noise.re[k] = high_band_noise_level * x;
        upper_band_noise.im[k] = high_band_noise_level * y;
    }
}

pub struct ComfortNoiseGenerator {
    optimization: Aec3Optimization,
    seed: u32,
    num_capture_channels: usize,
    n2_initial: Option<Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>>,
    y2_smoothed: Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
    n2: Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
    n2_counter: i32,
}

impl ComfortNoiseGenerator {
    pub fn new(optimization: Aec3Optimization, num_capture_channels: usize) -> Self {
        assert!(num_capture_channels > 0);
        let mut n2_initial = Vec::with_capacity(num_capture_channels);
        for _ in 0..num_capture_channels {
            n2_initial.push([0.0f32; FFT_LENGTH_BY_2_PLUS_1]);
        }
        let y2_smoothed = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels];
        let n2 = vec![[1.0e6f32; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels];
        let instance = Self {
            optimization,
            seed: 42,
            num_capture_channels,
            n2_initial: Some(n2_initial),
            y2_smoothed,
            n2,
            n2_counter: 0,
        };
        instance
    }

    pub fn compute(
        &mut self,
        saturated_capture: bool,
        capture_spectrum: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        lower_band_noise: &mut [FftData],
        upper_band_noise: &mut [FftData],
    ) {
        assert_eq!(capture_spectrum.len(), self.num_capture_channels);
        assert_eq!(lower_band_noise.len(), self.num_capture_channels);
        assert_eq!(upper_band_noise.len(), self.num_capture_channels);

        if !saturated_capture {
            // Smooth Y2: a + 0.1*(b-a) = 0.9a + 0.1b
            for ch in 0..self.num_capture_channels {
                for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                    let a = self.y2_smoothed[ch][k];
                    let b = capture_spectrum[ch][k];
                    self.y2_smoothed[ch][k] = a + 0.1f32 * (b - a);
                }
            }

            if self.n2_counter > 50 {
                for ch in 0..self.num_capture_channels {
                    for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                        let a = self.n2[ch][k];
                        let b = self.y2_smoothed[ch][k];
                        self.n2[ch][k] = if b < a {
                            (0.9f32 * b + 0.1f32 * a) * 1.0002f32
                        } else {
                            a * 1.0002f32
                        };
                    }
                }
            }

            if let Some(initial) = &mut self.n2_initial {
                self.n2_counter += 1;
                if self.n2_counter == 1000 {
                    self.n2_initial = None;
                } else {
                    for ch in 0..self.num_capture_channels {
                        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                            let a = self.n2[ch][k];
                            let b = initial[ch][k];
                            initial[ch][k] = if a > b { b + 0.001f32 * (a - b) } else { a };
                        }
                    }
                }
            }

            // Limit the noise floor
            const K_NOISE_FLOOR: f32 = 17.1267f32;
            for ch in 0..self.num_capture_channels {
                for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                    self.n2[ch][k] = self.n2[ch][k].max(K_NOISE_FLOOR);
                }
                if let Some(initial) = &mut self.n2_initial {
                    for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                        initial[ch][k] = initial[ch][k].max(K_NOISE_FLOOR);
                    }
                }
            }
        }

        let n2_ref: &Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]> = if let Some(initial) = &self.n2_initial {
            initial
        } else {
            &self.n2
        };

        for ch in 0..self.num_capture_channels {
            generate_comfort_noise(
                self.optimization,
                &n2_ref[ch],
                &mut self.seed,
                &mut lower_band_noise[ch],
                &mut upper_band_noise[ch],
            );
        }
    }

    pub fn noise_spectrum(&self) -> &[[f32; FFT_LENGTH_BY_2_PLUS_1]] {
        if let Some(initial) = &self.n2_initial {
            &initial
        } else {
            &self.n2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::config::EchoCanceller3Config;
    use crate::audio_processing::aec3::aec_state::AecState;
    use crate::audio_processing::aec3::aec3_common::detect_optimization;

    fn power(optimization: Aec3Optimization, n: &FftData) -> f32 {
        let mut buf = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        n.spectrum(optimization, &mut buf);
        buf.iter().sum::<f32>() / buf.len() as f32
    }

    #[test]
    fn correct_level() {
        const NUM_CHANNELS: usize = 5;
        let mut cng = ComfortNoiseGenerator::new(detect_optimization(), NUM_CHANNELS);
        let _aec_state = AecState::new(EchoCanceller3Config::default(), NUM_CHANNELS);

        let mut n2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; NUM_CHANNELS];
        let mut n_lower = vec![FftData::default(); NUM_CHANNELS];
        let mut n_upper = vec![FftData::default(); NUM_CHANNELS];

        for ch in 0..NUM_CHANNELS {
            for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                n2[ch][k] = 1000.0f32 * 1000.0f32 / (ch as f32 + 1.0f32);
            }
            n_lower[ch].clear();
            n_upper[ch].clear();
        }

        // Ensure instantaneous update to nonzero noise.
        cng.compute(false, &n2, &mut n_lower, &mut n_upper);
        for ch in 0..NUM_CHANNELS {
            assert!(power(detect_optimization(), &n_lower[ch]) > 0.0);
            assert!(power(detect_optimization(), &n_upper[ch]) > 0.0);
        }

        for _ in 0..10_000 {
            cng.compute(false, &n2, &mut n_lower, &mut n_upper);
        }

        for ch in 0..NUM_CHANNELS {
            let expected = 2.0f32 * n2[ch][0];
            let p_low = power(detect_optimization(), &n_lower[ch]);
            let p_high = power(detect_optimization(), &n_upper[ch]);
            let tol = n2[ch][0] / 10.0f32;
            assert!(
                (p_low - expected).abs() <= tol,
                "low ch {} p={} exp={}",
                ch,
                p_low,
                expected
            );
            assert!(
                (p_high - expected).abs() <= tol,
                "high ch {} p={} exp={}",
                ch,
                p_high,
                expected
            );
        }
    }
}
