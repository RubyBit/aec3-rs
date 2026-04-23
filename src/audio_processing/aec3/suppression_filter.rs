use crate::audio_processing::aec3::aec3_common::{
    Aec3Optimization, BLOCK_SIZE, FFT_LENGTH, FFT_LENGTH_BY_2, FFT_LENGTH_BY_2_PLUS_1,
    num_bands_for_rate, valid_full_band_rate,
};
use crate::audio_processing::aec3::aec3_fft::Aec3Fft;
use crate::audio_processing::aec3::fft_data::FftData;
use crate::audio_processing::aec3::vector_math::VectorMath;

pub struct SuppressionFilter {
    #[allow(dead_code)]
    optimization: Aec3Optimization,
    sample_rate_hz: i32,
    num_capture_channels: usize,
    fft: Aec3Fft,
    e_output_old: Vec<Vec<[f32; FFT_LENGTH_BY_2]>>,
}

impl SuppressionFilter {
    pub fn new(
        optimization: Aec3Optimization,
        sample_rate_hz: i32,
        num_capture_channels: usize,
    ) -> Self {
        assert!(
            valid_full_band_rate(sample_rate_hz),
            "invalid sample rate {sample_rate_hz}"
        );
        assert!(num_capture_channels > 0);
        let num_bands = num_bands_for_rate(sample_rate_hz);
        let e_output_old = (0..num_bands)
            .map(|_| vec![[0.0f32; FFT_LENGTH_BY_2]; num_capture_channels])
            .collect();
        Self {
            optimization,
            sample_rate_hz,
            num_capture_channels,
            fft: Aec3Fft::new(),
            e_output_old,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn apply_gain(
        &mut self,
        comfort_noise: &[FftData],
        comfort_noise_high_bands: &[FftData],
        suppression_gain: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        high_bands_gain: f32,
        e_lowest_band: &[FftData],
        mut e: Option<&mut Vec<Vec<Vec<f32>>>>,
    ) {
        let e = e
            .as_deref_mut()
            .expect("suppressor output must be provided");
        assert_eq!(comfort_noise.len(), self.num_capture_channels);
        assert_eq!(comfort_noise_high_bands.len(), self.num_capture_channels);
        assert_eq!(e_lowest_band.len(), self.num_capture_channels);
        let expected_bands = num_bands_for_rate(self.sample_rate_hz);
        assert_eq!(expected_bands, self.e_output_old.len());
        assert_eq!(e.len(), expected_bands);
        let num_bands = expected_bands;
        for band in 0..num_bands {
            assert_eq!(e[band].len(), self.num_capture_channels);
            for channel in 0..self.num_capture_channels {
                assert_eq!(e[band][channel].len(), BLOCK_SIZE);
            }
        }

        let vector_math = VectorMath::new(self.optimization);
        let mut suppression_gain_sq = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let mut noise_gain = [1.0f32; FFT_LENGTH_BY_2_PLUS_1];
        vector_math.multiply(suppression_gain, suppression_gain, &mut suppression_gain_sq);
        for (dst, &value) in noise_gain.iter_mut().zip(suppression_gain_sq.iter()) {
            *dst = (1.0 - value).max(0.0);
        }
        vector_math.sqrt(&mut noise_gain);
        let high_bands_noise_scaling =
            0.4f32 * (1.0 - high_bands_gain * high_bands_gain).max(0.0).sqrt();
        const IFFT_NORM: f32 = 2.0 / FFT_LENGTH as f32;

        for ch in 0..self.num_capture_channels {
            let mut spectrum = FftData::default();
            spectrum.assign(&e_lowest_band[ch]);
            let mut suppressed_re = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
            let mut suppressed_im = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
            let mut comfort_re = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
            let mut comfort_im = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
            vector_math.multiply(&spectrum.re, suppression_gain, &mut suppressed_re);
            vector_math.multiply(&spectrum.im, suppression_gain, &mut suppressed_im);
            vector_math.multiply(&noise_gain, &comfort_noise[ch].re, &mut comfort_re);
            vector_math.multiply(&noise_gain, &comfort_noise[ch].im, &mut comfort_im);
            spectrum.re.copy_from_slice(&suppressed_re);
            spectrum.im.copy_from_slice(&suppressed_im);
            vector_math.accumulate(&comfort_re, &mut spectrum.re);
            vector_math.accumulate(&comfort_im, &mut spectrum.im);

            let mut e_extended = [0.0f32; FFT_LENGTH];
            self.fft.ifft(&spectrum, &mut e_extended);

            let e0 = &mut e[0][ch];
            let e0_old = &mut self.e_output_old[0][ch];
            for i in 0..FFT_LENGTH_BY_2 {
                e0[i] = e0_old[i] * K_SQRT_HANNING[FFT_LENGTH_BY_2 + i]
                    + e_extended[i] * K_SQRT_HANNING[i];
                e0[i] *= IFFT_NORM;
            }
            e0_old.copy_from_slice(&e_extended[FFT_LENGTH_BY_2..]);

            for band in 1..num_bands {
                for sample in &mut e[band][ch] {
                    *sample *= high_bands_gain;
                }
            }

            if num_bands > 1 {
                let mut noise_fft = FftData::default();
                noise_fft.assign(&comfort_noise_high_bands[ch]);
                let mut noise_td = [0.0f32; FFT_LENGTH];
                self.fft.ifft(&noise_fft, &mut noise_td);
                let e1 = &mut e[1][ch];
                let gain = high_bands_noise_scaling * IFFT_NORM;
                for i in 0..FFT_LENGTH_BY_2 {
                    e1[i] += noise_td[i] * gain;
                }
            }

            for band in 1..num_bands {
                let current = &mut e[band][ch];
                let previous_storage = &mut self.e_output_old[band][ch];
                for i in 0..FFT_LENGTH_BY_2 {
                    std::mem::swap(&mut current[i], &mut previous_storage[i]);
                }
            }

            for band in 0..num_bands {
                for sample in &mut e[band][ch] {
                    *sample = sample.clamp(-32_768.0, 32_767.0);
                }
            }
        }
    }
}

const K_SQRT_HANNING: [f32; FFT_LENGTH] = [
    0.0,
    0.024541229,
    0.049067676,
    0.07356456,
    0.09801714,
    0.12241068,
    0.14673048,
    0.17096189,
    0.19509032,
    0.21910124,
    0.24298018,
    0.26671276,
    0.29028466,
    0.31368175,
    0.33688986,
    0.35989505,
    0.38268343,
    0.4052413,
    0.4275551,
    0.44961134,
    0.47139674,
    0.4928982,
    0.51410276,
    0.53499764,
    0.55557024,
    0.57580817,
    0.5956993,
    0.6152316,
    0.6343933,
    0.65317285,
    0.6715589,
    0.68954057,
    0.70710677,
    0.7242471,
    0.7409511,
    0.7572088,
    0.77301043,
    0.7883464,
    0.8032075,
    0.8175848,
    0.8314696,
    0.8448536,
    0.8577286,
    0.87008697,
    0.8819213,
    0.8932243,
    0.9039893,
    0.9142098,
    0.92387956,
    0.9329928,
    0.94154406,
    0.94952816,
    0.95694035,
    0.96377605,
    0.97003126,
    0.9757021,
    0.98078525,
    0.98527765,
    0.9891765,
    0.99247956,
    0.9951847,
    0.99729043,
    0.99879545,
    0.9996988,
    1.0,
    0.9996988,
    0.99879545,
    0.99729043,
    0.9951847,
    0.99247956,
    0.9891765,
    0.98527765,
    0.98078525,
    0.9757021,
    0.97003126,
    0.96377605,
    0.95694035,
    0.94952816,
    0.94154406,
    0.9329928,
    0.92387956,
    0.9142098,
    0.9039893,
    0.8932243,
    0.8819213,
    0.87008697,
    0.8577286,
    0.8448536,
    0.8314696,
    0.8175848,
    0.8032075,
    0.7883464,
    0.77301043,
    0.7572088,
    0.7409511,
    0.7242471,
    0.70710677,
    0.68954057,
    0.6715589,
    0.65317285,
    0.6343933,
    0.6152316,
    0.5956993,
    0.57580817,
    0.55557024,
    0.53499764,
    0.51410276,
    0.4928982,
    0.47139674,
    0.44961134,
    0.4275551,
    0.4052413,
    0.38268343,
    0.35989505,
    0.33688986,
    0.31368175,
    0.29028466,
    0.26671276,
    0.24298018,
    0.21910124,
    0.19509032,
    0.17096189,
    0.14673048,
    0.12241068,
    0.09801714,
    0.07356456,
    0.049067676,
    0.024541229,
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_processing::aec3::aec3_common::{
        Aec3Optimization, BLOCK_SIZE, FFT_LENGTH_BY_2, FFT_LENGTH_BY_2_PLUS_1, num_bands_for_rate,
    };
    use crate::audio_processing::aec3::aec3_fft::{Aec3Fft, Window};
    use std::f32::consts::PI;

    #[test]
    #[should_panic]
    fn null_output_panics() {
        let mut filter = SuppressionFilter::new(Aec3Optimization::None, 16_000, 1);
        let comfort = vec![FftData::default()];
        let cn_high = vec![FftData::default()];
        let e = vec![FftData::default()];
        let gain = [1.0f32; FFT_LENGTH_BY_2_PLUS_1];
        filter.apply_gain(&comfort, &cn_high, &gain, 1.0, &e, None);
    }

    #[test]
    #[should_panic]
    fn proper_sample_rate() {
        SuppressionFilter::new(Aec3Optimization::None, 16_001, 1);
    }

    #[test]
    fn comfort_noise_in_unity_gain() {
        let mut filter = SuppressionFilter::new(Aec3Optimization::None, 48_000, 1);
        let mut cn = vec![FftData::default()];
        let mut cn_high = vec![FftData::default()];
        cn[0].re.fill(1.0);
        cn[0].im.fill(1.0);
        cn_high[0].re.fill(1.0);
        cn_high[0].im.fill(1.0);
        let gain = [1.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let num_bands = num_bands_for_rate(48_000);
        let mut e_time = vec![vec![vec![0.0f32; BLOCK_SIZE]; 1]; num_bands];
        let e_ref = e_time.clone();
        let mut e_old = [0.0f32; FFT_LENGTH_BY_2];
        let fft = Aec3Fft::new();
        let mut spectrum = vec![FftData::default()];
        fft.padded_fft_with_window(&e_time[0][0], &e_old, Window::SqrtHanning, &mut spectrum[0]);
        e_old.copy_from_slice(&e_time[0][0]);
        filter.apply_gain(&cn, &cn_high, &gain, 1.0, &spectrum, Some(&mut e_time));
        assert_eq!(e_time, e_ref);
    }

    fn produce_sinusoid(
        sample_rate_hz: i32,
        sinusoidal_frequency_hz: f32,
        sample_counter: &mut usize,
        x: &mut [Vec<Vec<f32>>],
    ) {
        for (j, k) in (*sample_counter..(*sample_counter + BLOCK_SIZE)).enumerate() {
            let angle = 2.0 * PI * sinusoidal_frequency_hz * k as f32 / sample_rate_hz as f32;
            for channel in 0..x[0].len() {
                x[0][channel][j] = 32_767.0f32 * angle.sin();
            }
        }
        *sample_counter += BLOCK_SIZE;
        for band in x.iter_mut().skip(1) {
            for channel in band.iter_mut() {
                channel.fill(0.0);
            }
        }
    }

    #[test]
    fn signal_suppression() {
        const SAMPLE_RATE_HZ: i32 = 48_000;
        const NUM_CHANNELS: usize = 1;
        let mut filter =
            SuppressionFilter::new(Aec3Optimization::None, SAMPLE_RATE_HZ, NUM_CHANNELS);
        let mut cn = vec![FftData::default()];
        let mut cn_high = vec![FftData::default()];
        let mut gain = [1.0f32; FFT_LENGTH_BY_2_PLUS_1];
        for g in gain.iter_mut().skip(10) {
            *g = 0.0;
        }
        cn[0].re.fill(0.0);
        cn[0].im.fill(0.0);
        cn_high[0].re.fill(0.0);
        cn_high[0].im.fill(0.0);
        let num_bands = num_bands_for_rate(SAMPLE_RATE_HZ);
        let mut e_time = vec![vec![vec![0.0f32; BLOCK_SIZE]; NUM_CHANNELS]; num_bands];
        let fft = Aec3Fft::new();
        let mut e_old = [0.0f32; FFT_LENGTH_BY_2];
        let mut spectrum = vec![FftData::default(); NUM_CHANNELS];
        let mut sample_counter = 0usize;
        let mut e0_input = 0.0f32;
        let mut e0_output = 0.0f32;
        for _ in 0..100 {
            produce_sinusoid(
                16_000,
                16_000.0f32 * 40.0 / FFT_LENGTH_BY_2 as f32 / 2.0,
                &mut sample_counter,
                &mut e_time,
            );
            e0_input += e_time[0][0].iter().map(|v| v * v).sum::<f32>();
            fft.padded_fft_with_window(
                &e_time[0][0],
                &e_old,
                Window::SqrtHanning,
                &mut spectrum[0],
            );
            e_old.copy_from_slice(&e_time[0][0]);
            filter.apply_gain(&cn, &cn_high, &gain, 1.0, &spectrum, Some(&mut e_time));
            e0_output += e_time[0][0].iter().map(|v| v * v).sum::<f32>();
        }
        assert!(e0_output < e0_input / 1000.0f32);
    }

    #[test]
    fn signal_transparency() {
        const SAMPLE_RATE_HZ: i32 = 48_000;
        const NUM_CHANNELS: usize = 1;
        let mut filter =
            SuppressionFilter::new(Aec3Optimization::None, SAMPLE_RATE_HZ, NUM_CHANNELS);
        let mut cn = vec![FftData::default()];
        let mut cn_high = vec![FftData::default()];
        let mut gain = [1.0f32; FFT_LENGTH_BY_2_PLUS_1];
        for g in gain.iter_mut().skip(30) {
            *g = 0.0;
        }
        cn[0].re.fill(0.0);
        cn[0].im.fill(0.0);
        cn_high[0].re.fill(0.0);
        cn_high[0].im.fill(0.0);
        let num_bands = num_bands_for_rate(SAMPLE_RATE_HZ);
        let mut e_time = vec![vec![vec![0.0f32; BLOCK_SIZE]; NUM_CHANNELS]; num_bands];
        let fft = Aec3Fft::new();
        let mut e_old = [0.0f32; FFT_LENGTH_BY_2];
        let mut spectrum = vec![FftData::default(); NUM_CHANNELS];
        let mut sample_counter = 0usize;
        let mut e0_input = 0.0f32;
        let mut e0_output = 0.0f32;
        for _ in 0..100 {
            produce_sinusoid(
                16_000,
                16_000.0f32 * 10.0 / FFT_LENGTH_BY_2 as f32 / 2.0,
                &mut sample_counter,
                &mut e_time,
            );
            e0_input += e_time[0][0].iter().map(|v| v * v).sum::<f32>();
            fft.padded_fft_with_window(
                &e_time[0][0],
                &e_old,
                Window::SqrtHanning,
                &mut spectrum[0],
            );
            e_old.copy_from_slice(&e_time[0][0]);
            filter.apply_gain(&cn, &cn_high, &gain, 1.0, &spectrum, Some(&mut e_time));
            e0_output += e_time[0][0].iter().map(|v| v * v).sum::<f32>();
        }
        assert!(0.9f32 * e0_input < e0_output);
    }

    #[test]
    fn delay() {
        const SAMPLE_RATE_HZ: i32 = 48_000;
        const NUM_CHANNELS: usize = 1;
        let mut filter =
            SuppressionFilter::new(Aec3Optimization::None, SAMPLE_RATE_HZ, NUM_CHANNELS);
        let cn = vec![FftData::default(); NUM_CHANNELS];
        let cn_high = vec![FftData::default(); NUM_CHANNELS];
        let gain = [1.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let num_bands = num_bands_for_rate(SAMPLE_RATE_HZ);
        let mut e_time = vec![vec![vec![0.0f32; BLOCK_SIZE]; NUM_CHANNELS]; num_bands];
        let fft = Aec3Fft::new();
        let mut e_old = [0.0f32; FFT_LENGTH_BY_2];
        let mut spectrum = vec![FftData::default(); NUM_CHANNELS];

        for k in 0..100 {
            for band in 0..num_bands {
                for channel in 0..NUM_CHANNELS {
                    for sample in 0..BLOCK_SIZE {
                        e_time[band][channel][sample] = (k * BLOCK_SIZE + sample + channel) as f32;
                    }
                }
            }

            fft.padded_fft_with_window(
                &e_time[0][0],
                &e_old,
                Window::SqrtHanning,
                &mut spectrum[0],
            );
            e_old.copy_from_slice(&e_time[0][0]);
            filter.apply_gain(&cn, &cn_high, &gain, 1.0, &spectrum, Some(&mut e_time));

            if k > 2 {
                for band in 0..num_bands {
                    for channel in 0..NUM_CHANNELS {
                        for sample in 0..BLOCK_SIZE {
                            let expected = (k * BLOCK_SIZE + sample - BLOCK_SIZE + channel) as f32;
                            let actual = e_time[band][channel][sample];
                            assert!((actual - expected).abs() < 0.01);
                        }
                    }
                }
            }
        }
    }
}
