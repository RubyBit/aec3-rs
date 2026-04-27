//! Spectral feature extractor for RNN-VAD.

use num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

use crate::audio_processing::agc2::rnn_vad::common::{
    CEPSTRAL_COEFFS_HISTORY_SIZE, FRAME_SIZE_20_MS_24_KHZ, NUM_BANDS, NUM_LOWER_BANDS, PI,
};
use crate::audio_processing::agc2::rnn_vad::ring_buffer::RingBuffer;
use crate::audio_processing::agc2::rnn_vad::spectral_features_internal::{
    OPUS_BANDS_24_KHZ, SpectralCorrelator, compute_dct, compute_dct_table,
    compute_smoothed_log_magnitude_spectrum,
};
use crate::audio_processing::agc2::rnn_vad::symmetric_matrix_buffer::SymmetricMatrixBuffer;

const SILENCE_THRESHOLD: f32 = 0.04;

fn update_cepstral_difference_stats(
    new_cepstral_coeffs: &[f32; NUM_BANDS],
    ring_buf: &RingBuffer<f32, NUM_BANDS, CEPSTRAL_COEFFS_HISTORY_SIZE>,
    sym_matrix_buf: &mut SymmetricMatrixBuffer<f32, CEPSTRAL_COEFFS_HISTORY_SIZE>,
) {
    let mut distances = [0.0f32; CEPSTRAL_COEFFS_HISTORY_SIZE - 1];
    for i in 0..(CEPSTRAL_COEFFS_HISTORY_SIZE - 1) {
        let delay = i + 1;
        let old_cepstral_coeffs = ring_buf.get_array_view(delay);
        let mut d = 0.0f32;
        for k in 0..NUM_BANDS {
            let c = new_cepstral_coeffs[k] - old_cepstral_coeffs[k];
            d += c * c;
        }
        distances[i] = d;
    }
    sym_matrix_buf.push(&distances);
}

fn compute_scaled_half_vorbis_window(scaling: f32) -> [f32; FRAME_SIZE_20_MS_24_KHZ / 2] {
    const HALF_SIZE: usize = FRAME_SIZE_20_MS_24_KHZ / 2;
    let mut half_window = [0.0f32; HALF_SIZE];
    for (i, v) in half_window.iter_mut().enumerate().take(HALF_SIZE) {
        let t = (i as f64 + 0.5) / HALF_SIZE as f64;
        let s = (0.5 * PI * t).sin();
        *v = scaling * (0.5 * PI * s * s).sin() as f32;
    }
    half_window
}

fn compute_windowed_forward_fft(
    frame: &[f32; FRAME_SIZE_20_MS_24_KHZ],
    half_window: &[f32; FRAME_SIZE_20_MS_24_KHZ / 2],
    fft: &Arc<dyn Fft<f32>>,
    complex_buffer: &mut [Complex32],
    interleaved_output: &mut [f32; FRAME_SIZE_20_MS_24_KHZ],
) {
    debug_assert_eq!(frame.len(), 2 * half_window.len());

    for i in 0..half_window.len() {
        let j = FRAME_SIZE_20_MS_24_KHZ - 1 - i;
        complex_buffer[i].re = frame[i] * half_window[i];
        complex_buffer[i].im = 0.0;
        complex_buffer[j].re = frame[j] * half_window[i];
        complex_buffer[j].im = 0.0;
    }

    fft.process(complex_buffer);

    // Pack to WebRTC-like interleaved representation where index 1 is reserved
    // (Nyquist omitted).
    interleaved_output[0] = complex_buffer[0].re;
    interleaved_output[1] = 0.0;
    for k in 1..(FRAME_SIZE_20_MS_24_KHZ / 2) {
        interleaved_output[2 * k] = complex_buffer[k].re;
        interleaved_output[2 * k + 1] = complex_buffer[k].im;
    }
}

/// Computes spectral features from reference/lagged 20ms frames.
pub struct SpectralFeaturesExtractor {
    half_window: [f32; FRAME_SIZE_20_MS_24_KHZ / 2],
    fft_forward: Arc<dyn Fft<f32>>,
    fft_buffer: Vec<Complex32>,
    reference_frame_fft: [f32; FRAME_SIZE_20_MS_24_KHZ],
    lagged_frame_fft: [f32; FRAME_SIZE_20_MS_24_KHZ],
    spectral_correlator: SpectralCorrelator,
    reference_frame_bands_energy: [f32; OPUS_BANDS_24_KHZ],
    lagged_frame_bands_energy: [f32; OPUS_BANDS_24_KHZ],
    bands_cross_corr: [f32; OPUS_BANDS_24_KHZ],
    dct_table: [f32; NUM_BANDS * NUM_BANDS],
    cepstral_coeffs_ring_buf: RingBuffer<f32, NUM_BANDS, CEPSTRAL_COEFFS_HISTORY_SIZE>,
    cepstral_diffs_buf: SymmetricMatrixBuffer<f32, CEPSTRAL_COEFFS_HISTORY_SIZE>,
}

impl Default for SpectralFeaturesExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl SpectralFeaturesExtractor {
    pub fn new() -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let fft_forward = planner.plan_fft_forward(FRAME_SIZE_20_MS_24_KHZ);
        Self {
            half_window: compute_scaled_half_vorbis_window(1.0 / FRAME_SIZE_20_MS_24_KHZ as f32),
            fft_forward,
            fft_buffer: vec![Complex32::new(0.0, 0.0); FRAME_SIZE_20_MS_24_KHZ],
            reference_frame_fft: [0.0; FRAME_SIZE_20_MS_24_KHZ],
            lagged_frame_fft: [0.0; FRAME_SIZE_20_MS_24_KHZ],
            spectral_correlator: SpectralCorrelator::new(),
            reference_frame_bands_energy: [0.0; OPUS_BANDS_24_KHZ],
            lagged_frame_bands_energy: [0.0; OPUS_BANDS_24_KHZ],
            bands_cross_corr: [0.0; OPUS_BANDS_24_KHZ],
            dct_table: compute_dct_table(),
            cepstral_coeffs_ring_buf: RingBuffer::new(),
            cepstral_diffs_buf: SymmetricMatrixBuffer::new(),
        }
    }

    pub fn reset(&mut self) {
        self.cepstral_coeffs_ring_buf.reset();
        self.cepstral_diffs_buf.reset();
    }

    #[allow(clippy::too_many_arguments)]
    pub fn check_silence_compute_features(
        &mut self,
        reference_frame: &[f32; FRAME_SIZE_20_MS_24_KHZ],
        lagged_frame: &[f32; FRAME_SIZE_20_MS_24_KHZ],
        higher_bands_cepstrum: &mut [f32; NUM_BANDS - NUM_LOWER_BANDS],
        average: &mut [f32; NUM_LOWER_BANDS],
        first_derivative: &mut [f32; NUM_LOWER_BANDS],
        second_derivative: &mut [f32; NUM_LOWER_BANDS],
        bands_cross_corr: &mut [f32; NUM_LOWER_BANDS],
        variability: &mut f32,
    ) -> bool {
        compute_windowed_forward_fft(
            reference_frame,
            &self.half_window,
            &self.fft_forward,
            &mut self.fft_buffer,
            &mut self.reference_frame_fft,
        );
        self.spectral_correlator.compute_auto_correlation(
            &self.reference_frame_fft,
            &mut self.reference_frame_bands_energy,
        );

        let tot_energy: f32 = self.reference_frame_bands_energy.iter().sum();
        if tot_energy < SILENCE_THRESHOLD {
            return true;
        }

        compute_windowed_forward_fft(
            lagged_frame,
            &self.half_window,
            &self.fft_forward,
            &mut self.fft_buffer,
            &mut self.lagged_frame_fft,
        );
        self.spectral_correlator
            .compute_auto_correlation(&self.lagged_frame_fft, &mut self.lagged_frame_bands_energy);

        let mut log_bands_energy = [0.0f32; NUM_BANDS];
        compute_smoothed_log_magnitude_spectrum(
            &self.reference_frame_bands_energy,
            &mut log_bands_energy,
        );

        let mut cepstrum = [0.0f32; NUM_BANDS];
        compute_dct(&log_bands_energy, &self.dct_table, &mut cepstrum);
        cepstrum[0] -= 12.0;
        cepstrum[1] -= 4.0;

        self.cepstral_coeffs_ring_buf.push(&cepstrum);
        update_cepstral_difference_stats(
            &cepstrum,
            &self.cepstral_coeffs_ring_buf,
            &mut self.cepstral_diffs_buf,
        );

        higher_bands_cepstrum.copy_from_slice(&cepstrum[NUM_LOWER_BANDS..NUM_BANDS]);

        self.compute_avg_and_derivatives(average, first_derivative, second_derivative);
        self.compute_normalized_cepstral_correlation(bands_cross_corr);
        *variability = self.compute_variability();

        false
    }

    fn compute_avg_and_derivatives(
        &self,
        average: &mut [f32; NUM_LOWER_BANDS],
        first_derivative: &mut [f32; NUM_LOWER_BANDS],
        second_derivative: &mut [f32; NUM_LOWER_BANDS],
    ) {
        let curr = self.cepstral_coeffs_ring_buf.get_array_view(0);
        let prev1 = self.cepstral_coeffs_ring_buf.get_array_view(1);
        let prev2 = self.cepstral_coeffs_ring_buf.get_array_view(2);

        for i in 0..NUM_LOWER_BANDS {
            average[i] = curr[i] + prev1[i] + prev2[i];
            first_derivative[i] = curr[i] - prev2[i];
            second_derivative[i] = curr[i] - 2.0 * prev1[i] + prev2[i];
        }
    }

    fn compute_normalized_cepstral_correlation(
        &mut self,
        bands_cross_corr: &mut [f32; NUM_LOWER_BANDS],
    ) {
        self.spectral_correlator.compute_cross_correlation(
            &self.reference_frame_fft,
            &self.lagged_frame_fft,
            &mut self.bands_cross_corr,
        );

        for i in 0..self.bands_cross_corr.len() {
            self.bands_cross_corr[i] /= (0.001
                + self.reference_frame_bands_energy[i] * self.lagged_frame_bands_energy[i])
                .sqrt();
        }

        compute_dct(&self.bands_cross_corr, &self.dct_table, bands_cross_corr);
        bands_cross_corr[0] -= 1.3;
        bands_cross_corr[1] -= 0.9;
    }

    fn compute_variability(&self) -> f32 {
        let mut variability = 0.0f32;
        for delay1 in 0..CEPSTRAL_COEFFS_HISTORY_SIZE {
            let mut min_dist = f32::MAX;
            for delay2 in 0..CEPSTRAL_COEFFS_HISTORY_SIZE {
                if delay1 == delay2 {
                    continue;
                }
                min_dist = min_dist.min(self.cepstral_diffs_buf.get_value(delay1, delay2));
            }
            variability += min_dist;
        }
        variability / CEPSTRAL_COEFFS_HISTORY_SIZE as f32 - 2.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_FEATURE_VECTOR_SIZE: usize = NUM_BANDS + 3 * NUM_LOWER_BANDS + 1;
    const INITIAL_FEATURE_VAL: f32 = -9999.0;

    fn write_test_data(samples: &mut [f32; FRAME_SIZE_20_MS_24_KHZ]) {
        for (i, v) in samples.iter_mut().enumerate() {
            *v = (i % 100) as f32;
        }
    }

    fn split_feature_vector<'a>(
        feature_vector: &'a mut [f32; TEST_FEATURE_VECTOR_SIZE],
    ) -> (
        &'a mut [f32; NUM_BANDS - NUM_LOWER_BANDS],
        &'a mut [f32; NUM_LOWER_BANDS],
        &'a mut [f32; NUM_LOWER_BANDS],
        &'a mut [f32; NUM_LOWER_BANDS],
        &'a mut [f32; NUM_LOWER_BANDS],
        &'a mut f32,
    ) {
        let (a, rem) = feature_vector.split_at_mut(NUM_LOWER_BANDS);
        let (hb, rem) = rem.split_at_mut(NUM_BANDS - NUM_LOWER_BANDS);
        let (fd, rem) = rem.split_at_mut(NUM_LOWER_BANDS);
        let (sd, rem) = rem.split_at_mut(NUM_LOWER_BANDS);
        let (cc, rem) = rem.split_at_mut(NUM_LOWER_BANDS);
        let var = &mut rem[0];
        (
            hb.try_into().expect("higher bands shape"),
            a.try_into().expect("average shape"),
            fd.try_into().expect("first derivative shape"),
            sd.try_into().expect("second derivative shape"),
            cc.try_into().expect("cross corr shape"),
            var,
        )
    }

    #[test]
    fn spectral_features_with_and_without_silence() {
        let mut sfe = SpectralFeaturesExtractor::new();
        let mut samples = [0.0f32; FRAME_SIZE_20_MS_24_KHZ];
        let mut feature_vector = [INITIAL_FEATURE_VAL; TEST_FEATURE_VECTOR_SIZE];

        // With silence.
        let (hb, avg, fd, sd, cc, var) = split_feature_vector(&mut feature_vector);
        let is_silence =
            sfe.check_silence_compute_features(&samples, &samples, hb, avg, fd, sd, cc, var);
        assert!(is_silence);
        assert!(feature_vector.iter().all(|&x| x == INITIAL_FEATURE_VAL));

        // With no silence.
        write_test_data(&mut samples);
        let (hb, avg, fd, sd, cc, var) = split_feature_vector(&mut feature_vector);
        let is_silence =
            sfe.check_silence_compute_features(&samples, &samples, hb, avg, fd, sd, cc, var);
        assert!(!is_silence);
        assert!(!feature_vector.iter().all(|&x| x == INITIAL_FEATURE_VAL));
    }

    #[test]
    fn cepstral_features_constant_average_zero_derivative() {
        let mut sfe = SpectralFeaturesExtractor::new();
        let mut samples = [0.0f32; FRAME_SIZE_20_MS_24_KHZ];
        write_test_data(&mut samples);

        let mut feature_vector = [0.0f32; TEST_FEATURE_VECTOR_SIZE];
        for _ in 0..CEPSTRAL_COEFFS_HISTORY_SIZE {
            let (hb, avg, fd, sd, cc, var) = split_feature_vector(&mut feature_vector);
            let _ =
                sfe.check_silence_compute_features(&samples, &samples, hb, avg, fd, sd, cc, var);
        }

        let mut feature_vector_last = [0.0f32; TEST_FEATURE_VECTOR_SIZE];
        let (hb, avg_last, fd_last, sd_last, cc, var_last) =
            split_feature_vector(&mut feature_vector_last);
        let _ = sfe.check_silence_compute_features(
            &samples, &samples, hb, avg_last, fd_last, sd_last, cc, var_last,
        );

        // Average is unchanged.
        for (a, b) in feature_vector[..NUM_LOWER_BANDS]
            .iter()
            .zip(feature_vector_last[..NUM_LOWER_BANDS].iter())
        {
            assert!((a - b).abs() < 1e-6, "a={a}, b={b}");
        }

        // First and second derivatives are zero.
        let fd_off = NUM_BANDS;
        let sd_off = NUM_BANDS + NUM_LOWER_BANDS;
        for i in 0..NUM_LOWER_BANDS {
            assert!(feature_vector_last[fd_off + i].abs() < 1e-6);
            assert!(feature_vector_last[sd_off + i].abs() < 1e-6);
        }

        // Variability is unchanged.
        let var_idx = NUM_BANDS + 3 * NUM_LOWER_BANDS;
        assert!((feature_vector[var_idx] - feature_vector_last[var_idx]).abs() < 1e-6);
    }
}
