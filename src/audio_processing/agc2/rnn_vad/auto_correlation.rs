//! Auto-correlation on the decimated pitch buffer.

use std::sync::Arc;

use num_complex::Complex32;
use rustfft::{Fft, FftPlanner};

use crate::audio_processing::agc2::rnn_vad::common::{
    BUF_SIZE_12_KHZ, FRAME_SIZE_20_MS_12_KHZ, MAX_PITCH_12_KHZ, NUM_LAGS_12_KHZ,
};

const AUTO_CORRELATION_FFT_ORDER: usize = 9; // Length-512 FFT.
const FFT_FRAME_SIZE: usize = 1 << AUTO_CORRELATION_FFT_ORDER;
const CONVOLUTION_LENGTH: usize = BUF_SIZE_12_KHZ - MAX_PITCH_12_KHZ;

const _: () = {
    assert!(FFT_FRAME_SIZE > NUM_LAGS_12_KHZ + BUF_SIZE_12_KHZ - MAX_PITCH_12_KHZ);
    assert!(CONVOLUTION_LENGTH == FRAME_SIZE_20_MS_12_KHZ);
};

/// Class to compute auto-correlation on the pitch buffer for the target pitch
/// interval.
pub struct AutoCorrelationCalculator {
    fft_forward: Arc<dyn Fft<f32>>,
    fft_inverse: Arc<dyn Fft<f32>>,
    tmp: Vec<Complex32>,
    x: Vec<Complex32>,
    h: Vec<Complex32>,
}

impl Default for AutoCorrelationCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl AutoCorrelationCalculator {
    pub fn new() -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let fft_forward = planner.plan_fft_forward(FFT_FRAME_SIZE);
        let fft_inverse = planner.plan_fft_inverse(FFT_FRAME_SIZE);

        Self {
            fft_forward,
            fft_inverse,
            tmp: vec![Complex32::new(0.0, 0.0); FFT_FRAME_SIZE],
            x: vec![Complex32::new(0.0, 0.0); FFT_FRAME_SIZE],
            h: vec![Complex32::new(0.0, 0.0); FFT_FRAME_SIZE],
        }
    }

    /// Computes auto-correlation coefficients for target pitch interval.
    /// `auto_corr` indexes are inverted lags.
    pub fn compute_on_pitch_buffer(
        &mut self,
        pitch_buf: &[f32; BUF_SIZE_12_KHZ],
        auto_corr: &mut [f32; NUM_LAGS_12_KHZ],
    ) {
        // Compute FFT for reversed reference frame:
        // pitch_buf[-CONVOLUTION_LENGTH:]
        self.tmp.fill(Complex32::new(0.0, 0.0));
        for i in 0..CONVOLUTION_LENGTH {
            let src = BUF_SIZE_12_KHZ - 1 - i;
            self.tmp[i].re = pitch_buf[src];
        }
        self.h.copy_from_slice(&self.tmp);
        self.fft_forward.process(&mut self.h);

        // Compute FFT for sliding-frames chunk:
        // pitch_buf[:CONVOLUTION_LENGTH + NUM_LAGS_12_KHZ]
        self.tmp.fill(Complex32::new(0.0, 0.0));
        let chunk_len = CONVOLUTION_LENGTH + NUM_LAGS_12_KHZ;
        for i in 0..chunk_len {
            self.tmp[i].re = pitch_buf[i];
        }
        self.x.copy_from_slice(&self.tmp);
        self.fft_forward.process(&mut self.x);

        // Frequency-domain convolution.
        for i in 0..FFT_FRAME_SIZE {
            self.tmp[i] = self.x[i] * self.h[i];
        }

        self.fft_inverse.process(&mut self.tmp);

        let scaling = 1.0f32 / FFT_FRAME_SIZE as f32;
        for v in &mut self.tmp {
            *v *= scaling;
        }

        // Extract auto-correlation coefficients.
        let start = CONVOLUTION_LENGTH - 1;
        for i in 0..NUM_LAGS_12_KHZ {
            auto_corr[i] = self.tmp[start + i].re;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_processing::agc2::rnn_vad::pitch_search_internal::decimate_2x;
    use crate::audio_processing::agc2::rnn_vad::test_data::{PitchTestData, expect_near_absolute};

    fn reference_auto_correlation(
        pitch_buf: &[f32; BUF_SIZE_12_KHZ],
        out: &mut [f32; NUM_LAGS_12_KHZ],
    ) {
        // This follows the same x/y framing from the C++ description.
        // x: fixed tail frame of length CONVOLUTION_LENGTH.
        // y_i: sliding window of same length, i in [0, NUM_LAGS_12_KHZ).
        let x_start = BUF_SIZE_12_KHZ - CONVOLUTION_LENGTH;
        for lag_inv in 0..NUM_LAGS_12_KHZ {
            let mut acc = 0.0f32;
            for n in 0..CONVOLUTION_LENGTH {
                acc += pitch_buf[lag_inv + n] * pitch_buf[x_start + n];
            }
            out[lag_inv] = acc;
        }
    }

    #[test]
    fn check_auto_correlation_on_constant_pitch_buffer() {
        let pitch_buf_decimated = [1.0f32; BUF_SIZE_12_KHZ];
        let mut computed_output = [0.0f32; NUM_LAGS_12_KHZ];

        let mut auto_corr_calculator = AutoCorrelationCalculator::new();
        auto_corr_calculator.compute_on_pitch_buffer(&pitch_buf_decimated, &mut computed_output);

        let expected = FRAME_SIZE_20_MS_12_KHZ as f32;
        for &v in &computed_output {
            assert!((v - expected).abs() < 4e-5, "v={v}, expected={expected}");
        }
    }

    #[test]
    fn pitch_buffer_auto_correlation_within_tolerance_against_reference() {
        let mut pitch_buf_decimated = [0.0f32; BUF_SIZE_12_KHZ];
        for (i, v) in pitch_buf_decimated.iter_mut().enumerate() {
            let t = i as f32;
            *v = (0.031 * t).sin() + 0.37 * (0.067 * t).cos() + 0.13 * ((t * 0.011).sin());
        }

        let mut computed_output = [0.0f32; NUM_LAGS_12_KHZ];
        let mut expected_output = [0.0f32; NUM_LAGS_12_KHZ];

        let mut auto_corr_calculator = AutoCorrelationCalculator::new();
        auto_corr_calculator.compute_on_pitch_buffer(&pitch_buf_decimated, &mut computed_output);
        reference_auto_correlation(&pitch_buf_decimated, &mut expected_output);

        for (expected, computed) in expected_output.iter().zip(computed_output.iter()) {
            assert!(
                (expected - computed).abs() < 3e-3,
                "expected={expected}, computed={computed}"
            );
        }
    }

    #[test]
    fn pitch_buffer_auto_correlation_within_tolerance_fixture() {
        let test_data = PitchTestData::new();
        let mut pitch_buf_decimated = [0.0f32; BUF_SIZE_12_KHZ];
        decimate_2x(test_data.pitch_buffer_24khz_view(), &mut pitch_buf_decimated);

        let mut computed_output = [0.0f32; NUM_LAGS_12_KHZ];
        let mut auto_corr_calculator = AutoCorrelationCalculator::new();
        auto_corr_calculator.compute_on_pitch_buffer(&pitch_buf_decimated, &mut computed_output);

        expect_near_absolute(
            test_data.auto_correlation_12khz_view(),
            &computed_output,
            3e-3,
        );
    }
}
