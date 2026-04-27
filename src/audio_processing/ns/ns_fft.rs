//! 256-point FFT wrapper for the noise suppressor.

use std::sync::Arc;

use rustfft::{Fft, FftPlanner, num_complex::Complex32};

use super::ns_common::{FFT_SIZE, FFT_SIZE_BY_2_PLUS_1};

pub struct NrFft {
    fft: Arc<dyn Fft<f32>>,
    ifft: Arc<dyn Fft<f32>>,
}

impl Default for NrFft {
    fn default() -> Self {
        Self::new()
    }
}
// TODO: See aec3_fft.rs for a note on potential refactoring. Ns would be an easier candidate due to less code.
impl NrFft {
    pub fn new() -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);
        let ifft = planner.plan_fft_inverse(FFT_SIZE);
        Self { fft, ifft }
    }

    pub fn fft(
        &self,
        time_data: &mut [f32; FFT_SIZE],
        real: &mut [f32; FFT_SIZE],
        imag: &mut [f32; FFT_SIZE],
    ) {
        let mut spectrum = [Complex32::new(0.0, 0.0); FFT_SIZE];
        for (dst, &sample) in spectrum.iter_mut().zip(time_data.iter()) {
            *dst = Complex32::new(sample, 0.0);
        }

        self.fft.process(&mut spectrum);

        imag[0] = 0.0;
        real[0] = spectrum[0].re;

        imag[FFT_SIZE_BY_2_PLUS_1 - 1] = 0.0;
        real[FFT_SIZE_BY_2_PLUS_1 - 1] = spectrum[FFT_SIZE / 2].re;

        for i in 1..(FFT_SIZE_BY_2_PLUS_1 - 1) {
            real[i] = spectrum[i].re;
            imag[i] = spectrum[i].im;
        }
    }

    pub fn ifft(&self, real: &[f32], imag: &[f32], time_data: &mut [f32]) {
        assert!(real.len() >= FFT_SIZE_BY_2_PLUS_1);
        assert!(imag.len() >= FFT_SIZE_BY_2_PLUS_1);
        assert!(time_data.len() >= FFT_SIZE);

        let mut spectrum = [Complex32::new(0.0, 0.0); FFT_SIZE];
        spectrum[0] = Complex32::new(real[0], 0.0);
        spectrum[FFT_SIZE / 2] = Complex32::new(real[FFT_SIZE_BY_2_PLUS_1 - 1], 0.0);

        for i in 1..(FFT_SIZE_BY_2_PLUS_1 - 1) {
            spectrum[i] = Complex32::new(real[i], imag[i]);
            spectrum[FFT_SIZE - i] = Complex32::new(real[i], -imag[i]);
        }

        self.ifft.process(&mut spectrum);

        // Match the C++ scaling (2 / N) for Ooura real FFT inverse path.
        let scaling = 2.0 / FFT_SIZE as f32;
        for (dst, value) in time_data.iter_mut().zip(spectrum.iter()) {
            *dst = value.re * scaling;
        }
    }
}
