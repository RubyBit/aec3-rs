use std::sync::Arc;

use rustfft::{Fft, FftPlanner, num_complex::Complex32};

use super::aec3_common::{FFT_LENGTH, FFT_LENGTH_BY_2};
use super::fft_data::FftData;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Window {
    Rectangular,
    Hanning,
    SqrtHanning,
}

pub struct Aec3Fft {
    fft: Arc<dyn Fft<f32>>,
    ifft: Arc<dyn Fft<f32>>,
}

impl Default for Aec3Fft {
    fn default() -> Self {
        Self::new()
    }
}

// TODO: Since aec3 behaviour is equivalent to WebRTC's implementation at this point, consider redoing fft without this packing of complex
// values into real arrays, and instead just use the Complex32 arrays directly. This would simplify the code and reduce the chance of bugs in packing/unpacking.
// This would require significant refactoring of every place that uses Aec3Fft and FftData.
impl Aec3Fft {
    pub fn new() -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FFT_LENGTH);
        let ifft = planner.plan_fft_inverse(FFT_LENGTH);
        Self { fft, ifft }
    }

    pub fn fft(&self, x: &mut [f32; FFT_LENGTH], fft_data: &mut FftData) {
        let mut spectrum = [Complex32::new(0.0, 0.0); FFT_LENGTH];
        for (dst, &sample) in spectrum.iter_mut().zip(x.iter()) {
            *dst = Complex32::new(sample, 0.0);
        }
        self.fft.process(&mut spectrum);
        copy_from_complex_spectrum(fft_data, &spectrum);
        pack_complex_to_real_array(&spectrum, x);
    }

    pub fn ifft(&self, fft_data: &FftData, x: &mut [f32; FFT_LENGTH]) {
        let mut spectrum = [Complex32::new(0.0, 0.0); FFT_LENGTH];
        expand_fft_data(fft_data, &mut spectrum);
        self.ifft.process(&mut spectrum);
        for (dst, value) in x.iter_mut().zip(spectrum.iter()) {
            *dst = value.re * 0.5;
        }
    }

    pub fn zero_padded_fft(&self, input: &[f32], window: Window, fft_data: &mut FftData) {
        assert_eq!(input.len(), FFT_LENGTH_BY_2);
        let mut data = [0.0f32; FFT_LENGTH];
        match window {
            Window::Rectangular => {
                data[FFT_LENGTH_BY_2..].copy_from_slice(input);
            }
            Window::Hanning => {
                for (dst, (&sample, &win)) in data[FFT_LENGTH_BY_2..]
                    .iter_mut()
                    .zip(input.iter().zip(HANNING_64.iter()))
                {
                    *dst = sample * win;
                }
            }
            Window::SqrtHanning => {
                panic!("Window::SqrtHanning is not supported for zero padded FFTs");
            }
        }
        self.fft(&mut data, fft_data);
    }

    pub fn padded_fft(&self, x: &[f32], x_old: &[f32], fft_data: &mut FftData) {
        self.padded_fft_with_window(x, x_old, Window::Rectangular, fft_data);
    }

    pub fn padded_fft_with_window(
        &self,
        x: &[f32],
        x_old: &[f32],
        window: Window,
        fft_data: &mut FftData,
    ) {
        assert_eq!(x.len(), FFT_LENGTH_BY_2);
        assert_eq!(x_old.len(), FFT_LENGTH_BY_2);
        let mut data = [0.0f32; FFT_LENGTH];
        match window {
            Window::Rectangular => {
                data[..FFT_LENGTH_BY_2].copy_from_slice(x_old);
                data[FFT_LENGTH_BY_2..].copy_from_slice(x);
            }
            Window::Hanning => {
                panic!("Window::Hanning is not supported for padded FFTs");
            }
            Window::SqrtHanning => {
                for (dst, (&sample, &win)) in data[..FFT_LENGTH_BY_2]
                    .iter_mut()
                    .zip(x_old.iter().zip(SQRT_HANNING_128[..FFT_LENGTH_BY_2].iter()))
                {
                    *dst = sample * win;
                }
                for (dst, (&sample, &win)) in data[FFT_LENGTH_BY_2..]
                    .iter_mut()
                    .zip(x.iter().zip(SQRT_HANNING_128[FFT_LENGTH_BY_2..].iter()))
                {
                    *dst = sample * win;
                }
            }
        }
        self.fft(&mut data, fft_data);
    }
}

const HANNING_64: [f32; FFT_LENGTH_BY_2] = [
    0.0, 0.00248461, 0.00991376, 0.0222136, 0.03926189, 0.06088921, 0.08688061, 0.11697778,
    0.15088159, 0.1882551, 0.22872687, 0.27189467, 0.31732949, 0.36457977, 0.41317591, 0.46263495,
    0.51246535, 0.56217185, 0.61126047, 0.65924333, 0.70564355, 0.75, 0.79187184, 0.83084292,
    0.86652594, 0.89856625, 0.92664544, 0.95048443, 0.96984631, 0.98453864, 0.99441541, 0.99937846,
    0.99937846, 0.99441541, 0.98453864, 0.96984631, 0.95048443, 0.92664544, 0.89856625, 0.86652594,
    0.83084292, 0.79187184, 0.75, 0.70564355, 0.65924333, 0.61126047, 0.56217185, 0.51246535,
    0.46263495, 0.41317591, 0.36457977, 0.31732949, 0.27189467, 0.22872687, 0.1882551, 0.15088159,
    0.11697778, 0.08688061, 0.06088921, 0.03926189, 0.0222136, 0.00991376, 0.00248461, 0.0,
];

const SQRT_HANNING_128: [f32; FFT_LENGTH] = [
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

fn copy_from_complex_spectrum(dst: &mut FftData, spectrum: &[Complex32; FFT_LENGTH]) {
    dst.re[0] = spectrum[0].re;
    dst.im[0] = 0.0;
    dst.re[FFT_LENGTH_BY_2] = spectrum[FFT_LENGTH_BY_2].re;
    dst.im[FFT_LENGTH_BY_2] = 0.0;
    for k in 1..FFT_LENGTH_BY_2 {
        dst.re[k] = spectrum[k].re;
        dst.im[k] = spectrum[k].im;
    }
}

fn pack_complex_to_real_array(spectrum: &[Complex32; FFT_LENGTH], dest: &mut [f32; FFT_LENGTH]) {
    dest[0] = spectrum[0].re;
    dest[1] = spectrum[FFT_LENGTH_BY_2].re;
    let mut idx = 2;
    for k in 1..FFT_LENGTH_BY_2 {
        dest[idx] = spectrum[k].re;
        dest[idx + 1] = spectrum[k].im;
        idx += 2;
    }
}

fn expand_fft_data(src: &FftData, spectrum: &mut [Complex32; FFT_LENGTH]) {
    spectrum[0] = Complex32::new(src.re[0], 0.0);
    spectrum[FFT_LENGTH_BY_2] = Complex32::new(src.re[FFT_LENGTH_BY_2], 0.0);
    for k in 1..FFT_LENGTH_BY_2 {
        let re = src.re[k];
        let im = src.im[k];
        spectrum[k] = Complex32::new(re, im);
        spectrum[FFT_LENGTH - k] = Complex32::new(re, -im);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_processing::aec3::aec3_common::{FFT_LENGTH, FFT_LENGTH_BY_2};

    #[test]
    fn fft_matches_reference_cases() {
        let fft = Aec3Fft::default();
        let mut fft_data = FftData::default();
        let mut x = [0.0f32; FFT_LENGTH];

        fft.fft(&mut x, &mut fft_data);
        assert!(fft_data.re.iter().all(|&v| v.abs() < 1e-6));
        assert!(fft_data.im.iter().all(|&v| v.abs() < 1e-6));

        x.fill(0.0);
        x[0] = 1.0;
        fft.fft(&mut x, &mut fft_data);
        for k in 0..=FFT_LENGTH_BY_2 {
            assert!((fft_data.re[k] - 1.0).abs() < 1e-4);
            assert!(fft_data.im[k].abs() < 1e-4);
        }

        x.fill(1.0);
        fft.fft(&mut x, &mut fft_data);
        assert!((fft_data.re[0] - 128.0).abs() < 1e-3);
        for k in 1..=FFT_LENGTH_BY_2 {
            assert!(fft_data.re[k].abs() < 1e-3);
            assert!(fft_data.im[k].abs() < 1e-3);
        }
    }

    #[test]
    fn ifft_matches_reference_cases() {
        let fft = Aec3Fft::default();
        let mut fft_data = FftData::default();
        let mut x = [0.0f32; FFT_LENGTH];

        fft.ifft(&fft_data, &mut x);
        assert!(x.iter().all(|&v| v.abs() < 1e-6));

        fft_data.re.fill(1.0);
        fft_data.im.fill(0.0);
        fft.ifft(&fft_data, &mut x);
        assert!((x[0] - 64.0).abs() < 1e-3);
        for sample in x.iter().skip(1) {
            assert!(sample.abs() < 1e-3);
        }

        fft_data.re.fill(0.0);
        fft_data.im.fill(0.0);
        fft_data.re[0] = 128.0;
        fft.ifft(&fft_data, &mut x);
        for sample in x.iter() {
            assert!((sample - 64.0).abs() < 1e-3);
        }
    }

    #[test]
    fn fft_and_ifft_roundtrip_scales_like_reference() {
        let fft = Aec3Fft::default();
        let mut fft_data = FftData::default();
        let mut x = [0.0f32; FFT_LENGTH];
        let mut reference = [0.0f32; FFT_LENGTH];
        let mut value = 0.0f32;

        for _ in 0..20 {
            for j in 0..FFT_LENGTH {
                x[j] = value;
                reference[j] = value * 64.0;
                value += 1.0;
            }
            fft.fft(&mut x, &mut fft_data);
            fft.ifft(&fft_data, &mut x);
            for (a, b) in x.iter().zip(reference.iter()) {
                assert!((a - b).abs() < 1e-2);
            }
        }
    }

    #[test]
    fn zero_padded_fft_matches_reference_behavior() {
        let fft = Aec3Fft::default();
        let mut fft_data = FftData::default();
        let mut x_in = [0.0f32; FFT_LENGTH_BY_2];
        let mut x_out = [0.0f32; FFT_LENGTH];
        let mut x_ref = [0.0f32; FFT_LENGTH];
        let mut v = 0.0f32;

        for _ in 0..20 {
            for j in 0..FFT_LENGTH_BY_2 {
                x_in[j] = v;
                x_ref[j + FFT_LENGTH_BY_2] = x_in[j] * 64.0;
                v += 1.0;
            }

            fft.zero_padded_fft(&x_in, Window::Rectangular, &mut fft_data);
            fft.ifft(&fft_data, &mut x_out);
            for (a, b) in x_out.iter().zip(x_ref.iter()) {
                assert!((a - b).abs() < 1e-1);
            }

            x_ref.fill(0.0);
        }
    }

    #[test]
    fn padded_fft_matches_reference_behavior() {
        let fft = Aec3Fft::default();
        let mut fft_data = FftData::default();
        let mut x_in = [0.0f32; FFT_LENGTH_BY_2];
        let mut x_out = [0.0f32; FFT_LENGTH];
        let mut x_old = [0.0f32; FFT_LENGTH_BY_2];
        let mut x_old_ref = [0.0f32; FFT_LENGTH_BY_2];
        let mut x_ref = [0.0f32; FFT_LENGTH];
        let mut v = 0.0f32;

        for _ in 0..20 {
            for j in 0..FFT_LENGTH_BY_2 {
                x_in[j] = v;
                v += 1.0;
            }

            x_ref[..FFT_LENGTH_BY_2].copy_from_slice(&x_old);
            x_ref[FFT_LENGTH_BY_2..].copy_from_slice(&x_in);
            for sample in x_ref.iter_mut() {
                *sample *= 64.0;
            }
            x_old_ref.copy_from_slice(&x_in);

            fft.padded_fft(&x_in, &x_old, &mut fft_data);
            x_old.copy_from_slice(&x_in);
            fft.ifft(&fft_data, &mut x_out);

            for (a, b) in x_out.iter().zip(x_ref.iter()) {
                assert!((a - b).abs() < 1e-1);
            }

            assert_eq!(x_old, x_old_ref);
            x_ref.fill(0.0);
        }
    }
}
