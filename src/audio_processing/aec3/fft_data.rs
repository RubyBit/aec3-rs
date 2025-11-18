use super::aec3_common::{Aec3Optimization, FFT_LENGTH, FFT_LENGTH_BY_2, FFT_LENGTH_BY_2_PLUS_1};

/// Holds the positive-frequency bins (including DC and Nyquist) for a
/// 128-point real-valued FFT.
#[derive(Clone, Debug, PartialEq)]
pub struct FftData {
    pub re: [f32; FFT_LENGTH_BY_2_PLUS_1],
    pub im: [f32; FFT_LENGTH_BY_2_PLUS_1],
}

impl Default for FftData {
    fn default() -> Self {
        Self {
            re: [0.0; FFT_LENGTH_BY_2_PLUS_1],
            im: [0.0; FFT_LENGTH_BY_2_PLUS_1],
        }
    }
}

impl FftData {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn assign(&mut self, src: &FftData) {
        self.re = src.re;
        self.im = src.im;
        self.im[0] = 0.0;
        self.im[FFT_LENGTH_BY_2] = 0.0;
    }

    pub fn clear(&mut self) {
        self.re.fill(0.0);
        self.im.fill(0.0);
    }

    pub fn spectrum(&self, optimization: Aec3Optimization, power_spectrum: &mut [f32]) {
        assert_eq!(power_spectrum.len(), FFT_LENGTH_BY_2_PLUS_1);
        match optimization {
            Aec3Optimization::Sse2 | Aec3Optimization::Neon | Aec3Optimization::None => {
                for (dst, (&re, &im)) in power_spectrum
                    .iter_mut()
                    .zip(self.re.iter().zip(self.im.iter()))
                {
                    *dst = re * re + im * im;
                }
            }
        }
    }

    pub fn copy_from_packed_array(&mut self, packed: &[f32; FFT_LENGTH]) {
        self.re[0] = packed[0];
        self.re[FFT_LENGTH_BY_2] = packed[1];
        self.im[0] = 0.0;
        self.im[FFT_LENGTH_BY_2] = 0.0;

        let mut src_idx = 2;
        for k in 1..FFT_LENGTH_BY_2 {
            self.re[k] = packed[src_idx];
            self.im[k] = packed[src_idx + 1];
            src_idx += 2;
        }
    }

    pub fn copy_to_packed_array(&self, packed: &mut [f32; FFT_LENGTH]) {
        packed[0] = self.re[0];
        packed[1] = self.re[FFT_LENGTH_BY_2];
        let mut dst_idx = 2;
        for k in 1..FFT_LENGTH_BY_2 {
            packed[dst_idx] = self.re[k];
            packed[dst_idx + 1] = self.im[k];
            dst_idx += 2;
        }
    }
}
