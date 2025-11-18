use crate::audio_processing::aec3::aec3_common::{BLOCK_SIZE, FFT_LENGTH_BY_2_PLUS_1};
use crate::audio_processing::aec3::fft_data::FftData;

/// Stores the values returned from the echo subtractor for a single capture channel.
#[derive(Clone)]
pub struct SubtractorOutput {
    pub s_main: [f32; BLOCK_SIZE],
    pub s_shadow: [f32; BLOCK_SIZE],
    pub e_main: [f32; BLOCK_SIZE],
    pub e_shadow: [f32; BLOCK_SIZE],
    pub e_main_fft: FftData,
    pub e2_main_spectrum: [f32; FFT_LENGTH_BY_2_PLUS_1],
    pub e2_shadow_spectrum: [f32; FFT_LENGTH_BY_2_PLUS_1],
    pub s2_main: f32,
    pub s2_shadow: f32,
    pub e2_main: f32,
    pub e2_shadow: f32,
    pub y2: f32,
    pub s_main_max_abs: f32,
    pub s_shadow_max_abs: f32,
}

impl SubtractorOutput {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        self.s_main.fill(0.0);
        self.s_shadow.fill(0.0);
        self.e_main.fill(0.0);
        self.e_shadow.fill(0.0);
        self.e_main_fft.re.fill(0.0);
        self.e_main_fft.im.fill(0.0);
        self.e2_main_spectrum.fill(0.0);
        self.e2_shadow_spectrum.fill(0.0);
        self.e2_main = 0.0;
        self.e2_shadow = 0.0;
        self.s2_main = 0.0;
        self.s2_shadow = 0.0;
        self.y2 = 0.0;
        self.s_main_max_abs = 0.0;
        self.s_shadow_max_abs = 0.0;
    }

    pub fn compute_metrics(&mut self, y: &[f32]) {
        assert_eq!(y.len(), BLOCK_SIZE);
        let sum_of_squares = |acc: f32, sample: &f32| acc + sample * sample;
        self.y2 = y.iter().fold(0.0, sum_of_squares);
        self.e2_main = self.e_main.iter().fold(0.0, sum_of_squares);
        self.e2_shadow = self.e_shadow.iter().fold(0.0, sum_of_squares);
        self.s2_main = self.s_main.iter().fold(0.0, sum_of_squares);
        self.s2_shadow = self.s_shadow.iter().fold(0.0, sum_of_squares);
        self.s_main_max_abs = self
            .s_main
            .iter()
            .fold(0.0, |acc, &sample| acc.max(sample.abs()));
        self.s_shadow_max_abs = self
            .s_shadow
            .iter()
            .fold(0.0, |acc, &sample| acc.max(sample.abs()));
    }
}

impl Default for SubtractorOutput {
    fn default() -> Self {
        Self {
            s_main: [0.0; BLOCK_SIZE],
            s_shadow: [0.0; BLOCK_SIZE],
            e_main: [0.0; BLOCK_SIZE],
            e_shadow: [0.0; BLOCK_SIZE],
            e_main_fft: FftData::default(),
            e2_main_spectrum: [0.0; FFT_LENGTH_BY_2_PLUS_1],
            e2_shadow_spectrum: [0.0; FFT_LENGTH_BY_2_PLUS_1],
            s2_main: 0.0,
            s2_shadow: 0.0,
            e2_main: 0.0,
            e2_shadow: 0.0,
            y2: 0.0,
            s_main_max_abs: 0.0,
            s_shadow_max_abs: 0.0,
        }
    }
}
