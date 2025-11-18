use crate::audio_processing::aec3::aec3_common::FFT_LENGTH_BY_2_PLUS_1;

pub trait NearendDetector: Send {
    fn is_nearend_state(&self) -> bool;
    fn update(
        &mut self,
        nearend_spectrum: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        residual_echo_spectrum: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        comfort_noise_spectrum: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        initial_state: bool,
    );
}
