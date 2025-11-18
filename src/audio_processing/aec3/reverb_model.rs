use super::aec3_common::FFT_LENGTH_BY_2_PLUS_1;

/// Simple exponential reverberation model applied to render spectra.
pub struct ReverbModel {
    reverb: [f32; FFT_LENGTH_BY_2_PLUS_1],
}

impl ReverbModel {
    pub fn new() -> Self {
        let mut model = Self {
            reverb: [0.0; FFT_LENGTH_BY_2_PLUS_1],
        };
        model.reset();
        model
    }

    pub fn reset(&mut self) {
        self.reverb.fill(0.0);
    }

    pub fn reverb(&self) -> &[f32; FFT_LENGTH_BY_2_PLUS_1] {
        &self.reverb
    }

    pub fn update_reverb_no_freq_shaping(
        &mut self,
        power_spectrum: &[f32],
        power_spectrum_scaling: f32,
        reverb_decay: f32,
    ) {
        if reverb_decay <= 0.0 {
            return;
        }
        assert_eq!(power_spectrum.len(), FFT_LENGTH_BY_2_PLUS_1);
        for (dst, &src) in self.reverb.iter_mut().zip(power_spectrum.iter()) {
            *dst = (*dst + src * power_spectrum_scaling) * reverb_decay;
        }
    }

    pub fn update_reverb(
        &mut self,
        power_spectrum: &[f32],
        power_spectrum_scaling: &[f32],
        reverb_decay: f32,
    ) {
        if reverb_decay <= 0.0 {
            return;
        }
        assert_eq!(power_spectrum.len(), FFT_LENGTH_BY_2_PLUS_1);
        assert_eq!(power_spectrum_scaling.len(), FFT_LENGTH_BY_2_PLUS_1);
        for ((dst, &src), &scale) in self
            .reverb
            .iter_mut()
            .zip(power_spectrum.iter())
            .zip(power_spectrum_scaling.iter())
        {
            *dst = (*dst + src * scale) * reverb_decay;
        }
    }
}

impl Default for ReverbModel {
    fn default() -> Self {
        Self::new()
    }
}
