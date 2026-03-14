//! Feature extraction pipeline for feeding the RNN-VAD.

use crate::audio_processing::agc2::biquad_filter::{BiQuadFilter, Config as BiQuadConfig};
use crate::audio_processing::agc2::cpu_features::AvailableCpuFeatures;
use crate::audio_processing::agc2::rnn_vad::common::{
    BUF_SIZE_24_KHZ, FEATURE_VECTOR_SIZE, FRAME_SIZE_10_MS_24_KHZ, FRAME_SIZE_20_MS_24_KHZ,
    MAX_PITCH_24_KHZ, NUM_BANDS, NUM_LOWER_BANDS,
};
use crate::audio_processing::agc2::rnn_vad::lp_residual::{
    compute_and_post_process_lpc_coefficients, compute_lp_residual, NUM_LPC_COEFFICIENTS,
};
use crate::audio_processing::agc2::rnn_vad::pitch_search::PitchEstimator;
use crate::audio_processing::agc2::rnn_vad::sequence_buffer::SequenceBuffer;
use crate::audio_processing::agc2::rnn_vad::spectral_features::SpectralFeaturesExtractor;

// Computed as scipy.signal.butter(N=2, Wn=60/24000, btype='highpass').
const HPF_CONFIG_24K: BiQuadConfig = BiQuadConfig {
    b: [0.99446179, -1.98892358, 0.99446179],
    a: [-1.98889291, 0.98895425],
};

/// Feature extractor to feed the VAD RNN.
pub struct FeaturesExtractor {
    use_high_pass_filter: bool,
    hpf: BiQuadFilter,
    pitch_buf_24khz: SequenceBuffer<
        f32,
        BUF_SIZE_24_KHZ,
        FRAME_SIZE_10_MS_24_KHZ,
        FRAME_SIZE_20_MS_24_KHZ,
    >,
    lp_residual: [f32; BUF_SIZE_24_KHZ],
    pitch_estimator: PitchEstimator,
    spectral_features_extractor: SpectralFeaturesExtractor,
    pitch_period_48khz: i32,
}

impl FeaturesExtractor {
    pub fn new(cpu_features: AvailableCpuFeatures) -> Self {
        let mut s = Self {
            use_high_pass_filter: false,
            hpf: BiQuadFilter::new(HPF_CONFIG_24K),
            pitch_buf_24khz: SequenceBuffer::new(),
            lp_residual: [0.0; BUF_SIZE_24_KHZ],
            pitch_estimator: PitchEstimator::new(cpu_features),
            spectral_features_extractor: SpectralFeaturesExtractor::new(),
            pitch_period_48khz: 0,
        };
        s.reset();
        s
    }

    pub fn reset(&mut self) {
        self.pitch_buf_24khz.reset();
        self.spectral_features_extractor.reset();
        if self.use_high_pass_filter {
            self.hpf.reset();
        }
    }

    /// Analyzes samples, computes feature vector, and returns `true` if silence
    /// is detected. On silence, the feature vector is partially written and must
    /// not be fed to the RNN.
    pub fn check_silence_compute_features(
        &mut self,
        samples: &[f32; FRAME_SIZE_10_MS_24_KHZ],
        feature_vector: &mut [f32; FEATURE_VECTOR_SIZE],
    ) -> bool {
        // Pre-processing and pitch buffer push.
        if self.use_high_pass_filter {
            let mut samples_filtered = [0.0f32; FRAME_SIZE_10_MS_24_KHZ];
            self.hpf.process(samples, &mut samples_filtered);
            self.pitch_buf_24khz.push(&samples_filtered);
        } else {
            self.pitch_buf_24khz.push(samples);
        }

        let pitch_buf_view = self.pitch_buf_24khz.get_buffer_view();

        // LP residual.
        let mut lpc_coeffs = [0.0f32; NUM_LPC_COEFFICIENTS];
        compute_and_post_process_lpc_coefficients(pitch_buf_view, &mut lpc_coeffs);
        compute_lp_residual(&lpc_coeffs, pitch_buf_view, &mut self.lp_residual);

        // Pitch estimate normalized feature.
        self.pitch_period_48khz = self.pitch_estimator.estimate(&self.lp_residual);
        feature_vector[FEATURE_VECTOR_SIZE - 2] = 0.01 * (self.pitch_period_48khz as f32 - 300.0);

        // Lagged frame according to estimated pitch.
        debug_assert!(self.pitch_period_48khz / 2 <= MAX_PITCH_24_KHZ as i32);
        let lagged_start = MAX_PITCH_24_KHZ - (self.pitch_period_48khz as usize / 2);
        let lagged_end = lagged_start + FRAME_SIZE_20_MS_24_KHZ;

        let lagged_frame: [f32; FRAME_SIZE_20_MS_24_KHZ] = pitch_buf_view[lagged_start..lagged_end]
            .try_into()
            .expect("lagged frame shape must match");

        let reference_frame: [f32; FRAME_SIZE_20_MS_24_KHZ] = self
            .pitch_buf_24khz
            .get_most_recent_values_view()
            .try_into()
            .expect("reference frame shape must match");

        let (first, rem) = feature_vector.split_at_mut(NUM_LOWER_BANDS);
        let (higher_bands, rem) = rem.split_at_mut(NUM_BANDS - NUM_LOWER_BANDS);
        let (first_derivative, rem) = rem.split_at_mut(NUM_LOWER_BANDS);
        let (second_derivative, rem) = rem.split_at_mut(NUM_LOWER_BANDS);
        let (bands_cross_corr, rem) = rem.split_at_mut(NUM_LOWER_BANDS);
        // rem[0] is the normalized pitch-period slot already written above.
        // rem[1] is the variability slot.
        let variability = &mut rem[1];

        self.spectral_features_extractor.check_silence_compute_features(
            &reference_frame,
            &lagged_frame,
            higher_bands.try_into().expect("higher bands shape"),
            first.try_into().expect("average shape"),
            first_derivative.try_into().expect("first derivative shape"),
            second_derivative.try_into().expect("second derivative shape"),
            bands_cross_corr.try_into().expect("cross corr shape"),
            variability,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_processing::agc2::cpu_features::get_available_cpu_features;
    use crate::audio_processing::agc2::rnn_vad::common::{PI, SAMPLE_RATE_24_KHZ};

    const NUM_TEST_DATA_FRAMES: usize =
        (BUF_SIZE_24_KHZ + FRAME_SIZE_10_MS_24_KHZ - 1) / FRAME_SIZE_10_MS_24_KHZ;
    const NUM_TEST_DATA_SIZE: usize = NUM_TEST_DATA_FRAMES * FRAME_SIZE_10_MS_24_KHZ;

    fn pitch_is_valid(pitch_hz: f32) -> bool {
        let pitch_period = SAMPLE_RATE_24_KHZ as f32 / pitch_hz;
        let initial_min_pitch_24khz = crate::audio_processing::agc2::rnn_vad::common::INITIAL_MIN_PITCH_24_KHZ as f32;
        let max_pitch_24khz = crate::audio_processing::agc2::rnn_vad::common::MAX_PITCH_24_KHZ as f32;
        initial_min_pitch_24khz <= pitch_period && pitch_period <= max_pitch_24khz
    }

    fn create_pure_tone(amplitude: f32, freq_hz: f32, dst: &mut [f32]) {
        for (i, s) in dst.iter_mut().enumerate() {
            *s = amplitude * (2.0 * PI as f32 * freq_hz * i as f32 / SAMPLE_RATE_24_KHZ as f32).sin();
        }
    }

    fn feed_test_data(
        features_extractor: &mut FeaturesExtractor,
        samples: &[f32],
        feature_vector: &mut [f32; FEATURE_VECTOR_SIZE],
    ) -> bool {
        let num_frames = samples.len() / FRAME_SIZE_10_MS_24_KHZ;
        let mut is_silence = true;
        for i in 0..num_frames {
            let start = i * FRAME_SIZE_10_MS_24_KHZ;
            let end = start + FRAME_SIZE_10_MS_24_KHZ;
            let frame: [f32; FRAME_SIZE_10_MS_24_KHZ] =
                samples[start..end].try_into().expect("frame shape");
            is_silence = features_extractor.check_silence_compute_features(&frame, feature_vector);
        }
        is_silence
    }

    #[test]
    fn feature_extraction_low_high_pitch() {
        let amplitude = 1000.0f32;
        let low_pitch_hz = 150.0f32;
        let high_pitch_hz = 250.0f32;
        assert!(pitch_is_valid(low_pitch_hz));
        assert!(pitch_is_valid(high_pitch_hz));

        let cpu_features = get_available_cpu_features();
        let mut features_extractor = FeaturesExtractor::new(cpu_features);
        let mut samples = vec![0.0f32; NUM_TEST_DATA_SIZE];
        let mut feature_vector = [0.0f32; FEATURE_VECTOR_SIZE];

        let pitch_feature_index = FEATURE_VECTOR_SIZE - 2;

        create_pure_tone(amplitude, low_pitch_hz, &mut samples);
        assert!(!feed_test_data(&mut features_extractor, &samples, &mut feature_vector));
        let high_pitch_period = feature_vector[pitch_feature_index];

        features_extractor.reset();
        create_pure_tone(amplitude, high_pitch_hz, &mut samples);
        assert!(!feed_test_data(&mut features_extractor, &samples, &mut feature_vector));
        let low_pitch_period = feature_vector[pitch_feature_index];

        assert!(
            low_pitch_period < high_pitch_period,
            "expected low_pitch_period < high_pitch_period, got low_pitch_period={}, high_pitch_period={}",
            low_pitch_period,
            high_pitch_period
        );
    }
}
