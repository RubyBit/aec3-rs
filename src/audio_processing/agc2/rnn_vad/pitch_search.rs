//! Pitch estimator built on top of pitch-search internals.

use crate::audio_processing::agc2::cpu_features::AvailableCpuFeatures;
use crate::audio_processing::agc2::rnn_vad::auto_correlation::AutoCorrelationCalculator;
use crate::audio_processing::agc2::rnn_vad::common::{
    BUF_SIZE_12_KHZ, BUF_SIZE_24_KHZ, MAX_PITCH_48_KHZ, NUM_LAGS_12_KHZ, REFINE_NUM_LAGS_24_KHZ,
};
use crate::audio_processing::agc2::rnn_vad::pitch_search_internal::{
    PitchInfo, compute_extended_pitch_period_48khz, compute_pitch_period_12khz,
    compute_pitch_period_48khz, compute_sliding_frame_square_energies_24khz, decimate_2x,
};

/// Pitch estimator.
pub struct PitchEstimator {
    cpu_features: AvailableCpuFeatures,
    last_pitch_48khz: PitchInfo,
    auto_corr_calculator: AutoCorrelationCalculator,
    y_energy_24khz: [f32; REFINE_NUM_LAGS_24_KHZ],
    pitch_buffer_12khz: [f32; BUF_SIZE_12_KHZ],
    auto_correlation_12khz: [f32; NUM_LAGS_12_KHZ],
}

impl PitchEstimator {
    pub fn new(cpu_features: AvailableCpuFeatures) -> Self {
        Self {
            cpu_features,
            last_pitch_48khz: PitchInfo {
                period: 0,
                strength: 0.0,
            },
            auto_corr_calculator: AutoCorrelationCalculator::new(),
            y_energy_24khz: [0.0; REFINE_NUM_LAGS_24_KHZ],
            pitch_buffer_12khz: [0.0; BUF_SIZE_12_KHZ],
            auto_correlation_12khz: [0.0; NUM_LAGS_12_KHZ],
        }
    }

    /// Returns the estimated pitch period at 48 kHz.
    pub fn estimate(&mut self, pitch_buffer: &[f32; BUF_SIZE_24_KHZ]) -> i32 {
        // Initial pitch search at 12 kHz.
        decimate_2x(pitch_buffer, &mut self.pitch_buffer_12khz);
        self.auto_corr_calculator
            .compute_on_pitch_buffer(&self.pitch_buffer_12khz, &mut self.auto_correlation_12khz);
        let mut pitch_periods = compute_pitch_period_12khz(
            &self.pitch_buffer_12khz,
            &self.auto_correlation_12khz,
            self.cpu_features,
        );

        // Adapt inverted lags from 12 kHz to 24 kHz.
        pitch_periods.best *= 2;
        pitch_periods.second_best *= 2;

        // Refine from 12 kHz to 48 kHz.
        compute_sliding_frame_square_energies_24khz(
            pitch_buffer,
            &mut self.y_energy_24khz,
            self.cpu_features,
        );

        let pitch_lag_48khz = compute_pitch_period_48khz(
            pitch_buffer,
            &self.y_energy_24khz,
            pitch_periods,
            self.cpu_features,
        );

        self.last_pitch_48khz = compute_extended_pitch_period_48khz(
            pitch_buffer,
            &self.y_energy_24khz,
            MAX_PITCH_48_KHZ as i32 - pitch_lag_48khz,
            self.last_pitch_48khz,
            self.cpu_features,
        );

        self.last_pitch_48khz.period
    }

    pub fn last_pitch_strength_for_testing(&self) -> f32 {
        self.last_pitch_48khz.strength
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_processing::agc2::cpu_features::{
        get_available_cpu_features, no_available_cpu_features,
    };
    use crate::audio_processing::agc2::rnn_vad::common::{MAX_PITCH_48_KHZ, MIN_PITCH_48_KHZ};
    use crate::audio_processing::agc2::rnn_vad::test_data::create_lp_residual_and_pitch_info_reader;

    #[test]
    fn pitch_estimate_is_within_valid_range() {
        let cpu = no_available_cpu_features();
        let mut estimator = PitchEstimator::new(cpu);
        let mut lp_residual = [0.0f32; BUF_SIZE_24_KHZ];
        for (i, v) in lp_residual.iter_mut().enumerate() {
            *v = (0.022 * i as f32).sin() + 0.19 * (0.055 * i as f32).cos();
        }

        let pitch_period = estimator.estimate(&lp_residual);
        assert!(pitch_period >= MIN_PITCH_48_KHZ as i32);
        assert!(pitch_period <= MAX_PITCH_48_KHZ as i32);
        assert!(estimator.last_pitch_strength_for_testing().is_finite());
    }

    #[test]
    fn pitch_estimator_is_deterministic_for_same_input() {
        let cpu = no_available_cpu_features();
        let mut estimator = PitchEstimator::new(cpu);
        let mut lp_residual = [0.0f32; BUF_SIZE_24_KHZ];
        for (i, v) in lp_residual.iter_mut().enumerate() {
            *v = (0.015 * i as f32).sin() + 0.11 * (0.083 * i as f32).cos();
        }

        let p1 = estimator.estimate(&lp_residual);
        let s1 = estimator.last_pitch_strength_for_testing();

        let mut estimator2 = PitchEstimator::new(cpu);
        let p2 = estimator2.estimate(&lp_residual);
        let s2 = estimator2.last_pitch_strength_for_testing();

        assert_eq!(p1, p2);
        assert!((s1 - s2).abs() < 1e-6, "s1={s1}, s2={s2}");
    }

    #[test]
    fn pitch_search_multi_frame_within_valid_range_and_deterministic() {
        let cpu = no_available_cpu_features();
        let mut estimator_a = PitchEstimator::new(cpu);
        let mut estimator_b = PitchEstimator::new(cpu);

        // C++ test uses up to 300 fixture frames; mirror that cadence with
        // deterministic synthetic frames.
        for frame_idx in 0..300 {
            let mut lp_residual = [0.0f32; BUF_SIZE_24_KHZ];
            let phase = frame_idx as f32 * 0.03;
            for (i, v) in lp_residual.iter_mut().enumerate() {
                let t = i as f32;
                *v = (0.012 * t + phase).sin() + 0.17 * (0.037 * t - 0.4 * phase).cos();
            }

            let p_a = estimator_a.estimate(&lp_residual);
            let s_a = estimator_a.last_pitch_strength_for_testing();

            let p_b = estimator_b.estimate(&lp_residual);
            let s_b = estimator_b.last_pitch_strength_for_testing();

            assert!(
                p_a >= MIN_PITCH_48_KHZ as i32 && p_a <= MAX_PITCH_48_KHZ as i32,
                "frame={frame_idx}, p_a={p_a}"
            );
            assert!(
                p_b >= MIN_PITCH_48_KHZ as i32 && p_b <= MAX_PITCH_48_KHZ as i32,
                "frame={frame_idx}, p_b={p_b}"
            );
            assert!(s_a.is_finite(), "frame={frame_idx}, s_a={s_a}");
            assert!(s_b.is_finite(), "frame={frame_idx}, s_b={s_b}");

            assert_eq!(p_a, p_b, "frame={frame_idx}");
            assert!(
                (s_a - s_b).abs() < 1e-6,
                "frame={frame_idx}, s_a={s_a}, s_b={s_b}"
            );
        }
    }

    #[test]
    fn pitch_search_within_tolerance_fixture() {
        let mut reader = create_lp_residual_and_pitch_info_reader();
        let num_frames = reader.num_chunks().min(300);

        let cpu_features = get_available_cpu_features();
        let mut pitch_estimator = PitchEstimator::new(cpu_features);

        let mut chunk = vec![0.0f32; BUF_SIZE_24_KHZ + 2];
        for i in 0..num_frames {
            assert!(reader.read_chunk(&mut chunk), "frame={i}");

            let mut lp_residual = [0.0f32; BUF_SIZE_24_KHZ];
            lp_residual.copy_from_slice(&chunk[..BUF_SIZE_24_KHZ]);

            let expected_pitch_period = chunk[BUF_SIZE_24_KHZ];
            let expected_pitch_strength = chunk[BUF_SIZE_24_KHZ + 1];

            let pitch_period = pitch_estimator.estimate(&lp_residual);
            assert_eq!(expected_pitch_period as i32, pitch_period, "frame={i}");
            assert!(
                (expected_pitch_strength - pitch_estimator.last_pitch_strength_for_testing()).abs()
                    < 15e-6,
                "frame={i}, expected={}, got={}",
                expected_pitch_strength,
                pitch_estimator.last_pitch_strength_for_testing()
            );
        }
    }
}
