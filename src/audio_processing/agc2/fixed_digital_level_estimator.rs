//! Fixed digital level estimator mirrored from WebRTC AGC2.

use crate::audio_processing::agc2::agc2_common::SUB_FRAMES_IN_FRAME;

const INITIAL_FILTER_STATE_LEVEL: f32 = 0.0;
const ATTACK_FILTER_CONSTANT: f32 = 0.0;
// Computed in C++ as: 10 ** (-1/20 * subframe_duration / kDecayMs)
const DECAY_FILTER_CONSTANT: f32 = 0.9971259;

/// Produces a smooth signal level estimate from an input audio stream.
pub struct FixedDigitalLevelEstimator {
    filter_state_level: f32,
    samples_in_frame: usize,
    samples_in_sub_frame: usize,
}

impl FixedDigitalLevelEstimator {
    pub fn new(samples_per_channel: usize) -> Self {
        let mut s = Self {
            filter_state_level: INITIAL_FILTER_STATE_LEVEL,
            samples_in_frame: 0,
            samples_in_sub_frame: 0,
        };
        s.set_samples_per_channel(samples_per_channel);
        s
    }

    /// Input is expected as deinterleaved FloatS16 channels.
    pub fn compute_level(&mut self, float_frame: &[&[f32]]) -> [f32; SUB_FRAMES_IN_FRAME] {
        assert!(!float_frame.is_empty());
        for ch in float_frame {
            assert_eq!(ch.len(), self.samples_in_frame);
        }

        // Compute max envelope without smoothing.
        let mut envelope = [0.0f32; SUB_FRAMES_IN_FRAME];
        for channel in float_frame {
            for (sub_frame, slot) in envelope.iter_mut().enumerate() {
                for sample_in_sub_frame in 0..self.samples_in_sub_frame {
                    let idx = sub_frame * self.samples_in_sub_frame + sample_in_sub_frame;
                    *slot = (*slot).max(channel[idx].abs());
                }
            }
        }

        // Make sure envelope increases happen one step earlier.
        for sub_frame in 0..(SUB_FRAMES_IN_FRAME - 1) {
            if envelope[sub_frame] < envelope[sub_frame + 1] {
                envelope[sub_frame] = envelope[sub_frame + 1];
            }
        }

        // Add attack / decay smoothing.
        for slot in envelope.iter_mut().take(SUB_FRAMES_IN_FRAME) {
            let envelope_value = *slot;
            if envelope_value > self.filter_state_level {
                *slot = envelope_value * (1.0 - ATTACK_FILTER_CONSTANT)
                    + self.filter_state_level * ATTACK_FILTER_CONSTANT;
            } else {
                *slot = envelope_value * (1.0 - DECAY_FILTER_CONSTANT)
                    + self.filter_state_level * DECAY_FILTER_CONSTANT;
            }
            self.filter_state_level = *slot;
        }

        envelope
    }

    pub fn set_samples_per_channel(&mut self, samples_per_channel: usize) {
        self.samples_in_frame = samples_per_channel;
        self.samples_in_sub_frame = self.samples_in_frame / SUB_FRAMES_IN_FRAME;
        self.check_parameter_combination();
    }

    pub fn reset(&mut self) {
        self.filter_state_level = INITIAL_FILTER_STATE_LEVEL;
    }

    pub fn last_audio_level(&self) -> f32 {
        self.filter_state_level
    }

    fn check_parameter_combination(&self) {
        assert!(self.samples_in_frame > 0);
        assert!(SUB_FRAMES_IN_FRAME <= self.samples_in_frame);
        assert_eq!(self.samples_in_frame % SUB_FRAMES_IN_FRAME, 0);
        assert!(self.samples_in_sub_frame > 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_processing::agc2::vector_float_frame::VectorFloatFrame;

    const INPUT_LEVEL: f32 = 10000.0;
    const DECAY_MS: f32 = 20.0;

    fn sample_rate_to_default_channel_size(sample_rate_hz: usize) -> usize {
        sample_rate_hz / 100
    }

    fn db_to_ratio(v: f32) -> f32 {
        10.0f32.powf(v / 20.0)
    }

    fn dbfs_to_float_s16(v: f32) -> f32 {
        db_to_ratio(v) * crate::audio_processing::agc2::agc2_common::MAX_ABS_FLOAT_S16_VALUE
    }

    // Run audio at specified settings through the level estimator and verify bounds.
    fn test_level_estimator(
        samples_per_channel: usize,
        num_channels: usize,
        input_level_linear_scale: f32,
        expected_min: f32,
        expected_max: f32,
    ) {
        let mut level_estimator = FixedDigitalLevelEstimator::new(samples_per_channel);
        let vectors_with_float_frame =
            VectorFloatFrame::new(num_channels, samples_per_channel, input_level_linear_scale);

        for i in 0..500 {
            let channel_views = vectors_with_float_frame.view_const();
            let level = level_estimator.compute_level(&channel_views);

            if i < 50 {
                continue;
            }

            for &x in &level {
                assert!(expected_min <= x, "expected_min={expected_min}, got={x}");
                assert!(x <= expected_max, "expected_max={expected_max}, got={x}");
            }
        }
    }

    // Returns time for level estimator to decrease its level estimate by `level_reduction_db`.
    fn time_ms_to_decrease_level(
        samples_per_channel: usize,
        num_channels: usize,
        input_level_db: f32,
        level_reduction_db: f32,
    ) -> f32 {
        let input_level = dbfs_to_float_s16(input_level_db);
        assert!(level_reduction_db > 0.0);

        let vectors_with_float_frame =
            VectorFloatFrame::new(num_channels, samples_per_channel, input_level);

        let mut level_estimator = FixedDigitalLevelEstimator::new(samples_per_channel);

        // Give estimator plenty of time to ramp up and stabilize.
        let mut last_level = 0.0f32;
        for _ in 0..500 {
            let channel_views = vectors_with_float_frame.view_const();
            let level_envelope = level_estimator.compute_level(&channel_views);
            last_level = *level_envelope.last().expect("subframes are non-empty");
        }

        // Set input to 0.
        let vectors_with_zero_float_frame =
            VectorFloatFrame::new(num_channels, samples_per_channel, 0.0);

        let reduced_level_linear = dbfs_to_float_s16(input_level_db - level_reduction_db);
        let mut sub_frames_until_level_reduction = 0i32;
        while last_level > reduced_level_linear {
            let channel_views = vectors_with_zero_float_frame.view_const();
            let level_envelope = level_estimator.compute_level(&channel_views);
            for &v in &level_envelope {
                assert!(v < last_level);
                sub_frames_until_level_reduction += 1;
                last_level = v;
                if last_level <= reduced_level_linear {
                    break;
                }
            }
        }

        sub_frames_until_level_reduction as f32
            * crate::audio_processing::agc2::agc2_common::FRAME_DURATION_MS as f32
            / SUB_FRAMES_IN_FRAME as f32
    }

    #[test]
    fn estimator_should_not_crash() {
        test_level_estimator(
            sample_rate_to_default_channel_size(8000),
            1,
            0.0,
            f32::MIN,
            f32::MAX,
        );
    }

    #[test]
    fn estimator_should_estimate_constant_level() {
        test_level_estimator(
            sample_rate_to_default_channel_size(10000),
            1,
            INPUT_LEVEL,
            INPUT_LEVEL * 0.99,
            INPUT_LEVEL * 1.01,
        );
    }

    #[test]
    fn estimator_should_estimate_constant_level_for_many_channels() {
        const NUM_CHANNELS: usize = 10;
        test_level_estimator(
            sample_rate_to_default_channel_size(20000),
            NUM_CHANNELS,
            INPUT_LEVEL,
            INPUT_LEVEL * 0.99,
            INPUT_LEVEL * 1.01,
        );
    }

    #[test]
    fn time_to_decrease_for_low_level() {
        const LEVEL_REDUCTION_DB: f32 = 25.0;
        const INITIAL_LOW_LEVEL: f32 = -40.0;
        const EXPECTED_TIME: f32 = LEVEL_REDUCTION_DB * DECAY_MS;

        let time_to_decrease = time_ms_to_decrease_level(
            sample_rate_to_default_channel_size(22000),
            1,
            INITIAL_LOW_LEVEL,
            LEVEL_REDUCTION_DB,
        );

        assert!(EXPECTED_TIME * 0.9 <= time_to_decrease);
        assert!(time_to_decrease <= EXPECTED_TIME * 1.1);
    }

    #[test]
    fn time_to_decrease_for_full_scale_level() {
        const LEVEL_REDUCTION_DB: f32 = 25.0;
        const EXPECTED_TIME: f32 = LEVEL_REDUCTION_DB * DECAY_MS;

        let time_to_decrease = time_ms_to_decrease_level(
            sample_rate_to_default_channel_size(26000),
            1,
            0.0,
            LEVEL_REDUCTION_DB,
        );

        assert!(EXPECTED_TIME * 0.9 <= time_to_decrease);
        assert!(time_to_decrease <= EXPECTED_TIME * 1.1);
    }

    #[test]
    fn time_to_decrease_for_multiple_channels() {
        const LEVEL_REDUCTION_DB: f32 = 25.0;
        const EXPECTED_TIME: f32 = LEVEL_REDUCTION_DB * DECAY_MS;
        const NUM_CHANNELS: usize = 10;

        let time_to_decrease = time_ms_to_decrease_level(
            sample_rate_to_default_channel_size(28000),
            NUM_CHANNELS,
            0.0,
            LEVEL_REDUCTION_DB,
        );

        assert!(EXPECTED_TIME * 0.9 <= time_to_decrease);
        assert!(time_to_decrease <= EXPECTED_TIME * 1.1);
    }
}
