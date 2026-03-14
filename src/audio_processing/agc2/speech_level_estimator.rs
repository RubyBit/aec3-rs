//! Speech level estimator mirrored from WebRTC AGC2 `SpeechLevelEstimatorImpl`.

use crate::audio_processing::agc2::adaptive_digital_gain_controller::AdaptiveDigitalConfig;
use crate::audio_processing::agc2::agc2_common::{
    FRAME_DURATION_MS, LEVEL_ESTIMATOR_LEAK_FACTOR, LEVEL_ESTIMATOR_TIME_TO_CONFIDENCE_MS,
    SATURATION_PROTECTOR_INITIAL_HEADROOM_DB, VAD_CONFIDENCE_THRESHOLD,
};

pub trait SpeechLevelEstimator {
    fn update(&mut self, rms_dbfs: f32, speech_probability: f32);
    fn get_level_dbfs(&self) -> f32;
    fn is_confident(&self) -> bool;
    fn reset(&mut self);
}

#[derive(Clone, Copy)]
struct Ratio {
    numerator: f32,
    denominator: f32,
}

impl Ratio {
    fn get_ratio(&self) -> f32 {
        assert_ne!(self.denominator, 0.0);
        self.numerator / self.denominator
    }
}

#[derive(Clone, Copy)]
struct LevelEstimatorState {
    time_to_confidence_ms: i32,
    level_dbfs: Ratio,
}

pub struct SpeechLevelEstimatorImpl {
    initial_speech_level_dbfs: f32,
    adjacent_speech_frames_threshold: i32,
    preliminary_state: LevelEstimatorState,
    reliable_state: LevelEstimatorState,
    level_dbfs: f32,
    is_confident: bool,
    num_adjacent_speech_frames: i32,
}

fn clamp_level_estimate_dbfs(level_estimate_dbfs: f32) -> f32 {
    level_estimate_dbfs.clamp(-90.0, 30.0)
}

fn get_initial_speech_level_estimate_dbfs(config: AdaptiveDigitalConfig) -> f32 {
    clamp_level_estimate_dbfs(
        -SATURATION_PROTECTOR_INITIAL_HEADROOM_DB - config.initial_gain_db - config.headroom_db,
    )
}

impl SpeechLevelEstimatorImpl {
    pub fn new(config: AdaptiveDigitalConfig, adjacent_speech_frames_threshold: i32) -> Self {
        let initial_speech_level_dbfs = get_initial_speech_level_estimate_dbfs(config);
        assert!(adjacent_speech_frames_threshold >= 1);
        let mut s = Self {
            initial_speech_level_dbfs,
            adjacent_speech_frames_threshold,
            preliminary_state: LevelEstimatorState {
                time_to_confidence_ms: 0,
                level_dbfs: Ratio {
                    numerator: 0.0,
                    denominator: 1.0,
                },
            },
            reliable_state: LevelEstimatorState {
                time_to_confidence_ms: 0,
                level_dbfs: Ratio {
                    numerator: 0.0,
                    denominator: 1.0,
                },
            },
            level_dbfs: initial_speech_level_dbfs,
            is_confident: false,
            num_adjacent_speech_frames: 0,
        };
        s.reset();
        s
    }

    fn update_is_confident(&mut self) {
        if self.adjacent_speech_frames_threshold == 1 {
            self.is_confident = self.preliminary_state.time_to_confidence_ms == 0;
            return;
        }

        assert!(
            self.reliable_state.time_to_confidence_ms != 0
                || self.preliminary_state.time_to_confidence_ms == 0
        );

        self.is_confident = self.reliable_state.time_to_confidence_ms == 0
            || (self.num_adjacent_speech_frames >= self.adjacent_speech_frames_threshold
                && self.preliminary_state.time_to_confidence_ms == 0);
    }

    fn reset_level_estimator_state(&self, state: &mut LevelEstimatorState) {
        state.time_to_confidence_ms = LEVEL_ESTIMATOR_TIME_TO_CONFIDENCE_MS as i32;
        state.level_dbfs.numerator = self.initial_speech_level_dbfs;
        state.level_dbfs.denominator = 1.0;
    }
}

impl SpeechLevelEstimator for SpeechLevelEstimatorImpl {
    fn update(&mut self, rms_dbfs: f32, speech_probability: f32) {
        assert!(rms_dbfs > -150.0);
        assert!(rms_dbfs < 50.0);
        assert!((0.0..=1.0).contains(&speech_probability));

        if speech_probability < VAD_CONFIDENCE_THRESHOLD {
            // Not a speech frame.
            if self.adjacent_speech_frames_threshold > 1 {
                if self.num_adjacent_speech_frames >= self.adjacent_speech_frames_threshold {
                    // First non-speech frame after long enough speech sequence.
                    self.reliable_state = self.preliminary_state;
                } else if self.num_adjacent_speech_frames > 0 {
                    // First non-speech frame after too short sequence.
                    self.preliminary_state = self.reliable_state;
                }
            }
            self.num_adjacent_speech_frames = 0;
        } else {
            // Speech frame observed.
            self.num_adjacent_speech_frames += 1;

            assert!(self.preliminary_state.time_to_confidence_ms >= 0);
            let buffer_is_full = self.preliminary_state.time_to_confidence_ms == 0;
            if !buffer_is_full {
                self.preliminary_state.time_to_confidence_ms -= FRAME_DURATION_MS as i32;
                if self.preliminary_state.time_to_confidence_ms < 0 {
                    self.preliminary_state.time_to_confidence_ms = 0;
                }
            }

            // Weighted average with speech probability as weight.
            assert!(speech_probability > 0.0);
            let leak_factor = if buffer_is_full {
                LEVEL_ESTIMATOR_LEAK_FACTOR
            } else {
                1.0
            };
            self.preliminary_state.level_dbfs.numerator =
                self.preliminary_state.level_dbfs.numerator * leak_factor
                    + rms_dbfs * speech_probability;
            self.preliminary_state.level_dbfs.denominator =
                self.preliminary_state.level_dbfs.denominator * leak_factor + speech_probability;

            let level_dbfs = self.preliminary_state.level_dbfs.get_ratio();
            if self.num_adjacent_speech_frames >= self.adjacent_speech_frames_threshold {
                self.level_dbfs = clamp_level_estimate_dbfs(level_dbfs);
            }
        }
        self.update_is_confident();
    }

    fn get_level_dbfs(&self) -> f32 {
        self.level_dbfs
    }

    fn is_confident(&self) -> bool {
        self.is_confident
    }

    fn reset(&mut self) {
        let mut preliminary = self.preliminary_state;
        let mut reliable = self.reliable_state;
        self.reset_level_estimator_state(&mut preliminary);
        self.reset_level_estimator_state(&mut reliable);
        self.preliminary_state = preliminary;
        self.reliable_state = reliable;
        self.level_dbfs = self.initial_speech_level_dbfs;
        self.num_adjacent_speech_frames = 0;
    }
}

pub fn create_speech_level_estimator(
    config: AdaptiveDigitalConfig,
    adjacent_speech_frames_threshold: i32,
) -> Box<dyn SpeechLevelEstimator> {
    Box::new(SpeechLevelEstimatorImpl::new(
        config,
        adjacent_speech_frames_threshold,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Number of speech frames that the level estimator must observe to become confident.
    const NUM_FRAMES_TO_CONFIDENCE: i32 =
        LEVEL_ESTIMATOR_TIME_TO_CONFIDENCE_MS as i32 / FRAME_DURATION_MS as i32;
    const CONVERGENCE_SPEED_TESTS_LEVEL_TOLERANCE: f32 = 0.5;

    fn run_on_constant_level(
        num_iterations: i32,
        rms_dbfs: f32,
        speech_probability: f32,
        level_estimator: &mut SpeechLevelEstimatorImpl,
    ) {
        for _ in 0..num_iterations {
            level_estimator.update(rms_dbfs, speech_probability);
        }
    }

    const NO_SPEECH_PROBABILITY: f32 = 0.0;
    const LOW_SPEECH_PROBABILITY: f32 = VAD_CONFIDENCE_THRESHOLD / 2.0;
    const MAX_SPEECH_PROBABILITY: f32 = 1.0;

    struct TestLevelEstimator {
        estimator: SpeechLevelEstimatorImpl,
        initial_speech_level_dbfs: f32,
        level_rms_dbfs: f32,
        level_peak_dbfs: f32,
    }

    impl TestLevelEstimator {
        fn new(adjacent_speech_frames_threshold: i32) -> Self {
            let estimator =
                SpeechLevelEstimatorImpl::new(AdaptiveDigitalConfig::default(), adjacent_speech_frames_threshold);
            let initial_speech_level_dbfs = estimator.get_level_dbfs();
            let level_rms_dbfs = initial_speech_level_dbfs / 2.0;
            let level_peak_dbfs = initial_speech_level_dbfs / 3.0;
            assert!(level_rms_dbfs < level_peak_dbfs);
            assert!(initial_speech_level_dbfs < level_rms_dbfs);
            assert!(level_rms_dbfs - initial_speech_level_dbfs > 5.0);
            Self {
                estimator,
                initial_speech_level_dbfs,
                level_rms_dbfs,
                level_peak_dbfs,
            }
        }
    }

    #[test]
    fn level_stabilizes() {
        let mut level_estimator = TestLevelEstimator::new(1);
        run_on_constant_level(
            NUM_FRAMES_TO_CONFIDENCE,
            level_estimator.level_rms_dbfs,
            MAX_SPEECH_PROBABILITY,
            &mut level_estimator.estimator,
        );
        let estimated_level_dbfs = level_estimator.estimator.get_level_dbfs();
        run_on_constant_level(
            1,
            level_estimator.level_rms_dbfs,
            MAX_SPEECH_PROBABILITY,
            &mut level_estimator.estimator,
        );
        assert!((level_estimator.estimator.get_level_dbfs() - estimated_level_dbfs).abs() <= 0.1);
    }

    #[test]
    fn is_not_confident() {
        let mut level_estimator = TestLevelEstimator::new(1);
        run_on_constant_level(
            NUM_FRAMES_TO_CONFIDENCE / 2,
            level_estimator.level_rms_dbfs,
            MAX_SPEECH_PROBABILITY,
            &mut level_estimator.estimator,
        );
        assert!(!level_estimator.estimator.is_confident());
    }

    #[test]
    fn is_confident() {
        let mut level_estimator = TestLevelEstimator::new(1);
        run_on_constant_level(
            NUM_FRAMES_TO_CONFIDENCE,
            level_estimator.level_rms_dbfs,
            MAX_SPEECH_PROBABILITY,
            &mut level_estimator.estimator,
        );
        assert!(level_estimator.estimator.is_confident());
    }

    #[test]
    fn estimator_ignores_non_speech_frames() {
        let mut level_estimator = TestLevelEstimator::new(1);
        run_on_constant_level(
            NUM_FRAMES_TO_CONFIDENCE,
            level_estimator.level_rms_dbfs,
            MAX_SPEECH_PROBABILITY,
            &mut level_estimator.estimator,
        );
        let estimated_level_dbfs = level_estimator.estimator.get_level_dbfs();

        run_on_constant_level(
            NUM_FRAMES_TO_CONFIDENCE,
            0.0,
            NO_SPEECH_PROBABILITY,
            &mut level_estimator.estimator,
        );
        assert_eq!(level_estimator.estimator.get_level_dbfs(), estimated_level_dbfs);
    }

    #[test]
    fn convergence_speed_before_confidence() {
        let mut level_estimator = TestLevelEstimator::new(1);
        run_on_constant_level(
            NUM_FRAMES_TO_CONFIDENCE,
            level_estimator.level_rms_dbfs,
            MAX_SPEECH_PROBABILITY,
            &mut level_estimator.estimator,
        );
        assert!(
            (level_estimator.estimator.get_level_dbfs() - level_estimator.level_rms_dbfs).abs()
                <= CONVERGENCE_SPEED_TESTS_LEVEL_TOLERANCE
        );
    }

    #[test]
    fn convergence_speed_after_confidence() {
        let mut level_estimator = TestLevelEstimator::new(1);

        run_on_constant_level(
            NUM_FRAMES_TO_CONFIDENCE,
            level_estimator.initial_speech_level_dbfs,
            MAX_SPEECH_PROBABILITY,
            &mut level_estimator.estimator,
        );
        assert_eq!(
            level_estimator.estimator.get_level_dbfs(),
            level_estimator.initial_speech_level_dbfs
        );
        assert!(level_estimator.estimator.is_confident());

        const CONVERGENCE_TIME_AFTER_CONFIDENCE_NUM_FRAMES: i32 = 700;
        run_on_constant_level(
            CONVERGENCE_TIME_AFTER_CONFIDENCE_NUM_FRAMES,
            level_estimator.level_rms_dbfs,
            MAX_SPEECH_PROBABILITY,
            &mut level_estimator.estimator,
        );
        assert!(
            (level_estimator.estimator.get_level_dbfs() - level_estimator.level_rms_dbfs).abs()
                <= CONVERGENCE_SPEED_TESTS_LEVEL_TOLERANCE
        );
    }

    #[test]
    fn do_not_adapt_to_short_speech_segments() {
        for adjacent_speech_frames_threshold in [1, 9, 17] {
            let mut level_estimator = TestLevelEstimator::new(adjacent_speech_frames_threshold);
            let initial_level = level_estimator.estimator.get_level_dbfs();
            assert!(initial_level < level_estimator.level_peak_dbfs);
            for _ in 0..(adjacent_speech_frames_threshold - 1) {
                level_estimator
                    .estimator
                    .update(level_estimator.level_rms_dbfs, MAX_SPEECH_PROBABILITY);
                assert_eq!(initial_level, level_estimator.estimator.get_level_dbfs());
            }
            level_estimator
                .estimator
                .update(level_estimator.level_rms_dbfs, LOW_SPEECH_PROBABILITY);
            assert_eq!(initial_level, level_estimator.estimator.get_level_dbfs());
        }
    }

    #[test]
    fn adapt_to_enough_speech_segments() {
        for adjacent_speech_frames_threshold in [1, 9, 17] {
            let mut level_estimator = TestLevelEstimator::new(adjacent_speech_frames_threshold);
            let initial_level = level_estimator.estimator.get_level_dbfs();
            assert!(initial_level < level_estimator.level_peak_dbfs);
            for _ in 0..adjacent_speech_frames_threshold {
                level_estimator
                    .estimator
                    .update(level_estimator.level_rms_dbfs, MAX_SPEECH_PROBABILITY);
            }
            assert!(initial_level < level_estimator.estimator.get_level_dbfs());
        }
    }
}
