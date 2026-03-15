//! Adaptive digital gain controller mirrored from WebRTC AGC2.

use crate::audio_processing::agc2::agc2_common::{
    ADJACENT_SPEECH_FRAMES_THRESHOLD, FRAME_DURATION_MS, LIMITER_THRESHOLD_FOR_AGC_GAIN_DBFS,
    VAD_CONFIDENCE_THRESHOLD,
};
use crate::audio_processing::agc2::gain_applier::GainApplier;

#[derive(Debug, Clone, Copy)]
pub struct AdaptiveDigitalConfig {
    pub max_gain_db: f32,
    pub headroom_db: f32,
    pub max_gain_change_db_per_second: f32,
    pub max_output_noise_level_dbfs: f32,
    pub initial_gain_db: f32,
}

impl Default for AdaptiveDigitalConfig {
    fn default() -> Self {
        // WebRTC AGC2 defaults (from `api/audio/audio_processing.h`).
        Self {
            headroom_db: 5.0,
            max_gain_db: 50.0,
            initial_gain_db: 15.0,
            max_gain_change_db_per_second: 6.0,
            max_output_noise_level_dbfs: -50.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FrameInfo {
    pub speech_probability: f32,
    pub speech_level_dbfs: f32,
    pub speech_level_reliable: bool,
    pub noise_rms_dbfs: f32,
    pub headroom_db: f32,
    pub limiter_envelope_dbfs: f32,
}

pub struct AdaptiveDigitalGainController {
    gain_applier: GainApplier,
    config: AdaptiveDigitalConfig,
    adjacent_speech_frames_threshold: i32,
    max_gain_change_db_per_10ms: f32,
    calls_since_last_gain_log: i32,
    frames_to_gain_increase_allowed: i32,
    last_gain_db: f32,
}

impl AdaptiveDigitalGainController {
    pub fn new(config: AdaptiveDigitalConfig, adjacent_speech_frames_threshold: i32) -> Self {
        let max_gain_change_db_per_10ms =
            config.max_gain_change_db_per_second * FRAME_DURATION_MS as f32 / 1000.0;
        assert!(max_gain_change_db_per_10ms > 0.0);
        assert!(adjacent_speech_frames_threshold >= 1);
        assert!((-90.0..=0.0).contains(&config.max_output_noise_level_dbfs));
        Self {
            gain_applier: GainApplier::new(false, db_to_ratio(config.initial_gain_db)),
            config,
            adjacent_speech_frames_threshold,
            max_gain_change_db_per_10ms,
            calls_since_last_gain_log: 0,
            frames_to_gain_increase_allowed: adjacent_speech_frames_threshold,
            last_gain_db: config.initial_gain_db,
        }
    }

    pub fn process(&mut self, info: &FrameInfo, frame: &mut [&mut [f32]]) {
        assert!(info.speech_level_dbfs >= -150.0);
        assert!(!frame.is_empty());
        let samples_per_channel = frame[0].len();
        assert!(matches!(samples_per_channel, 80 | 160 | 320 | 480));
        for ch in frame.iter() {
            assert_eq!(ch.len(), samples_per_channel);
        }

        // Compute level used to select desired gain.
        assert!(info.headroom_db > 0.0);
        let input_level_dbfs = info.speech_level_dbfs + info.headroom_db;

        let target_gain_db = limit_gain_by_low_confidence(
            limit_gain_by_noise(
                compute_gain_db(input_level_dbfs, &self.config),
                info.noise_rms_dbfs,
                self.config.max_output_noise_level_dbfs,
            ),
            self.last_gain_db,
            info.limiter_envelope_dbfs,
            info.speech_level_reliable,
        );

        // Forbid increasing gain until enough adjacent speech frames are observed.
        let mut first_confident_speech_frame = false;
        if info.speech_probability < VAD_CONFIDENCE_THRESHOLD {
            self.frames_to_gain_increase_allowed = self.adjacent_speech_frames_threshold;
        } else if self.frames_to_gain_increase_allowed > 0 {
            self.frames_to_gain_increase_allowed -= 1;
            first_confident_speech_frame = self.frames_to_gain_increase_allowed == 0;
        }

        let gain_increase_allowed = self.frames_to_gain_increase_allowed == 0;

        let mut max_gain_increase_db = self.max_gain_change_db_per_10ms;
        if first_confident_speech_frame {
            max_gain_increase_db *= self.adjacent_speech_frames_threshold as f32;
        }

        let gain_change_this_frame_db = compute_gain_change_this_frame_db(
            target_gain_db,
            self.last_gain_db,
            gain_increase_allowed,
            self.max_gain_change_db_per_10ms,
            max_gain_increase_db,
        );

        if gain_change_this_frame_db != 0.0 {
            self.gain_applier
                .set_gain_factor(db_to_ratio(self.last_gain_db + gain_change_this_frame_db));
        }
        self.gain_applier.apply_gain(frame);

        self.last_gain_db += gain_change_this_frame_db;

        self.calls_since_last_gain_log += 1;
        if self.calls_since_last_gain_log == 1000 {
            self.calls_since_last_gain_log = 0;
        }
    }

}

impl Default for AdaptiveDigitalGainController {
    fn default() -> Self {
        Self::new(
            AdaptiveDigitalConfig::default(),
            ADJACENT_SPEECH_FRAMES_THRESHOLD as i32,
        )
    }
}

fn db_to_ratio(v: f32) -> f32 {
    10.0f32.powf(v / 20.0)
}

fn compute_gain_db(input_level_dbfs: f32, config: &AdaptiveDigitalConfig) -> f32 {
    // If level is very low, apply maximum gain.
    if input_level_dbfs < -(config.headroom_db + config.max_gain_db) {
        return config.max_gain_db;
    }
    // Level is below -headroom, boost it to -headroom.
    if input_level_dbfs < -config.headroom_db {
        return -config.headroom_db - input_level_dbfs;
    }
    // Level is too high and we can't boost.
    0.0
}

fn limit_gain_by_noise(
    target_gain_db: f32,
    input_noise_level_dbfs: f32,
    max_output_noise_level_dbfs: f32,
) -> f32 {
    let max_allowed_gain_db = max_output_noise_level_dbfs - input_noise_level_dbfs;
    target_gain_db.min(max_allowed_gain_db.max(0.0))
}

fn limit_gain_by_low_confidence(
    target_gain_db: f32,
    last_gain_db: f32,
    limiter_audio_level_dbfs: f32,
    estimate_is_confident: bool,
) -> f32 {
    if estimate_is_confident || limiter_audio_level_dbfs <= LIMITER_THRESHOLD_FOR_AGC_GAIN_DBFS {
        return target_gain_db;
    }
    let limiter_level_dbfs_before_gain = limiter_audio_level_dbfs - last_gain_db;
    let new_target_gain_db = (LIMITER_THRESHOLD_FOR_AGC_GAIN_DBFS - limiter_level_dbfs_before_gain)
        .max(0.0);
    new_target_gain_db.min(target_gain_db)
}

fn compute_gain_change_this_frame_db(
    target_gain_db: f32,
    last_gain_db: f32,
    gain_increase_allowed: bool,
    max_gain_decrease_db: f32,
    max_gain_increase_db: f32,
) -> f32 {
    assert!(max_gain_decrease_db > 0.0);
    assert!(max_gain_increase_db > 0.0);
    let mut target_gain_difference_db = target_gain_db - last_gain_db;
    if !gain_increase_allowed {
        target_gain_difference_db = target_gain_difference_db.min(0.0);
    }
    target_gain_difference_db.clamp(-max_gain_decrease_db, max_gain_increase_db)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_processing::agc2::agc2_common::{
        ADJACENT_SPEECH_FRAMES_THRESHOLD, FRAME_DURATION_MS, MIN_LEVEL_DBFS,
    };
    use crate::audio_processing::agc2::vector_float_frame::VectorFloatFrame;

    const MONO: usize = 1;
    const STEREO: usize = 2;
    const FRAME_LEN_10MS_8KHZ: usize = 80;
    const FRAME_LEN_10MS_48KHZ: usize = 480;

    const MAX_SPEECH_PROBABILITY: f32 = 1.0;
    const NO_NOISE_DBFS: f32 = MIN_LEVEL_DBFS;
    const WITH_NOISE_DBFS: f32 = -20.0;
    const NUM_EXTRA_FRAMES: i32 = 10;

    fn get_max_gain_change_per_frame_db(max_gain_change_db_per_second: f32) -> f32 {
        max_gain_change_db_per_second * FRAME_DURATION_MS as f32 / 1000.0
    }

    fn get_frame_info_to_not_adapt(config: AdaptiveDigitalConfig) -> FrameInfo {
        FrameInfo {
            speech_probability: MAX_SPEECH_PROBABILITY,
            speech_level_dbfs: -config.initial_gain_db - config.headroom_db,
            speech_level_reliable: true,
            noise_rms_dbfs: NO_NOISE_DBFS,
            headroom_db: config.headroom_db,
            limiter_envelope_dbfs: -2.0,
        }
    }

    #[test]
    fn gain_applier_should_not_crash() {
        let mut gain_applier =
            AdaptiveDigitalGainController::new(AdaptiveDigitalConfig::default(), ADJACENT_SPEECH_FRAMES_THRESHOLD as i32);
        let mut fake_audio = VectorFloatFrame::new(STEREO, FRAME_LEN_10MS_48KHZ, 10000.0);
        let mut fake_view = fake_audio.view();
        gain_applier.process(&get_frame_info_to_not_adapt(AdaptiveDigitalConfig::default()), &mut fake_view);
    }

    #[test]
    fn max_gain_applied() {
        let default_config = AdaptiveDigitalConfig::default();
        let num_frames_to_adapt =
            (default_config.max_gain_db / get_max_gain_change_per_frame_db(default_config.max_gain_change_db_per_second)) as i32
                + NUM_EXTRA_FRAMES;
        let config = AdaptiveDigitalConfig {
            max_output_noise_level_dbfs: -40.0,
            ..default_config
        };
        let mut gain_applier = AdaptiveDigitalGainController::new(
            config,
            ADJACENT_SPEECH_FRAMES_THRESHOLD as i32,
        );
        let mut info = get_frame_info_to_not_adapt(config);
        info.speech_level_dbfs = -60.0;

        let mut applied_gain = 0.0;
        for _ in 0..num_frames_to_adapt {
            let mut fake_audio = VectorFloatFrame::new(MONO, FRAME_LEN_10MS_8KHZ, 1.0);
            let mut fake_view = fake_audio.view();
            gain_applier.process(&info, &mut fake_view);
            applied_gain = fake_audio.sample(0, 0);
        }
        let applied_gain_db = 20.0 * applied_gain.log10();
        assert!((applied_gain_db - default_config.max_gain_db).abs() <= 0.1);
    }

    #[test]
    fn gain_does_not_change_fast() {
        let default_config = AdaptiveDigitalConfig::default();
        let mut gain_applier = AdaptiveDigitalGainController::new(
            default_config,
            ADJACENT_SPEECH_FRAMES_THRESHOLD as i32,
        );

        const INITIAL_LEVEL_DBFS: f32 = -25.0;
        let max_gain_change_db_per_frame =
            get_max_gain_change_per_frame_db(default_config.max_gain_change_db_per_second);
        let num_frames_to_adapt = (INITIAL_LEVEL_DBFS.abs() / max_gain_change_db_per_frame) as i32
            + NUM_EXTRA_FRAMES;

        let max_change_per_frame_linear = db_to_ratio(max_gain_change_db_per_frame);

        let mut last_gain_linear = db_to_ratio(default_config.initial_gain_db);
        for _ in 0..num_frames_to_adapt {
            let mut fake_audio = VectorFloatFrame::new(MONO, FRAME_LEN_10MS_8KHZ, 1.0);
            let mut info = get_frame_info_to_not_adapt(default_config);
            info.speech_level_dbfs = INITIAL_LEVEL_DBFS;
            let mut fake_view = fake_audio.view();
            gain_applier.process(&info, &mut fake_view);
            let current_gain_linear = fake_audio.sample(0, 0);
            assert!((current_gain_linear - last_gain_linear).abs() <= max_change_per_frame_linear);
            last_gain_linear = current_gain_linear;
        }

        for _ in 0..num_frames_to_adapt {
            let mut fake_audio = VectorFloatFrame::new(MONO, FRAME_LEN_10MS_8KHZ, 1.0);
            let mut info = get_frame_info_to_not_adapt(default_config);
            info.speech_level_dbfs = 0.0;
            let mut fake_view = fake_audio.view();
            gain_applier.process(&info, &mut fake_view);
            let current_gain_linear = fake_audio.sample(0, 0);
            assert!((current_gain_linear - last_gain_linear).abs() <= max_change_per_frame_linear);
            last_gain_linear = current_gain_linear;
        }
    }

    #[test]
    fn gain_is_ramped_in_a_frame() {
        let default_config = AdaptiveDigitalConfig::default();
        let mut gain_applier = AdaptiveDigitalGainController::new(
            default_config,
            ADJACENT_SPEECH_FRAMES_THRESHOLD as i32,
        );

        const INITIAL_LEVEL_DBFS: f32 = -25.0;
        let mut fake_audio = VectorFloatFrame::new(MONO, FRAME_LEN_10MS_48KHZ, 1.0);
        let mut info = get_frame_info_to_not_adapt(default_config);
        info.speech_level_dbfs = INITIAL_LEVEL_DBFS;
        let mut fake_view = fake_audio.view();
        gain_applier.process(&info, &mut fake_view);

        let channel = fake_audio.view_const();
        let mut maximal_difference: f32 = 0.0;
        let mut current_value = db_to_ratio(default_config.initial_gain_db);
        for &x in channel[0] {
            let difference = (x - current_value).abs();
            maximal_difference = maximal_difference.max(difference);
            current_value = x;
        }

        let max_change_per_frame_linear =
            db_to_ratio(get_max_gain_change_per_frame_db(default_config.max_gain_change_db_per_second));
        let max_change_per_sample = max_change_per_frame_linear / FRAME_LEN_10MS_48KHZ as f32;
        assert!(maximal_difference <= max_change_per_sample);
    }

    #[test]
    fn noise_limits_gain() {
        let default_config = AdaptiveDigitalConfig::default();
        let mut gain_applier = AdaptiveDigitalGainController::new(
            default_config,
            ADJACENT_SPEECH_FRAMES_THRESHOLD as i32,
        );

        const INITIAL_LEVEL_DBFS: f32 = -25.0;
        let num_initial_frames =
            (default_config.initial_gain_db / get_max_gain_change_per_frame_db(default_config.max_gain_change_db_per_second)) as i32;
        const NUM_FRAMES: i32 = 50;

        assert!(WITH_NOISE_DBFS > default_config.max_output_noise_level_dbfs);

        for i in 0..(num_initial_frames + NUM_FRAMES) {
            let mut fake_audio = VectorFloatFrame::new(MONO, FRAME_LEN_10MS_48KHZ, 1.0);
            let mut info = get_frame_info_to_not_adapt(default_config);
            info.speech_level_dbfs = INITIAL_LEVEL_DBFS;
            info.noise_rms_dbfs = WITH_NOISE_DBFS;
            let mut fake_view = fake_audio.view();
            gain_applier.process(&info, &mut fake_view);

            if i > num_initial_frames {
                let max_ratio = fake_audio
                    .view_const()[0]
                    .iter()
                    .copied()
                    .fold(f32::MIN, f32::max);
                assert!((max_ratio - 1.0).abs() <= 0.001);
            }
        }
    }

    #[test]
    fn can_handle_positive_speech_levels() {
        let default_config = AdaptiveDigitalConfig::default();
        let mut gain_applier = AdaptiveDigitalGainController::new(
            default_config,
            ADJACENT_SPEECH_FRAMES_THRESHOLD as i32,
        );

        let mut fake_audio = VectorFloatFrame::new(STEREO, FRAME_LEN_10MS_48KHZ, 10000.0);
        let mut info = get_frame_info_to_not_adapt(default_config);
        info.speech_level_dbfs = 5.0;
        let mut fake_view = fake_audio.view();
        gain_applier.process(&info, &mut fake_view);
    }

    #[test]
    fn audio_level_limits_gain() {
        let default_config = AdaptiveDigitalConfig::default();
        let mut gain_applier = AdaptiveDigitalGainController::new(
            default_config,
            ADJACENT_SPEECH_FRAMES_THRESHOLD as i32,
        );

        const INITIAL_LEVEL_DBFS: f32 = -25.0;
        let num_initial_frames =
            (default_config.initial_gain_db / get_max_gain_change_per_frame_db(default_config.max_gain_change_db_per_second)) as i32;
        const NUM_FRAMES: i32 = 50;

        assert!(WITH_NOISE_DBFS > default_config.max_output_noise_level_dbfs);

        for i in 0..(num_initial_frames + NUM_FRAMES) {
            let mut fake_audio = VectorFloatFrame::new(MONO, FRAME_LEN_10MS_48KHZ, 1.0);
            let mut info = get_frame_info_to_not_adapt(default_config);
            info.speech_level_dbfs = INITIAL_LEVEL_DBFS;
            info.limiter_envelope_dbfs = 1.0;
            info.speech_level_reliable = false;
            let mut fake_view = fake_audio.view();
            gain_applier.process(&info, &mut fake_view);

            if i > num_initial_frames {
                let max_ratio = fake_audio
                    .view_const()[0]
                    .iter()
                    .copied()
                    .fold(f32::MIN, f32::max);
                assert!((max_ratio - 1.0).abs() <= 0.001);
            }
        }
    }

    #[test]
    fn do_not_increase_gain_with_too_few_speech_frames() {
        for adjacent_speech_frames_threshold in [
            1,
            7,
            31,
            ADJACENT_SPEECH_FRAMES_THRESHOLD as i32,
        ] {
            let default_config = AdaptiveDigitalConfig::default();
            let mut gain_applier =
                AdaptiveDigitalGainController::new(default_config, adjacent_speech_frames_threshold);

            let mut info = get_frame_info_to_not_adapt(default_config);
            info.speech_level_dbfs -= 12.0;

            let mut prev_gain = 0.0;
            for i in 0..adjacent_speech_frames_threshold {
                let mut audio = VectorFloatFrame::new(MONO, FRAME_LEN_10MS_48KHZ, 1.0);
                let mut audio_view = audio.view();
                gain_applier.process(&info, &mut audio_view);
                let gain = audio.sample(0, 0);
                if i > 0 {
                    assert_eq!(prev_gain, gain);
                }
                prev_gain = gain;
            }
        }
    }

    #[test]
    fn increase_gain_with_enough_speech_frames() {
        for adjacent_speech_frames_threshold in [
            1,
            7,
            31,
            ADJACENT_SPEECH_FRAMES_THRESHOLD as i32,
        ] {
            let default_config = AdaptiveDigitalConfig::default();
            let mut gain_applier =
                AdaptiveDigitalGainController::new(default_config, adjacent_speech_frames_threshold);

            let mut info = get_frame_info_to_not_adapt(default_config);
            info.speech_level_dbfs -= 12.0;

            let mut prev_gain = 0.0;
            for _ in 0..adjacent_speech_frames_threshold {
                let mut audio = VectorFloatFrame::new(MONO, FRAME_LEN_10MS_48KHZ, 1.0);
                let mut audio_view = audio.view();
                gain_applier.process(&info, &mut audio_view);
                prev_gain = audio.sample(0, 0);
            }

            let mut audio = VectorFloatFrame::new(MONO, FRAME_LEN_10MS_48KHZ, 1.0);
            let mut audio_view = audio.view();
            gain_applier.process(&info, &mut audio_view);
            assert!(audio.sample(0, 0) > prev_gain);
        }
    }
}
