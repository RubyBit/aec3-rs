//! High-level AGC2 orchestrator mirrored from WebRTC `gain_controller2`.

use crate::audio_processing::agc2::adaptive_digital_gain_controller::{
    AdaptiveDigitalConfig, AdaptiveDigitalGainController, FrameInfo,
};
use crate::audio_processing::agc2::agc2_common::{
    ADJACENT_SPEECH_FRAMES_THRESHOLD, MAX_ABS_FLOAT_S16_VALUE,
    SATURATION_PROTECTOR_INITIAL_HEADROOM_DB,
};
use crate::audio_processing::agc2::cpu_features::{get_available_cpu_features, AvailableCpuFeatures};
use crate::audio_processing::agc2::gain_applier::GainApplier;
use crate::audio_processing::agc2::input_volume_controller::{
    Config as InputVolumeControllerConfig, InputVolumeController,
};
use crate::audio_processing::agc2::limiter::Limiter;
use crate::audio_processing::agc2::noise_level_estimator::{
    create_noise_floor_estimator, NoiseLevelEstimator,
};
use crate::audio_processing::agc2::saturation_protector::{
    create_saturation_protector, SaturationProtector,
};
use crate::audio_processing::agc2::speech_level_estimator::{
    create_speech_level_estimator, SpeechLevelEstimator,
};
use crate::audio_processing::agc2::vad_wrapper::VoiceActivityDetectorWrapper;
use crate::audio_processing::audio_buffer::AudioBuffer;

#[derive(Debug, Clone, Copy)]
pub struct FixedDigitalConfig {
    pub gain_db: f32,
}

impl Default for FixedDigitalConfig {
    fn default() -> Self {
        Self { gain_db: 0.0 }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AdaptiveDigitalControllerConfig {
    pub enabled: bool,
    pub headroom_db: f32,
    pub max_gain_db: f32,
    pub initial_gain_db: f32,
    pub max_gain_change_db_per_second: f32,
    pub max_output_noise_level_dbfs: f32,
}

impl Default for AdaptiveDigitalControllerConfig {
    fn default() -> Self {
        let d = AdaptiveDigitalConfig::default();
        Self {
            enabled: true,
            headroom_db: d.headroom_db,
            max_gain_db: d.max_gain_db,
            initial_gain_db: d.initial_gain_db,
            max_gain_change_db_per_second: d.max_gain_change_db_per_second,
            max_output_noise_level_dbfs: d.max_output_noise_level_dbfs,
        }
    }
}

impl AdaptiveDigitalControllerConfig {
    fn as_adaptive_digital_config(self) -> AdaptiveDigitalConfig {
        AdaptiveDigitalConfig {
            max_gain_db: self.max_gain_db,
            headroom_db: self.headroom_db,
            max_gain_change_db_per_second: self.max_gain_change_db_per_second,
            max_output_noise_level_dbfs: self.max_output_noise_level_dbfs,
            initial_gain_db: self.initial_gain_db,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct InputVolumeControllerRuntimeConfig {
    pub enabled: bool,
}

impl Default for InputVolumeControllerRuntimeConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct GainController2Config {
    pub fixed_digital: FixedDigitalConfig,
    pub adaptive_digital: AdaptiveDigitalControllerConfig,
    pub input_volume_controller: InputVolumeControllerRuntimeConfig,
}

#[derive(Debug, Clone, Copy)]
struct AudioLevels {
    peak_dbfs: f32,
    rms_dbfs: f32,
}

#[derive(Debug, Clone, Copy)]
struct SpeechLevel {
    is_confident: bool,
    rms_dbfs: f32,
}

pub struct GainController2 {
    cpu_features: AvailableCpuFeatures,
    fixed_gain_applier: GainApplier,
    noise_level_estimator: Option<Box<dyn NoiseLevelEstimator>>,
    vad: Option<VoiceActivityDetectorWrapper>,
    speech_level_estimator: Option<Box<dyn SpeechLevelEstimator>>,
    input_volume_controller: Option<InputVolumeController>,
    saturation_protector: Option<Box<dyn SaturationProtector>>,
    adaptive_digital_controller: Option<AdaptiveDigitalGainController>,
    limiter: Limiter,
    recommended_input_volume: Option<i32>,
}

impl GainController2 {
    pub fn new(
        config: GainController2Config,
        input_volume_controller_config: InputVolumeControllerConfig,
        sample_rate_hz: usize,
        num_channels: usize,
        use_internal_vad: bool,
    ) -> Self {
        assert!(Self::validate(&config));

        let cpu_features = get_available_cpu_features();
        let adaptive_config = config.adaptive_digital.as_adaptive_digital_config();

        let mut speech_level_estimator = None;
        let mut vad = None;
        if config.input_volume_controller.enabled || config.adaptive_digital.enabled {
            speech_level_estimator = Some(create_speech_level_estimator(
                adaptive_config,
                ADJACENT_SPEECH_FRAMES_THRESHOLD as i32,
            ));
            if use_internal_vad {
                vad = Some(VoiceActivityDetectorWrapper::new(cpu_features, sample_rate_hz));
            }
        }

        let mut input_volume_controller = None;
        if config.input_volume_controller.enabled {
            let mut c = InputVolumeController::new(num_channels as i32, input_volume_controller_config);
            c.initialize();
            input_volume_controller = Some(c);
        }

        let mut noise_level_estimator = None;
        let mut saturation_protector = None;
        let mut adaptive_digital_controller = None;
        if config.adaptive_digital.enabled {
            noise_level_estimator = Some(create_noise_floor_estimator());
            saturation_protector = Some(create_saturation_protector(
                SATURATION_PROTECTOR_INITIAL_HEADROOM_DB,
                ADJACENT_SPEECH_FRAMES_THRESHOLD as i32,
            ));
            adaptive_digital_controller = Some(AdaptiveDigitalGainController::new(
                adaptive_config,
                ADJACENT_SPEECH_FRAMES_THRESHOLD as i32,
            ));
        }

        Self {
            cpu_features,
            fixed_gain_applier: GainApplier::new(false, db_to_ratio(config.fixed_digital.gain_db)),
            noise_level_estimator,
            vad,
            speech_level_estimator,
            input_volume_controller,
            saturation_protector,
            adaptive_digital_controller,
            limiter: Limiter::new(sample_rate_hz / 100),
            recommended_input_volume: None,
        }
    }

    pub fn set_capture_output_used(&mut self, capture_output_used: bool) {
        if let Some(c) = self.input_volume_controller.as_mut() {
            c.handle_capture_output_used_change(capture_output_used);
        }
    }

    pub fn set_fixed_gain_db(&mut self, gain_db: f32) {
        let gain_factor = db_to_ratio(gain_db);
        if self.fixed_gain_applier.get_gain_factor() != gain_factor {
            self.limiter.reset();
        }
        self.fixed_gain_applier.set_gain_factor(gain_factor);
    }

    pub fn analyze(&mut self, applied_input_volume: i32, audio_buffer: &AudioBuffer) {
        assert!((0..=255).contains(&applied_input_volume));
        self.recommended_input_volume = None;
        if let Some(c) = self.input_volume_controller.as_mut() {
            c.analyze_input_audio(applied_input_volume, audio_buffer);
        }
    }

    pub fn process(&mut self, input_volume_changed: bool, audio: &mut AudioBuffer) {
        self.recommended_input_volume = None;

        if input_volume_changed {
            if let Some(s) = self.speech_level_estimator.as_mut() {
                s.reset();
            }
            if let Some(s) = self.saturation_protector.as_mut() {
                s.reset();
            }
        }

        let mut speech_probability = 0.0;
        if let Some(vad) = self.vad.as_mut() {
            let frame: Vec<&[f32]> = (0..audio.num_channels()).map(|ch| audio.channel(ch)).collect();
            speech_probability = vad.analyze(&frame);
        }

        let audio_levels = compute_audio_levels(audio);

        let mut noise_rms_dbfs = None;
        if let Some(n) = self.noise_level_estimator.as_mut() {
            let frame: Vec<&[f32]> = (0..audio.num_channels()).map(|ch| audio.channel(ch)).collect();
            noise_rms_dbfs = Some(n.analyze(&frame));
        }

        let mut speech_level = None;
        if let Some(s) = self.speech_level_estimator.as_mut() {
            s.update(audio_levels.rms_dbfs, speech_probability);
            speech_level = Some(SpeechLevel {
                is_confident: s.is_confident(),
                rms_dbfs: s.get_level_dbfs(),
            });
        }

        if let Some(c) = self.input_volume_controller.as_mut() {
            let s = speech_level.expect("speech level estimator must exist when IVC is enabled");
            self.recommended_input_volume = c.recommend_input_volume(
                speech_probability,
                if s.is_confident { Some(s.rms_dbfs) } else { None },
            );
        }

        let num_channels = audio.num_channels();
        let mut working: Vec<Vec<f32>> = (0..num_channels)
            .map(|ch| audio.channel(ch).to_vec())
            .collect();

        {
            let mut frame_mut: Vec<&mut [f32]> =
                working.iter_mut().map(|ch| ch.as_mut_slice()).collect();

            if let Some(adaptive) = self.adaptive_digital_controller.as_mut() {
                let saturation = self
                    .saturation_protector
                    .as_mut()
                    .expect("saturation protector must exist with adaptive digital");
                let s = speech_level.expect("speech level estimator must exist with adaptive digital");
                saturation.analyze(speech_probability, audio_levels.peak_dbfs, s.rms_dbfs);
                let limiter_envelope_dbfs = float_s16_to_dbfs(self.limiter.last_audio_level());
                let noise = noise_rms_dbfs.expect("noise estimator must exist with adaptive digital");

                let frame_info = FrameInfo {
                    speech_probability,
                    speech_level_dbfs: s.rms_dbfs,
                    speech_level_reliable: s.is_confident,
                    noise_rms_dbfs: noise,
                    headroom_db: saturation.headroom_db(),
                    limiter_envelope_dbfs,
                };
                adaptive.process(
                    &frame_info,
                    &mut frame_mut,
                );
            }

            self.fixed_gain_applier.apply_gain(&mut frame_mut);
            self.limiter.set_samples_per_channel(audio.num_frames());
            self.limiter.process(&mut frame_mut);
        }

        for (ch, channel) in working.iter().enumerate() {
            audio.channel_mut(ch).copy_from_slice(channel);
        }
    }

    pub fn validate(config: &GainController2Config) -> bool {
        config.fixed_digital.gain_db >= 0.0
            && config.fixed_digital.gain_db < 50.0
            && config.adaptive_digital.headroom_db >= 0.0
            && config.adaptive_digital.max_gain_db > 0.0
            && config.adaptive_digital.initial_gain_db >= 0.0
            && config.adaptive_digital.max_gain_change_db_per_second > 0.0
            && config.adaptive_digital.max_output_noise_level_dbfs <= 0.0
    }

    pub fn cpu_features(&self) -> AvailableCpuFeatures {
        self.cpu_features
    }

    pub fn recommended_input_volume(&self) -> Option<i32> {
        self.recommended_input_volume
    }
}

fn db_to_ratio(v: f32) -> f32 {
    10.0f32.powf(v / 20.0)
}

fn float_s16_to_dbfs(v: f32) -> f32 {
    if v <= 1.0 {
        return -90.31;
    }
    20.0 * (v / MAX_ABS_FLOAT_S16_VALUE).log10()
}

fn compute_audio_levels(audio: &AudioBuffer) -> AudioLevels {
    let first_ch = audio.channel(0);
    let mut peak = 0.0f32;
    let mut rms = 0.0f32;
    for &x in first_ch {
        peak = peak.max(x.abs());
        rms += x * x;
    }
    let rms = (rms / first_ch.len() as f32).sqrt();
    AudioLevels {
        peak_dbfs: float_s16_to_dbfs(peak),
        rms_dbfs: float_s16_to_dbfs(rms),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_processing::agc2::limiter_db_gain_curve::LIMITER_MAX_INPUT_LEVEL_DBFS;
    use crate::audio_processing::stream_config::StreamConfig;

    const TEST_SAMPLES_PCM_48K_STEREO: &[u8] =
        include_bytes!("../../tests/test_data/audio_processing/resources/near48_stereo.pcm");

    fn set_audio_buffer_samples(value: f32, ab: &mut AudioBuffer) {
        for ch in 0..ab.num_channels() {
            ab.channel_mut(ch).fill(value);
        }
    }

    fn run_agc2_with_constant_input(
        agc2: &mut GainController2,
        input_level: f32,
        num_frames: i32,
        sample_rate_hz: usize,
        num_channels: usize,
        applied_initial_volume: i32,
    ) -> f32 {
        let mut ab = AudioBuffer::from_sample_rates(
            sample_rate_hz,
            num_channels,
            sample_rate_hz,
            num_channels,
            sample_rate_hz,
        );

        for _ in 0..(num_frames + 1) {
            set_audio_buffer_samples(input_level, &mut ab);
            let applied_volume = agc2.recommended_input_volume().unwrap_or(applied_initial_volume);
            agc2.analyze(applied_volume, &ab);
            agc2.process(false, &mut ab);
        }

        let num_samples = sample_rate_hz / 100;
        ab.channel(0)[num_samples - 1]
    }

    fn create_agc2_fixed_digital_mode(fixed_gain_db: f32, sample_rate_hz: usize) -> GainController2 {
        let mut config = GainController2Config::default();
        config.adaptive_digital.enabled = false;
        config.input_volume_controller.enabled = false;
        config.fixed_digital.gain_db = fixed_gain_db;
        assert!(GainController2::validate(&config));
        GainController2::new(
            config,
            InputVolumeControllerConfig::default(),
            sample_rate_hz,
            1,
            true,
        )
    }

    fn test_input_volume_controller_config() -> InputVolumeControllerConfig {
        InputVolumeControllerConfig {
            min_input_volume: InputVolumeControllerConfig::default().min_input_volume,
            clipped_level_min: 20,
            clipped_level_step: 30,
            clipped_ratio_threshold: 0.4,
            clipped_wait_frames: 50,
            enable_clipping_predictor: true,
            target_range_max_dbfs: -6,
            target_range_experimental_max_dbfs: InputVolumeControllerConfig::default()
                .target_range_experimental_max_dbfs,
            target_range_min_dbfs: -70,
            update_input_volume_wait_frames: 100,
            speech_probability_threshold: 0.9,
            speech_ratio_threshold: 1.0,
        }
    }

    fn lin_space(min: f32, max: f32, n: usize) -> Vec<f32> {
        assert!(n >= 2);
        let step = (max - min) / (n - 1) as f32;
        (0..n).map(|i| min + i as f32 * step).collect()
    }

    fn i16_samples_from_le_bytes(bytes: &[u8]) -> Vec<i16> {
        assert_eq!(bytes.len() % 2, 0);
        bytes.chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect()
    }

    fn read_float_samples_from_stereo_pcm(
        samples_per_channel: usize,
        num_channels: usize,
        pcm: &[i16],
        pos: &mut usize,
        data: &mut [f32],
    ) {
        assert!(num_channels <= 2);
        assert_eq!(data.len(), samples_per_channel * num_channels);
        for sample in 0..samples_per_channel {
            for channel in 0..2 {
                let v = pcm[*pos] as f32 / 32768.0;
                *pos += 1;
                if *pos >= pcm.len() {
                    *pos = 0;
                }
                if channel < num_channels {
                    data[sample * num_channels + channel] = v;
                }
            }
        }
    }

    #[test]
    fn check_default_config() {
        let config = GainController2Config::default();
        assert!(GainController2::validate(&config));
    }

    #[test]
    fn check_fixed_digital_config() {
        let mut config = GainController2Config::default();
        config.fixed_digital.gain_db = -5.0;
        assert!(!GainController2::validate(&config));

        config.fixed_digital.gain_db = 0.0;
        assert!(GainController2::validate(&config));

        config.fixed_digital.gain_db = 15.0;
        assert!(GainController2::validate(&config));
    }

    #[test]
    fn check_headroom_db() {
        let mut config = GainController2Config::default();
        config.adaptive_digital.headroom_db = -1.0;
        assert!(!GainController2::validate(&config));
        config.adaptive_digital.headroom_db = 0.0;
        assert!(GainController2::validate(&config));
        config.adaptive_digital.headroom_db = 5.0;
        assert!(GainController2::validate(&config));
    }

    #[test]
    fn check_max_gain_db() {
        let mut config = GainController2Config::default();
        config.adaptive_digital.max_gain_db = -1.0;
        assert!(!GainController2::validate(&config));
        config.adaptive_digital.max_gain_db = 0.0;
        assert!(!GainController2::validate(&config));
        config.adaptive_digital.max_gain_db = 5.0;
        assert!(GainController2::validate(&config));
    }

    #[test]
    fn check_initial_gain_db() {
        let mut config = GainController2Config::default();
        config.adaptive_digital.initial_gain_db = -1.0;
        assert!(!GainController2::validate(&config));
        config.adaptive_digital.initial_gain_db = 0.0;
        assert!(GainController2::validate(&config));
        config.adaptive_digital.initial_gain_db = 5.0;
        assert!(GainController2::validate(&config));
    }

    #[test]
    fn check_adaptive_digital_max_gain_change_speed_config() {
        let mut config = GainController2Config::default();
        config.adaptive_digital.max_gain_change_db_per_second = -1.0;
        assert!(!GainController2::validate(&config));
        config.adaptive_digital.max_gain_change_db_per_second = 0.0;
        assert!(!GainController2::validate(&config));
        config.adaptive_digital.max_gain_change_db_per_second = 5.0;
        assert!(GainController2::validate(&config));
    }

    #[test]
    fn check_adaptive_digital_max_output_noise_level_config() {
        let mut config = GainController2Config::default();
        config.adaptive_digital.max_output_noise_level_dbfs = 5.0;
        assert!(!GainController2::validate(&config));
        config.adaptive_digital.max_output_noise_level_dbfs = 0.0;
        assert!(GainController2::validate(&config));
        config.adaptive_digital.max_output_noise_level_dbfs = -5.0;
        assert!(GainController2::validate(&config));
    }

    #[test]
    fn check_get_recommended_input_volume_when_input_volume_controller_not_enabled() {
        const HIGH_INPUT_LEVEL: f32 = 32767.0;
        const LOW_INPUT_LEVEL: f32 = 1000.0;
        const INITIAL_INPUT_VOLUME: i32 = 100;
        const NUM_CHANNELS: usize = 2;
        const NUM_FRAMES: i32 = 5;
        const SAMPLE_RATE_HZ: usize = 16_000;

        let mut config = GainController2Config::default();
        config.input_volume_controller.enabled = false;

        let mut gain_controller = GainController2::new(
            config,
            InputVolumeControllerConfig::default(),
            SAMPLE_RATE_HZ,
            NUM_CHANNELS,
            true,
        );

        assert_eq!(gain_controller.recommended_input_volume(), None);

        run_agc2_with_constant_input(
            &mut gain_controller,
            LOW_INPUT_LEVEL,
            NUM_FRAMES,
            SAMPLE_RATE_HZ,
            NUM_CHANNELS,
            INITIAL_INPUT_VOLUME,
        );
        assert_eq!(gain_controller.recommended_input_volume(), None);

        run_agc2_with_constant_input(
            &mut gain_controller,
            HIGH_INPUT_LEVEL,
            NUM_FRAMES,
            SAMPLE_RATE_HZ,
            NUM_CHANNELS,
            INITIAL_INPUT_VOLUME,
        );
        assert_eq!(gain_controller.recommended_input_volume(), None);
    }

    #[test]
    fn check_get_recommended_input_volume_when_input_volume_controller_not_enabled_and_specific_config_used()
    {
        const HIGH_INPUT_LEVEL: f32 = 32767.0;
        const LOW_INPUT_LEVEL: f32 = 1000.0;
        const INITIAL_INPUT_VOLUME: i32 = 100;
        const NUM_CHANNELS: usize = 2;
        const NUM_FRAMES: i32 = 5;
        const SAMPLE_RATE_HZ: usize = 16_000;

        let mut config = GainController2Config::default();
        config.input_volume_controller.enabled = false;

        let mut gain_controller = GainController2::new(
            config,
            test_input_volume_controller_config(),
            SAMPLE_RATE_HZ,
            NUM_CHANNELS,
            true,
        );

        assert_eq!(gain_controller.recommended_input_volume(), None);

        run_agc2_with_constant_input(
            &mut gain_controller,
            LOW_INPUT_LEVEL,
            NUM_FRAMES,
            SAMPLE_RATE_HZ,
            NUM_CHANNELS,
            INITIAL_INPUT_VOLUME,
        );
        assert_eq!(gain_controller.recommended_input_volume(), None);

        run_agc2_with_constant_input(
            &mut gain_controller,
            HIGH_INPUT_LEVEL,
            NUM_FRAMES,
            SAMPLE_RATE_HZ,
            NUM_CHANNELS,
            INITIAL_INPUT_VOLUME,
        );
        assert_eq!(gain_controller.recommended_input_volume(), None);
    }

    #[test]
    fn check_get_recommended_input_volume_when_input_volume_controller_enabled() {
        const HIGH_INPUT_LEVEL: f32 = 32767.0;
        const LOW_INPUT_LEVEL: f32 = 1000.0;
        const INITIAL_INPUT_VOLUME: i32 = 100;
        const NUM_CHANNELS: usize = 2;
        const NUM_FRAMES: i32 = 5;
        const SAMPLE_RATE_HZ: usize = 16_000;

        let mut config = GainController2Config::default();
        config.input_volume_controller.enabled = true;
        config.adaptive_digital.enabled = true;

        let mut gain_controller = GainController2::new(
            config,
            InputVolumeControllerConfig::default(),
            SAMPLE_RATE_HZ,
            NUM_CHANNELS,
            true,
        );

        assert_eq!(gain_controller.recommended_input_volume(), None);

        run_agc2_with_constant_input(
            &mut gain_controller,
            LOW_INPUT_LEVEL,
            NUM_FRAMES,
            SAMPLE_RATE_HZ,
            NUM_CHANNELS,
            INITIAL_INPUT_VOLUME,
        );
        assert!(gain_controller.recommended_input_volume().is_some());

        run_agc2_with_constant_input(
            &mut gain_controller,
            HIGH_INPUT_LEVEL,
            NUM_FRAMES,
            SAMPLE_RATE_HZ,
            NUM_CHANNELS,
            INITIAL_INPUT_VOLUME,
        );
        assert!(gain_controller.recommended_input_volume().is_some());
    }

    #[test]
    fn check_get_recommended_input_volume_when_input_volume_controller_enabled_and_specific_config_used()
    {
        const HIGH_INPUT_LEVEL: f32 = 32767.0;
        const LOW_INPUT_LEVEL: f32 = 1000.0;
        const INITIAL_INPUT_VOLUME: i32 = 100;
        const NUM_CHANNELS: usize = 2;
        const NUM_FRAMES: i32 = 5;
        const SAMPLE_RATE_HZ: usize = 16_000;

        let mut config = GainController2Config::default();
        config.input_volume_controller.enabled = true;
        config.adaptive_digital.enabled = true;

        let mut gain_controller = GainController2::new(
            config,
            test_input_volume_controller_config(),
            SAMPLE_RATE_HZ,
            NUM_CHANNELS,
            true,
        );

        assert_eq!(gain_controller.recommended_input_volume(), None);

        run_agc2_with_constant_input(
            &mut gain_controller,
            LOW_INPUT_LEVEL,
            NUM_FRAMES,
            SAMPLE_RATE_HZ,
            NUM_CHANNELS,
            INITIAL_INPUT_VOLUME,
        );
        assert!(gain_controller.recommended_input_volume().is_some());

        run_agc2_with_constant_input(
            &mut gain_controller,
            HIGH_INPUT_LEVEL,
            NUM_FRAMES,
            SAMPLE_RATE_HZ,
            NUM_CHANNELS,
            INITIAL_INPUT_VOLUME,
        );
        assert!(gain_controller.recommended_input_volume().is_some());
    }

    #[test]
    fn apply_default_config() {
        let gain_controller2 = GainController2::new(
            GainController2Config::default(),
            InputVolumeControllerConfig::default(),
            16_000,
            2,
            true,
        );
        let _ = gain_controller2.cpu_features();
    }

    #[test]
    fn gain_should_change_on_set_gain() {
        const INPUT_LEVEL: f32 = 1000.0;
        const NUM_FRAMES: i32 = 5;
        const SAMPLE_RATE_HZ: usize = 8000;
        const GAIN_0_DB: f32 = 0.0;
        const GAIN_20_DB: f32 = 20.0;

        let mut agc2_fixed = create_agc2_fixed_digital_mode(GAIN_0_DB, SAMPLE_RATE_HZ);

        let unchanged = run_agc2_with_constant_input(
            &mut agc2_fixed,
            INPUT_LEVEL,
            NUM_FRAMES,
            SAMPLE_RATE_HZ,
            1,
            0,
        );
        assert_eq!(unchanged, INPUT_LEVEL);

        agc2_fixed.set_fixed_gain_db(GAIN_20_DB);
        let amplified = run_agc2_with_constant_input(
            &mut agc2_fixed,
            INPUT_LEVEL,
            NUM_FRAMES,
            SAMPLE_RATE_HZ,
            1,
            0,
        );
        assert_eq!(amplified, INPUT_LEVEL * 10.0);
    }

    #[test]
    fn change_fixed_gain_should_be_fast_and_time_invariant() {
        const NUM_FRAMES: i32 = 5;
        const INPUT_LEVEL: f32 = 1000.0;
        const SAMPLE_RATE_HZ: usize = 8000;
        const GAIN_DB_LOW: f32 = 0.0;
        const GAIN_DB_HIGH: f32 = 25.0;

        let mut agc2_fixed = create_agc2_fixed_digital_mode(GAIN_DB_LOW, SAMPLE_RATE_HZ);

        let output_level_pre = run_agc2_with_constant_input(
            &mut agc2_fixed,
            INPUT_LEVEL,
            NUM_FRAMES,
            SAMPLE_RATE_HZ,
            1,
            0,
        );

        agc2_fixed.set_fixed_gain_db(GAIN_DB_HIGH);
        let _ = run_agc2_with_constant_input(
            &mut agc2_fixed,
            INPUT_LEVEL,
            NUM_FRAMES,
            SAMPLE_RATE_HZ,
            1,
            0,
        );

        agc2_fixed.set_fixed_gain_db(GAIN_DB_LOW);
        let output_level_post = run_agc2_with_constant_input(
            &mut agc2_fixed,
            INPUT_LEVEL,
            NUM_FRAMES,
            SAMPLE_RATE_HZ,
            1,
            0,
        );

        assert_eq!(output_level_pre, output_level_post);
    }

    #[test]
    fn check_saturation_behavior_with_limiter() {
        let cases = [
            (0.1f32, LIMITER_MAX_INPUT_LEVEL_DBFS as f32 - 0.01, 8000usize, false),
            (0.1f32, LIMITER_MAX_INPUT_LEVEL_DBFS as f32 - 0.01, 48000usize, false),
            (
                LIMITER_MAX_INPUT_LEVEL_DBFS as f32 + 0.01,
                10.0,
                8000usize,
                true,
            ),
            (
                LIMITER_MAX_INPUT_LEVEL_DBFS as f32 + 0.01,
                10.0,
                48000usize,
                true,
            ),
        ];

        for &(gain_db_min, gain_db_max, sample_rate_hz, saturation_expected) in &cases {
            for gain_db in lin_space(gain_db_min, gain_db_max, 10) {
                let mut agc2_fixed = create_agc2_fixed_digital_mode(gain_db, sample_rate_hz);
                let processed_sample = run_agc2_with_constant_input(
                    &mut agc2_fixed,
                    32767.0,
                    5,
                    sample_rate_hz,
                    1,
                    0,
                );
                if saturation_expected {
                    assert!((processed_sample - 32767.0).abs() < 1e-3);
                } else {
                    assert!(processed_sample < 32767.0);
                }
            }
        }
    }

    #[test]
    fn check_final_gain_with_adaptive_digital_controller() {
        const SAMPLE_RATE_HZ: usize = 48_000;
        const STEREO: usize = 2;
        const FRAME_DURATION_MS: i32 = 10;
        const DURATION_MS: i32 = 10_000;
        const NUM_FRAMES_TO_PROCESS: i32 = DURATION_MS / FRAME_DURATION_MS;

        let mut config = GainController2Config::default();
        config.fixed_digital.gain_db = 0.0;
        config.adaptive_digital.enabled = true;
        config.input_volume_controller.enabled = false;

        let mut agc2 = GainController2::new(
            config,
            InputVolumeControllerConfig::default(),
            SAMPLE_RATE_HZ,
            STEREO,
            true,
        );

        let pcm = i16_samples_from_le_bytes(TEST_SAMPLES_PCM_48K_STEREO);
        let mut pcm_pos = 0usize;
        let stream_config = StreamConfig::new(SAMPLE_RATE_HZ, STEREO, false);
        let mut capture_input =
            vec![0.0f32; stream_config.num_frames() * stream_config.num_channels()];

        let mut audio_buffer = AudioBuffer::from_sample_rates(
            SAMPLE_RATE_HZ,
            STEREO,
            SAMPLE_RATE_HZ,
            STEREO,
            SAMPLE_RATE_HZ,
        );

        let gain = 10.0f32.powf(-6.0 / 20.0);
        for _ in 0..NUM_FRAMES_TO_PROCESS {
            read_float_samples_from_stereo_pcm(
                stream_config.num_frames(),
                stream_config.num_channels(),
                &pcm,
                &mut pcm_pos,
                &mut capture_input,
            );

            for x in &mut capture_input {
                *x *= gain;
            }

            // Mirror the upstream WebRTC test setup exactly, including its
            // `CopyVectorToAudioBuffer`/`SetupFrame` quirk: `capture_input` is
            // interleaved (L0, R0, L1, R1, ...) but is passed to `AudioBuffer`
            // as if it were planar by splitting it into consecutive channel
            // blocks. This produces a "garbled" per-channel signal and is what
            // the upstream test converges on.
            // TODO: Honestly, this seems like a bug in the upstream test,
            // I want to remove it eventually but better mirror it for now to ensure parity.
            let frames = stream_config.num_frames();
            let channels = stream_config.num_channels();
            let mut capture_refs: Vec<&[f32]> = Vec::with_capacity(channels);
            for ch in 0..channels {
                let start = ch * frames;
                let end = start + frames;
                capture_refs.push(&capture_input[start..end]);
            }
            audio_buffer.copy_from(&capture_refs, &stream_config);
            agc2.process(false, &mut audio_buffer);
        }

        set_audio_buffer_samples(1.0, &mut audio_buffer);
        agc2.process(false, &mut audio_buffer);
        let applied_gain_db = 20.0 * audio_buffer.channel(0)[0].log10();

        const EXPECTED_GAIN_DB: f32 = 7.0;
        const TOLERANCE_DB: f32 = 0.3;
        assert!(
            (applied_gain_db - EXPECTED_GAIN_DB).abs() <= TOLERANCE_DB,
            "applied_gain_db={applied_gain_db}, expected={EXPECTED_GAIN_DB}, tol={TOLERANCE_DB}"
        );
    }
}
