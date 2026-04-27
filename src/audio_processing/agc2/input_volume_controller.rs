//! Input volume controller mirrored from WebRTC AGC2.

use std::cmp::{max, min};

use crate::audio_processing::agc2::clipping_predictor::{
    ClippingPredictor, ClippingPredictorConfig, ClippingPredictorMode, create_clipping_predictor,
};
use crate::audio_processing::agc2::input_volume_stats_reporter::update_histogram_on_recommended_input_volume_change_to_match_target;
use crate::audio_processing::audio_buffer::AudioBuffer;

const VOLUME_QUANTIZATION_SLACK: i32 = 25;
const MAX_INPUT_VOLUME: i32 = 255;
const MAX_ABS_RMS_ERROR_DBFS: i32 = 15;

#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub min_input_volume: i32,
    pub clipped_level_min: i32,
    pub clipped_level_step: i32,
    pub clipped_ratio_threshold: f32,
    pub clipped_wait_frames: i32,
    pub enable_clipping_predictor: bool,
    pub target_range_max_dbfs: i32,
    pub target_range_experimental_max_dbfs: i32,
    pub target_range_min_dbfs: i32,
    pub update_input_volume_wait_frames: i32,
    pub speech_probability_threshold: f32,
    pub speech_ratio_threshold: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            min_input_volume: 20,
            clipped_level_min: 70,
            clipped_level_step: 15,
            clipped_ratio_threshold: 0.1,
            clipped_wait_frames: 300,
            enable_clipping_predictor: true,
            target_range_max_dbfs: -30,
            target_range_experimental_max_dbfs: -12,
            target_range_min_dbfs: -50,
            update_input_volume_wait_frames: 100,
            speech_probability_threshold: 0.7,
            speech_ratio_threshold: 0.6,
        }
    }
}

const GAIN_MAP: [i32; 256] = [
    -56, -54, -52, -50, -48, -47, -45, -43, -42, -40, -38, -37, -35, -34, -33, -31, -30, -29, -27,
    -26, -25, -24, -23, -22, -20, -19, -18, -17, -16, -15, -14, -14, -13, -12, -11, -10, -9, -8,
    -8, -7, -6, -5, -5, -4, -3, -2, -2, -1, 0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9,
    9, 10, 10, 11, 11, 12, 12, 13, 13, 13, 14, 14, 15, 15, 15, 16, 16, 17, 17, 17, 18, 18, 18, 19,
    19, 19, 20, 20, 21, 21, 21, 22, 22, 22, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 26, 26, 26, 27,
    27, 27, 28, 28, 28, 28, 29, 29, 29, 30, 30, 30, 30, 31, 31, 31, 32, 32, 32, 32, 33, 33, 33, 33,
    34, 34, 34, 35, 35, 35, 35, 36, 36, 36, 36, 37, 37, 37, 38, 38, 38, 38, 39, 39, 39, 39, 40, 40,
    40, 40, 41, 41, 41, 41, 42, 42, 42, 42, 43, 43, 43, 44, 44, 44, 44, 45, 45, 45, 45, 46, 46, 46,
    46, 47, 47, 47, 47, 48, 48, 48, 48, 49, 49, 49, 49, 50, 50, 50, 50, 51, 51, 51, 51, 52, 52, 52,
    52, 53, 53, 53, 53, 54, 54, 54, 54, 55, 55, 55, 55, 56, 56, 56, 56, 57, 57, 57, 57, 58, 58, 58,
    58, 59, 59, 59, 59, 60, 60, 60, 60, 61, 61, 61, 61, 62, 62, 62, 62, 63, 63, 63, 63, 64,
];

fn compute_volume_update(gain_error_db: i32, input_volume: i32, min_input_volume: i32) -> i32 {
    assert!((0..=MAX_INPUT_VOLUME).contains(&input_volume));
    if gain_error_db == 0 {
        return input_volume;
    }

    let mut new_volume = input_volume;
    if gain_error_db > 0 {
        while GAIN_MAP[new_volume as usize] - GAIN_MAP[input_volume as usize] < gain_error_db
            && new_volume < MAX_INPUT_VOLUME
        {
            new_volume += 1;
        }
    } else {
        while GAIN_MAP[new_volume as usize] - GAIN_MAP[input_volume as usize] > gain_error_db
            && new_volume > min_input_volume
        {
            new_volume -= 1;
        }
    }
    new_volume
}

fn compute_clipped_ratio(audio_buffer: &AudioBuffer) -> f32 {
    let samples_per_channel = audio_buffer.num_frames();
    assert!(samples_per_channel > 0);

    let mut num_clipped = 0usize;
    for ch in 0..audio_buffer.num_channels() {
        let channel = audio_buffer.channel(ch);
        let mut num_clipped_in_ch = 0usize;
        for &sample in channel {
            if sample >= 32767.0 || sample <= -32768.0 {
                num_clipped_in_ch += 1;
            }
        }
        num_clipped = num_clipped.max(num_clipped_in_ch);
    }
    num_clipped as f32 / samples_per_channel as f32
}

fn get_speech_level_rms_error_db(
    speech_level_dbfs: f32,
    target_range_min_dbfs: i32,
    target_range_max_dbfs: i32,
) -> i32 {
    let speech_level_dbfs = speech_level_dbfs.clamp(-90.0, 30.0);
    if speech_level_dbfs > target_range_max_dbfs as f32 {
        (target_range_max_dbfs as f32 - speech_level_dbfs).round() as i32
    } else if speech_level_dbfs < target_range_min_dbfs as f32 {
        (target_range_min_dbfs as f32 - speech_level_dbfs).round() as i32
    } else {
        0
    }
}

pub struct MonoInputVolumeController {
    min_input_volume: i32,
    min_input_volume_after_clipping: i32,
    max_input_volume: i32,
    last_recommended_input_volume: i32,
    capture_output_used: bool,
    check_volume_on_next_process: bool,
    startup: bool,
    recommended_input_volume: i32,
    update_input_volume_wait_frames: i32,
    frames_since_update_input_volume: i32,
    speech_frames_since_update_input_volume: i32,
    is_first_frame: bool,
    speech_probability_threshold: f32,
    speech_ratio_threshold: f32,
}

impl MonoInputVolumeController {
    pub fn new(
        min_input_volume_after_clipping: i32,
        min_input_volume: i32,
        update_input_volume_wait_frames: i32,
        speech_probability_threshold: f32,
        speech_ratio_threshold: f32,
    ) -> Self {
        assert!((0..=255).contains(&min_input_volume));
        assert!((0..=255).contains(&min_input_volume_after_clipping));
        assert!(speech_probability_threshold >= 0.0 && speech_probability_threshold <= 1.0);
        assert!(speech_ratio_threshold >= 0.0 && speech_ratio_threshold <= 1.0);
        Self {
            min_input_volume,
            min_input_volume_after_clipping,
            max_input_volume: MAX_INPUT_VOLUME,
            last_recommended_input_volume: 0,
            capture_output_used: true,
            check_volume_on_next_process: true,
            startup: true,
            recommended_input_volume: 0,
            update_input_volume_wait_frames: max(update_input_volume_wait_frames, 1),
            frames_since_update_input_volume: 0,
            speech_frames_since_update_input_volume: 0,
            is_first_frame: true,
            speech_probability_threshold,
            speech_ratio_threshold,
        }
    }

    pub fn initialize(&mut self) {
        self.max_input_volume = MAX_INPUT_VOLUME;
        self.capture_output_used = true;
        self.check_volume_on_next_process = true;
        self.frames_since_update_input_volume = 0;
        self.speech_frames_since_update_input_volume = 0;
        self.is_first_frame = true;
    }

    pub fn handle_capture_output_used_change(&mut self, capture_output_used: bool) {
        if self.capture_output_used == capture_output_used {
            return;
        }
        self.capture_output_used = capture_output_used;
        if capture_output_used {
            self.check_volume_on_next_process = true;
        }
    }

    pub fn set_stream_analog_level(&mut self, input_volume: i32) {
        self.recommended_input_volume = input_volume;
    }

    pub fn handle_clipping(&mut self, clipped_level_step: i32) {
        assert!(clipped_level_step > 0);
        self.set_max_level(max(
            self.min_input_volume_after_clipping,
            self.max_input_volume - clipped_level_step,
        ));

        if self.last_recommended_input_volume > self.min_input_volume_after_clipping {
            self.set_input_volume(max(
                self.min_input_volume_after_clipping,
                self.last_recommended_input_volume - clipped_level_step,
            ));
            self.frames_since_update_input_volume = 0;
            self.speech_frames_since_update_input_volume = 0;
            self.is_first_frame = false;
        }
    }

    pub fn process(&mut self, rms_error_db: Option<i32>, speech_probability: f32) {
        if self.check_volume_on_next_process {
            self.check_volume_on_next_process = false;
            let _ = self.check_volume_and_reset();
        }

        if speech_probability >= self.speech_probability_threshold {
            self.speech_frames_since_update_input_volume += 1;
        }

        self.frames_since_update_input_volume += 1;
        if self.frames_since_update_input_volume >= self.update_input_volume_wait_frames {
            let speech_ratio = self.speech_frames_since_update_input_volume as f32
                / self.update_input_volume_wait_frames as f32;

            self.frames_since_update_input_volume = 0;
            self.speech_frames_since_update_input_volume = 0;

            if !self.is_first_frame && speech_ratio >= self.speech_ratio_threshold {
                if let Some(rms_error_db) = rms_error_db {
                    self.update_input_volume(rms_error_db);
                }
            }
        }

        self.is_first_frame = false;
    }

    pub fn recommended_analog_level(&self) -> i32 {
        self.recommended_input_volume
    }

    pub fn min_input_volume_after_clipping(&self) -> i32 {
        self.min_input_volume_after_clipping
    }

    fn set_input_volume(&mut self, mut new_volume: i32) {
        let applied_input_volume = self.recommended_input_volume;
        if applied_input_volume == 0 {
            return;
        }
        if !(0..=MAX_INPUT_VOLUME).contains(&applied_input_volume) {
            return;
        }

        if applied_input_volume > self.last_recommended_input_volume + VOLUME_QUANTIZATION_SLACK
            || applied_input_volume < self.last_recommended_input_volume - VOLUME_QUANTIZATION_SLACK
        {
            self.last_recommended_input_volume = applied_input_volume;
            if self.last_recommended_input_volume > self.max_input_volume {
                self.set_max_level(self.last_recommended_input_volume);
            }
            self.frames_since_update_input_volume = 0;
            self.speech_frames_since_update_input_volume = 0;
            self.is_first_frame = false;
            return;
        }

        new_volume = min(new_volume, self.max_input_volume);
        if new_volume == self.last_recommended_input_volume {
            return;
        }

        self.recommended_input_volume = new_volume;
        self.last_recommended_input_volume = new_volume;
    }

    fn set_max_level(&mut self, level: i32) {
        assert!(level >= self.min_input_volume_after_clipping);
        self.max_input_volume = level;
    }

    fn check_volume_and_reset(&mut self) -> i32 {
        let mut input_volume = self.recommended_input_volume;
        if input_volume == 0 && !self.startup {
            return 0;
        }
        if !(0..=MAX_INPUT_VOLUME).contains(&input_volume) {
            return -1;
        }

        if input_volume < self.min_input_volume {
            input_volume = self.min_input_volume;
            self.recommended_input_volume = input_volume;
        }

        self.last_recommended_input_volume = input_volume;
        self.startup = false;
        self.frames_since_update_input_volume = 0;
        self.speech_frames_since_update_input_volume = 0;
        self.is_first_frame = true;
        0
    }

    fn update_input_volume(&mut self, rms_error_db: i32) {
        let rms_error_db = rms_error_db.clamp(-MAX_ABS_RMS_ERROR_DBFS, MAX_ABS_RMS_ERROR_DBFS);
        if rms_error_db == 0 {
            return;
        }
        self.set_input_volume(compute_volume_update(
            rms_error_db,
            self.last_recommended_input_volume,
            self.min_input_volume,
        ));
    }
}

pub struct InputVolumeController {
    num_capture_channels: i32,
    min_input_volume: i32,
    recommended_input_volume: i32,
    applied_input_volume: Option<i32>,
    capture_output_used: bool,
    clipped_level_step: i32,
    clipped_ratio_threshold: f32,
    clipped_wait_frames: i32,
    clipping_predictor: Option<Box<dyn ClippingPredictor>>,
    use_clipping_predictor_step: bool,
    frames_since_clipped: i32,
    target_range_max_dbfs: i32,
    target_range_min_dbfs: i32,
    channel_controllers: Vec<MonoInputVolumeController>,
    channel_controlling_gain: i32,
}

impl InputVolumeController {
    pub fn new(num_capture_channels: i32, config: Config) -> Self {
        let clipping_predictor = create_clipping_predictor(
            num_capture_channels,
            ClippingPredictorConfig {
                enabled: config.enable_clipping_predictor,
                mode: ClippingPredictorMode::AdaptiveStepClippingPeakPrediction,
                window_length: 5,
                reference_window_length: 5,
                reference_window_delay: 5,
                clipping_threshold: -1.0,
                crest_factor_margin: 3.0,
            },
        );
        let use_clipping_predictor_step = clipping_predictor.is_some();

        let mut channel_controllers = Vec::with_capacity(num_capture_channels as usize);
        for _ in 0..num_capture_channels {
            channel_controllers.push(MonoInputVolumeController::new(
                config.clipped_level_min,
                config.min_input_volume,
                config.update_input_volume_wait_frames,
                config.speech_probability_threshold,
                config.speech_ratio_threshold,
            ));
        }

        Self {
            num_capture_channels,
            min_input_volume: config.min_input_volume,
            recommended_input_volume: 0,
            applied_input_volume: None,
            capture_output_used: true,
            clipped_level_step: config.clipped_level_step,
            clipped_ratio_threshold: config.clipped_ratio_threshold,
            clipped_wait_frames: config.clipped_wait_frames,
            clipping_predictor,
            use_clipping_predictor_step,
            frames_since_clipped: config.clipped_wait_frames,
            target_range_max_dbfs: config.target_range_max_dbfs,
            target_range_min_dbfs: config.target_range_min_dbfs,
            channel_controllers,
            channel_controlling_gain: 0,
        }
    }

    pub fn initialize(&mut self) {
        for controller in &mut self.channel_controllers {
            controller.initialize();
        }
        self.capture_output_used = true;
        self.aggregate_channel_levels();
        self.applied_input_volume = None;
    }

    pub fn analyze_input_audio(&mut self, applied_input_volume: i32, audio_buffer: &AudioBuffer) {
        assert!((0..=255).contains(&applied_input_volume));
        self.set_applied_input_volume(applied_input_volume);
        assert_eq!(audio_buffer.num_channels(), self.channel_controllers.len());

        self.aggregate_channel_levels();
        if !self.capture_output_used {
            return;
        }

        if let Some(clipping_predictor) = self.clipping_predictor.as_mut() {
            let frame: Vec<&[f32]> = (0..audio_buffer.num_channels())
                .map(|ch| audio_buffer.channel(ch))
                .collect();
            clipping_predictor.analyze(&frame);
        }

        let clipped_ratio = compute_clipped_ratio(audio_buffer);

        if self.frames_since_clipped < self.clipped_wait_frames {
            self.frames_since_clipped += 1;
            return;
        }

        let clipping_detected = clipped_ratio > self.clipped_ratio_threshold;
        let mut clipping_predicted = false;
        let mut predicted_step = 0;
        if let Some(clipping_predictor) = self.clipping_predictor.as_ref() {
            for channel in 0..self.num_capture_channels {
                let step = clipping_predictor.estimate_clipped_level_step(
                    channel,
                    self.recommended_input_volume,
                    self.clipped_level_step,
                    self.channel_controllers[channel as usize].min_input_volume_after_clipping(),
                    MAX_INPUT_VOLUME,
                );
                if let Some(step) = step {
                    predicted_step = predicted_step.max(step);
                    clipping_predicted = true;
                }
            }
        }

        let mut step = self.clipped_level_step;
        if clipping_predicted {
            predicted_step = predicted_step.max(self.clipped_level_step);
            if self.use_clipping_predictor_step {
                step = predicted_step;
            }
        }

        if clipping_detected || (clipping_predicted && self.use_clipping_predictor_step) {
            for controller in &mut self.channel_controllers {
                controller.handle_clipping(step);
            }
            self.frames_since_clipped = 0;
            if let Some(clipping_predictor) = self.clipping_predictor.as_mut() {
                clipping_predictor.reset();
            }
        }

        self.aggregate_channel_levels();
    }

    pub fn recommend_input_volume(
        &mut self,
        speech_probability: f32,
        speech_level_dbfs: Option<f32>,
    ) -> Option<i32> {
        if self.applied_input_volume.is_none() {
            return None;
        }

        self.aggregate_channel_levels();
        let volume_after_clipping_handling = self.recommended_input_volume;

        if !self.capture_output_used {
            return self.applied_input_volume;
        }

        let rms_error_db = speech_level_dbfs.map(|speech_level_dbfs| {
            get_speech_level_rms_error_db(
                speech_level_dbfs,
                self.target_range_min_dbfs,
                self.target_range_max_dbfs,
            )
        });

        for controller in &mut self.channel_controllers {
            controller.process(rms_error_db, speech_probability);
        }

        self.aggregate_channel_levels();
        if volume_after_clipping_handling != self.recommended_input_volume {
            update_histogram_on_recommended_input_volume_change_to_match_target(
                self.recommended_input_volume,
            );
        }
        self.applied_input_volume = None;
        Some(self.recommended_input_volume)
    }

    pub fn handle_capture_output_used_change(&mut self, capture_output_used: bool) {
        for controller in &mut self.channel_controllers {
            controller.handle_capture_output_used_change(capture_output_used);
        }
        self.capture_output_used = capture_output_used;
    }

    pub fn recommended_input_volume(&self) -> i32 {
        self.recommended_input_volume
    }

    fn set_applied_input_volume(&mut self, input_volume: i32) {
        self.applied_input_volume = Some(input_volume);
        for controller in &mut self.channel_controllers {
            controller.set_stream_analog_level(input_volume);
        }
        self.aggregate_channel_levels();
    }

    fn aggregate_channel_levels(&mut self) {
        let mut new_recommended = self.channel_controllers[0].recommended_analog_level();
        self.channel_controlling_gain = 0;
        for (ch, controller) in self.channel_controllers.iter().enumerate().skip(1) {
            let input_volume = controller.recommended_analog_level();
            if input_volume < new_recommended {
                new_recommended = input_volume;
                self.channel_controlling_gain = ch as i32;
            }
        }

        if let Some(applied_input_volume) = self.applied_input_volume {
            if applied_input_volume > 0 {
                new_recommended = max(new_recommended, self.min_input_volume);
            }
        }

        self.recommended_input_volume = new_recommended;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_processing::agc2::input_volume_stats_reporter::{
        on_change_to_match_target_samples, reset_on_change_to_match_target_samples,
    };

    const SAMPLE_RATE_HZ: usize = 32_000;
    const NUM_CHANNELS: usize = 1;
    const DEFAULT_INITIAL_INPUT_VOLUME: i32 = 128;
    const HIGH_SPEECH_PROBABILITY: f32 = 0.7;
    const LOW_SPEECH_PROBABILITY: f32 = 0.1;
    const SPEECH_LEVEL: f32 = -25.0;
    const MAX_SAMPLE_S16: f32 = 32767.0;

    fn create_audio_buffer() -> AudioBuffer {
        AudioBuffer::from_sample_rates(
            SAMPLE_RATE_HZ,
            NUM_CHANNELS,
            SAMPLE_RATE_HZ,
            NUM_CHANNELS,
            SAMPLE_RATE_HZ,
        )
    }

    fn write_audio_buffer_samples(
        samples_value: f32,
        clipped_ratio: f32,
        audio_buffer: &mut AudioBuffer,
    ) {
        assert!(clipped_ratio >= 0.0 && clipped_ratio <= 1.0);
        let num_samples = audio_buffer.num_frames();
        let num_clipping_samples = (clipped_ratio * num_samples as f32) as usize;
        for ch in 0..audio_buffer.num_channels() {
            let channel = audio_buffer.channel_mut(ch);
            for (i, sample) in channel.iter_mut().enumerate() {
                if i < num_clipping_samples {
                    *sample = 32767.0;
                } else {
                    *sample = samples_value;
                }
            }
        }
    }

    fn write_alternating_audio_buffer_samples(samples_value: f32, audio_buffer: &mut AudioBuffer) {
        for ch in 0..audio_buffer.num_channels() {
            let channel = audio_buffer.channel_mut(ch);
            for i in (0..channel.len()).step_by(2) {
                channel[i] = samples_value;
                if i + 1 < channel.len() {
                    channel[i + 1] = 0.0;
                }
            }
        }
    }

    fn parity_config(min_input_volume: i32) -> Config {
        Config {
            min_input_volume,
            clipped_level_min: 165,
            clipped_level_step: 15,
            clipped_ratio_threshold: 0.1,
            clipped_wait_frames: 300,
            enable_clipping_predictor: false,
            target_range_max_dbfs: -18,
            target_range_experimental_max_dbfs: -12,
            target_range_min_dbfs: -30,
            update_input_volume_wait_frames: 0,
            speech_probability_threshold: 0.5,
            speech_ratio_threshold: 1.0,
        }
    }

    fn call_agc_sequence(
        controller: &mut InputVolumeController,
        audio_buffer: &AudioBuffer,
        applied_input_volume: i32,
        speech_probability: f32,
        speech_level_dbfs: Option<f32>,
        num_calls: i32,
    ) -> Option<i32> {
        assert!(num_calls >= 1);
        let mut volume = Some(applied_input_volume);
        for _ in 0..num_calls {
            controller.analyze_input_audio(volume.unwrap_or(applied_input_volume), audio_buffer);
            volume = controller.recommend_input_volume(speech_probability, speech_level_dbfs);
            if let Some(v) = volume {
                assert_eq!(v, controller.recommended_input_volume());
            }
        }
        volume
    }

    fn call_recommend_input_volume(
        controller: &mut InputVolumeController,
        audio_buffer: &AudioBuffer,
        num_calls: i32,
        initial_volume: i32,
        speech_probability: f32,
        speech_level_dbfs: Option<f32>,
    ) -> i32 {
        call_agc_sequence(
            controller,
            audio_buffer,
            initial_volume,
            speech_probability,
            speech_level_dbfs,
            num_calls,
        )
        .expect("volume expected")
    }

    fn update_recommended_input_volume(
        mono_controller: &mut MonoInputVolumeController,
        applied_input_volume: i32,
        speech_probability: f32,
        rms_error_dbfs: Option<i32>,
    ) -> i32 {
        mono_controller.set_stream_analog_level(applied_input_volume);
        assert_eq!(
            mono_controller.recommended_analog_level(),
            applied_input_volume
        );
        mono_controller.process(rms_error_dbfs, speech_probability);
        mono_controller.recommended_analog_level()
    }

    fn feed_frames(
        controller: &mut InputVolumeController,
        audio_buffer: &AudioBuffer,
        num_frames: i32,
        mut applied_input_volume: i32,
        speech_probability: f32,
        speech_level_dbfs: Option<f32>,
    ) -> i32 {
        for _ in 0..num_frames {
            controller.analyze_input_audio(applied_input_volume, audio_buffer);
            applied_input_volume = controller
                .recommend_input_volume(speech_probability, speech_level_dbfs)
                .expect("volume expected");
        }
        applied_input_volume
    }

    #[test]
    fn check_handle_clipping_lowers_volume() {
        const INITIAL_INPUT_VOLUME: i32 = 100;
        const INPUT_VOLUME_STEP: i32 = 29;
        let mut mono_controller = MonoInputVolumeController::new(70, 32, 3, 0.7, 0.8);
        mono_controller.initialize();

        update_recommended_input_volume(&mut mono_controller, INITIAL_INPUT_VOLUME, 0.1, Some(-10));
        mono_controller.handle_clipping(INPUT_VOLUME_STEP);

        assert_eq!(
            mono_controller.recommended_analog_level(),
            INITIAL_INPUT_VOLUME - INPUT_VOLUME_STEP
        );
    }

    #[test]
    fn check_process_negative_rms_error_decreases_input_volume() {
        const INITIAL_INPUT_VOLUME: i32 = 100;
        let mut mono_controller = MonoInputVolumeController::new(64, 32, 3, 0.7, 0.8);
        mono_controller.initialize();

        let mut volume = update_recommended_input_volume(
            &mut mono_controller,
            INITIAL_INPUT_VOLUME,
            0.7,
            Some(-10),
        );
        volume = update_recommended_input_volume(&mut mono_controller, volume, 0.7, Some(-10));
        volume = update_recommended_input_volume(&mut mono_controller, volume, 0.7, Some(-10));

        assert!(volume < INITIAL_INPUT_VOLUME);
    }

    #[test]
    fn check_process_positive_rms_error_increases_input_volume() {
        const INITIAL_INPUT_VOLUME: i32 = 100;
        let mut mono_controller = MonoInputVolumeController::new(64, 32, 3, 0.7, 0.8);
        mono_controller.initialize();

        let mut volume = update_recommended_input_volume(
            &mut mono_controller,
            INITIAL_INPUT_VOLUME,
            0.7,
            Some(10),
        );
        volume = update_recommended_input_volume(&mut mono_controller, volume, 0.7, Some(10));
        volume = update_recommended_input_volume(&mut mono_controller, volume, 0.7, Some(10));

        assert!(volume > INITIAL_INPUT_VOLUME);
    }

    #[test]
    fn check_process_negative_rms_error_decreases_input_volume_with_limit() {
        const INITIAL_INPUT_VOLUME: i32 = 100;
        let mut mono_controller_1 = MonoInputVolumeController::new(64, 32, 2, 0.7, 0.8);
        let mut mono_controller_2 = MonoInputVolumeController::new(64, 32, 2, 0.7, 0.8);
        let mut mono_controller_3 = MonoInputVolumeController::new(64, 32, 2, 0.7, 0.8);
        mono_controller_1.initialize();
        mono_controller_2.initialize();
        mono_controller_3.initialize();

        let mut volume_1 = update_recommended_input_volume(
            &mut mono_controller_1,
            INITIAL_INPUT_VOLUME,
            0.7,
            Some(-14),
        );
        volume_1 =
            update_recommended_input_volume(&mut mono_controller_1, volume_1, 0.7, Some(-14));

        let mut volume_2 = update_recommended_input_volume(
            &mut mono_controller_2,
            INITIAL_INPUT_VOLUME,
            0.7,
            Some(-15),
        );
        let mut volume_3 = update_recommended_input_volume(
            &mut mono_controller_3,
            INITIAL_INPUT_VOLUME,
            0.7,
            Some(-30),
        );
        volume_2 =
            update_recommended_input_volume(&mut mono_controller_2, volume_2, 0.7, Some(-15));
        volume_3 =
            update_recommended_input_volume(&mut mono_controller_3, volume_3, 0.7, Some(-30));

        assert!(volume_1 < INITIAL_INPUT_VOLUME);
        assert!(volume_2 < volume_1);
        assert_eq!(volume_2, volume_3);
    }

    #[test]
    fn check_process_positive_rms_error_increases_input_volume_with_limit() {
        const INITIAL_INPUT_VOLUME: i32 = 100;
        let mut mono_controller_1 = MonoInputVolumeController::new(64, 32, 2, 0.7, 0.8);
        let mut mono_controller_2 = MonoInputVolumeController::new(64, 32, 2, 0.7, 0.8);
        let mut mono_controller_3 = MonoInputVolumeController::new(64, 32, 2, 0.7, 0.8);
        mono_controller_1.initialize();
        mono_controller_2.initialize();
        mono_controller_3.initialize();

        let mut volume_1 = update_recommended_input_volume(
            &mut mono_controller_1,
            INITIAL_INPUT_VOLUME,
            0.7,
            Some(14),
        );
        volume_1 = update_recommended_input_volume(&mut mono_controller_1, volume_1, 0.7, Some(14));

        let mut volume_2 = update_recommended_input_volume(
            &mut mono_controller_2,
            INITIAL_INPUT_VOLUME,
            0.7,
            Some(15),
        );
        let mut volume_3 = update_recommended_input_volume(
            &mut mono_controller_3,
            INITIAL_INPUT_VOLUME,
            0.7,
            Some(30),
        );
        volume_2 = update_recommended_input_volume(&mut mono_controller_2, volume_2, 0.7, Some(15));
        volume_3 = update_recommended_input_volume(&mut mono_controller_3, volume_3, 0.7, Some(30));

        assert!(volume_1 > INITIAL_INPUT_VOLUME);
        assert!(volume_2 > volume_1);
        assert_eq!(volume_2, volume_3);
    }

    #[test]
    fn check_clipped_level_min_is_effective() {
        const INITIAL_INPUT_VOLUME: i32 = 100;
        const CLIPPED_LEVEL_MIN: i32 = 70;
        let mut mono_controller_1 =
            MonoInputVolumeController::new(CLIPPED_LEVEL_MIN, 84, 2, 0.7, 0.8);
        let mut mono_controller_2 =
            MonoInputVolumeController::new(CLIPPED_LEVEL_MIN, 84, 2, 0.7, 0.8);
        mono_controller_1.initialize();
        mono_controller_2.initialize();

        assert_eq!(
            update_recommended_input_volume(
                &mut mono_controller_1,
                INITIAL_INPUT_VOLUME,
                0.1,
                Some(-10)
            ),
            INITIAL_INPUT_VOLUME
        );
        assert_eq!(
            update_recommended_input_volume(
                &mut mono_controller_2,
                INITIAL_INPUT_VOLUME,
                0.1,
                Some(-10)
            ),
            INITIAL_INPUT_VOLUME
        );

        mono_controller_1.handle_clipping(29);
        mono_controller_2.handle_clipping(31);

        assert_eq!(
            mono_controller_2.recommended_analog_level(),
            CLIPPED_LEVEL_MIN
        );
        assert!(
            mono_controller_2.recommended_analog_level()
                < mono_controller_1.recommended_analog_level()
        );
    }

    #[test]
    fn check_process_empty_rms_error_does_not_lower_volume() {
        const INITIAL_INPUT_VOLUME: i32 = 100;
        let mut mono_controller_1 = MonoInputVolumeController::new(64, 84, 2, 0.7, 0.8);
        let mut mono_controller_2 = MonoInputVolumeController::new(64, 84, 2, 0.7, 0.8);
        mono_controller_1.initialize();
        mono_controller_2.initialize();

        let mut volume_1 = update_recommended_input_volume(
            &mut mono_controller_1,
            INITIAL_INPUT_VOLUME,
            0.7,
            Some(-10),
        );
        let mut volume_2 = update_recommended_input_volume(
            &mut mono_controller_2,
            INITIAL_INPUT_VOLUME,
            0.7,
            Some(-10),
        );

        assert_eq!(volume_1, INITIAL_INPUT_VOLUME);
        assert_eq!(volume_2, INITIAL_INPUT_VOLUME);

        volume_1 = update_recommended_input_volume(&mut mono_controller_1, volume_1, 0.7, None);
        volume_2 =
            update_recommended_input_volume(&mut mono_controller_2, volume_2, 0.7, Some(-10));

        assert_eq!(volume_1, INITIAL_INPUT_VOLUME);
        assert!(volume_2 < volume_1);
    }

    #[test]
    fn no_clipping_has_no_impact() {
        let config = Config {
            min_input_volume: 20,
            enable_clipping_predictor: false,
            ..Default::default()
        };
        let mut controller = InputVolumeController::new(1, config);
        controller.initialize();

        let mut audio_buffer = create_audio_buffer();
        write_audio_buffer_samples(0.0, 0.0, &mut audio_buffer);

        let _ = call_agc_sequence(
            &mut controller,
            &audio_buffer,
            DEFAULT_INITIAL_INPUT_VOLUME,
            HIGH_SPEECH_PROBABILITY,
            Some(SPEECH_LEVEL),
            1,
        );

        for _ in 0..100 {
            let current_volume = controller.recommended_input_volume();
            let volume = call_agc_sequence(
                &mut controller,
                &audio_buffer,
                current_volume,
                LOW_SPEECH_PROBABILITY,
                Some(SPEECH_LEVEL),
                1,
            )
            .expect("volume expected");
            assert_eq!(volume, DEFAULT_INITIAL_INPUT_VOLUME);
        }
    }

    #[test]
    fn clipping_under_threshold_has_no_impact() {
        let config = Config {
            min_input_volume: 20,
            enable_clipping_predictor: false,
            clipped_ratio_threshold: 0.1,
            ..Default::default()
        };
        let mut controller = InputVolumeController::new(1, config);
        controller.initialize();

        let mut audio_buffer = create_audio_buffer();
        write_audio_buffer_samples(0.0, 0.099, &mut audio_buffer);

        let _ = call_agc_sequence(
            &mut controller,
            &audio_buffer,
            DEFAULT_INITIAL_INPUT_VOLUME,
            HIGH_SPEECH_PROBABILITY,
            Some(SPEECH_LEVEL),
            1,
        );

        assert_eq!(
            controller.recommended_input_volume(),
            DEFAULT_INITIAL_INPUT_VOLUME
        );
    }

    #[test]
    fn clipping_lowers_volume() {
        let config = Config {
            min_input_volume: 20,
            enable_clipping_predictor: false,
            clipped_level_step: 15,
            clipped_ratio_threshold: 0.1,
            ..Default::default()
        };
        let mut controller = InputVolumeController::new(1, config);
        controller.initialize();

        let mut audio_buffer = create_audio_buffer();
        // Prime the controller state on non-clipping audio first.
        write_audio_buffer_samples(1000.0, 0.0, &mut audio_buffer);
        let _ = call_agc_sequence(
            &mut controller,
            &audio_buffer,
            255,
            HIGH_SPEECH_PROBABILITY,
            Some(SPEECH_LEVEL),
            1,
        );

        write_audio_buffer_samples(0.0, 0.2, &mut audio_buffer);

        let _ = call_agc_sequence(
            &mut controller,
            &audio_buffer,
            255,
            HIGH_SPEECH_PROBABILITY,
            Some(SPEECH_LEVEL),
            1,
        );

        assert_eq!(controller.recommended_input_volume(), 240);
    }

    #[test]
    fn waiting_period_between_clipping_checks() {
        let config = Config {
            min_input_volume: 20,
            enable_clipping_predictor: false,
            clipped_level_step: 15,
            clipped_ratio_threshold: 0.1,
            clipped_wait_frames: 300,
            ..Default::default()
        };
        let mut controller = InputVolumeController::new(1, config);
        controller.initialize();

        let mut audio_buffer = create_audio_buffer();
        // Prime the controller state on non-clipping audio first.
        write_audio_buffer_samples(1000.0, 0.0, &mut audio_buffer);
        let _ = call_agc_sequence(
            &mut controller,
            &audio_buffer,
            255,
            HIGH_SPEECH_PROBABILITY,
            Some(SPEECH_LEVEL),
            1,
        );

        write_audio_buffer_samples(0.0, 0.2, &mut audio_buffer);

        let mut volume = call_agc_sequence(
            &mut controller,
            &audio_buffer,
            255,
            LOW_SPEECH_PROBABILITY,
            Some(SPEECH_LEVEL),
            1,
        )
        .expect("volume expected");
        assert_eq!(volume, 240);

        for _ in 0..300 {
            volume = call_agc_sequence(
                &mut controller,
                &audio_buffer,
                volume,
                LOW_SPEECH_PROBABILITY,
                Some(SPEECH_LEVEL),
                1,
            )
            .expect("volume expected");
        }
        assert_eq!(volume, 240);

        volume = call_agc_sequence(
            &mut controller,
            &audio_buffer,
            volume,
            LOW_SPEECH_PROBABILITY,
            Some(SPEECH_LEVEL),
            1,
        )
        .expect("volume expected");
        assert_eq!(volume, 225);
    }

    #[test]
    fn takes_no_action_on_zero_mic_volume_after_startup() {
        let config = Config {
            min_input_volume: 20,
            enable_clipping_predictor: false,
            ..Default::default()
        };
        let mut controller = InputVolumeController::new(1, config);
        controller.initialize();

        let mut audio_buffer = create_audio_buffer();
        write_audio_buffer_samples(1000.0, 0.0, &mut audio_buffer);

        let _ = call_agc_sequence(
            &mut controller,
            &audio_buffer,
            DEFAULT_INITIAL_INPUT_VOLUME,
            HIGH_SPEECH_PROBABILITY,
            Some(SPEECH_LEVEL),
            1,
        );

        let volume = call_agc_sequence(
            &mut controller,
            &audio_buffer,
            0,
            HIGH_SPEECH_PROBABILITY,
            Some(-48.0),
            10,
        )
        .expect("volume expected");
        assert_eq!(volume, 0);
    }

    #[test]
    fn startup_min_volume_configuration_respected_when_applied_input_volume_above_min() {
        for min_input_volume in [12, 20] {
            let config = parity_config(min_input_volume);
            let mut controller = InputVolumeController::new(1, config);
            controller.initialize();

            let mut audio_buffer = create_audio_buffer();
            write_audio_buffer_samples(0.0, 0.0, &mut audio_buffer);

            let volume =
                call_agc_sequence(&mut controller, &audio_buffer, 128, 0.9, Some(-80.0), 1)
                    .expect("volume expected");
            assert_eq!(volume, 128);
        }
    }

    #[test]
    fn startup_min_volume_configuration_respected_when_applied_input_volume_maybe_below_min() {
        for min_input_volume in [12, 20] {
            let config = parity_config(min_input_volume);
            let mut controller = InputVolumeController::new(1, config);
            controller.initialize();

            let mut audio_buffer = create_audio_buffer();
            write_audio_buffer_samples(0.0, 0.0, &mut audio_buffer);

            let volume = call_agc_sequence(&mut controller, &audio_buffer, 10, 0.9, Some(-80.0), 1)
                .expect("volume expected");
            assert!(volume >= 10);
        }
    }

    #[test]
    fn startup_min_volume_respected_once_when_applied_volume_zero() {
        for min_input_volume in [12, 20] {
            let mut config = parity_config(min_input_volume);
            config.update_input_volume_wait_frames = 1;
            config.speech_ratio_threshold = 0.5;

            let mut controller = InputVolumeController::new(1, config);
            controller.initialize();

            let mut audio_buffer = create_audio_buffer();
            write_audio_buffer_samples(0.0, 0.0, &mut audio_buffer);

            let volume = call_agc_sequence(&mut controller, &audio_buffer, 0, 0.9, Some(-80.0), 1)
                .expect("volume expected");
            assert_eq!(volume, min_input_volume);

            let volume = call_agc_sequence(&mut controller, &audio_buffer, 0, 0.9, Some(-80.0), 1)
                .expect("volume expected");
            assert_eq!(volume, 0);
        }
    }

    #[test]
    fn no_action_while_muted() {
        for min_input_volume in [12, 20] {
            let config = parity_config(min_input_volume);
            let mut controller_1 = InputVolumeController::new(1, config);
            let mut controller_2 = InputVolumeController::new(1, config);
            controller_1.initialize();
            controller_2.initialize();

            let mut audio_buffer_1 = create_audio_buffer();
            let mut audio_buffer_2 = create_audio_buffer();
            write_audio_buffer_samples(1000.0, 0.0, &mut audio_buffer_1);
            write_audio_buffer_samples(1000.0, 0.0, &mut audio_buffer_2);

            let mut volume_1 = call_agc_sequence(
                &mut controller_1,
                &audio_buffer_1,
                255,
                HIGH_SPEECH_PROBABILITY,
                Some(SPEECH_LEVEL),
                1,
            )
            .expect("volume expected");
            let mut volume_2 = call_agc_sequence(
                &mut controller_2,
                &audio_buffer_2,
                255,
                HIGH_SPEECH_PROBABILITY,
                Some(SPEECH_LEVEL),
                1,
            )
            .expect("volume expected");

            assert_eq!(volume_1, 255);
            assert_eq!(volume_2, 255);

            controller_2.handle_capture_output_used_change(false);

            write_alternating_audio_buffer_samples(32767.0, &mut audio_buffer_1);
            write_alternating_audio_buffer_samples(32767.0, &mut audio_buffer_2);

            volume_1 = call_agc_sequence(
                &mut controller_1,
                &audio_buffer_1,
                volume_1,
                HIGH_SPEECH_PROBABILITY,
                Some(SPEECH_LEVEL),
                1,
            )
            .expect("volume expected");
            volume_2 = call_agc_sequence(
                &mut controller_2,
                &audio_buffer_2,
                volume_2,
                HIGH_SPEECH_PROBABILITY,
                Some(SPEECH_LEVEL),
                1,
            )
            .expect("volume expected");

            assert!(volume_1 < 255);
            assert_eq!(volume_2, 255);
        }
    }

    #[test]
    fn unmuting_checks_volume_without_raising() {
        for min_input_volume in [12, 20] {
            let config = parity_config(min_input_volume);
            let mut controller = InputVolumeController::new(1, config);
            controller.initialize();

            let mut audio_buffer = create_audio_buffer();
            write_audio_buffer_samples(1000.0, 0.0, &mut audio_buffer);

            let _ = call_agc_sequence(
                &mut controller,
                &audio_buffer,
                DEFAULT_INITIAL_INPUT_VOLUME,
                HIGH_SPEECH_PROBABILITY,
                Some(SPEECH_LEVEL),
                1,
            );

            controller.handle_capture_output_used_change(false);
            controller.handle_capture_output_used_change(true);

            let volume = call_agc_sequence(
                &mut controller,
                &audio_buffer,
                127,
                HIGH_SPEECH_PROBABILITY,
                Some(SPEECH_LEVEL),
                1,
            )
            .expect("volume expected");
            assert_eq!(volume, 127);
        }
    }

    #[test]
    fn unmuting_raises_too_low_volume() {
        for min_input_volume in [12, 20] {
            let config = parity_config(min_input_volume);
            let mut controller = InputVolumeController::new(1, config);
            controller.initialize();

            let mut audio_buffer = create_audio_buffer();
            write_audio_buffer_samples(1000.0, 0.0, &mut audio_buffer);

            let _ = call_agc_sequence(
                &mut controller,
                &audio_buffer,
                DEFAULT_INITIAL_INPUT_VOLUME,
                HIGH_SPEECH_PROBABILITY,
                Some(SPEECH_LEVEL),
                1,
            );

            controller.handle_capture_output_used_change(false);
            controller.handle_capture_output_used_change(true);

            let volume = call_agc_sequence(
                &mut controller,
                &audio_buffer,
                11,
                HIGH_SPEECH_PROBABILITY,
                Some(SPEECH_LEVEL),
                1,
            )
            .expect("volume expected");
            assert_eq!(volume, min_input_volume);
        }
    }

    #[test]
    fn manual_level_change_results_in_no_set_mic_call() {
        for min_input_volume in [12, 20] {
            let config = parity_config(min_input_volume);
            let mut controller = InputVolumeController::new(1, config);
            controller.initialize();

            let mut audio_buffer = create_audio_buffer();
            write_alternating_audio_buffer_samples(3276.7, &mut audio_buffer);

            let mut volume = call_agc_sequence(
                &mut controller,
                &audio_buffer,
                DEFAULT_INITIAL_INPUT_VOLUME,
                HIGH_SPEECH_PROBABILITY,
                Some(SPEECH_LEVEL),
                1,
            )
            .expect("volume expected");

            assert_ne!(volume, 154);

            volume = call_agc_sequence(
                &mut controller,
                &audio_buffer,
                154,
                HIGH_SPEECH_PROBABILITY,
                Some(-29.0),
                1,
            )
            .expect("volume expected");
            assert_eq!(volume, 154);

            volume = call_agc_sequence(
                &mut controller,
                &audio_buffer,
                100,
                HIGH_SPEECH_PROBABILITY,
                Some(-17.0),
                1,
            )
            .expect("volume expected");
            assert_eq!(volume, 100);

            volume = call_agc_sequence(
                &mut controller,
                &audio_buffer,
                volume,
                HIGH_SPEECH_PROBABILITY,
                Some(-17.0),
                1,
            )
            .expect("volume expected");
            assert_eq!(volume, 99);
        }
    }

    #[test]
    fn recovery_after_manual_level_change_from_max() {
        for min_input_volume in [12, 20] {
            let config = parity_config(min_input_volume);
            let mut controller = InputVolumeController::new(1, config);
            controller.initialize();

            let mut audio_buffer = create_audio_buffer();
            write_alternating_audio_buffer_samples(3276.7, &mut audio_buffer);

            let mut volume = call_agc_sequence(
                &mut controller,
                &audio_buffer,
                DEFAULT_INITIAL_INPUT_VOLUME,
                HIGH_SPEECH_PROBABILITY,
                Some(SPEECH_LEVEL),
                1,
            )
            .expect("volume expected");

            volume = call_agc_sequence(
                &mut controller,
                &audio_buffer,
                volume,
                HIGH_SPEECH_PROBABILITY,
                Some(-48.0),
                1,
            )
            .expect("volume expected");
            assert_eq!(volume, 183);

            volume = call_agc_sequence(
                &mut controller,
                &audio_buffer,
                volume,
                HIGH_SPEECH_PROBABILITY,
                Some(-48.0),
                1,
            )
            .expect("volume expected");
            assert_eq!(volume, 243);

            volume = call_agc_sequence(
                &mut controller,
                &audio_buffer,
                volume,
                HIGH_SPEECH_PROBABILITY,
                Some(-48.0),
                1,
            )
            .expect("volume expected");
            assert_eq!(volume, 255);

            volume = call_agc_sequence(
                &mut controller,
                &audio_buffer,
                50,
                HIGH_SPEECH_PROBABILITY,
                Some(-17.0),
                1,
            )
            .expect("volume expected");
            assert_eq!(controller.recommended_input_volume(), 50);

            volume = call_agc_sequence(
                &mut controller,
                &audio_buffer,
                volume,
                HIGH_SPEECH_PROBABILITY,
                Some(-38.0),
                1,
            )
            .expect("volume expected");

            assert_eq!(volume, 65);
        }
    }

    #[test]
    fn clipping_lowering_is_limited() {
        for min_input_volume in [12, 20] {
            let config = parity_config(min_input_volume);
            let mut controller = InputVolumeController::new(1, config);
            controller.initialize();

            let mut audio_buffer = create_audio_buffer();
            write_audio_buffer_samples(1000.0, 0.0, &mut audio_buffer);
            let _ = call_agc_sequence(
                &mut controller,
                &audio_buffer,
                180,
                HIGH_SPEECH_PROBABILITY,
                Some(SPEECH_LEVEL),
                1,
            );

            write_audio_buffer_samples(0.0, 0.2, &mut audio_buffer);
            let _ = call_agc_sequence(
                &mut controller,
                &audio_buffer,
                180,
                HIGH_SPEECH_PROBABILITY,
                Some(SPEECH_LEVEL),
                1,
            );
            assert_eq!(controller.recommended_input_volume(), 165);

            let mut volume = controller.recommended_input_volume();
            for _ in 0..1000 {
                volume = call_agc_sequence(
                    &mut controller,
                    &audio_buffer,
                    volume,
                    HIGH_SPEECH_PROBABILITY,
                    Some(SPEECH_LEVEL),
                    1,
                )
                .expect("volume expected");
            }
            assert_eq!(volume, 165);
        }
    }

    #[test]
    fn clipping_max_is_respected_when_equal_to_level() {
        for min_input_volume in [12, 20] {
            let config = parity_config(min_input_volume);
            let mut controller = InputVolumeController::new(1, config);
            controller.initialize();

            let mut audio_buffer = create_audio_buffer();
            write_audio_buffer_samples(1000.0, 0.0, &mut audio_buffer);
            let _ = call_agc_sequence(
                &mut controller,
                &audio_buffer,
                255,
                HIGH_SPEECH_PROBABILITY,
                Some(SPEECH_LEVEL),
                1,
            );

            write_audio_buffer_samples(0.0, 0.2, &mut audio_buffer);
            let _ = call_agc_sequence(
                &mut controller,
                &audio_buffer,
                255,
                HIGH_SPEECH_PROBABILITY,
                Some(SPEECH_LEVEL),
                1,
            );
            assert_eq!(controller.recommended_input_volume(), 240);

            write_alternating_audio_buffer_samples(3276.7, &mut audio_buffer);
            let mut volume = controller.recommended_input_volume();
            for _ in 0..10 {
                volume = call_agc_sequence(
                    &mut controller,
                    &audio_buffer,
                    volume,
                    HIGH_SPEECH_PROBABILITY,
                    Some(-48.0),
                    1,
                )
                .expect("volume expected");
            }
            assert_eq!(volume, 240);
        }
    }

    #[test]
    fn clipping_max_is_respected_when_higher_than_level() {
        for min_input_volume in [12, 20] {
            let config = parity_config(min_input_volume);
            let mut controller = InputVolumeController::new(1, config);
            controller.initialize();

            let mut audio_buffer = create_audio_buffer();
            write_audio_buffer_samples(1000.0, 0.0, &mut audio_buffer);
            let _ = call_agc_sequence(
                &mut controller,
                &audio_buffer,
                200,
                HIGH_SPEECH_PROBABILITY,
                Some(SPEECH_LEVEL),
                1,
            );

            write_audio_buffer_samples(0.0, 0.2, &mut audio_buffer);
            let mut volume = call_agc_sequence(
                &mut controller,
                &audio_buffer,
                200,
                HIGH_SPEECH_PROBABILITY,
                Some(SPEECH_LEVEL),
                1,
            )
            .expect("volume expected");
            assert_eq!(volume, 185);

            write_alternating_audio_buffer_samples(3276.7, &mut audio_buffer);
            volume = call_agc_sequence(
                &mut controller,
                &audio_buffer,
                volume,
                HIGH_SPEECH_PROBABILITY,
                Some(-58.0),
                1,
            )
            .expect("volume expected");
            assert_eq!(volume, 240);

            volume = call_agc_sequence(
                &mut controller,
                &audio_buffer,
                volume,
                HIGH_SPEECH_PROBABILITY,
                Some(-58.0),
                10,
            )
            .expect("volume expected");
            assert_eq!(volume, 240);
        }
    }

    #[test]
    fn user_can_raise_volume_after_clipping() {
        for min_input_volume in [12, 20] {
            let config = parity_config(min_input_volume);
            let mut controller = InputVolumeController::new(1, config);
            controller.initialize();

            let mut audio_buffer = create_audio_buffer();
            write_audio_buffer_samples(1000.0, 0.0, &mut audio_buffer);
            let _ = call_agc_sequence(
                &mut controller,
                &audio_buffer,
                225,
                HIGH_SPEECH_PROBABILITY,
                Some(SPEECH_LEVEL),
                1,
            );

            write_audio_buffer_samples(0.0, 0.2, &mut audio_buffer);
            let _ = call_agc_sequence(
                &mut controller,
                &audio_buffer,
                225,
                HIGH_SPEECH_PROBABILITY,
                Some(SPEECH_LEVEL),
                1,
            );
            assert_eq!(controller.recommended_input_volume(), 210);

            write_alternating_audio_buffer_samples(3276.7, &mut audio_buffer);
            let mut volume = call_agc_sequence(
                &mut controller,
                &audio_buffer,
                250,
                HIGH_SPEECH_PROBABILITY,
                Some(-32.0),
                1,
            )
            .expect("volume expected");
            assert_eq!(volume, 250);

            volume = call_agc_sequence(
                &mut controller,
                &audio_buffer,
                volume,
                HIGH_SPEECH_PROBABILITY,
                Some(-8.0),
                1,
            )
            .expect("volume expected");
            assert_eq!(volume, 210);

            volume = call_agc_sequence(
                &mut controller,
                &audio_buffer,
                volume,
                HIGH_SPEECH_PROBABILITY,
                Some(-58.0),
                1,
            )
            .expect("volume expected");
            assert_eq!(volume, 250);

            volume = call_agc_sequence(
                &mut controller,
                &audio_buffer,
                volume,
                HIGH_SPEECH_PROBABILITY,
                Some(-48.0),
                1,
            )
            .expect("volume expected");
            assert_eq!(volume, 250);
        }
    }

    #[test]
    fn clipping_does_not_pull_low_volume_back_up() {
        for min_input_volume in [12, 20] {
            let config = parity_config(min_input_volume);
            let mut controller = InputVolumeController::new(1, config);
            controller.initialize();

            let mut audio_buffer = create_audio_buffer();
            write_audio_buffer_samples(1000.0, 0.0, &mut audio_buffer);
            let _ = call_agc_sequence(
                &mut controller,
                &audio_buffer,
                80,
                HIGH_SPEECH_PROBABILITY,
                Some(SPEECH_LEVEL),
                1,
            );

            let initial_volume = controller.recommended_input_volume();
            write_audio_buffer_samples(0.0, 0.2, &mut audio_buffer);
            let _ = call_agc_sequence(
                &mut controller,
                &audio_buffer,
                initial_volume,
                HIGH_SPEECH_PROBABILITY,
                Some(SPEECH_LEVEL),
                1,
            );

            assert_eq!(controller.recommended_input_volume(), initial_volume);
        }
    }

    #[test]
    fn enforce_min_input_volume_during_upwards_adjustment() {
        for min_input_volume in [12, 20] {
            let mut config = parity_config(min_input_volume);
            config.target_range_min_dbfs = -30;
            config.update_input_volume_wait_frames = 1;
            config.speech_probability_threshold = 0.5;
            config.speech_ratio_threshold = 0.5;

            let mut controller = InputVolumeController::new(1, config);
            controller.initialize();

            let mut audio_buffer = create_audio_buffer();
            write_alternating_audio_buffer_samples(3276.7, &mut audio_buffer);

            let _ = call_agc_sequence(
                &mut controller,
                &audio_buffer,
                DEFAULT_INITIAL_INPUT_VOLUME,
                HIGH_SPEECH_PROBABILITY,
                Some(SPEECH_LEVEL),
                1,
            );

            let mut volume = call_agc_sequence(
                &mut controller,
                &audio_buffer,
                1,
                HIGH_SPEECH_PROBABILITY,
                Some(-17.0),
                1,
            )
            .expect("volume expected");

            assert_eq!(volume, min_input_volume);

            volume = call_agc_sequence(
                &mut controller,
                &audio_buffer,
                volume,
                HIGH_SPEECH_PROBABILITY,
                Some(-29.0),
                1,
            )
            .expect("volume expected");
            assert_eq!(volume, min_input_volume);

            volume = call_agc_sequence(
                &mut controller,
                &audio_buffer,
                volume,
                HIGH_SPEECH_PROBABILITY,
                Some(-30.0),
                1,
            )
            .expect("volume expected");
            assert_eq!(volume, min_input_volume);

            volume = call_agc_sequence(
                &mut controller,
                &audio_buffer,
                volume,
                HIGH_SPEECH_PROBABILITY,
                Some(-38.0),
                10,
            )
            .expect("volume expected");
            assert!(volume > min_input_volume);
        }
    }

    #[test]
    fn recovery_after_manual_level_change_below_min() {
        for min_input_volume in [12, 20] {
            let config = parity_config(min_input_volume);
            let mut controller = InputVolumeController::new(1, config);
            controller.initialize();

            let mut audio_buffer = create_audio_buffer();
            write_alternating_audio_buffer_samples(3276.7, &mut audio_buffer);

            let _ = call_agc_sequence(
                &mut controller,
                &audio_buffer,
                DEFAULT_INITIAL_INPUT_VOLUME,
                HIGH_SPEECH_PROBABILITY,
                Some(SPEECH_LEVEL),
                1,
            );

            let volume = call_agc_sequence(
                &mut controller,
                &audio_buffer,
                1,
                HIGH_SPEECH_PROBABILITY,
                Some(-17.0),
                1,
            )
            .expect("volume expected");

            assert_eq!(volume, min_input_volume);
        }
    }

    #[test]
    fn mic_volume_response_to_rms_error() {
        for min_input_volume in [12, 20] {
            let config = parity_config(min_input_volume);
            let mut controller = InputVolumeController::new(1, config);
            controller.initialize();

            let mut audio_buffer = create_audio_buffer();
            write_alternating_audio_buffer_samples(3276.7, &mut audio_buffer);

            let mut volume = call_agc_sequence(
                &mut controller,
                &audio_buffer,
                DEFAULT_INITIAL_INPUT_VOLUME,
                HIGH_SPEECH_PROBABILITY,
                Some(SPEECH_LEVEL),
                1,
            )
            .expect("volume expected");

            volume = call_recommend_input_volume(
                &mut controller,
                &audio_buffer,
                1,
                volume,
                HIGH_SPEECH_PROBABILITY,
                Some(-23.0),
            );
            volume = call_recommend_input_volume(
                &mut controller,
                &audio_buffer,
                1,
                volume,
                HIGH_SPEECH_PROBABILITY,
                Some(-28.0),
            );

            volume = call_recommend_input_volume(
                &mut controller,
                &audio_buffer,
                1,
                volume,
                HIGH_SPEECH_PROBABILITY,
                Some(-29.0),
            );
            assert_eq!(volume, 128);

            volume = call_recommend_input_volume(
                &mut controller,
                &audio_buffer,
                1,
                volume,
                HIGH_SPEECH_PROBABILITY,
                Some(-38.0),
            );
            assert_eq!(volume, 156);

            volume = call_recommend_input_volume(
                &mut controller,
                &audio_buffer,
                1,
                volume,
                HIGH_SPEECH_PROBABILITY,
                Some(-23.0),
            );
            volume = call_recommend_input_volume(
                &mut controller,
                &audio_buffer,
                1,
                volume,
                HIGH_SPEECH_PROBABILITY,
                Some(-18.0),
            );

            volume = call_recommend_input_volume(
                &mut controller,
                &audio_buffer,
                1,
                volume,
                HIGH_SPEECH_PROBABILITY,
                Some(-17.0),
            );
            assert_eq!(volume, 155);

            volume = call_recommend_input_volume(
                &mut controller,
                &audio_buffer,
                1,
                volume,
                HIGH_SPEECH_PROBABILITY,
                Some(-17.0),
            );
            assert_eq!(volume, 151);

            volume = call_recommend_input_volume(
                &mut controller,
                &audio_buffer,
                1,
                volume,
                HIGH_SPEECH_PROBABILITY,
                Some(-9.0),
            );
            assert_eq!(volume, 119);
        }
    }

    #[test]
    fn mic_volume_is_limited() {
        for min_input_volume in [12, 20] {
            let config = parity_config(min_input_volume);
            let mut controller = InputVolumeController::new(1, config);
            controller.initialize();

            let mut audio_buffer = create_audio_buffer();
            write_alternating_audio_buffer_samples(3276.7, &mut audio_buffer);

            let mut volume = call_agc_sequence(
                &mut controller,
                &audio_buffer,
                DEFAULT_INITIAL_INPUT_VOLUME,
                HIGH_SPEECH_PROBABILITY,
                Some(SPEECH_LEVEL),
                1,
            )
            .expect("volume expected");

            volume = call_recommend_input_volume(
                &mut controller,
                &audio_buffer,
                1,
                volume,
                HIGH_SPEECH_PROBABILITY,
                Some(-48.0),
            );
            assert_eq!(volume, 183);

            volume = call_recommend_input_volume(
                &mut controller,
                &audio_buffer,
                1,
                volume,
                HIGH_SPEECH_PROBABILITY,
                Some(-48.0),
            );
            assert_eq!(volume, 243);

            volume = call_recommend_input_volume(
                &mut controller,
                &audio_buffer,
                1,
                volume,
                HIGH_SPEECH_PROBABILITY,
                Some(-48.0),
            );
            assert_eq!(volume, 255);

            volume = call_recommend_input_volume(
                &mut controller,
                &audio_buffer,
                1,
                volume,
                HIGH_SPEECH_PROBABILITY,
                Some(-17.0),
            );
            assert_eq!(volume, 254);

            volume = call_recommend_input_volume(
                &mut controller,
                &audio_buffer,
                1,
                volume,
                HIGH_SPEECH_PROBABILITY,
                Some(22.0),
            );
            assert_eq!(volume, 194);

            volume = call_recommend_input_volume(
                &mut controller,
                &audio_buffer,
                1,
                volume,
                HIGH_SPEECH_PROBABILITY,
                Some(22.0),
            );
            assert_eq!(volume, 137);

            volume = call_recommend_input_volume(
                &mut controller,
                &audio_buffer,
                1,
                volume,
                HIGH_SPEECH_PROBABILITY,
                Some(22.0),
            );
            assert_eq!(volume, 88);

            volume = call_recommend_input_volume(
                &mut controller,
                &audio_buffer,
                1,
                volume,
                HIGH_SPEECH_PROBABILITY,
                Some(22.0),
            );
            assert_eq!(volume, 54);

            volume = call_recommend_input_volume(
                &mut controller,
                &audio_buffer,
                1,
                volume,
                HIGH_SPEECH_PROBABILITY,
                Some(22.0),
            );
            assert_eq!(volume, 33);

            volume = call_recommend_input_volume(
                &mut controller,
                &audio_buffer,
                1,
                volume,
                HIGH_SPEECH_PROBABILITY,
                Some(22.0),
            );
            assert_eq!(volume, max(18, min_input_volume));

            volume = call_recommend_input_volume(
                &mut controller,
                &audio_buffer,
                1,
                volume,
                HIGH_SPEECH_PROBABILITY,
                Some(22.0),
            );
            assert_eq!(volume, max(12, min_input_volume));
        }
    }

    #[test]
    fn update_input_volume_wait_frames_is_effective_controller_level() {
        let mut config_wait_0 = parity_config(20);
        config_wait_0.update_input_volume_wait_frames = 0;
        config_wait_0.speech_probability_threshold = 0.5;
        config_wait_0.speech_ratio_threshold = 0.8;

        let mut config_wait_100 = config_wait_0;
        config_wait_100.update_input_volume_wait_frames = 100;

        let mut controller_wait_0 = InputVolumeController::new(1, config_wait_0);
        let mut controller_wait_100 = InputVolumeController::new(1, config_wait_100);
        controller_wait_0.initialize();
        controller_wait_100.initialize();

        let mut audio_buffer = create_audio_buffer();
        write_alternating_audio_buffer_samples(3276.7, &mut audio_buffer);

        let input_volume = DEFAULT_INITIAL_INPUT_VOLUME;
        let mut volume_wait_0 = feed_frames(
            &mut controller_wait_0,
            &audio_buffer,
            99,
            input_volume,
            HIGH_SPEECH_PROBABILITY,
            Some(-42.0),
        );
        let mut volume_wait_100 = feed_frames(
            &mut controller_wait_100,
            &audio_buffer,
            99,
            input_volume,
            HIGH_SPEECH_PROBABILITY,
            Some(-42.0),
        );

        assert!(volume_wait_0 > input_volume);
        assert_eq!(volume_wait_100, input_volume);

        volume_wait_0 = feed_frames(
            &mut controller_wait_0,
            &audio_buffer,
            1,
            volume_wait_0,
            HIGH_SPEECH_PROBABILITY,
            Some(-42.0),
        );
        volume_wait_100 = feed_frames(
            &mut controller_wait_100,
            &audio_buffer,
            1,
            volume_wait_100,
            HIGH_SPEECH_PROBABILITY,
            Some(-42.0),
        );

        assert!(volume_wait_0 > input_volume);
        assert!(volume_wait_100 > input_volume);
    }

    #[test]
    fn speech_ratio_threshold_is_effective_controller_level() {
        let mut config = parity_config(20);
        config.update_input_volume_wait_frames = 10;
        config.speech_probability_threshold = 0.5;
        config.speech_ratio_threshold = 0.8;

        let mut controller_1 = InputVolumeController::new(1, config);
        let mut controller_2 = InputVolumeController::new(1, config);
        controller_1.initialize();
        controller_2.initialize();

        let mut audio_buffer = create_audio_buffer();
        write_alternating_audio_buffer_samples(3276.7, &mut audio_buffer);

        let input_volume = DEFAULT_INITIAL_INPUT_VOLUME;

        let mut volume_1 = feed_frames(
            &mut controller_1,
            &audio_buffer,
            1,
            input_volume,
            0.7,
            Some(-42.0),
        );
        let mut volume_2 = feed_frames(
            &mut controller_2,
            &audio_buffer,
            1,
            input_volume,
            0.4,
            Some(-42.0),
        );
        assert_eq!(volume_1, input_volume);
        assert_eq!(volume_2, input_volume);

        volume_1 = feed_frames(
            &mut controller_1,
            &audio_buffer,
            2,
            volume_1,
            0.4,
            Some(-42.0),
        );
        volume_2 = feed_frames(
            &mut controller_2,
            &audio_buffer,
            2,
            volume_2,
            0.4,
            Some(-42.0),
        );
        assert_eq!(volume_1, input_volume);
        assert_eq!(volume_2, input_volume);

        volume_1 = feed_frames(
            &mut controller_1,
            &audio_buffer,
            7,
            volume_1,
            0.7,
            Some(-42.0),
        );
        volume_2 = feed_frames(
            &mut controller_2,
            &audio_buffer,
            7,
            volume_2,
            0.7,
            Some(-42.0),
        );

        assert!(volume_1 > input_volume);
        assert_eq!(volume_2, input_volume);
    }

    #[test]
    fn speech_probability_threshold_is_effective_controller_level() {
        let mut config = parity_config(20);
        config.update_input_volume_wait_frames = 10;
        config.speech_probability_threshold = 0.5;
        config.speech_ratio_threshold = 0.8;

        let mut controller_1 = InputVolumeController::new(1, config);
        let mut controller_2 = InputVolumeController::new(1, config);
        controller_1.initialize();
        controller_2.initialize();

        let mut audio_buffer = create_audio_buffer();
        write_alternating_audio_buffer_samples(3276.7, &mut audio_buffer);

        let input_volume = DEFAULT_INITIAL_INPUT_VOLUME;

        let mut volume_1 = feed_frames(
            &mut controller_1,
            &audio_buffer,
            1,
            input_volume,
            0.5,
            Some(-42.0),
        );
        let mut volume_2 = feed_frames(
            &mut controller_2,
            &audio_buffer,
            1,
            input_volume,
            0.49,
            Some(-42.0),
        );
        assert_eq!(volume_1, input_volume);
        assert_eq!(volume_2, input_volume);

        volume_1 = feed_frames(
            &mut controller_1,
            &audio_buffer,
            2,
            volume_1,
            0.49,
            Some(-42.0),
        );
        volume_2 = feed_frames(
            &mut controller_2,
            &audio_buffer,
            2,
            volume_2,
            0.49,
            Some(-42.0),
        );
        assert_eq!(volume_1, input_volume);
        assert_eq!(volume_2, input_volume);

        volume_1 = feed_frames(
            &mut controller_1,
            &audio_buffer,
            7,
            volume_1,
            0.5,
            Some(-42.0),
        );
        volume_2 = feed_frames(
            &mut controller_2,
            &audio_buffer,
            7,
            volume_2,
            0.5,
            Some(-42.0),
        );

        assert!(volume_1 > input_volume);
        assert_eq!(volume_2, input_volume);
    }

    #[test]
    fn min_input_volume_enforced_with_clipping_when_above_clipped_level_min() {
        let mut config = parity_config(80);
        config.clipped_level_min = 70;

        let mut controller = InputVolumeController::new(1, config);
        controller.initialize();

        let mut audio_buffer = create_audio_buffer();
        write_audio_buffer_samples(4000.0, 0.8, &mut audio_buffer);

        let _ = call_agc_sequence(
            &mut controller,
            &audio_buffer,
            100,
            LOW_SPEECH_PROBABILITY,
            Some(-18.0),
            800,
        )
        .expect("volume expected");

        assert_eq!(controller.recommended_input_volume(), 80);
    }

    #[test]
    fn clipped_level_min_enforced_with_clipping_when_above_min_input_volume() {
        let mut config = parity_config(70);
        config.clipped_level_min = 80;

        let mut controller = InputVolumeController::new(1, config);
        controller.initialize();

        let mut audio_buffer = create_audio_buffer();
        write_audio_buffer_samples(4000.0, 0.8, &mut audio_buffer);

        let _ = call_agc_sequence(
            &mut controller,
            &audio_buffer,
            100,
            LOW_SPEECH_PROBABILITY,
            Some(-18.0),
            800,
        )
        .expect("volume expected");
        assert_eq!(controller.recommended_input_volume(), 80);
    }

    #[test]
    fn do_not_log_recommended_input_volume_on_change_to_match_target() {
        reset_on_change_to_match_target_samples();

        let mut config = parity_config(20);
        config.enable_clipping_predictor = false;
        let mut controller = InputVolumeController::new(1, config);
        controller.initialize();

        let mut audio_buffer = create_audio_buffer();
        write_audio_buffer_samples(1000.0, 0.0, &mut audio_buffer);

        let startup_volume = 255;
        let _ = call_agc_sequence(
            &mut controller,
            &audio_buffer,
            startup_volume,
            HIGH_SPEECH_PROBABILITY,
            Some(-20.0),
            1,
        )
        .expect("volume expected");

        write_audio_buffer_samples(32767.0, 1.0, &mut audio_buffer);
        let volume = call_agc_sequence(
            &mut controller,
            &audio_buffer,
            startup_volume,
            HIGH_SPEECH_PROBABILITY,
            Some(-20.0),
            1,
        )
        .expect("volume expected");

        assert!(volume < startup_volume);
        assert_eq!(on_change_to_match_target_samples(), vec![]);
    }

    #[test]
    fn log_recommended_input_volume_on_upward_change_to_match_target() {
        reset_on_change_to_match_target_samples();

        let mut config = parity_config(20);
        config.enable_clipping_predictor = false;
        let mut controller = InputVolumeController::new(1, config);
        controller.initialize();

        let mut audio_buffer = create_audio_buffer();
        write_audio_buffer_samples(0.99 * 32767.0, 0.0, &mut audio_buffer);

        let startup_volume = 100;
        let volume = call_agc_sequence(
            &mut controller,
            &audio_buffer,
            startup_volume,
            HIGH_SPEECH_PROBABILITY,
            Some(-50.0),
            14,
        )
        .expect("volume expected");
        assert!(volume > startup_volume);
        assert!(!on_change_to_match_target_samples().is_empty());
    }

    #[test]
    fn log_recommended_input_volume_on_downward_change_to_match_target() {
        reset_on_change_to_match_target_samples();

        let mut config = parity_config(20);
        config.enable_clipping_predictor = false;
        let mut controller = InputVolumeController::new(1, config);
        controller.initialize();

        let mut audio_buffer = create_audio_buffer();
        write_audio_buffer_samples(0.99 * 32767.0, 0.0, &mut audio_buffer);

        let startup_volume = 100;
        let volume = call_agc_sequence(
            &mut controller,
            &audio_buffer,
            startup_volume,
            HIGH_SPEECH_PROBABILITY,
            Some(-5.0),
            14,
        )
        .expect("volume expected");
        assert!(volume < startup_volume);
        assert!(!on_change_to_match_target_samples().is_empty());
    }

    #[test]
    fn disable_clipping_predictor_disables_clipping_predictor() {
        let mut config = parity_config(20);
        config.enable_clipping_predictor = false;
        let mut controller = InputVolumeController::new(1, config);
        controller.initialize();

        assert!(controller.clipping_predictor.is_none());
        assert!(!controller.use_clipping_predictor_step);
    }

    #[test]
    fn enable_clipping_predictor_enables_clipping_predictor() {
        let mut config = parity_config(20);
        config.enable_clipping_predictor = true;
        let mut controller = InputVolumeController::new(1, config);
        controller.initialize();

        assert!(controller.clipping_predictor.is_some());
        assert!(controller.use_clipping_predictor_step);
    }

    #[test]
    fn disable_clipping_predictor_does_not_lower_volume() {
        let mut volume = 255;
        let mut config = parity_config(20);
        config.enable_clipping_predictor = false;
        let mut controller = InputVolumeController::new(1, config);
        controller.initialize();

        assert!(controller.clipping_predictor.is_none());
        assert!(!controller.use_clipping_predictor_step);

        let mut audio_buffer = create_audio_buffer();
        for _ in 0..31 {
            write_alternating_audio_buffer_samples(0.99 * MAX_SAMPLE_S16, &mut audio_buffer);
            volume = call_agc_sequence(
                &mut controller,
                &audio_buffer,
                volume,
                LOW_SPEECH_PROBABILITY,
                Some(SPEECH_LEVEL),
                5,
            )
            .expect("volume expected");

            write_audio_buffer_samples(0.99 * MAX_SAMPLE_S16, 0.0, &mut audio_buffer);
            volume = call_agc_sequence(
                &mut controller,
                &audio_buffer,
                volume,
                LOW_SPEECH_PROBABILITY,
                Some(SPEECH_LEVEL),
                5,
            )
            .expect("volume expected");

            assert_eq!(volume, 255);
        }
    }

    #[test]
    fn used_clipping_predictions_produce_lower_analog_levels() {
        const INITIAL_LEVEL: i32 = 255;
        const CLOSE_TO_CLIPPING_PEAK_RATIO: f32 = 0.99;
        let mut volume_1 = INITIAL_LEVEL;
        let mut volume_2 = INITIAL_LEVEL;

        let mut config_1 = parity_config(20);
        let mut config_2 = parity_config(20);
        config_1.enable_clipping_predictor = true;
        config_2.enable_clipping_predictor = false;

        let mut controller_1 = InputVolumeController::new(1, config_1);
        let mut controller_2 = InputVolumeController::new(1, config_2);
        controller_1.initialize();
        controller_2.initialize();

        assert!(controller_1.clipping_predictor.is_some());
        assert!(controller_1.use_clipping_predictor_step);
        assert!(controller_2.clipping_predictor.is_none());

        let mut audio_buffer_1 = create_audio_buffer();
        let mut audio_buffer_2 = create_audio_buffer();

        write_alternating_audio_buffer_samples(
            CLOSE_TO_CLIPPING_PEAK_RATIO * MAX_SAMPLE_S16,
            &mut audio_buffer_1,
        );
        write_alternating_audio_buffer_samples(
            CLOSE_TO_CLIPPING_PEAK_RATIO * MAX_SAMPLE_S16,
            &mut audio_buffer_2,
        );
        volume_1 = call_agc_sequence(
            &mut controller_1,
            &audio_buffer_1,
            volume_1,
            LOW_SPEECH_PROBABILITY,
            Some(SPEECH_LEVEL),
            5,
        )
        .expect("volume expected");
        volume_2 = call_agc_sequence(
            &mut controller_2,
            &audio_buffer_2,
            volume_2,
            LOW_SPEECH_PROBABILITY,
            Some(SPEECH_LEVEL),
            5,
        )
        .expect("volume expected");

        write_audio_buffer_samples(
            CLOSE_TO_CLIPPING_PEAK_RATIO * MAX_SAMPLE_S16,
            0.0,
            &mut audio_buffer_1,
        );
        write_audio_buffer_samples(
            CLOSE_TO_CLIPPING_PEAK_RATIO * MAX_SAMPLE_S16,
            0.0,
            &mut audio_buffer_2,
        );
        volume_1 = call_agc_sequence(
            &mut controller_1,
            &audio_buffer_1,
            volume_1,
            LOW_SPEECH_PROBABILITY,
            Some(SPEECH_LEVEL),
            5,
        )
        .expect("volume expected");
        volume_2 = call_agc_sequence(
            &mut controller_2,
            &audio_buffer_2,
            volume_2,
            LOW_SPEECH_PROBABILITY,
            Some(SPEECH_LEVEL),
            5,
        )
        .expect("volume expected");

        assert_eq!(volume_1, INITIAL_LEVEL - 15);
        assert_eq!(volume_2, INITIAL_LEVEL);

        for _ in 0..30 {
            write_alternating_audio_buffer_samples(
                CLOSE_TO_CLIPPING_PEAK_RATIO * MAX_SAMPLE_S16,
                &mut audio_buffer_1,
            );
            write_alternating_audio_buffer_samples(
                CLOSE_TO_CLIPPING_PEAK_RATIO * MAX_SAMPLE_S16,
                &mut audio_buffer_2,
            );
            volume_1 = call_agc_sequence(
                &mut controller_1,
                &audio_buffer_1,
                volume_1,
                LOW_SPEECH_PROBABILITY,
                Some(SPEECH_LEVEL),
                5,
            )
            .expect("volume expected");
            volume_2 = call_agc_sequence(
                &mut controller_2,
                &audio_buffer_2,
                volume_2,
                LOW_SPEECH_PROBABILITY,
                Some(SPEECH_LEVEL),
                5,
            )
            .expect("volume expected");

            write_audio_buffer_samples(
                CLOSE_TO_CLIPPING_PEAK_RATIO * MAX_SAMPLE_S16,
                0.0,
                &mut audio_buffer_1,
            );
            write_audio_buffer_samples(
                CLOSE_TO_CLIPPING_PEAK_RATIO * MAX_SAMPLE_S16,
                0.0,
                &mut audio_buffer_2,
            );
            volume_1 = call_agc_sequence(
                &mut controller_1,
                &audio_buffer_1,
                volume_1,
                LOW_SPEECH_PROBABILITY,
                Some(SPEECH_LEVEL),
                5,
            )
            .expect("volume expected");
            volume_2 = call_agc_sequence(
                &mut controller_2,
                &audio_buffer_2,
                volume_2,
                LOW_SPEECH_PROBABILITY,
                Some(SPEECH_LEVEL),
                5,
            )
            .expect("volume expected");

            assert_eq!(volume_1, INITIAL_LEVEL - 15);
            assert_eq!(volume_2, INITIAL_LEVEL);
        }

        write_alternating_audio_buffer_samples(
            CLOSE_TO_CLIPPING_PEAK_RATIO * MAX_SAMPLE_S16,
            &mut audio_buffer_1,
        );
        write_alternating_audio_buffer_samples(
            CLOSE_TO_CLIPPING_PEAK_RATIO * MAX_SAMPLE_S16,
            &mut audio_buffer_2,
        );
        volume_1 = call_agc_sequence(
            &mut controller_1,
            &audio_buffer_1,
            volume_1,
            LOW_SPEECH_PROBABILITY,
            Some(SPEECH_LEVEL),
            5,
        )
        .expect("volume expected");
        volume_2 = call_agc_sequence(
            &mut controller_2,
            &audio_buffer_2,
            volume_2,
            LOW_SPEECH_PROBABILITY,
            Some(SPEECH_LEVEL),
            5,
        )
        .expect("volume expected");

        write_audio_buffer_samples(
            CLOSE_TO_CLIPPING_PEAK_RATIO * MAX_SAMPLE_S16,
            0.0,
            &mut audio_buffer_1,
        );
        write_audio_buffer_samples(
            CLOSE_TO_CLIPPING_PEAK_RATIO * MAX_SAMPLE_S16,
            0.0,
            &mut audio_buffer_2,
        );
        volume_1 = call_agc_sequence(
            &mut controller_1,
            &audio_buffer_1,
            volume_1,
            LOW_SPEECH_PROBABILITY,
            Some(SPEECH_LEVEL),
            5,
        )
        .expect("volume expected");
        volume_2 = call_agc_sequence(
            &mut controller_2,
            &audio_buffer_2,
            volume_2,
            LOW_SPEECH_PROBABILITY,
            Some(SPEECH_LEVEL),
            5,
        )
        .expect("volume expected");

        assert_eq!(volume_1, INITIAL_LEVEL - 2 * 15);
        assert_eq!(volume_2, INITIAL_LEVEL);

        for _ in 0..60 {
            write_alternating_audio_buffer_samples(0.0, &mut audio_buffer_1);
            write_alternating_audio_buffer_samples(0.0, &mut audio_buffer_2);
            volume_1 = call_agc_sequence(
                &mut controller_1,
                &audio_buffer_1,
                volume_1,
                LOW_SPEECH_PROBABILITY,
                Some(SPEECH_LEVEL),
                5,
            )
            .expect("volume expected");
            volume_2 = call_agc_sequence(
                &mut controller_2,
                &audio_buffer_2,
                volume_2,
                LOW_SPEECH_PROBABILITY,
                Some(SPEECH_LEVEL),
                5,
            )
            .expect("volume expected");

            write_audio_buffer_samples(0.0, 0.0, &mut audio_buffer_1);
            write_audio_buffer_samples(0.0, 0.0, &mut audio_buffer_2);
            volume_1 = call_agc_sequence(
                &mut controller_1,
                &audio_buffer_1,
                volume_1,
                LOW_SPEECH_PROBABILITY,
                Some(SPEECH_LEVEL),
                5,
            )
            .expect("volume expected");
            volume_2 = call_agc_sequence(
                &mut controller_2,
                &audio_buffer_2,
                volume_2,
                LOW_SPEECH_PROBABILITY,
                Some(SPEECH_LEVEL),
                5,
            )
            .expect("volume expected");
        }

        assert_eq!(volume_1, INITIAL_LEVEL - 2 * 15);
        assert_eq!(volume_2, INITIAL_LEVEL);

        write_alternating_audio_buffer_samples(MAX_SAMPLE_S16, &mut audio_buffer_1);
        write_alternating_audio_buffer_samples(MAX_SAMPLE_S16, &mut audio_buffer_2);
        volume_1 = call_agc_sequence(
            &mut controller_1,
            &audio_buffer_1,
            volume_1,
            LOW_SPEECH_PROBABILITY,
            Some(SPEECH_LEVEL),
            1,
        )
        .expect("volume expected");
        volume_2 = call_agc_sequence(
            &mut controller_2,
            &audio_buffer_2,
            volume_2,
            LOW_SPEECH_PROBABILITY,
            Some(SPEECH_LEVEL),
            1,
        )
        .expect("volume expected");

        assert_eq!(volume_1, INITIAL_LEVEL - 3 * 15);
        assert_eq!(volume_2, INITIAL_LEVEL - 15);

        for _ in 0..30 {
            write_alternating_audio_buffer_samples(MAX_SAMPLE_S16, &mut audio_buffer_1);
            write_alternating_audio_buffer_samples(MAX_SAMPLE_S16, &mut audio_buffer_2);
            volume_1 = call_agc_sequence(
                &mut controller_1,
                &audio_buffer_1,
                volume_1,
                LOW_SPEECH_PROBABILITY,
                Some(SPEECH_LEVEL),
                5,
            )
            .expect("volume expected");
            volume_2 = call_agc_sequence(
                &mut controller_2,
                &audio_buffer_2,
                volume_2,
                LOW_SPEECH_PROBABILITY,
                Some(SPEECH_LEVEL),
                5,
            )
            .expect("volume expected");

            write_audio_buffer_samples(MAX_SAMPLE_S16, 1.0, &mut audio_buffer_1);
            write_audio_buffer_samples(MAX_SAMPLE_S16, 1.0, &mut audio_buffer_2);
            volume_1 = call_agc_sequence(
                &mut controller_1,
                &audio_buffer_1,
                volume_1,
                LOW_SPEECH_PROBABILITY,
                Some(SPEECH_LEVEL),
                5,
            )
            .expect("volume expected");
            volume_2 = call_agc_sequence(
                &mut controller_2,
                &audio_buffer_2,
                volume_2,
                LOW_SPEECH_PROBABILITY,
                Some(SPEECH_LEVEL),
                5,
            )
            .expect("volume expected");
        }

        assert_eq!(volume_1, INITIAL_LEVEL - 3 * 15);
        assert_eq!(volume_2, INITIAL_LEVEL - 15);

        write_alternating_audio_buffer_samples(MAX_SAMPLE_S16, &mut audio_buffer_1);
        write_alternating_audio_buffer_samples(MAX_SAMPLE_S16, &mut audio_buffer_2);
        volume_1 = call_agc_sequence(
            &mut controller_1,
            &audio_buffer_1,
            volume_1,
            LOW_SPEECH_PROBABILITY,
            Some(SPEECH_LEVEL),
            1,
        )
        .expect("volume expected");
        volume_2 = call_agc_sequence(
            &mut controller_2,
            &audio_buffer_2,
            volume_2,
            LOW_SPEECH_PROBABILITY,
            Some(SPEECH_LEVEL),
            1,
        )
        .expect("volume expected");

        assert_eq!(volume_1, INITIAL_LEVEL - 4 * 15);
        assert_eq!(volume_2, INITIAL_LEVEL - 2 * 15);
    }
}
