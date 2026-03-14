//! Frame-wise clipping prediction and clipped level step estimation for AGC2.

use crate::audio_processing::agc2::agc2_common::MIN_LEVEL_DBFS;
use crate::audio_processing::agc2::clipping_predictor_level_buffer::{
    ClippingPredictorLevelBuffer, Level,
};

const CLIPPING_PREDICTOR_MAX_GAIN_CHANGE: i32 = 15;

// Maps input volumes [0,255] to gains in dB (ported from gain_map_internal.h).
const GAIN_MAP: [i32; 256] = [
    -56, -54, -52, -50, -48, -47, -45, -43, -42, -40, -38, -37, -35, -34, -33, -31, -30, -29,
    -27, -26, -25, -24, -23, -22, -20, -19, -18, -17, -16, -15, -14, -14, -13, -12, -11, -10,
    -9, -8, -8, -7, -6, -5, -5, -4, -3, -2, -2, -1, 0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6,
    7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 13, 14, 14, 15, 15, 15, 16, 16, 17, 17,
    17, 18, 18, 18, 19, 19, 19, 20, 20, 21, 21, 21, 22, 22, 22, 23, 23, 23, 24, 24, 24, 24, 25,
    25, 25, 26, 26, 26, 27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 30, 30, 30, 30, 31, 31, 31, 32,
    32, 32, 32, 33, 33, 33, 33, 34, 34, 34, 35, 35, 35, 35, 36, 36, 36, 36, 37, 37, 37, 38, 38,
    38, 38, 39, 39, 39, 39, 40, 40, 40, 40, 41, 41, 41, 41, 42, 42, 42, 42, 43, 43, 43, 44, 44,
    44, 44, 45, 45, 45, 45, 46, 46, 46, 46, 47, 47, 47, 47, 48, 48, 48, 48, 49, 49, 49, 49, 50,
    50, 50, 50, 51, 51, 51, 51, 52, 52, 52, 52, 53, 53, 53, 53, 54, 54, 54, 54, 55, 55, 55, 55,
    56, 56, 56, 56, 57, 57, 57, 57, 58, 58, 58, 58, 59, 59, 59, 59, 60, 60, 60, 60, 61, 61, 61,
    61, 62, 62, 62, 62, 63, 63, 63, 63, 64,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClippingPredictorMode {
    ClippingEventPrediction,
    AdaptiveStepClippingPeakPrediction,
    FixedStepClippingPeakPrediction,
}

#[derive(Debug, Clone, Copy)]
pub struct ClippingPredictorConfig {
    pub enabled: bool,
    pub mode: ClippingPredictorMode,
    pub window_length: i32,
    pub reference_window_length: i32,
    pub reference_window_delay: i32,
    pub clipping_threshold: f32,
    pub crest_factor_margin: f32,
}

impl Default for ClippingPredictorConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: ClippingPredictorMode::AdaptiveStepClippingPeakPrediction,
            window_length: 5,
            reference_window_length: 5,
            reference_window_delay: 5,
            clipping_threshold: -1.0,
            crest_factor_margin: 3.0,
        }
    }
}

pub trait ClippingPredictor {
    fn reset(&mut self);
    fn analyze(&mut self, frame: &[&[f32]]);
    fn estimate_clipped_level_step(
        &self,
        channel: i32,
        level: i32,
        default_step: i32,
        min_mic_level: i32,
        max_mic_level: i32,
    ) -> Option<i32>;
}

pub fn create_clipping_predictor(
    num_channels: i32,
    config: ClippingPredictorConfig,
) -> Option<Box<dyn ClippingPredictor>> {
    if !config.enabled {
        return None;
    }

    match config.mode {
        ClippingPredictorMode::ClippingEventPrediction => Some(Box::new(
            ClippingEventPredictor::new(
                num_channels,
                config.window_length,
                config.reference_window_length,
                config.reference_window_delay,
                config.clipping_threshold,
                config.crest_factor_margin,
            ),
        )),
        ClippingPredictorMode::AdaptiveStepClippingPeakPrediction => {
            Some(Box::new(ClippingPeakPredictor::new(
                num_channels,
                config.window_length,
                config.reference_window_length,
                config.reference_window_delay,
                config.clipping_threshold as i32,
                true,
            )))
        }
        ClippingPredictorMode::FixedStepClippingPeakPrediction => {
            Some(Box::new(ClippingPeakPredictor::new(
                num_channels,
                config.window_length,
                config.reference_window_length,
                config.reference_window_delay,
                config.clipping_threshold as i32,
                false,
            )))
        }
    }
}

fn float_s16_to_dbfs(v: f32) -> f32 {
    if v <= 1.0 {
        return MIN_LEVEL_DBFS;
    }
    20.0 * v.log10() + MIN_LEVEL_DBFS
}

fn compute_crest_factor(level: Level) -> f32 {
    float_s16_to_dbfs(level.max) - float_s16_to_dbfs(level.average.sqrt())
}

fn compute_volume_update(
    gain_error_db: i32,
    input_volume: i32,
    min_input_volume: i32,
    max_input_volume: i32,
) -> i32 {
    assert!(input_volume >= 0);
    assert!(input_volume <= max_input_volume);
    if gain_error_db == 0 {
        return input_volume;
    }

    let mut new_volume = input_volume;
    if gain_error_db > 0 {
        while GAIN_MAP[new_volume as usize] - GAIN_MAP[input_volume as usize] < gain_error_db
            && new_volume < max_input_volume
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

struct ClippingEventPredictor {
    ch_buffers: Vec<ClippingPredictorLevelBuffer>,
    window_length: i32,
    reference_window_length: i32,
    reference_window_delay: i32,
    clipping_threshold: f32,
    crest_factor_margin: f32,
}

impl ClippingEventPredictor {
    fn new(
        num_channels: i32,
        window_length: i32,
        reference_window_length: i32,
        reference_window_delay: i32,
        clipping_threshold: f32,
        crest_factor_margin: f32,
    ) -> Self {
        assert!(num_channels > 0);
        assert!(window_length > 0);
        assert!(reference_window_length > 0);
        assert!(reference_window_delay >= 0);
        assert!(reference_window_length + reference_window_delay > window_length);
        let buffer_length = reference_window_delay + reference_window_length;
        assert!(buffer_length > 0);
        let mut ch_buffers = Vec::with_capacity(num_channels as usize);
        for _ in 0..num_channels {
            ch_buffers.push(ClippingPredictorLevelBuffer::new(buffer_length));
        }
        Self {
            ch_buffers,
            window_length,
            reference_window_length,
            reference_window_delay,
            clipping_threshold,
            crest_factor_margin,
        }
    }

    fn predict_clipping_event(&self, channel: i32) -> bool {
        let metrics = self.ch_buffers[channel as usize].compute_partial_metrics(0, self.window_length);
        let Some(metrics) = metrics else {
            return false;
        };
        if float_s16_to_dbfs(metrics.max) <= self.clipping_threshold {
            return false;
        }

        let reference_metrics = self.ch_buffers[channel as usize].compute_partial_metrics(
            self.reference_window_delay,
            self.reference_window_length,
        );
        let Some(reference_metrics) = reference_metrics else {
            return false;
        };

        let crest_factor = compute_crest_factor(metrics);
        let reference_crest_factor = compute_crest_factor(reference_metrics);
        crest_factor < reference_crest_factor - self.crest_factor_margin
    }
}

impl ClippingPredictor for ClippingEventPredictor {
    fn reset(&mut self) {
        for buf in &mut self.ch_buffers {
            buf.reset();
        }
    }

    fn analyze(&mut self, frame: &[&[f32]]) {
        let num_channels = frame.len();
        assert_eq!(num_channels, self.ch_buffers.len());
        let samples_per_channel = frame[0].len();
        assert!(samples_per_channel > 0);

        for channel in 0..num_channels {
            let mut sum_squares = 0.0f32;
            let mut peak = 0.0f32;
            for &sample in frame[channel] {
                sum_squares += sample * sample;
                peak = peak.max(sample.abs());
            }
            self.ch_buffers[channel].push(Level {
                average: sum_squares / samples_per_channel as f32,
                max: peak,
            });
        }
    }

    fn estimate_clipped_level_step(
        &self,
        channel: i32,
        level: i32,
        default_step: i32,
        min_mic_level: i32,
        max_mic_level: i32,
    ) -> Option<i32> {
        assert!(channel >= 0);
        assert!((channel as usize) < self.ch_buffers.len());
        assert!((0..=255).contains(&level));
        assert!((1..=255).contains(&default_step));
        assert!((0..=255).contains(&min_mic_level));
        assert!((0..=255).contains(&max_mic_level));
        if level <= min_mic_level {
            return None;
        }

        if self.predict_clipping_event(channel) {
            let new_level = (level - default_step).clamp(min_mic_level, max_mic_level);
            let step = level - new_level;
            if step > 0 {
                return Some(step);
            }
        }
        None
    }
}

struct ClippingPeakPredictor {
    ch_buffers: Vec<ClippingPredictorLevelBuffer>,
    window_length: i32,
    reference_window_length: i32,
    reference_window_delay: i32,
    clipping_threshold: i32,
    adaptive_step_estimation: bool,
}

impl ClippingPeakPredictor {
    fn new(
        num_channels: i32,
        window_length: i32,
        reference_window_length: i32,
        reference_window_delay: i32,
        clipping_threshold: i32,
        adaptive_step_estimation: bool,
    ) -> Self {
        assert!(num_channels > 0);
        assert!(window_length > 0);
        assert!(reference_window_length > 0);
        assert!(reference_window_delay >= 0);
        assert!(reference_window_length + reference_window_delay > window_length);
        let buffer_length = reference_window_delay + reference_window_length;
        assert!(buffer_length > 0);
        let mut ch_buffers = Vec::with_capacity(num_channels as usize);
        for _ in 0..num_channels {
            ch_buffers.push(ClippingPredictorLevelBuffer::new(buffer_length));
        }

        Self {
            ch_buffers,
            window_length,
            reference_window_length,
            reference_window_delay,
            clipping_threshold,
            adaptive_step_estimation,
        }
    }

    fn estimate_peak_value(&self, channel: i32) -> Option<f32> {
        let reference_metrics = self.ch_buffers[channel as usize].compute_partial_metrics(
            self.reference_window_delay,
            self.reference_window_length,
        )?;

        let metrics = self.ch_buffers[channel as usize].compute_partial_metrics(0, self.window_length)?;
        if float_s16_to_dbfs(metrics.max) <= self.clipping_threshold as f32 {
            return None;
        }

        let reference_crest_factor = compute_crest_factor(reference_metrics);
        let projected_peak = reference_crest_factor + float_s16_to_dbfs(metrics.average.sqrt());
        Some(projected_peak)
    }
}

impl ClippingPredictor for ClippingPeakPredictor {
    fn reset(&mut self) {
        for buf in &mut self.ch_buffers {
            buf.reset();
        }
    }

    fn analyze(&mut self, frame: &[&[f32]]) {
        let num_channels = frame.len();
        assert_eq!(num_channels, self.ch_buffers.len());
        let samples_per_channel = frame[0].len();
        assert!(samples_per_channel > 0);

        for channel in 0..num_channels {
            let mut sum_squares = 0.0f32;
            let mut peak = 0.0f32;
            for &sample in frame[channel] {
                sum_squares += sample * sample;
                peak = peak.max(sample.abs());
            }
            self.ch_buffers[channel].push(Level {
                average: sum_squares / samples_per_channel as f32,
                max: peak,
            });
        }
    }

    fn estimate_clipped_level_step(
        &self,
        channel: i32,
        level: i32,
        default_step: i32,
        min_mic_level: i32,
        max_mic_level: i32,
    ) -> Option<i32> {
        assert!(channel >= 0);
        assert!((channel as usize) < self.ch_buffers.len());
        assert!((0..=255).contains(&level));
        assert!((1..=255).contains(&default_step));
        assert!((0..=255).contains(&min_mic_level));
        assert!((0..=255).contains(&max_mic_level));
        if level <= min_mic_level {
            return None;
        }

        if let Some(estimate_db) = self.estimate_peak_value(channel)
            && estimate_db > self.clipping_threshold as f32
        {
            let step = if !self.adaptive_step_estimation {
                default_step
            } else {
                let estimated_gain_change = (-(estimate_db.ceil() as i32))
                    .clamp(-CLIPPING_PREDICTOR_MAX_GAIN_CHANGE, 0);
                (level
                    - compute_volume_update(
                        estimated_gain_change,
                        level,
                        min_mic_level,
                        max_mic_level,
                    ))
                .max(default_step)
            };

            let new_level = (level - step).clamp(min_mic_level, max_mic_level);
            if level > new_level {
                return Some(level - new_level);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE_HZ: usize = 32_000;
    const NUM_CHANNELS: i32 = 1;
    const SAMPLES_PER_CHANNEL: usize = SAMPLE_RATE_HZ / 100;
    const MAX_MIC_LEVEL: i32 = 255;
    const MIN_MIC_LEVEL: i32 = 12;
    const DEFAULT_CLIPPED_LEVEL_STEP: i32 = 15;
    const MAX_SAMPLE_S16: f32 = i16::MAX as f32;

    // `20 * log10(0.99)`.
    const CLIPPING_THRESHOLD_DB: f32 = -0.08729611;

    fn call_analyze(num_calls: i32, frame: &[&[f32]], predictor: &mut dyn ClippingPredictor) {
        for _ in 0..num_calls {
            predictor.analyze(frame);
        }
    }

    fn analyze_non_zero_crest_factor_audio(
        num_calls: i32,
        num_channels: i32,
        peak_ratio: f32,
        predictor: &mut dyn ClippingPredictor,
    ) {
        assert!(num_calls > 0);
        assert!(num_channels > 0);
        assert!(peak_ratio <= 1.0);
        let mut audio_data = vec![vec![0.0f32; SAMPLES_PER_CHANNEL]; num_channels as usize];
        for channel in audio_data.iter_mut().take(num_channels as usize) {
            for sample in (0..SAMPLES_PER_CHANNEL).step_by(10) {
                channel[sample] = 0.1 * peak_ratio * MAX_SAMPLE_S16;
                channel[sample + 1] = 0.2 * peak_ratio * MAX_SAMPLE_S16;
                channel[sample + 2] = 0.3 * peak_ratio * MAX_SAMPLE_S16;
                channel[sample + 3] = 0.4 * peak_ratio * MAX_SAMPLE_S16;
                channel[sample + 4] = 0.5 * peak_ratio * MAX_SAMPLE_S16;
                channel[sample + 5] = 0.6 * peak_ratio * MAX_SAMPLE_S16;
                channel[sample + 6] = 0.7 * peak_ratio * MAX_SAMPLE_S16;
                channel[sample + 7] = 0.8 * peak_ratio * MAX_SAMPLE_S16;
                channel[sample + 8] = 0.9 * peak_ratio * MAX_SAMPLE_S16;
                channel[sample + 9] = 1.0 * peak_ratio * MAX_SAMPLE_S16;
            }
        }
        let frame: Vec<&[f32]> = audio_data.iter().map(Vec::as_slice).collect();
        call_analyze(num_calls, &frame, predictor);
    }

    fn analyze_zero_crest_factor_audio(
        num_calls: i32,
        num_channels: i32,
        peak_ratio: f32,
        predictor: &mut dyn ClippingPredictor,
    ) {
        assert!(num_calls > 0);
        assert!(num_channels > 0);
        assert!(peak_ratio <= 1.0);
        let mut audio_data = vec![vec![0.0f32; SAMPLES_PER_CHANNEL]; num_channels as usize];
        for channel in audio_data.iter_mut().take(num_channels as usize) {
            for sample in channel.iter_mut() {
                *sample = peak_ratio * MAX_SAMPLE_S16;
            }
        }
        let frame: Vec<&[f32]> = audio_data.iter().map(Vec::as_slice).collect();
        call_analyze(num_calls, &frame, predictor);
    }

    fn check_channel_estimates_with_value(
        num_channels: i32,
        level: i32,
        default_step: i32,
        min_mic_level: i32,
        max_mic_level: i32,
        predictor: &dyn ClippingPredictor,
        expected: i32,
    ) {
        for i in 0..num_channels {
            assert_eq!(
                predictor.estimate_clipped_level_step(
                    i,
                    level,
                    default_step,
                    min_mic_level,
                    max_mic_level,
                ),
                Some(expected)
            );
        }
    }

    fn check_channel_estimates_without_value(
        num_channels: i32,
        level: i32,
        default_step: i32,
        min_mic_level: i32,
        max_mic_level: i32,
        predictor: &dyn ClippingPredictor,
    ) {
        for i in 0..num_channels {
            assert_eq!(
                predictor.estimate_clipped_level_step(
                    i,
                    level,
                    default_step,
                    min_mic_level,
                    max_mic_level,
                ),
                None
            );
        }
    }

    fn get_config(
        mode: ClippingPredictorMode,
        window_length: i32,
        reference_window_length: i32,
        reference_window_delay: i32,
    ) -> ClippingPredictorConfig {
        ClippingPredictorConfig {
            enabled: true,
            mode,
            window_length,
            reference_window_length,
            reference_window_delay,
            clipping_threshold: -1.0,
            crest_factor_margin: 0.5,
        }
    }

    #[test]
    fn no_predictor_created() {
        let predictor = create_clipping_predictor(
            NUM_CHANNELS,
            ClippingPredictorConfig {
                enabled: false,
                ..Default::default()
            },
        );
        assert!(predictor.is_none());
    }

    #[test]
    fn clipping_event_prediction_created() {
        let predictor = create_clipping_predictor(
            NUM_CHANNELS,
            ClippingPredictorConfig {
                enabled: true,
                mode: ClippingPredictorMode::ClippingEventPrediction,
                ..Default::default()
            },
        );
        assert!(predictor.is_some());
    }

    #[test]
    fn adaptive_step_clipping_peak_prediction_created() {
        let predictor = create_clipping_predictor(
            NUM_CHANNELS,
            ClippingPredictorConfig {
                enabled: true,
                mode: ClippingPredictorMode::AdaptiveStepClippingPeakPrediction,
                ..Default::default()
            },
        );
        assert!(predictor.is_some());
    }

    #[test]
    fn fixed_step_clipping_peak_prediction_created() {
        let predictor = create_clipping_predictor(
            NUM_CHANNELS,
            ClippingPredictorConfig {
                enabled: true,
                mode: ClippingPredictorMode::FixedStepClippingPeakPrediction,
                ..Default::default()
            },
        );
        assert!(predictor.is_some());
    }

    #[test]
    fn check_clipping_event_predictor_estimate_after_crest_factor_drop() {
        for num_channels in [1, 5] {
            for window_length in [1, 5, 10] {
                for reference_window_length in [1, 5] {
                    for reference_window_delay in [0, 1, 5] {
                        let config = get_config(
                            ClippingPredictorMode::ClippingEventPrediction,
                            window_length,
                            reference_window_length,
                            reference_window_delay,
                        );
                        if config.reference_window_length + config.reference_window_delay
                            <= config.window_length
                        {
                            continue;
                        }

                        let mut predictor =
                            create_clipping_predictor(num_channels, config).expect("predictor");
                        analyze_non_zero_crest_factor_audio(
                            config.reference_window_length + config.reference_window_delay
                                - config.window_length,
                            num_channels,
                            0.99,
                            predictor.as_mut(),
                        );
                        check_channel_estimates_without_value(
                            num_channels,
                            255,
                            DEFAULT_CLIPPED_LEVEL_STEP,
                            MIN_MIC_LEVEL,
                            MAX_MIC_LEVEL,
                            predictor.as_ref(),
                        );
                        analyze_zero_crest_factor_audio(
                            config.window_length,
                            num_channels,
                            0.99,
                            predictor.as_mut(),
                        );
                        check_channel_estimates_with_value(
                            num_channels,
                            255,
                            DEFAULT_CLIPPED_LEVEL_STEP,
                            MIN_MIC_LEVEL,
                            MAX_MIC_LEVEL,
                            predictor.as_ref(),
                            DEFAULT_CLIPPED_LEVEL_STEP,
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn check_clipping_event_predictor_no_estimate_after_constant_crest_factor() {
        for num_channels in [1, 5] {
            for window_length in [1, 5, 10] {
                for reference_window_length in [1, 5] {
                    for reference_window_delay in [0, 1, 5] {
                        let config = get_config(
                            ClippingPredictorMode::ClippingEventPrediction,
                            window_length,
                            reference_window_length,
                            reference_window_delay,
                        );
                        if config.reference_window_length + config.reference_window_delay
                            <= config.window_length
                        {
                            continue;
                        }

                        let mut predictor =
                            create_clipping_predictor(num_channels, config).expect("predictor");
                        analyze_non_zero_crest_factor_audio(
                            config.reference_window_length + config.reference_window_delay
                                - config.window_length,
                            num_channels,
                            0.99,
                            predictor.as_mut(),
                        );
                        check_channel_estimates_without_value(
                            num_channels,
                            255,
                            DEFAULT_CLIPPED_LEVEL_STEP,
                            MIN_MIC_LEVEL,
                            MAX_MIC_LEVEL,
                            predictor.as_ref(),
                        );
                        analyze_non_zero_crest_factor_audio(
                            config.window_length,
                            num_channels,
                            0.99,
                            predictor.as_mut(),
                        );
                        check_channel_estimates_without_value(
                            num_channels,
                            255,
                            DEFAULT_CLIPPED_LEVEL_STEP,
                            MIN_MIC_LEVEL,
                            MAX_MIC_LEVEL,
                            predictor.as_ref(),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn check_clipping_peak_predictor_estimate_after_high_crest_factor() {
        for num_channels in [1, 5] {
            for window_length in [1, 5, 10] {
                for reference_window_length in [1, 5] {
                    for reference_window_delay in [0, 1, 5] {
                        let config = get_config(
                            ClippingPredictorMode::AdaptiveStepClippingPeakPrediction,
                            window_length,
                            reference_window_length,
                            reference_window_delay,
                        );
                        if config.reference_window_length + config.reference_window_delay
                            <= config.window_length
                        {
                            continue;
                        }

                        let mut predictor =
                            create_clipping_predictor(num_channels, config).expect("predictor");
                        analyze_non_zero_crest_factor_audio(
                            config.reference_window_length + config.reference_window_delay
                                - config.window_length,
                            num_channels,
                            0.99,
                            predictor.as_mut(),
                        );
                        check_channel_estimates_without_value(
                            num_channels,
                            255,
                            DEFAULT_CLIPPED_LEVEL_STEP,
                            MIN_MIC_LEVEL,
                            MAX_MIC_LEVEL,
                            predictor.as_ref(),
                        );
                        analyze_non_zero_crest_factor_audio(
                            config.window_length,
                            num_channels,
                            0.99,
                            predictor.as_mut(),
                        );
                        check_channel_estimates_with_value(
                            num_channels,
                            255,
                            DEFAULT_CLIPPED_LEVEL_STEP,
                            MIN_MIC_LEVEL,
                            MAX_MIC_LEVEL,
                            predictor.as_ref(),
                            DEFAULT_CLIPPED_LEVEL_STEP,
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn check_clipping_peak_predictor_no_estimate_after_low_crest_factor() {
        for num_channels in [1, 5] {
            for window_length in [1, 5, 10] {
                for reference_window_length in [1, 5] {
                    for reference_window_delay in [0, 1, 5] {
                        let config = get_config(
                            ClippingPredictorMode::AdaptiveStepClippingPeakPrediction,
                            window_length,
                            reference_window_length,
                            reference_window_delay,
                        );
                        if config.reference_window_length + config.reference_window_delay
                            <= config.window_length
                        {
                            continue;
                        }

                        let mut predictor =
                            create_clipping_predictor(num_channels, config).expect("predictor");
                        analyze_zero_crest_factor_audio(
                            config.reference_window_length + config.reference_window_delay
                                - config.window_length,
                            num_channels,
                            0.99,
                            predictor.as_mut(),
                        );
                        check_channel_estimates_without_value(
                            num_channels,
                            255,
                            DEFAULT_CLIPPED_LEVEL_STEP,
                            MIN_MIC_LEVEL,
                            MAX_MIC_LEVEL,
                            predictor.as_ref(),
                        );
                        analyze_non_zero_crest_factor_audio(
                            config.window_length,
                            num_channels,
                            0.99,
                            predictor.as_mut(),
                        );
                        check_channel_estimates_without_value(
                            num_channels,
                            255,
                            DEFAULT_CLIPPED_LEVEL_STEP,
                            MIN_MIC_LEVEL,
                            MAX_MIC_LEVEL,
                            predictor.as_ref(),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn check_estimate_after_crest_factor_drop_parametrized() {
        for clipping_threshold in [-1.0f32, 0.0f32] {
            for crest_factor_margin in [3.0f32, 4.16f32] {
                let config = ClippingPredictorConfig {
                    enabled: true,
                    mode: ClippingPredictorMode::ClippingEventPrediction,
                    window_length: 5,
                    reference_window_length: 5,
                    reference_window_delay: 5,
                    clipping_threshold,
                    crest_factor_margin,
                };
                let mut predictor = create_clipping_predictor(NUM_CHANNELS, config).expect("predictor");
                analyze_non_zero_crest_factor_audio(
                    config.reference_window_length,
                    NUM_CHANNELS,
                    0.99,
                    predictor.as_mut(),
                );
                check_channel_estimates_without_value(
                    NUM_CHANNELS,
                    255,
                    DEFAULT_CLIPPED_LEVEL_STEP,
                    MIN_MIC_LEVEL,
                    MAX_MIC_LEVEL,
                    predictor.as_ref(),
                );
                analyze_zero_crest_factor_audio(
                    config.window_length,
                    NUM_CHANNELS,
                    0.99,
                    predictor.as_mut(),
                );
                if config.clipping_threshold < CLIPPING_THRESHOLD_DB
                    && config.crest_factor_margin < 4.15
                {
                    check_channel_estimates_with_value(
                        NUM_CHANNELS,
                        255,
                        DEFAULT_CLIPPED_LEVEL_STEP,
                        MIN_MIC_LEVEL,
                        MAX_MIC_LEVEL,
                        predictor.as_ref(),
                        DEFAULT_CLIPPED_LEVEL_STEP,
                    );
                } else {
                    check_channel_estimates_without_value(
                        NUM_CHANNELS,
                        255,
                        DEFAULT_CLIPPED_LEVEL_STEP,
                        MIN_MIC_LEVEL,
                        MAX_MIC_LEVEL,
                        predictor.as_ref(),
                    );
                }
            }
        }
    }

    #[test]
    fn check_estimate_after_high_crest_factor_with_no_clipping_margin() {
        for mode in [
            ClippingPredictorMode::AdaptiveStepClippingPeakPrediction,
            ClippingPredictorMode::FixedStepClippingPeakPrediction,
        ] {
            let config = ClippingPredictorConfig {
                enabled: true,
                mode,
                window_length: 5,
                reference_window_length: 5,
                reference_window_delay: 5,
                clipping_threshold: 0.0,
                crest_factor_margin: 3.0,
            };
            let mut predictor = create_clipping_predictor(NUM_CHANNELS, config).expect("predictor");
            analyze_non_zero_crest_factor_audio(
                config.reference_window_length,
                NUM_CHANNELS,
                0.99,
                predictor.as_mut(),
            );
            check_channel_estimates_without_value(
                NUM_CHANNELS,
                255,
                DEFAULT_CLIPPED_LEVEL_STEP,
                MIN_MIC_LEVEL,
                MAX_MIC_LEVEL,
                predictor.as_ref(),
            );
            analyze_zero_crest_factor_audio(config.window_length, NUM_CHANNELS, 0.99, predictor.as_mut());
            check_channel_estimates_without_value(
                NUM_CHANNELS,
                255,
                DEFAULT_CLIPPED_LEVEL_STEP,
                MIN_MIC_LEVEL,
                MAX_MIC_LEVEL,
                predictor.as_ref(),
            );
        }
    }

    #[test]
    fn check_estimate_after_high_crest_factor_with_clipping_margin() {
        for mode in [
            ClippingPredictorMode::AdaptiveStepClippingPeakPrediction,
            ClippingPredictorMode::FixedStepClippingPeakPrediction,
        ] {
            let config = ClippingPredictorConfig {
                enabled: true,
                mode,
                window_length: 5,
                reference_window_length: 5,
                reference_window_delay: 5,
                clipping_threshold: -1.0,
                crest_factor_margin: 3.0,
            };
            let mut predictor = create_clipping_predictor(NUM_CHANNELS, config).expect("predictor");
            analyze_non_zero_crest_factor_audio(
                config.reference_window_length,
                NUM_CHANNELS,
                0.99,
                predictor.as_mut(),
            );
            check_channel_estimates_without_value(
                NUM_CHANNELS,
                255,
                DEFAULT_CLIPPED_LEVEL_STEP,
                MIN_MIC_LEVEL,
                MAX_MIC_LEVEL,
                predictor.as_ref(),
            );
            analyze_zero_crest_factor_audio(config.window_length, NUM_CHANNELS, 0.99, predictor.as_mut());
            let expected_step = if mode
                == ClippingPredictorMode::AdaptiveStepClippingPeakPrediction
            {
                17
            } else {
                DEFAULT_CLIPPED_LEVEL_STEP
            };
            check_channel_estimates_with_value(
                NUM_CHANNELS,
                255,
                DEFAULT_CLIPPED_LEVEL_STEP,
                MIN_MIC_LEVEL,
                MAX_MIC_LEVEL,
                predictor.as_ref(),
                expected_step,
            );
        }
    }

    #[test]
    fn check_estimate_after_reset_event_predictor() {
        let config = ClippingPredictorConfig {
            enabled: true,
            mode: ClippingPredictorMode::ClippingEventPrediction,
            window_length: 5,
            reference_window_length: 5,
            reference_window_delay: 5,
            clipping_threshold: -1.0,
            crest_factor_margin: 3.0,
        };
        let mut predictor = create_clipping_predictor(NUM_CHANNELS, config).expect("predictor");
        analyze_non_zero_crest_factor_audio(
            config.reference_window_length,
            NUM_CHANNELS,
            0.99,
            predictor.as_mut(),
        );
        check_channel_estimates_without_value(
            NUM_CHANNELS,
            255,
            DEFAULT_CLIPPED_LEVEL_STEP,
            MIN_MIC_LEVEL,
            MAX_MIC_LEVEL,
            predictor.as_ref(),
        );
        predictor.reset();
        analyze_zero_crest_factor_audio(config.window_length, NUM_CHANNELS, 0.99, predictor.as_mut());
        check_channel_estimates_without_value(
            NUM_CHANNELS,
            255,
            DEFAULT_CLIPPED_LEVEL_STEP,
            MIN_MIC_LEVEL,
            MAX_MIC_LEVEL,
            predictor.as_ref(),
        );
    }

    #[test]
    fn check_no_estimate_after_reset_peak_predictor() {
        let config = ClippingPredictorConfig {
            enabled: true,
            mode: ClippingPredictorMode::AdaptiveStepClippingPeakPrediction,
            window_length: 5,
            reference_window_length: 5,
            reference_window_delay: 5,
            clipping_threshold: -1.0,
            crest_factor_margin: 3.0,
        };
        let mut predictor = create_clipping_predictor(NUM_CHANNELS, config).expect("predictor");
        analyze_non_zero_crest_factor_audio(
            config.reference_window_length,
            NUM_CHANNELS,
            0.99,
            predictor.as_mut(),
        );
        check_channel_estimates_without_value(
            NUM_CHANNELS,
            255,
            DEFAULT_CLIPPED_LEVEL_STEP,
            MIN_MIC_LEVEL,
            MAX_MIC_LEVEL,
            predictor.as_ref(),
        );
        predictor.reset();
        analyze_zero_crest_factor_audio(config.window_length, NUM_CHANNELS, 0.99, predictor.as_mut());
        check_channel_estimates_without_value(
            NUM_CHANNELS,
            255,
            DEFAULT_CLIPPED_LEVEL_STEP,
            MIN_MIC_LEVEL,
            MAX_MIC_LEVEL,
            predictor.as_ref(),
        );
    }

    #[test]
    fn check_adaptive_step_estimate() {
        let config = ClippingPredictorConfig {
            enabled: true,
            mode: ClippingPredictorMode::AdaptiveStepClippingPeakPrediction,
            window_length: 5,
            reference_window_length: 5,
            reference_window_delay: 5,
            clipping_threshold: -1.0,
            crest_factor_margin: 3.0,
        };
        let mut predictor = create_clipping_predictor(NUM_CHANNELS, config).expect("predictor");
        analyze_non_zero_crest_factor_audio(
            config.reference_window_length,
            NUM_CHANNELS,
            0.99,
            predictor.as_mut(),
        );
        check_channel_estimates_without_value(
            NUM_CHANNELS,
            255,
            DEFAULT_CLIPPED_LEVEL_STEP,
            MIN_MIC_LEVEL,
            MAX_MIC_LEVEL,
            predictor.as_ref(),
        );
        analyze_zero_crest_factor_audio(config.window_length, NUM_CHANNELS, 0.99, predictor.as_mut());
        check_channel_estimates_with_value(
            NUM_CHANNELS,
            255,
            DEFAULT_CLIPPED_LEVEL_STEP,
            MIN_MIC_LEVEL,
            MAX_MIC_LEVEL,
            predictor.as_ref(),
            17,
        );
    }

    #[test]
    fn check_fixed_step_estimate() {
        let config = ClippingPredictorConfig {
            enabled: true,
            mode: ClippingPredictorMode::FixedStepClippingPeakPrediction,
            window_length: 5,
            reference_window_length: 5,
            reference_window_delay: 5,
            clipping_threshold: -1.0,
            crest_factor_margin: 3.0,
        };
        let mut predictor = create_clipping_predictor(NUM_CHANNELS, config).expect("predictor");
        analyze_non_zero_crest_factor_audio(
            config.reference_window_length,
            NUM_CHANNELS,
            0.99,
            predictor.as_mut(),
        );
        check_channel_estimates_without_value(
            NUM_CHANNELS,
            255,
            DEFAULT_CLIPPED_LEVEL_STEP,
            MIN_MIC_LEVEL,
            MAX_MIC_LEVEL,
            predictor.as_ref(),
        );
        analyze_zero_crest_factor_audio(config.window_length, NUM_CHANNELS, 0.99, predictor.as_mut());
        check_channel_estimates_with_value(
            NUM_CHANNELS,
            255,
            DEFAULT_CLIPPED_LEVEL_STEP,
            MIN_MIC_LEVEL,
            MAX_MIC_LEVEL,
            predictor.as_ref(),
            DEFAULT_CLIPPED_LEVEL_STEP,
        );
    }
}
