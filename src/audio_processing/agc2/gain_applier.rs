//! Gain applier mirrored from WebRTC AGC2 `gain_applier`.

use crate::audio_processing::agc2::agc2_common::{MAX_FLOAT_S16_VALUE, MIN_FLOAT_S16_VALUE};

/// Applies a (possibly ramped) gain to deinterleaved floating-point frames.
pub struct GainApplier {
    /// Whether to clip samples after gain is applied.
    hard_clip_samples: bool,
    last_gain_factor: f32,
    /// If this differs from `last_gain_factor`, gain ramps during next `apply_gain`.
    current_gain_factor: f32,
    samples_per_channel: i32,
    inverse_samples_per_channel: f32,
}

impl GainApplier {
    pub fn new(hard_clip_samples: bool, initial_gain_factor: f32) -> Self {
        Self {
            hard_clip_samples,
            last_gain_factor: initial_gain_factor,
            current_gain_factor: initial_gain_factor,
            samples_per_channel: -1,
            inverse_samples_per_channel: -1.0,
        }
    }

    pub fn apply_gain(&mut self, signal: &mut [&mut [f32]]) {
        assert!(!signal.is_empty());
        let samples_per_channel = signal[0].len() as i32;
        assert!(samples_per_channel > 0);
        for channel in signal.iter() {
            assert_eq!(channel.len() as i32, samples_per_channel);
        }

        if samples_per_channel != self.samples_per_channel {
            self.initialize(samples_per_channel);
        }

        apply_gain_with_ramping(
            self.last_gain_factor,
            self.current_gain_factor,
            self.inverse_samples_per_channel,
            signal,
        );

        self.last_gain_factor = self.current_gain_factor;

        if self.hard_clip_samples {
            clip_signal(signal);
        }
    }

    pub fn set_gain_factor(&mut self, gain_factor: f32) {
        assert!(gain_factor > 0.0);
        self.current_gain_factor = gain_factor;
    }

    pub fn get_gain_factor(&self) -> f32 {
        self.current_gain_factor
    }

    fn initialize(&mut self, samples_per_channel: i32) {
        assert!(samples_per_channel > 0);
        self.samples_per_channel = samples_per_channel;
        self.inverse_samples_per_channel = 1.0 / self.samples_per_channel as f32;
    }
}

fn gain_close_to_one(gain_factor: f32) -> bool {
    1.0 - 1.0 / MAX_FLOAT_S16_VALUE <= gain_factor && gain_factor <= 1.0 + 1.0 / MAX_FLOAT_S16_VALUE
}

fn clip_signal(signal: &mut [&mut [f32]]) {
    for channel in signal.iter_mut() {
        for sample in channel.iter_mut() {
            *sample = sample.clamp(MIN_FLOAT_S16_VALUE, MAX_FLOAT_S16_VALUE);
        }
    }
}

fn apply_gain_with_ramping(
    last_gain_linear: f32,
    gain_at_end_of_frame_linear: f32,
    inverse_samples_per_channel: f32,
    float_frame: &mut [&mut [f32]],
) {
    // Do not modify signal.
    if last_gain_linear == gain_at_end_of_frame_linear
        && gain_close_to_one(gain_at_end_of_frame_linear)
    {
        return;
    }

    // Gain is constant and different from 1.
    if last_gain_linear == gain_at_end_of_frame_linear {
        for channel in float_frame.iter_mut() {
            for sample in channel.iter_mut() {
                *sample *= gain_at_end_of_frame_linear;
            }
        }
        return;
    }

    // Gain changes: ramp over frame.
    let increment = (gain_at_end_of_frame_linear - last_gain_linear) * inverse_samples_per_channel;
    for channel in float_frame.iter_mut() {
        let mut gain = last_gain_linear;
        for sample in channel.iter_mut() {
            *sample *= gain;
            gain += increment;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_processing::agc2::vector_float_frame::VectorFloatFrame;

    #[test]
    fn initial_gain_is_respected() {
        let initial_signal_level = 123.0f32;
        let gain_factor = 10.0f32;
        let mut fake_audio = VectorFloatFrame::new(1, 1, initial_signal_level);
        let mut gain_applier = GainApplier::new(true, gain_factor);

        let mut fake_view = fake_audio.view();
        gain_applier.apply_gain(&mut fake_view);
        assert!((fake_audio.sample(0, 0) - initial_signal_level * gain_factor).abs() < 0.1);
    }

    #[test]
    fn clipping_is_done() {
        let initial_signal_level = 30000.0f32;
        let gain_factor = 10.0f32;
        let mut fake_audio = VectorFloatFrame::new(1, 1, initial_signal_level);
        let mut gain_applier = GainApplier::new(true, gain_factor);

        let mut view = fake_audio.view();
        gain_applier.apply_gain(&mut view);
        assert!((fake_audio.sample(0, 0) - i16::MAX as f32).abs() < 0.1);
    }

    #[test]
    fn clipping_is_not_done() {
        let initial_signal_level = 30000.0f32;
        let gain_factor = 10.0f32;
        let mut fake_audio = VectorFloatFrame::new(1, 1, initial_signal_level);
        let mut gain_applier = GainApplier::new(false, gain_factor);

        let mut view = fake_audio.view();
        gain_applier.apply_gain(&mut view);
        assert!((fake_audio.sample(0, 0) - initial_signal_level * gain_factor).abs() < 0.1);
    }

    #[test]
    fn ramping_is_done() {
        let initial_signal_level = 30000.0f32;
        let initial_gain_factor = 1.0f32;
        let target_gain_factor = 0.5f32;
        let num_channels = 3usize;
        let samples_per_channel = 4usize;
        let mut fake_audio =
            VectorFloatFrame::new(num_channels, samples_per_channel, initial_signal_level);
        let mut gain_applier = GainApplier::new(false, initial_gain_factor);

        gain_applier.set_gain_factor(target_gain_factor);
        let mut view = fake_audio.view();
        gain_applier.apply_gain(&mut view);

        for channel in 0..num_channels {
            let channel_view = fake_audio.view_const();
            let mut max_signal_change = 0.0f32;
            let mut last_signal_level = initial_signal_level;
            for &sample in channel_view[channel] {
                let current_change = (last_signal_level - sample).abs();
                max_signal_change = max_signal_change.max(current_change);
                last_signal_level = sample;
            }
            let total_gain_change =
                ((initial_gain_factor - target_gain_factor) * initial_signal_level).abs();
            assert!(
                (max_signal_change - total_gain_change / samples_per_channel as f32).abs() < 0.1
            );
        }

        let mut next_fake_audio_frame =
            VectorFloatFrame::new(num_channels, samples_per_channel, initial_signal_level);
        let mut next_view = next_fake_audio_frame.view();
        gain_applier.apply_gain(&mut next_view);

        assert!(
            (next_fake_audio_frame.sample(0, 0) - initial_signal_level * target_gain_factor).abs()
                < 0.1
        );
    }
}
