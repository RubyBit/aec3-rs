//! Limiter mirrored from WebRTC AGC2.

use crate::audio_processing::agc2::agc2_common::{
    MAX_FLOAT_S16_VALUE, MAXIMAL_NUMBER_OF_SAMPLES_PER_CHANNEL, MIN_FLOAT_S16_VALUE,
    SUB_FRAMES_IN_FRAME,
};
use crate::audio_processing::agc2::fixed_digital_level_estimator::FixedDigitalLevelEstimator;
use crate::audio_processing::agc2::interpolated_gain_curve::{InterpolatedGainCurve, Stats};

const ATTACK_FIRST_SUBFRAME_INTERPOLATION_POWER: f32 = 8.0;

pub struct Limiter {
    interp_gain_curve: InterpolatedGainCurve,
    level_estimator: FixedDigitalLevelEstimator,
    scaling_factors: [f32; SUB_FRAMES_IN_FRAME + 1],
    per_sample_scaling_factors: [f32; MAXIMAL_NUMBER_OF_SAMPLES_PER_CHANNEL],
    last_scaling_factor: f32,
}

impl Limiter {
    pub fn new(samples_per_channel: usize) -> Self {
        assert!(samples_per_channel <= MAXIMAL_NUMBER_OF_SAMPLES_PER_CHANNEL);
        Self {
            interp_gain_curve: InterpolatedGainCurve::new(),
            level_estimator: FixedDigitalLevelEstimator::new(samples_per_channel),
            scaling_factors: [1.0; SUB_FRAMES_IN_FRAME + 1],
            per_sample_scaling_factors: [1.0; MAXIMAL_NUMBER_OF_SAMPLES_PER_CHANNEL],
            last_scaling_factor: 1.0,
        }
    }

    pub fn process(&mut self, signal: &mut [&mut [f32]]) {
        assert!(!signal.is_empty());
        let samples_per_channel = signal[0].len();
        assert!(samples_per_channel <= MAXIMAL_NUMBER_OF_SAMPLES_PER_CHANNEL);
        for channel in signal.iter() {
            assert_eq!(channel.len(), samples_per_channel);
        }

        let signal_const: Vec<&[f32]> = signal.iter().map(|ch| &ch[..]).collect();
        let level_estimate = self.level_estimator.compute_level(&signal_const);

        self.scaling_factors[0] = self.last_scaling_factor;
        for (i, &x) in level_estimate.iter().enumerate() {
            self.scaling_factors[i + 1] = self.interp_gain_curve.look_up_gain_to_apply(x);
        }

        let per_sample = &mut self.per_sample_scaling_factors[..samples_per_channel];
        compute_per_sample_subframe_factors(&self.scaling_factors, per_sample);
        scale_samples(per_sample, signal);

        self.last_scaling_factor = self.scaling_factors[SUB_FRAMES_IN_FRAME];
    }

    pub fn get_gain_curve_stats(&self) -> Stats {
        self.interp_gain_curve.get_stats()
    }

    pub fn set_samples_per_channel(&mut self, samples_per_channel: usize) {
        assert!(samples_per_channel <= MAXIMAL_NUMBER_OF_SAMPLES_PER_CHANNEL);
        self.level_estimator
            .set_samples_per_channel(samples_per_channel);
    }

    pub fn reset(&mut self) {
        self.level_estimator.reset();
    }

    pub fn last_audio_level(&self) -> f32 {
        self.level_estimator.last_audio_level()
    }
}

fn interpolate_first_subframe(last_factor: f32, current_factor: f32, subframe: &mut [f32]) {
    let n = subframe.len() as f32;
    let p = ATTACK_FIRST_SUBFRAME_INTERPOLATION_POWER;
    for (i, v) in subframe.iter_mut().enumerate() {
        let t = i as f32 / n;
        *v = (1.0 - t).powf(p) * (last_factor - current_factor) + current_factor;
    }
}

fn compute_per_sample_subframe_factors(
    scaling_factors: &[f32; SUB_FRAMES_IN_FRAME + 1],
    per_sample_scaling_factors: &mut [f32],
) {
    let num_subframes = scaling_factors.len() - 1;
    assert_eq!(per_sample_scaling_factors.len() % num_subframes, 0);
    let subframe_size = per_sample_scaling_factors.len() / num_subframes;

    let is_attack = scaling_factors[0] > scaling_factors[1];
    if is_attack {
        interpolate_first_subframe(
            scaling_factors[0],
            scaling_factors[1],
            &mut per_sample_scaling_factors[..subframe_size],
        );
    }

    for i in if is_attack { 1 } else { 0 }..num_subframes {
        let subframe_start = i * subframe_size;
        let scaling_start = scaling_factors[i];
        let scaling_end = scaling_factors[i + 1];
        let scaling_diff = (scaling_end - scaling_start) / subframe_size as f32;
        for j in 0..subframe_size {
            per_sample_scaling_factors[subframe_start + j] =
                scaling_start + scaling_diff * j as f32;
        }
    }
}

fn scale_samples(per_sample_scaling_factors: &[f32], signal: &mut [&mut [f32]]) {
    let samples_per_channel = signal[0].len();
    assert_eq!(samples_per_channel, per_sample_scaling_factors.len());
    for channel in signal.iter_mut() {
        for i in 0..samples_per_channel {
            channel[i] = (channel[i] * per_sample_scaling_factors[i])
                .clamp(MIN_FLOAT_S16_VALUE, MAX_FLOAT_S16_VALUE);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_processing::agc2::agc2_common::MAX_ABS_FLOAT_S16_VALUE;
    use crate::audio_processing::agc2::limiter_db_gain_curve::LIMITER_MAX_INPUT_LEVEL_DBFS;
    use crate::audio_processing::agc2::vector_float_frame::VectorFloatFrame;

    fn db_to_ratio(v: f32) -> f32 {
        10.0f32.powf(v / 20.0)
    }

    fn dbfs_to_float_s16(v: f32) -> f32 {
        db_to_ratio(v) * MAX_ABS_FLOAT_S16_VALUE
    }

    #[test]
    fn limiter_should_construct_and_run() {
        const SAMPLES_PER_CHANNEL: usize = 480;
        let mut limiter = Limiter::new(SAMPLES_PER_CHANNEL);

        let mut frame = VectorFloatFrame::new(1, SAMPLES_PER_CHANNEL, MAX_ABS_FLOAT_S16_VALUE);
        let mut view = frame.view();
        limiter.process(&mut view);
    }

    #[test]
    fn output_volume_above_threshold() {
        const SAMPLES_PER_CHANNEL: usize = 480;
        let input_level = (MAX_ABS_FLOAT_S16_VALUE
            + dbfs_to_float_s16(LIMITER_MAX_INPUT_LEVEL_DBFS as f32))
            / 2.0;
        let mut limiter = Limiter::new(SAMPLES_PER_CHANNEL);

        let mut frame = VectorFloatFrame::new(1, SAMPLES_PER_CHANNEL, input_level);

        for _ in 0..5 {
            {
                let mut view = frame.view();
                view[0].fill(input_level);
                limiter.process(&mut view);
            }
        }

        {
            let mut view = frame.view();
            view[0].fill(input_level);
            limiter.process(&mut view);
        }

        for &sample in frame.view_const()[0] {
            assert!(0.9 * MAX_ABS_FLOAT_S16_VALUE < sample);
        }
    }
}
