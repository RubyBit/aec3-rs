//! Interpolated gain curve mirrored from WebRTC AGC2.

use crate::audio_processing::agc2::agc2_common::{
    INTERPOLATED_GAIN_CURVE_KNEE_POINTS, INTERPOLATED_GAIN_CURVE_TOTAL_POINTS,
};

pub const INPUT_LEVEL_SCALING_FACTOR: f32 = 32768.0;
pub const MAX_INPUT_LEVEL_LINEAR: f32 = 36766.300710566735f32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GainCurveRegion {
    Identity = 0,
    Knee = 1,
    Limiter = 2,
    Saturation = 3,
}

#[derive(Debug, Clone, Copy)]
pub struct Stats {
    pub look_ups_identity_region: usize,
    pub look_ups_knee_region: usize,
    pub look_ups_limiter_region: usize,
    pub look_ups_saturation_region: usize,
    pub available: bool,
    pub region: GainCurveRegion,
    pub region_duration_frames: i64,
}

impl Default for Stats {
    fn default() -> Self {
        Self {
            look_ups_identity_region: 0,
            look_ups_knee_region: 0,
            look_ups_limiter_region: 0,
            look_ups_saturation_region: 0,
            available: false,
            region: GainCurveRegion::Identity,
            region_duration_frames: 0,
        }
    }
}

pub struct InterpolatedGainCurve {
    stats: Stats,
}

impl Default for InterpolatedGainCurve {
    fn default() -> Self {
        Self::new()
    }
}

impl InterpolatedGainCurve {
    pub const APPROXIMATION_PARAMS_X: [f32; INTERPOLATED_GAIN_CURVE_TOTAL_POINTS] = [
        30057.296875,
        30148.986328125,
        30240.67578125,
        30424.052734375,
        30607.4296875,
        30790.806640625,
        30974.18359375,
        31157.560546875,
        31340.939453125,
        31524.31640625,
        31707.693359375,
        31891.0703125,
        32074.447265625,
        32257.82421875,
        32441.201171875,
        32624.580078125,
        32807.95703125,
        32991.33203125,
        33174.7109375,
        33358.08984375,
        33541.46484375,
        33724.84375,
        33819.53515625,
        34009.5390625,
        34200.05859375,
        34389.81640625,
        34674.48828125,
        35054.375,
        35434.86328125,
        35814.81640625,
        36195.16796875,
        36575.03125,
    ];

    pub const APPROXIMATION_PARAMS_M: [f32; INTERPOLATED_GAIN_CURVE_TOTAL_POINTS] = [
        -3.515235675877192989e-07,
        -1.050251626111275982e-06,
        -2.085213736791047268e-06,
        -3.443004743530764244e-06,
        -4.773849468620028347e-06,
        -6.077375928725814447e-06,
        -7.353257842623861507e-06,
        -8.601219633419532329e-06,
        -9.821013009059242904e-06,
        -1.101243378798244521e-05,
        -1.217532644659513608e-05,
        -1.330956911260727793e-05,
        -1.441507538402220234e-05,
        -1.549179251014720649e-05,
        -1.653970684856176376e-05,
        -1.755882840370759368e-05,
        -1.854918446042574942e-05,
        -1.951086778717581183e-05,
        -2.044398024736437947e-05,
        -2.134862734237685800e-05,
        -2.222496914328075945e-05,
        -2.265374678245279938e-05,
        -2.242570917587727308e-05,
        -2.220122041762806475e-05,
        -2.198020956711843610e-05,
        -2.176260204578284174e-05,
        -2.133731686626560986e-05,
        -2.092481918225530535e-05,
        -2.052459603874012828e-05,
        -2.013615448959171772e-05,
        -1.975903069251216948e-05,
        -1.939277899509761482e-05,
    ];

    pub const APPROXIMATION_PARAMS_Q: [f32; INTERPOLATED_GAIN_CURVE_TOTAL_POINTS] = [
        1.010565876960754395,
        1.031631827354431152,
        1.062929749488830566,
        1.104239225387573242,
        1.144973039627075195,
        1.185109615325927734,
        1.224629044532775879,
        1.263512492179870605,
        1.301741957664489746,
        1.339300632476806641,
        1.376173257827758789,
        1.412345528602600098,
        1.447803974151611328,
        1.482536554336547852,
        1.516532182693481445,
        1.549780607223510742,
        1.582272171974182129,
        1.613999366760253906,
        1.644955039024353027,
        1.675132393836975098,
        1.704526185989379883,
        1.718986630439758301,
        1.711274504661560059,
        1.703639745712280273,
        1.696081161499023438,
        1.688597679138183594,
        1.673851132392883301,
        1.659391283988952637,
        1.645209431648254395,
        1.631297469139099121,
        1.617647409439086914,
        1.604251742362976074,
    ];

    pub fn new() -> Self {
        Self {
            stats: Stats::default(),
        }
    }

    pub fn get_stats(&self) -> Stats {
        self.stats
    }

    pub fn look_up_gain_to_apply(&mut self, input_level: f32) -> f32 {
        self.update_stats(input_level);

        if input_level <= Self::APPROXIMATION_PARAMS_X[0] {
            return 1.0;
        }

        if input_level >= MAX_INPUT_LEVEL_LINEAR {
            return 32768.0 / input_level;
        }

        // Mirror C++ `std::lower_bound`: first element that is `>= input_level`.
        let pos = Self::APPROXIMATION_PARAMS_X.partition_point(|&x| x < input_level);
        let index = pos - 1;

        let gain =
            Self::APPROXIMATION_PARAMS_M[index] * input_level + Self::APPROXIMATION_PARAMS_Q[index];
        debug_assert!(gain >= 0.0);
        gain
    }

    fn update_stats(&mut self, input_level: f32) {
        self.stats.available = true;

        let region = if input_level < Self::APPROXIMATION_PARAMS_X[0] {
            self.stats.look_ups_identity_region += 1;
            GainCurveRegion::Identity
        } else if input_level
            < Self::APPROXIMATION_PARAMS_X[INTERPOLATED_GAIN_CURVE_KNEE_POINTS - 1]
        {
            self.stats.look_ups_knee_region += 1;
            GainCurveRegion::Knee
        } else if input_level < MAX_INPUT_LEVEL_LINEAR {
            self.stats.look_ups_limiter_region += 1;
            GainCurveRegion::Limiter
        } else {
            self.stats.look_ups_saturation_region += 1;
            GainCurveRegion::Saturation
        };

        if region == self.stats.region {
            self.stats.region_duration_frames += 1;
        } else {
            self.stats.region_duration_frames = 0;
            self.stats.region = region;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_processing::agc2::agc2_common::MAX_ABS_FLOAT_S16_VALUE;
    use crate::audio_processing::agc2::limiter_db_gain_curve::LimiterDbGainCurve;

    const LEVEL_EPSILON: f64 = 1e-2 * MAX_ABS_FLOAT_S16_VALUE as f64;
    const INTERPOLATED_GAIN_CURVE_TOLERANCE: f64 = 1.0 / 32768.0;

    fn lin_space(l: f64, r: f64, num_points: usize) -> Vec<f64> {
        assert!(num_points >= 2);
        let step = (r - l) / (num_points as f64 - 1.0);
        (0..num_points).map(|i| l + i as f64 * step).collect()
    }

    #[test]
    fn create_use() {
        let mut igc = InterpolatedGainCurve::new();

        let limiter = LimiterDbGainCurve::new();
        let levels = lin_space(LEVEL_EPSILON, limiter.max_input_level_linear() + 1.0, 500);
        for level in levels {
            assert!(igc.look_up_gain_to_apply(level as f32) >= 0.0);
        }
    }

    #[test]
    fn check_valid_output() {
        let mut igc = InterpolatedGainCurve::new();
        let limiter = LimiterDbGainCurve::new();

        let levels = lin_space(LEVEL_EPSILON, limiter.max_input_level_linear() * 2.0, 500);
        for level in levels {
            let gain = igc.look_up_gain_to_apply(level as f32);
            assert!(gain >= 0.0);
            assert!(gain <= 1.0);
        }
    }

    #[test]
    fn check_monotonicity() {
        let mut igc = InterpolatedGainCurve::new();
        let limiter = LimiterDbGainCurve::new();

        let levels = lin_space(
            LEVEL_EPSILON,
            limiter.max_input_level_linear() + LEVEL_EPSILON + 0.5,
            500,
        );
        let mut prev_gain = igc.look_up_gain_to_apply(0.0);
        for level in levels {
            let gain = igc.look_up_gain_to_apply(level as f32);
            assert!(prev_gain >= gain);
            prev_gain = gain;
        }
    }

    #[test]
    fn check_approximation() {
        let mut igc = InterpolatedGainCurve::new();
        let limiter = LimiterDbGainCurve::new();

        let levels = lin_space(
            LEVEL_EPSILON,
            limiter.max_input_level_linear() - LEVEL_EPSILON,
            500,
        );
        for level in levels {
            assert!(
                (limiter.get_gain_linear(level) - igc.look_up_gain_to_apply(level as f32) as f64)
                    .abs()
                    < INTERPOLATED_GAIN_CURVE_TOLERANCE
            );
        }
    }

    #[test]
    fn check_region_boundaries() {
        let mut igc = InterpolatedGainCurve::new();
        let limiter = LimiterDbGainCurve::new();

        let levels = vec![
            LEVEL_EPSILON,
            limiter.knee_start_linear() + LEVEL_EPSILON,
            limiter.limiter_start_linear() + LEVEL_EPSILON,
            limiter.max_input_level_linear() + LEVEL_EPSILON,
        ];
        for level in levels {
            igc.look_up_gain_to_apply(level as f32);
        }

        let stats = igc.get_stats();
        assert_eq!(1, stats.look_ups_identity_region);
        assert_eq!(1, stats.look_ups_knee_region);
        assert_eq!(1, stats.look_ups_limiter_region);
        assert_eq!(1, stats.look_ups_saturation_region);
    }

    #[test]
    fn check_identity_region() {
        const NUM_STEPS: usize = 10;
        let mut igc = InterpolatedGainCurve::new();
        let limiter = LimiterDbGainCurve::new();

        let levels = lin_space(LEVEL_EPSILON, limiter.knee_start_linear(), NUM_STEPS);
        for level in levels {
            assert_eq!(1.0, igc.look_up_gain_to_apply(level as f32));
        }

        let stats = igc.get_stats();
        assert_eq!(NUM_STEPS - 1, stats.look_ups_identity_region);
        assert_eq!(1, stats.look_ups_knee_region);
        assert_eq!(0, stats.look_ups_limiter_region);
        assert_eq!(0, stats.look_ups_saturation_region);
    }

    #[test]
    fn check_no_over_approximation_knee() {
        const NUM_STEPS: usize = 10;
        let mut igc = InterpolatedGainCurve::new();
        let limiter = LimiterDbGainCurve::new();

        let levels = lin_space(
            limiter.knee_start_linear() + LEVEL_EPSILON,
            limiter.limiter_start_linear(),
            NUM_STEPS,
        );
        for level in levels {
            assert!(
                igc.look_up_gain_to_apply(level as f32) as f64
                    <= limiter.get_gain_linear(level) + 1e-7
            );
        }

        let stats = igc.get_stats();
        assert_eq!(0, stats.look_ups_identity_region);
        assert_eq!(NUM_STEPS - 1, stats.look_ups_knee_region);
        assert_eq!(1, stats.look_ups_limiter_region);
        assert_eq!(0, stats.look_ups_saturation_region);
    }

    #[test]
    fn check_no_over_approximation_beyond_knee() {
        const NUM_STEPS: usize = 10;
        let mut igc = InterpolatedGainCurve::new();
        let limiter = LimiterDbGainCurve::new();

        let levels = lin_space(
            limiter.limiter_start_linear() + LEVEL_EPSILON,
            limiter.max_input_level_linear() - LEVEL_EPSILON,
            NUM_STEPS,
        );
        for level in levels {
            assert!(
                igc.look_up_gain_to_apply(level as f32) as f64
                    <= limiter.get_gain_linear(level) + 1e-7
            );
        }

        let stats = igc.get_stats();
        assert_eq!(0, stats.look_ups_identity_region);
        assert_eq!(0, stats.look_ups_knee_region);
        assert_eq!(NUM_STEPS, stats.look_ups_limiter_region);
        assert_eq!(0, stats.look_ups_saturation_region);
    }

    #[test]
    fn check_no_over_approximation_with_saturation() {
        const NUM_STEPS: usize = 3;
        let mut igc = InterpolatedGainCurve::new();
        let limiter = LimiterDbGainCurve::new();

        let levels = lin_space(
            limiter.max_input_level_linear() + LEVEL_EPSILON,
            limiter.max_input_level_linear() + LEVEL_EPSILON + 0.5,
            NUM_STEPS,
        );
        for level in levels {
            assert!(
                igc.look_up_gain_to_apply(level as f32) as f64 <= limiter.get_gain_linear(level)
            );
        }

        let stats = igc.get_stats();
        assert_eq!(0, stats.look_ups_identity_region);
        assert_eq!(0, stats.look_ups_knee_region);
        assert_eq!(0, stats.look_ups_limiter_region);
        assert_eq!(NUM_STEPS, stats.look_ups_saturation_region);
    }
}
