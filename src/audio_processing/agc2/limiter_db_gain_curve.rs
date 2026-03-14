//! Limiter gain curve in dB domain mirrored from WebRTC AGC2.

use crate::audio_processing::agc2::agc2_common::MAX_ABS_FLOAT_S16_VALUE;

pub const LIMITER_MAX_INPUT_LEVEL_DBFS: f64 = 1.0;
pub const LIMITER_KNEE_SMOOTHNESS_DB: f64 = 1.0;
pub const LIMITER_COMPRESSION_RATIO: f64 = 5.0;

fn dbfs_to_float_s16(dbfs: f64) -> f64 {
    10.0f64.powf(dbfs / 20.0) * MAX_ABS_FLOAT_S16_VALUE as f64
}

fn float_s16_to_dbfs(linear: f64) -> f64 {
    if linear <= 0.0 {
        f64::NEG_INFINITY
    } else {
        20.0 * (linear / MAX_ABS_FLOAT_S16_VALUE as f64).log10()
    }
}

fn compute_knee_start(
    max_input_level_db: f64,
    knee_smoothness_db: f64,
    compression_ratio: f64,
) -> f64 {
    assert!(
        (compression_ratio - 1.0) * knee_smoothness_db / (2.0 * compression_ratio)
            < max_input_level_db
    );
    -knee_smoothness_db / 2.0 - max_input_level_db / (compression_ratio - 1.0)
}

fn compute_knee_region_polynomial(
    knee_start_dbfs: f64,
    knee_smoothness_db: f64,
    compression_ratio: f64,
) -> [f64; 3] {
    let a = (1.0 - compression_ratio) / (2.0 * knee_smoothness_db * compression_ratio);
    let b = 1.0 - 2.0 * a * knee_start_dbfs;
    let c = a * knee_start_dbfs * knee_start_dbfs;
    [a, b, c]
}

fn compute_limiter_d1(max_input_level_db: f64, compression_ratio: f64) -> f64 {
    (10.0f64.powf(-max_input_level_db / (20.0 * compression_ratio)) * (1.0 - compression_ratio)
        / compression_ratio)
        / MAX_ABS_FLOAT_S16_VALUE as f64
}

fn compute_limiter_d2(compression_ratio: f64) -> f64 {
    (1.0 - 2.0 * compression_ratio) / compression_ratio
}

fn compute_limiter_i2(
    max_input_level_db: f64,
    compression_ratio: f64,
    gain_curve_limiter_i1: f64,
) -> f64 {
    assert!(gain_curve_limiter_i1 != 0.0);
    10.0f64.powf(-max_input_level_db / (20.0 * compression_ratio))
        / gain_curve_limiter_i1
        / (MAX_ABS_FLOAT_S16_VALUE as f64).powf(gain_curve_limiter_i1 - 1.0)
}

#[derive(Debug, Clone)]
pub struct LimiterDbGainCurve {
    max_input_level_db: f64,
    max_input_level_linear: f64,
    knee_start_dbfs: f64,
    knee_start_linear: f64,
    limiter_start_dbfs: f64,
    limiter_start_linear: f64,
    knee_region_polynomial: [f64; 3],
    gain_curve_limiter_d1: f64,
    gain_curve_limiter_d2: f64,
    gain_curve_limiter_i1: f64,
    gain_curve_limiter_i2: f64,
}

impl Default for LimiterDbGainCurve {
    fn default() -> Self {
        Self::new()
    }
}

impl LimiterDbGainCurve {
    pub fn new() -> Self {
        let max_input_level_db = LIMITER_MAX_INPUT_LEVEL_DBFS;
        let knee_smoothness_db = LIMITER_KNEE_SMOOTHNESS_DB;
        let compression_ratio = LIMITER_COMPRESSION_RATIO;

        assert!(knee_smoothness_db > 0.0);
        assert!(compression_ratio > 1.0);

        let max_input_level_linear = dbfs_to_float_s16(max_input_level_db);
        let knee_start_dbfs =
            compute_knee_start(max_input_level_db, knee_smoothness_db, compression_ratio);
        let knee_start_linear = dbfs_to_float_s16(knee_start_dbfs);
        let limiter_start_dbfs = knee_start_dbfs + knee_smoothness_db;
        let limiter_start_linear = dbfs_to_float_s16(limiter_start_dbfs);
        let knee_region_polynomial =
            compute_knee_region_polynomial(knee_start_dbfs, knee_smoothness_db, compression_ratio);

        let gain_curve_limiter_d1 = compute_limiter_d1(max_input_level_db, compression_ratio);
        let gain_curve_limiter_d2 = compute_limiter_d2(compression_ratio);
        let gain_curve_limiter_i1 = 1.0 / compression_ratio;
        let gain_curve_limiter_i2 =
            compute_limiter_i2(max_input_level_db, compression_ratio, gain_curve_limiter_i1);

        assert!(max_input_level_db >= knee_start_dbfs + knee_smoothness_db);

        Self {
            max_input_level_db,
            max_input_level_linear,
            knee_start_dbfs,
            knee_start_linear,
            limiter_start_dbfs,
            limiter_start_linear,
            knee_region_polynomial,
            gain_curve_limiter_d1,
            gain_curve_limiter_d2,
            gain_curve_limiter_i1,
            gain_curve_limiter_i2,
        }
    }

    pub fn max_input_level_db(&self) -> f64 {
        self.max_input_level_db
    }

    pub fn max_input_level_linear(&self) -> f64 {
        self.max_input_level_linear
    }

    pub fn knee_start_linear(&self) -> f64 {
        self.knee_start_linear
    }

    pub fn limiter_start_linear(&self) -> f64 {
        self.limiter_start_linear
    }

    pub fn get_output_level_dbfs(&self, input_level_dbfs: f64) -> f64 {
        if input_level_dbfs < self.knee_start_dbfs {
            input_level_dbfs
        } else if input_level_dbfs < self.limiter_start_dbfs {
            self.get_knee_region_output_level_dbfs(input_level_dbfs)
        } else {
            self.get_compressor_region_output_level_dbfs(input_level_dbfs)
        }
    }

    pub fn get_gain_linear(&self, input_level_linear: f64) -> f64 {
        if input_level_linear < self.knee_start_linear {
            return 1.0;
        }
        dbfs_to_float_s16(self.get_output_level_dbfs(float_s16_to_dbfs(input_level_linear)))
            / input_level_linear
    }

    pub fn get_gain_first_derivative_linear(&self, x: f64) -> f64 {
        assert!(x >= self.limiter_start_linear - 1e-7 * MAX_ABS_FLOAT_S16_VALUE as f64);
        self.gain_curve_limiter_d1
            * (x / MAX_ABS_FLOAT_S16_VALUE as f64).powf(self.gain_curve_limiter_d2)
    }

    pub fn get_gain_integral_linear(&self, x0: f64, x1: f64) -> f64 {
        assert!(x0 <= x1);
        assert!(x0 >= self.limiter_start_linear);
        let limiter_integral = |x: f64| self.gain_curve_limiter_i2 * x.powf(self.gain_curve_limiter_i1);
        limiter_integral(x1) - limiter_integral(x0)
    }

    fn get_knee_region_output_level_dbfs(&self, input_level_dbfs: f64) -> f64 {
        self.knee_region_polynomial[0] * input_level_dbfs * input_level_dbfs
            + self.knee_region_polynomial[1] * input_level_dbfs
            + self.knee_region_polynomial[2]
    }

    fn get_compressor_region_output_level_dbfs(&self, input_level_dbfs: f64) -> f64 {
        (input_level_dbfs - self.max_input_level_db) / LIMITER_COMPRESSION_RATIO
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct_destruct() {
        let _l = LimiterDbGainCurve::new();
    }

    #[test]
    fn gain_curve_should_be_monotone() {
        let l = LimiterDbGainCurve::new();
        let mut level = -90.0;
        let mut last_output_level = 0.0;
        let mut has_last_output_level = false;
        while level <= l.max_input_level_db() {
            let current_output_level = l.get_output_level_dbfs(level);
            if !has_last_output_level {
                last_output_level = current_output_level;
                has_last_output_level = true;
            }
            assert!(last_output_level <= current_output_level);
            last_output_level = current_output_level;
            level += 0.5;
        }
    }

    #[test]
    fn gain_curve_should_be_continuous() {
        let l = LimiterDbGainCurve::new();
        let mut level = -90.0;
        let mut last_output_level = 0.0;
        let mut has_last_output_level = false;
        const MAX_DELTA: f64 = 0.5;
        while level <= l.max_input_level_db() {
            let current_output_level = l.get_output_level_dbfs(level);
            if !has_last_output_level {
                last_output_level = current_output_level;
                has_last_output_level = true;
            }
            assert!(current_output_level <= last_output_level + MAX_DELTA);
            last_output_level = current_output_level;
            level += 0.5;
        }
    }

    #[test]
    fn output_gain_should_be_less_than_full_scale() {
        let l = LimiterDbGainCurve::new();
        let mut level = -90.0;
        while level <= l.max_input_level_db() {
            let current_output_level = l.get_output_level_dbfs(level);
            assert!(current_output_level <= 0.0);
            level += 0.5;
        }
    }
}
