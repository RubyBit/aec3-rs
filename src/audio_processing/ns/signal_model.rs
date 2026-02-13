//! Signal model features used by the NS speech-probability estimator.

use super::ns_common::{FFT_SIZE_BY_2_PLUS_1, LRT_FEATURE_THR};

#[derive(Debug, Clone)]
pub struct SignalModel {
    pub lrt: f32,
    pub spectral_diff: f32,
    pub spectral_flatness: f32,
    pub avg_log_lrt: [f32; FFT_SIZE_BY_2_PLUS_1],
}

impl Default for SignalModel {
    fn default() -> Self {
        Self {
            lrt: LRT_FEATURE_THR,
            spectral_diff: 0.5,
            spectral_flatness: 0.5,
            avg_log_lrt: [LRT_FEATURE_THR; FFT_SIZE_BY_2_PLUS_1],
        }
    }
}
