//! Suppression parameter mapping from NS suppression level.

use super::ns_config::SuppressionLevel;

#[derive(Debug, Clone, Copy)]
pub struct SuppressionParams {
    pub over_subtraction_factor: f32,
    pub minimum_attenuating_gain: f32,
    pub use_attenuation_adjustment: bool,
}

impl SuppressionParams {
    pub fn new(suppression_level: SuppressionLevel) -> Self {
        match suppression_level {
            SuppressionLevel::K6dB => Self {
                over_subtraction_factor: 1.0,
                minimum_attenuating_gain: 0.5,
                use_attenuation_adjustment: false,
            },
            SuppressionLevel::K12dB => Self {
                over_subtraction_factor: 1.0,
                minimum_attenuating_gain: 0.25,
                use_attenuation_adjustment: true,
            },
            SuppressionLevel::K18dB => Self {
                over_subtraction_factor: 1.1,
                minimum_attenuating_gain: 0.125,
                use_attenuation_adjustment: true,
            },
            SuppressionLevel::K21dB => Self {
                over_subtraction_factor: 1.25,
                minimum_attenuating_gain: 0.09,
                use_attenuation_adjustment: true,
            },
        }
    }
}