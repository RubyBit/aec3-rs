//! EchoCanceller3 configuration ported from the WebRTC reference implementation.
//!
//! The structure layout, default values, and validation logic mirror
//! `reference_aec_cpp/api/echo_canceller3_config.{h,cc}` as closely as possible.

use std::cmp::{max, min};

#[derive(Debug, Clone, PartialEq)]
pub struct EchoCanceller3Config {
    pub buffering: Buffering,
    pub delay: Delay,
    pub filter: Filter,
    pub erle: Erle,
    pub ep_strength: EpStrength,
    pub echo_audibility: EchoAudibility,
    pub render_levels: RenderLevels,
    pub echo_removal_control: EchoRemovalControl,
    pub echo_model: EchoModel,
    pub suppressor: Suppressor,
}

impl Default for EchoCanceller3Config {
    fn default() -> Self {
        Self {
            buffering: Buffering::default(),
            delay: Delay::default(),
            filter: Filter::default(),
            erle: Erle::default(),
            ep_strength: EpStrength::default(),
            echo_audibility: EchoAudibility::default(),
            render_levels: RenderLevels::default(),
            echo_removal_control: EchoRemovalControl::default(),
            echo_model: EchoModel::default(),
            suppressor: Suppressor::default(),
        }
    }
}

impl EchoCanceller3Config {
    /// Validates the configuration by clamping all fields to the same ranges as the
    /// reference implementation. Returns true iff no field required adjustment.
    pub fn validate(&mut self) -> bool {
        let mut res = true;
        let c = self;

        if c.delay.down_sampling_factor != 4 && c.delay.down_sampling_factor != 8 {
            c.delay.down_sampling_factor = 4;
            res = false;
        }

        res &= limit_usize(&mut c.delay.default_delay, 0, 5000);
        res &= limit_usize(&mut c.delay.num_filters, 0, 5000);
        res &= limit_usize(&mut c.delay.delay_headroom_samples, 0, 5000);
        res &= limit_usize(&mut c.delay.hysteresis_limit_blocks, 0, 5000);
        res &= limit_usize(&mut c.delay.fixed_capture_delay_samples, 0, 5000);
        res &= limit_f32(&mut c.delay.delay_estimate_smoothing, 0.0, 1.0);
        res &= limit_f32(&mut c.delay.delay_candidate_detection_threshold, 0.0, 1.0);
        res &= limit_i32(&mut c.delay.delay_selection_thresholds.initial, 1, 250);
        res &= limit_i32(&mut c.delay.delay_selection_thresholds.converged, 1, 250);

        res &= floor_limit_usize(&mut c.filter.main.length_blocks, 1);
        res &= limit_f32(&mut c.filter.main.leakage_converged, 0.0, 1000.0);
        res &= limit_f32(&mut c.filter.main.leakage_diverged, 0.0, 1000.0);
        res &= limit_f32(&mut c.filter.main.error_floor, 0.0, 1000.0);
        res &= limit_f32(&mut c.filter.main.error_ceil, 0.0, 100_000_000.0);
        res &= limit_f32(&mut c.filter.main.noise_gate, 0.0, 100_000_000.0);

        res &= floor_limit_usize(&mut c.filter.main_initial.length_blocks, 1);
        res &= limit_f32(&mut c.filter.main_initial.leakage_converged, 0.0, 1000.0);
        res &= limit_f32(&mut c.filter.main_initial.leakage_diverged, 0.0, 1000.0);
        res &= limit_f32(&mut c.filter.main_initial.error_floor, 0.0, 1000.0);
        res &= limit_f32(&mut c.filter.main_initial.error_ceil, 0.0, 100_000_000.0);
        res &= limit_f32(&mut c.filter.main_initial.noise_gate, 0.0, 100_000_000.0);

        if c.filter.main.length_blocks < c.filter.main_initial.length_blocks {
            c.filter.main_initial.length_blocks = c.filter.main.length_blocks;
            res = false;
        }

        res &= floor_limit_usize(&mut c.filter.shadow.length_blocks, 1);
        res &= limit_f32(&mut c.filter.shadow.rate, 0.0, 1.0);
        res &= limit_f32(&mut c.filter.shadow.noise_gate, 0.0, 100_000_000.0);

        res &= floor_limit_usize(&mut c.filter.shadow_initial.length_blocks, 1);
        res &= limit_f32(&mut c.filter.shadow_initial.rate, 0.0, 1.0);
        res &= limit_f32(&mut c.filter.shadow_initial.noise_gate, 0.0, 100_000_000.0);

        if c.filter.shadow.length_blocks < c.filter.shadow_initial.length_blocks {
            c.filter.shadow_initial.length_blocks = c.filter.shadow.length_blocks;
            res = false;
        }

        res &= limit_usize(&mut c.filter.config_change_duration_blocks, 0, 100_000);
        res &= limit_f32(&mut c.filter.initial_state_seconds, 0.0, 100.0);

        res &= limit_f32(&mut c.erle.min, 1.0, 100_000.0);
        res &= limit_f32(&mut c.erle.max_l, 1.0, 100_000.0);
        res &= limit_f32(&mut c.erle.max_h, 1.0, 100_000.0);
        if c.erle.min > c.erle.max_l || c.erle.min > c.erle.max_h {
            c.erle.min = c.erle.max_l.min(c.erle.max_h);
            res = false;
        }
        res &= limit_usize(&mut c.erle.num_sections, 1, c.filter.main.length_blocks);

        res &= limit_f32(&mut c.ep_strength.default_gain, 0.0, 1_000_000.0);
        res &= limit_f32(&mut c.ep_strength.default_len, -1.0, 1.0);

        const MAX_POWER: f32 = 32_768.0 * 32_768.0;
        res &= limit_f32(&mut c.echo_audibility.low_render_limit, 0.0, MAX_POWER);
        res &= limit_f32(&mut c.echo_audibility.normal_render_limit, 0.0, MAX_POWER);
        res &= limit_f32(&mut c.echo_audibility.floor_power, 0.0, MAX_POWER);
        res &= limit_f32(
            &mut c.echo_audibility.audibility_threshold_lf,
            0.0,
            MAX_POWER,
        );
        res &= limit_f32(
            &mut c.echo_audibility.audibility_threshold_mf,
            0.0,
            MAX_POWER,
        );
        res &= limit_f32(
            &mut c.echo_audibility.audibility_threshold_hf,
            0.0,
            MAX_POWER,
        );

        res &= limit_f32(&mut c.render_levels.active_render_limit, 0.0, MAX_POWER);
        res &= limit_f32(
            &mut c.render_levels.poor_excitation_render_limit,
            0.0,
            MAX_POWER,
        );
        res &= limit_f32(
            &mut c.render_levels.poor_excitation_render_limit_ds8,
            0.0,
            MAX_POWER,
        );

        res &= limit_usize(&mut c.echo_model.noise_floor_hold, 0, 1000);
        res &= limit_f32(&mut c.echo_model.min_noise_floor_power, 0.0, 2_000_000.0);
        res &= limit_f32(&mut c.echo_model.stationary_gate_slope, 0.0, 1_000_000.0);
        res &= limit_f32(&mut c.echo_model.noise_gate_power, 0.0, 1_000_000.0);
        res &= limit_f32(&mut c.echo_model.noise_gate_slope, 0.0, 1_000_000.0);
        res &= limit_usize(&mut c.echo_model.render_pre_window_size, 0, 100);
        res &= limit_usize(&mut c.echo_model.render_post_window_size, 0, 100);

        res &= limit_usize(&mut c.suppressor.nearend_average_blocks, 1, 5000);

        res &= limit_f32(
            &mut c.suppressor.normal_tuning.mask_lf.enr_transparent,
            0.0,
            100.0,
        );
        res &= limit_f32(
            &mut c.suppressor.normal_tuning.mask_lf.enr_suppress,
            0.0,
            100.0,
        );
        res &= limit_f32(
            &mut c.suppressor.normal_tuning.mask_lf.emr_transparent,
            0.0,
            100.0,
        );
        res &= limit_f32(
            &mut c.suppressor.normal_tuning.mask_hf.enr_transparent,
            0.0,
            100.0,
        );
        res &= limit_f32(
            &mut c.suppressor.normal_tuning.mask_hf.enr_suppress,
            0.0,
            100.0,
        );
        res &= limit_f32(
            &mut c.suppressor.normal_tuning.mask_hf.emr_transparent,
            0.0,
            100.0,
        );
        res &= limit_f32(&mut c.suppressor.normal_tuning.max_inc_factor, 0.0, 100.0);
        res &= limit_f32(
            &mut c.suppressor.normal_tuning.max_dec_factor_lf,
            0.0,
            100.0,
        );

        res &= limit_f32(
            &mut c.suppressor.nearend_tuning.mask_lf.enr_transparent,
            0.0,
            100.0,
        );
        res &= limit_f32(
            &mut c.suppressor.nearend_tuning.mask_lf.enr_suppress,
            0.0,
            100.0,
        );
        res &= limit_f32(
            &mut c.suppressor.nearend_tuning.mask_lf.emr_transparent,
            0.0,
            100.0,
        );
        res &= limit_f32(
            &mut c.suppressor.nearend_tuning.mask_hf.enr_transparent,
            0.0,
            100.0,
        );
        res &= limit_f32(
            &mut c.suppressor.nearend_tuning.mask_hf.enr_suppress,
            0.0,
            100.0,
        );
        res &= limit_f32(
            &mut c.suppressor.nearend_tuning.mask_hf.emr_transparent,
            0.0,
            100.0,
        );
        res &= limit_f32(&mut c.suppressor.nearend_tuning.max_inc_factor, 0.0, 100.0);
        res &= limit_f32(
            &mut c.suppressor.nearend_tuning.max_dec_factor_lf,
            0.0,
            100.0,
        );

        res &= limit_f32(
            &mut c.suppressor.dominant_nearend_detection.enr_threshold,
            0.0,
            1_000_000.0,
        );
        res &= limit_f32(
            &mut c.suppressor.dominant_nearend_detection.snr_threshold,
            0.0,
            1_000_000.0,
        );
        res &= limit_usize(
            &mut c.suppressor.dominant_nearend_detection.hold_duration,
            0,
            10_000,
        );
        res &= limit_usize(
            &mut c.suppressor.dominant_nearend_detection.trigger_threshold,
            0,
            10_000,
        );

        res &= limit_usize(
            &mut c
                .suppressor
                .subband_nearend_detection
                .nearend_average_blocks,
            1,
            1024,
        );
        res &= limit_usize(
            &mut c.suppressor.subband_nearend_detection.subband1.low,
            0,
            65,
        );
        res &= limit_usize(
            &mut c.suppressor.subband_nearend_detection.subband1.high,
            c.suppressor.subband_nearend_detection.subband1.low,
            65,
        );
        res &= limit_usize(
            &mut c.suppressor.subband_nearend_detection.subband2.low,
            0,
            65,
        );
        res &= limit_usize(
            &mut c.suppressor.subband_nearend_detection.subband2.high,
            c.suppressor.subband_nearend_detection.subband2.low,
            65,
        );
        res &= limit_f32(
            &mut c.suppressor.subband_nearend_detection.nearend_threshold,
            0.0,
            1.0e24,
        );
        res &= limit_f32(
            &mut c.suppressor.subband_nearend_detection.snr_threshold,
            0.0,
            1.0e24,
        );

        res &= limit_f32(
            &mut c.suppressor.high_bands_suppression.enr_threshold,
            0.0,
            1_000_000.0,
        );
        res &= limit_f32(
            &mut c.suppressor.high_bands_suppression.max_gain_during_echo,
            0.0,
            1.0,
        );
        res &= limit_f32(
            &mut c
                .suppressor
                .high_bands_suppression
                .anti_howling_activation_threshold,
            0.0,
            MAX_POWER,
        );
        res &= limit_f32(
            &mut c.suppressor.high_bands_suppression.anti_howling_gain,
            0.0,
            1.0,
        );

        res &= limit_f32(&mut c.suppressor.floor_first_increase, 0.0, 1_000_000.0);

        res
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Buffering {
    pub excess_render_detection_interval_blocks: usize,
    pub max_allowed_excess_render_blocks: usize,
}

impl Default for Buffering {
    fn default() -> Self {
        Self {
            excess_render_detection_interval_blocks: 250,
            max_allowed_excess_render_blocks: 8,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Delay {
    pub default_delay: usize,
    pub down_sampling_factor: usize,
    pub num_filters: usize,
    pub delay_headroom_samples: usize,
    pub hysteresis_limit_blocks: usize,
    pub fixed_capture_delay_samples: usize,
    pub delay_estimate_smoothing: f32,
    pub delay_candidate_detection_threshold: f32,
    pub delay_selection_thresholds: DelaySelectionThresholds,
    pub use_external_delay_estimator: bool,
    pub log_warning_on_delay_changes: bool,
    pub render_alignment_mixing: AlignmentMixing,
    pub capture_alignment_mixing: AlignmentMixing,
}

impl Default for Delay {
    fn default() -> Self {
        Self {
            default_delay: 5,
            down_sampling_factor: 4,
            num_filters: 5,
            delay_headroom_samples: 32,
            hysteresis_limit_blocks: 1,
            fixed_capture_delay_samples: 0,
            delay_estimate_smoothing: 0.7,
            delay_candidate_detection_threshold: 0.2,
            delay_selection_thresholds: DelaySelectionThresholds {
                initial: 5,
                converged: 20,
            },
            use_external_delay_estimator: false,
            log_warning_on_delay_changes: false,
            render_alignment_mixing: AlignmentMixing {
                downmix: false,
                adaptive_selection: true,
                activity_power_threshold: 10_000.0,
                prefer_first_two_channels: true,
            },
            capture_alignment_mixing: AlignmentMixing {
                downmix: false,
                adaptive_selection: true,
                activity_power_threshold: 10_000.0,
                prefer_first_two_channels: false,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DelaySelectionThresholds {
    pub initial: i32,
    pub converged: i32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AlignmentMixing {
    pub downmix: bool,
    pub adaptive_selection: bool,
    pub activity_power_threshold: f32,
    pub prefer_first_two_channels: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Filter {
    pub main: MainConfiguration,
    pub shadow: ShadowConfiguration,
    pub main_initial: MainConfiguration,
    pub shadow_initial: ShadowConfiguration,
    pub config_change_duration_blocks: usize,
    pub initial_state_seconds: f32,
    pub conservative_initial_phase: bool,
    pub enable_shadow_filter_output_usage: bool,
    pub use_linear_filter: bool,
    pub export_linear_aec_output: bool,
}

impl Default for Filter {
    fn default() -> Self {
        Self {
            main: MainConfiguration {
                length_blocks: 13,
                leakage_converged: 0.00005,
                leakage_diverged: 0.05,
                error_floor: 0.001,
                error_ceil: 2.0,
                noise_gate: 20_075_344.0,
            },
            shadow: ShadowConfiguration {
                length_blocks: 13,
                rate: 0.7,
                noise_gate: 20_075_344.0,
            },
            main_initial: MainConfiguration {
                length_blocks: 12,
                leakage_converged: 0.005,
                leakage_diverged: 0.5,
                error_floor: 0.001,
                error_ceil: 2.0,
                noise_gate: 20_075_344.0,
            },
            shadow_initial: ShadowConfiguration {
                length_blocks: 12,
                rate: 0.9,
                noise_gate: 20_075_344.0,
            },
            config_change_duration_blocks: 250,
            initial_state_seconds: 2.5,
            conservative_initial_phase: false,
            enable_shadow_filter_output_usage: true,
            use_linear_filter: true,
            export_linear_aec_output: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MainConfiguration {
    pub length_blocks: usize,
    pub leakage_converged: f32,
    pub leakage_diverged: f32,
    pub error_floor: f32,
    pub error_ceil: f32,
    pub noise_gate: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ShadowConfiguration {
    pub length_blocks: usize,
    pub rate: f32,
    pub noise_gate: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Erle {
    pub min: f32,
    pub max_l: f32,
    pub max_h: f32,
    pub onset_detection: bool,
    pub num_sections: usize,
    pub clamp_quality_estimate_to_zero: bool,
    pub clamp_quality_estimate_to_one: bool,
}

impl Default for Erle {
    fn default() -> Self {
        Self {
            min: 1.0,
            max_l: 4.0,
            max_h: 1.5,
            onset_detection: true,
            num_sections: 1,
            clamp_quality_estimate_to_zero: true,
            clamp_quality_estimate_to_one: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EpStrength {
    pub default_gain: f32,
    pub default_len: f32,
    pub echo_can_saturate: bool,
    pub bounded_erl: bool,
}

impl Default for EpStrength {
    fn default() -> Self {
        Self {
            default_gain: 1.0,
            default_len: 0.83,
            echo_can_saturate: true,
            bounded_erl: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EchoAudibility {
    pub low_render_limit: f32,
    pub normal_render_limit: f32,
    pub floor_power: f32,
    pub audibility_threshold_lf: f32,
    pub audibility_threshold_mf: f32,
    pub audibility_threshold_hf: f32,
    pub use_stationarity_properties: bool,
    pub use_stationarity_properties_at_init: bool,
}

impl Default for EchoAudibility {
    fn default() -> Self {
        Self {
            low_render_limit: 4.0 * 64.0,
            normal_render_limit: 64.0,
            floor_power: 2.0 * 64.0,
            audibility_threshold_lf: 10.0,
            audibility_threshold_mf: 10.0,
            audibility_threshold_hf: 10.0,
            use_stationarity_properties: false,
            use_stationarity_properties_at_init: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RenderLevels {
    pub active_render_limit: f32,
    pub poor_excitation_render_limit: f32,
    pub poor_excitation_render_limit_ds8: f32,
    pub render_power_gain_db: f32,
}

impl Default for RenderLevels {
    fn default() -> Self {
        Self {
            active_render_limit: 100.0,
            poor_excitation_render_limit: 150.0,
            poor_excitation_render_limit_ds8: 20.0,
            render_power_gain_db: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EchoRemovalControl {
    pub has_clock_drift: bool,
    pub linear_and_stable_echo_path: bool,
}

impl Default for EchoRemovalControl {
    fn default() -> Self {
        Self {
            has_clock_drift: false,
            linear_and_stable_echo_path: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EchoModel {
    pub noise_floor_hold: usize,
    pub min_noise_floor_power: f32,
    pub stationary_gate_slope: f32,
    pub noise_gate_power: f32,
    pub noise_gate_slope: f32,
    pub render_pre_window_size: usize,
    pub render_post_window_size: usize,
}

impl Default for EchoModel {
    fn default() -> Self {
        Self {
            noise_floor_hold: 50,
            min_noise_floor_power: 1_638_400.0,
            stationary_gate_slope: 10.0,
            noise_gate_power: 27_509.42,
            noise_gate_slope: 0.3,
            render_pre_window_size: 1,
            render_post_window_size: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Suppressor {
    pub nearend_average_blocks: usize,
    pub normal_tuning: Tuning,
    pub nearend_tuning: Tuning,
    pub dominant_nearend_detection: DominantNearendDetection,
    pub subband_nearend_detection: SubbandNearendDetection,
    pub use_subband_nearend_detection: bool,
    pub high_bands_suppression: HighBandsSuppression,
    pub floor_first_increase: f32,
}

impl Default for Suppressor {
    fn default() -> Self {
        Self {
            nearend_average_blocks: 4,
            normal_tuning: Tuning::new(
                MaskingThresholds::new(0.3, 0.4, 0.3),
                MaskingThresholds::new(0.07, 0.1, 0.3),
                2.0,
                0.25,
            ),
            nearend_tuning: Tuning::new(
                MaskingThresholds::new(1.09, 1.1, 0.3),
                MaskingThresholds::new(0.1, 0.3, 0.3),
                2.0,
                0.25,
            ),
            dominant_nearend_detection: DominantNearendDetection {
                enr_threshold: 0.25,
                enr_exit_threshold: 10.0,
                snr_threshold: 30.0,
                hold_duration: 50,
                trigger_threshold: 12,
                use_during_initial_phase: true,
            },
            subband_nearend_detection: SubbandNearendDetection {
                nearend_average_blocks: 1,
                subband1: SubbandRegion { low: 1, high: 1 },
                subband2: SubbandRegion { low: 1, high: 1 },
                nearend_threshold: 1.0,
                snr_threshold: 1.0,
            },
            use_subband_nearend_detection: false,
            high_bands_suppression: HighBandsSuppression {
                enr_threshold: 1.0,
                max_gain_during_echo: 1.0,
                anti_howling_activation_threshold: 25.0,
                anti_howling_gain: 0.01,
            },
            floor_first_increase: 0.00001,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MaskingThresholds {
    pub enr_transparent: f32,
    pub enr_suppress: f32,
    pub emr_transparent: f32,
}

impl MaskingThresholds {
    pub const fn new(enr_transparent: f32, enr_suppress: f32, emr_transparent: f32) -> Self {
        Self {
            enr_transparent,
            enr_suppress,
            emr_transparent,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tuning {
    pub mask_lf: MaskingThresholds,
    pub mask_hf: MaskingThresholds,
    pub max_inc_factor: f32,
    pub max_dec_factor_lf: f32,
}

impl Tuning {
    pub const fn new(
        mask_lf: MaskingThresholds,
        mask_hf: MaskingThresholds,
        max_inc_factor: f32,
        max_dec_factor_lf: f32,
    ) -> Self {
        Self {
            mask_lf,
            mask_hf,
            max_inc_factor,
            max_dec_factor_lf,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DominantNearendDetection {
    pub enr_threshold: f32,
    pub enr_exit_threshold: f32,
    pub snr_threshold: f32,
    pub hold_duration: usize,
    pub trigger_threshold: usize,
    pub use_during_initial_phase: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SubbandNearendDetection {
    pub nearend_average_blocks: usize,
    pub subband1: SubbandRegion,
    pub subband2: SubbandRegion,
    pub nearend_threshold: f32,
    pub snr_threshold: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SubbandRegion {
    pub low: usize,
    pub high: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HighBandsSuppression {
    pub enr_threshold: f32,
    pub max_gain_during_echo: f32,
    pub anti_howling_activation_threshold: f32,
    pub anti_howling_gain: f32,
}

fn limit_f32(value: &mut f32, min_value: f32, max_value: f32) -> bool {
    let clamped = (*value).max(min_value).min(max_value);
    let clamped = if clamped.is_finite() {
        clamped
    } else {
        min_value
    };
    let unchanged = *value == clamped;
    *value = clamped;
    unchanged
}

fn limit_usize(value: &mut usize, min_value: usize, max_value: usize) -> bool {
    let clamped = min(max(*value, min_value), max_value);
    let unchanged = *value == clamped;
    *value = clamped;
    unchanged
}

fn floor_limit_usize(value: &mut usize, min_value: usize) -> bool {
    if *value < min_value {
        *value = min_value;
        false
    } else {
        true
    }
}

fn limit_i32(value: &mut i32, min_value: i32, max_value: i32) -> bool {
    let clamped = (*value).max(min_value).min(max_value);
    let unchanged = *value == clamped;
    *value = clamped;
    unchanged
}
