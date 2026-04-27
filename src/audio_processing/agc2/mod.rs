//! Rust implementation of WebRTC AGC2 components.

pub mod adaptive_digital_gain_controller;
pub mod agc2_common;
pub mod biquad_filter;
pub mod clipping_predictor;
pub mod clipping_predictor_level_buffer;
pub mod cpu_features;
pub mod fixed_digital_level_estimator;
pub mod gain_applier;
pub mod input_volume_controller;
pub mod input_volume_stats_reporter;
pub mod interpolated_gain_curve;
pub mod limiter;
pub mod limiter_db_gain_curve;
pub mod noise_level_estimator;
pub mod rnn_vad;
pub mod saturation_protector;
pub mod saturation_protector_buffer;
pub mod speech_level_estimator;
pub mod speech_probability_buffer;
pub mod vad_wrapper;
pub mod vector_float_frame;
