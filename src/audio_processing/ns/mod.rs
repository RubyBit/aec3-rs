//! Noise suppression (NS) implementation ported from the WebRTC reference
//! module with ergonomic Rust APIs.

pub mod fast_math;
pub mod histograms;
pub mod noise_suppressor;
pub mod noise_estimator;
pub mod ns_common;
pub mod ns_config;
pub mod ns_fft;
pub mod prior_signal_model;
pub mod prior_signal_model_estimator;
pub mod quantile_noise_estimator;
pub mod signal_model;
pub mod signal_model_estimator;
pub mod speech_probability_estimator;
pub mod suppression_params;
pub mod wiener_filter;

pub use noise_suppressor::NoiseSuppressor;
pub use ns_config::{NsConfig, SuppressionLevel};