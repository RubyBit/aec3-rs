//! Shared constants for the NS pipeline.

pub const FFT_SIZE: usize = 256;
pub const FFT_SIZE_BY_2_PLUS_1: usize = FFT_SIZE / 2 + 1;
pub const NS_FRAME_SIZE: usize = 160;
pub const OVERLAP_SIZE: usize = FFT_SIZE - NS_FRAME_SIZE;

pub const SHORT_STARTUP_PHASE_BLOCKS: i32 = 50;
pub const LONG_STARTUP_PHASE_BLOCKS: i32 = 200;
pub const FEATURE_UPDATE_WINDOW_SIZE: i32 = 500;

pub const LRT_FEATURE_THR: f32 = 0.5;
pub const BIN_SIZE_LRT: f32 = 0.1;
pub const BIN_SIZE_SPEC_FLAT: f32 = 0.05;
pub const BIN_SIZE_SPEC_DIFF: f32 = 0.1;
