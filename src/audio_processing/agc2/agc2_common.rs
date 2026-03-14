//! Shared AGC2 constants mirrored from WebRTC `agc2_common.h`.

pub const MIN_FLOAT_S16_VALUE: f32 = -32768.0;
pub const MAX_FLOAT_S16_VALUE: f32 = 32767.0;
pub const MAX_ABS_FLOAT_S16_VALUE: f32 = 32768.0;

/// Minimum audio level in dBFS scale for S16 samples.
pub const MIN_LEVEL_DBFS: f32 = -90.31;

pub const FRAME_DURATION_MS: usize = 10;
pub const SUB_FRAMES_IN_FRAME: usize = 20;
pub const MAXIMAL_NUMBER_OF_SAMPLES_PER_CHANNEL: usize = 480;

// Adaptive digital gain applier settings.

/// At what limiter levels should we start decreasing the adaptive digital gain.
pub const LIMITER_THRESHOLD_FOR_AGC_GAIN_DBFS: f32 = -1.0;

/// Number of milliseconds to wait to periodically reset the VAD.
pub const VAD_RESET_PERIOD_MS: usize = 1500;

/// Speech probability threshold to detect speech activity.
pub const VAD_CONFIDENCE_THRESHOLD: f32 = 0.95;

/// Minimum number of adjacent speech frames with high confidence.
pub const ADJACENT_SPEECH_FRAMES_THRESHOLD: usize = 12;

/// Number of milliseconds of speech frames to observe to make the estimator confident.
pub const LEVEL_ESTIMATOR_TIME_TO_CONFIDENCE_MS: f32 = 400.0;
pub const LEVEL_ESTIMATOR_LEAK_FACTOR: f32 = 1.0 - 1.0 / LEVEL_ESTIMATOR_TIME_TO_CONFIDENCE_MS;

// Saturation protector settings.
pub const SATURATION_PROTECTOR_INITIAL_HEADROOM_DB: f32 = 20.0;
pub const SATURATION_PROTECTOR_BUFFER_SIZE: usize = 4;

// Number of interpolation points for each region of the limiter.
pub const INTERPOLATED_GAIN_CURVE_KNEE_POINTS: usize = 22;
pub const INTERPOLATED_GAIN_CURVE_BEYOND_KNEE_POINTS: usize = 10;
pub const INTERPOLATED_GAIN_CURVE_TOTAL_POINTS: usize =
    INTERPOLATED_GAIN_CURVE_KNEE_POINTS + INTERPOLATED_GAIN_CURVE_BEYOND_KNEE_POINTS;
