//! Shared constants for WebRTC AGC2 RNN-VAD.

pub const PI: f64 = std::f64::consts::PI;

pub const SAMPLE_RATE_24_KHZ: usize = 24_000;
pub const FRAME_SIZE_10_MS_24_KHZ: usize = SAMPLE_RATE_24_KHZ / 100;
pub const FRAME_SIZE_20_MS_24_KHZ: usize = FRAME_SIZE_10_MS_24_KHZ * 2;

// Pitch buffer.
pub const MIN_PITCH_24_KHZ: usize = SAMPLE_RATE_24_KHZ / 800; // 0.00125 s
pub const MAX_PITCH_24_KHZ: usize = SAMPLE_RATE_24_KHZ / 125 * 2; // 0.016 s
pub const BUF_SIZE_24_KHZ: usize = MAX_PITCH_24_KHZ + FRAME_SIZE_20_MS_24_KHZ;

// 24 kHz analysis.
pub const INITIAL_MIN_PITCH_24_KHZ: usize = 3 * MIN_PITCH_24_KHZ;
pub const INITIAL_NUM_LAGS_24_KHZ: usize = MAX_PITCH_24_KHZ - INITIAL_MIN_PITCH_24_KHZ;
pub const REFINE_NUM_LAGS_24_KHZ: usize = MAX_PITCH_24_KHZ + 1;

// 12 kHz analysis.
pub const SAMPLE_RATE_12_KHZ: usize = 12_000;
pub const FRAME_SIZE_10_MS_12_KHZ: usize = SAMPLE_RATE_12_KHZ / 100;
pub const FRAME_SIZE_20_MS_12_KHZ: usize = FRAME_SIZE_10_MS_12_KHZ * 2;
pub const BUF_SIZE_12_KHZ: usize = BUF_SIZE_24_KHZ / 2;
pub const INITIAL_MIN_PITCH_12_KHZ: usize = INITIAL_MIN_PITCH_24_KHZ / 2;
pub const MAX_PITCH_12_KHZ: usize = MAX_PITCH_24_KHZ / 2;
pub const NUM_LAGS_12_KHZ: usize = MAX_PITCH_12_KHZ - INITIAL_MIN_PITCH_12_KHZ;

// 48 kHz constants.
pub const MIN_PITCH_48_KHZ: usize = MIN_PITCH_24_KHZ * 2;
pub const MAX_PITCH_48_KHZ: usize = MAX_PITCH_24_KHZ * 2;

// Spectral features.
pub const NUM_BANDS: usize = 22;
pub const NUM_LOWER_BANDS: usize = 6;
pub const CEPSTRAL_COEFFS_HISTORY_SIZE: usize = 8;

pub const FEATURE_VECTOR_SIZE: usize = 42;

const _: () = {
    assert!(BUF_SIZE_24_KHZ % 2 == 0, "The buffer size must be even.");
    assert!(MIN_PITCH_24_KHZ < INITIAL_MIN_PITCH_24_KHZ);
    assert!(INITIAL_MIN_PITCH_24_KHZ < MAX_PITCH_24_KHZ);
    assert!(MAX_PITCH_24_KHZ > INITIAL_MIN_PITCH_24_KHZ);
    assert!(REFINE_NUM_LAGS_24_KHZ > INITIAL_NUM_LAGS_24_KHZ);
    assert!(MAX_PITCH_12_KHZ > INITIAL_MIN_PITCH_12_KHZ);
    assert!(0 < NUM_LOWER_BANDS && NUM_LOWER_BANDS < NUM_BANDS);
    assert!(CEPSTRAL_COEFFS_HISTORY_SIZE > 2);
};
