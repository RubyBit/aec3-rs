//! Internal spectral-feature primitives for RNN-VAD.

use crate::audio_processing::agc2::rnn_vad::common::{
    FRAME_SIZE_20_MS_24_KHZ, NUM_BANDS, PI,
};

/// At 24 kHz, the last 3 Opus bands are above Nyquist.
pub const OPUS_BANDS_24_KHZ: usize = 20;

const _: () = {
    assert!(OPUS_BANDS_24_KHZ < NUM_BANDS);
};

/// Number of FFT frequency bins covered by each Opus band at 24 kHz for 20 ms.
pub const fn get_opus_scale_num_bins_24khz_20ms() -> [usize; OPUS_BANDS_24_KHZ - 1] {
    [4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 16, 16, 16, 24, 24, 32, 48]
}

const OPUS_BAND_WEIGHTS_24_KHZ_20_MS: [f32; FRAME_SIZE_20_MS_24_KHZ / 2] = [
    // Band 0..7 (8 bands x 4 bins)
    0.0, 0.25, 0.5, 0.75, 0.0, 0.25, 0.5, 0.75, 0.0, 0.25, 0.5, 0.75, 0.0, 0.25, 0.5, 0.75,
    0.0, 0.25, 0.5, 0.75, 0.0, 0.25, 0.5, 0.75, 0.0, 0.25, 0.5, 0.75, 0.0, 0.25, 0.5, 0.75,
    // Band 8..11 (4 bands x 8 bins)
    0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.0, 0.125, 0.25, 0.375, 0.5, 0.625,
    0.75, 0.875, 0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.0, 0.125, 0.25, 0.375,
    0.5, 0.625, 0.75, 0.875,
    // Band 12..14 (3 bands x 16 bins)
    0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875,
    0.75, 0.8125, 0.875, 0.9375, 0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375,
    0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 0.0, 0.0625, 0.125, 0.1875,
    0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375,
    // Band 15..16 (2 bands x 24 bins)
    0.0, 0.0416667, 0.0833333, 0.125, 0.166667, 0.208333, 0.25, 0.291667, 0.333333, 0.375,
    0.416667, 0.458333, 0.5, 0.541667, 0.583333, 0.625, 0.666667, 0.708333, 0.75, 0.791667,
    0.833333, 0.875, 0.916667, 0.958333, 0.0, 0.0416667, 0.0833333, 0.125, 0.166667,
    0.208333, 0.25, 0.291667, 0.333333, 0.375, 0.416667, 0.458333, 0.5, 0.541667, 0.583333,
    0.625, 0.666667, 0.708333, 0.75, 0.791667, 0.833333, 0.875, 0.916667, 0.958333,
    // Band 17 (32 bins)
    0.0, 0.03125, 0.0625, 0.09375, 0.125, 0.15625, 0.1875, 0.21875, 0.25, 0.28125, 0.3125,
    0.34375, 0.375, 0.40625, 0.4375, 0.46875, 0.5, 0.53125, 0.5625, 0.59375, 0.625, 0.65625,
    0.6875, 0.71875, 0.75, 0.78125, 0.8125, 0.84375, 0.875, 0.90625, 0.9375, 0.96875,
    // Band 18 (48 bins)
    0.0, 0.0208333, 0.0416667, 0.0625, 0.0833333, 0.104167, 0.125, 0.145833, 0.166667, 0.1875,
    0.208333, 0.229167, 0.25, 0.270833, 0.291667, 0.3125, 0.333333, 0.354167, 0.375, 0.395833,
    0.416667, 0.4375, 0.458333, 0.479167, 0.5, 0.520833, 0.541667, 0.5625, 0.583333, 0.604167,
    0.625, 0.645833, 0.666667, 0.6875, 0.708333, 0.729167, 0.75, 0.770833, 0.791667, 0.8125,
    0.833333, 0.854167, 0.875, 0.895833, 0.916667, 0.9375, 0.958333, 0.979167,
];

/// Computes band-wise spectral correlations in the Opus perceptual scale.
pub struct SpectralCorrelator {
    weights: [f32; FRAME_SIZE_20_MS_24_KHZ / 2],
}

impl Default for SpectralCorrelator {
    fn default() -> Self {
        Self::new()
    }
}

impl SpectralCorrelator {
    pub fn new() -> Self {
        Self {
            weights: OPUS_BAND_WEIGHTS_24_KHZ_20_MS,
        }
    }

    pub fn compute_auto_correlation(
        &self,
        x: &[f32],
        auto_corr: &mut [f32; OPUS_BANDS_24_KHZ],
    ) {
        self.compute_cross_correlation(x, x, auto_corr);
    }

    pub fn compute_cross_correlation(
        &self,
        x: &[f32],
        y: &[f32],
        cross_corr: &mut [f32; OPUS_BANDS_24_KHZ],
    ) {
        assert_eq!(x.len(), FRAME_SIZE_20_MS_24_KHZ);
        assert_eq!(x.len(), y.len());
        assert_eq!(x[1], 0.0, "Nyquist coefficient must be zeroed");
        assert_eq!(y[1], 0.0, "Nyquist coefficient must be zeroed");

        let bins = get_opus_scale_num_bins_24khz_20ms();
        let mut k = 0usize;
        cross_corr[0] = 0.0;
        for i in 0..(OPUS_BANDS_24_KHZ - 1) {
            cross_corr[i + 1] = 0.0;
            for _ in 0..bins[i] {
                let v = x[2 * k] * y[2 * k] + x[2 * k + 1] * y[2 * k + 1];
                let tmp = self.weights[k] * v;
                cross_corr[i] += v - tmp;
                cross_corr[i + 1] += tmp;
                k += 1;
            }
        }
        cross_corr[0] *= 2.0;
        assert_eq!(k, FRAME_SIZE_20_MS_24_KHZ / 2);
    }
}

/// Computes smoothed log-magnitude spectrum from band energies.
pub fn compute_smoothed_log_magnitude_spectrum(
    bands_energy: &[f32],
    log_bands_energy: &mut [f32; NUM_BANDS],
) {
    assert!(bands_energy.len() <= NUM_BANDS);

    const ONE_BY_HUNDRED: f32 = 1e-2;
    const LOG_ONE_BY_HUNDRED: f32 = -2.0;

    let mut log_max = LOG_ONE_BY_HUNDRED;
    let mut follow = LOG_ONE_BY_HUNDRED;

    let mut smooth = |x: f32| -> f32 {
        let x = (log_max - 7.0).max((follow - 1.5).max(x));
        log_max = log_max.max(x);
        follow = (follow - 1.5).max(x);
        x
    };

    for i in 0..bands_energy.len() {
        log_bands_energy[i] = smooth((ONE_BY_HUNDRED + bands_energy[i]).log10());
    }
    for item in log_bands_energy.iter_mut().take(NUM_BANDS).skip(bands_energy.len()) {
        *item = smooth(LOG_ONE_BY_HUNDRED);
    }
}

/// Computes DCT table for vectors of size `NUM_BANDS`.
pub fn compute_dct_table() -> [f32; NUM_BANDS * NUM_BANDS] {
    let mut dct_table = [0.0f32; NUM_BANDS * NUM_BANDS];
    let k = 0.5f64.sqrt();
    for i in 0..NUM_BANDS {
        for j in 0..NUM_BANDS {
            dct_table[i * NUM_BANDS + j] = ((i as f64 + 0.5) * j as f64 * PI / NUM_BANDS as f64)
                .cos() as f32;
        }
        dct_table[i * NUM_BANDS] *= k as f32;
    }
    dct_table
}

/// Computes DCT for `input` using precomputed table.
/// In-place operation is not supported.
pub fn compute_dct(input: &[f32], dct_table: &[f32; NUM_BANDS * NUM_BANDS], out: &mut [f32]) {
    const DCT_SCALING_FACTOR: f32 = 0.301_511_35; // sqrt(2 / NUM_BANDS)

    assert!(input.len() <= NUM_BANDS);
    assert!(!out.is_empty());
    assert!(out.len() <= input.len());

    for i in 0..out.len() {
        out[i] = 0.0;
        for j in 0..input.len() {
            out[i] += input[j] * dct_table[j * NUM_BANDS + i];
        }
        out[i] *= DCT_SCALING_FACTOR;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_processing::agc2::rnn_vad::common::{SAMPLE_RATE_24_KHZ, FRAME_SIZE_20_MS_24_KHZ};

    #[test]
    fn test_opus_scale_boundaries() {
        let band_frequency_boundaries_hz = [
            200, 400, 600, 800, 1000, 1200, 1400, 1600, 2000, 2400, 2800, 3200, 4000, 4800,
            5600, 6800, 8000, 9600, 12000, 15600, 20000,
        ];
        let bins = get_opus_scale_num_bins_24khz_20ms();

        let mut prev = 0usize;
        for i in 0..bins.len() {
            let boundary = (band_frequency_boundaries_hz[i] as usize) * FRAME_SIZE_20_MS_24_KHZ
                / SAMPLE_RATE_24_KHZ;
            assert_eq!(bins[i], boundary - prev);
            prev = boundary;
        }
    }

    #[test]
    fn spectral_correlator_valid_output() {
        let mut input = vec![1.0f32; FRAME_SIZE_20_MS_24_KHZ];
        input[1] = 0.0; // Nyquist frequency.

        let mut out = [0.0f32; OPUS_BANDS_24_KHZ];
        let c = SpectralCorrelator::new();
        c.compute_auto_correlation(&input, &mut out);

        for &v in &out {
            assert!(v > 0.0, "v={v}");
        }
    }

    #[test]
    fn compute_smoothed_log_magnitude_spectrum_within_tolerance() {
        let input = [
            86.060539245605, 275.668334960938, 43.4065284729, 6.541896820068, 17.964015960693,
            8.090919494629, 1.26192009449, 1.21270263195, 1.619154453278, 0.508935272694,
            0.346316039562, 0.237035423517, 0.172424271703, 0.271657168865, 0.126088857651,
            0.139967113733, 0.207200810313, 0.155893072486, 0.091090843081, 0.033391401172,
            0.013879744336, 0.011973354965,
        ];
        let expected = [
            1.934854507446,
            2.440402746201,
            1.637655138969,
            0.816367030144,
            1.254645109177,
            0.908534288406,
            0.104459829628,
            0.087320849299,
            0.211962252855,
            -0.284886807203,
            -0.448164641857,
            -0.607240796089,
            -0.738917350769,
            -0.550279200077,
            -0.86617743969,
            -0.824003994465,
            -0.663138568401,
            -0.780171751976,
            -0.995288193226,
            -1.362596273422,
            -1.621970295906,
            -1.658103585243,
        ];

        let mut computed = [0.0f32; NUM_BANDS];
        compute_smoothed_log_magnitude_spectrum(&input, &mut computed);

        for (e, c) in expected.iter().zip(computed.iter()) {
            assert!((e - c).abs() < 1e-5, "e={e}, c={c}");
        }
    }

    #[test]
    fn compute_dct_within_tolerance() {
        let input = [
            0.232155621052,
            0.678957760334,
            0.220818966627,
            -0.077363930643,
            -0.559227049351,
            0.432545185089,
            0.353900641203,
            0.398993015289,
            0.409774333239,
            0.45497789979,
            0.300520688295,
            -0.010286616161,
            0.272525429726,
            0.098067551851,
            0.083649002016,
            0.04622688517,
            -0.033228103071,
            0.144773483276,
            -0.117661058903,
            -0.00562880002,
            -0.00954768993,
            -0.045382082462,
        ];
        let expected = [
            0.697072803974,
            0.442710995674,
            -0.293156713247,
            -0.060711503029,
            0.292050391436,
            0.489301353693,
            0.402255415916,
            0.134404733777,
            -0.086305990815,
            -0.199605688453,
            -0.234511867166,
            -0.413774639368,
            -0.388507157564,
            -0.032798115164,
            0.0446055457,
            0.112466648221,
            -0.050096966326,
            0.045971218497,
            -0.029815061018,
            -0.410366982222,
            -0.209233760834,
            -0.128037497401,
        ];

        let table = compute_dct_table();
        let mut computed = [0.0f32; NUM_BANDS];
        compute_dct(&input, &table, &mut computed);

        for (e, c) in expected.iter().zip(computed.iter()) {
            assert!((e - c).abs() < 1e-5, "e={e}, c={c}");
        }
    }
}
