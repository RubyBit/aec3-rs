//! Internal pitch-search primitives for RNN-VAD.

use crate::audio_processing::agc2::cpu_features::AvailableCpuFeatures;
use crate::audio_processing::agc2::rnn_vad::common::{
    BUF_SIZE_12_KHZ, BUF_SIZE_24_KHZ, FRAME_SIZE_20_MS_12_KHZ, FRAME_SIZE_20_MS_24_KHZ,
    INITIAL_NUM_LAGS_24_KHZ, MAX_PITCH_24_KHZ, MAX_PITCH_48_KHZ, MIN_PITCH_24_KHZ,
    MIN_PITCH_48_KHZ, NUM_LAGS_12_KHZ, REFINE_NUM_LAGS_24_KHZ,
};
use crate::audio_processing::agc2::rnn_vad::vector_math::VectorMath;

/// Performs 2x decimation without anti-aliasing filtering.
pub fn decimate_2x(src: &[f32; BUF_SIZE_24_KHZ], dst: &mut [f32; BUF_SIZE_12_KHZ]) {
    for i in 0..BUF_SIZE_12_KHZ {
        dst[i] = src[2 * i];
    }
}

/// Top-2 pitch-period candidates (inverted lag units).
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct CandidatePitchPeriods {
    pub best: i32,
    pub second_best: i32,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct PitchInfo {
    pub period: i32,
    pub strength: f32,
}

#[derive(Debug, Copy, Clone)]
struct Range {
    min: usize,
    max: usize,
}

const PITCH_NEIGHBORHOOD_RADIUS: usize = 2;
const SUB_HARMONIC_MULTIPLIERS: [i32; 14] = [3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5, 2, 3, 2];
const INITIAL_PITCH_PERIOD_THRESHOLDS: [i32; 14] = [
    20, 45, 80, 125, 180, 245, 320, 405, 500, 605, 720, 845, 980, 1125,
];

const _: () = {
    assert!(SUB_HARMONIC_MULTIPLIERS.len() == INITIAL_PITCH_PERIOD_THRESHOLDS.len());
};

fn compute_auto_correlation_24k(
    inverted_lag: usize,
    pitch_buffer: &[f32; BUF_SIZE_24_KHZ],
    vector_math: &VectorMath,
) -> f32 {
    debug_assert!(inverted_lag < BUF_SIZE_24_KHZ);
    debug_assert!(inverted_lag < REFINE_NUM_LAGS_24_KHZ);
    let x = &pitch_buffer[MAX_PITCH_24_KHZ..(MAX_PITCH_24_KHZ + FRAME_SIZE_20_MS_24_KHZ)];
    let y = &pitch_buffer[inverted_lag..(inverted_lag + FRAME_SIZE_20_MS_24_KHZ)];
    vector_math.dot_product(x, y)
}

fn get_pitch_pseudo_interpolation_offset(prev: f32, curr: f32, next: f32) -> i32 {
    if (next - prev) > 0.7 * (curr - prev) {
        1
    } else if (prev - next) > 0.7 * (curr - next) {
        -1
    } else {
        0
    }
}

fn pitch_pseudo_interpolation_lag_pitch_buf(
    lag: i32,
    pitch_buffer: &[f32; BUF_SIZE_24_KHZ],
    vector_math: &VectorMath,
) -> i32 {
    let mut offset = 0;
    if lag > 0 && (lag as usize) < MAX_PITCH_24_KHZ {
        let inverted_lag = MAX_PITCH_24_KHZ as i32 - lag;
        let inv = inverted_lag as usize;
        offset = get_pitch_pseudo_interpolation_offset(
            compute_auto_correlation_24k(inv + 1, pitch_buffer, vector_math),
            compute_auto_correlation_24k(inv, pitch_buffer, vector_math),
            compute_auto_correlation_24k(inv - 1, pitch_buffer, vector_math),
        );
    }
    2 * lag + offset
}

fn create_inverted_lag_range(inverted_lag: usize) -> Range {
    Range {
        min: inverted_lag.saturating_sub(PITCH_NEIGHBORHOOD_RADIUS),
        max: (inverted_lag + PITCH_NEIGHBORHOOD_RADIUS).min(INITIAL_NUM_LAGS_24_KHZ - 1),
    }
}

fn compute_auto_correlation_range(
    inverted_lags: Range,
    pitch_buffer: &[f32; BUF_SIZE_24_KHZ],
    auto_correlation: &mut [f32; INITIAL_NUM_LAGS_24_KHZ],
    inverted_lags_index: &mut Vec<usize>,
    vector_math: &VectorMath,
) {
    if inverted_lags.min > 0 {
        auto_correlation[inverted_lags.min - 1] = 0.0;
    }
    if inverted_lags.max < INITIAL_NUM_LAGS_24_KHZ - 1 {
        auto_correlation[inverted_lags.max + 1] = 0.0;
    }

    for inverted_lag in inverted_lags.min..=inverted_lags.max {
        auto_correlation[inverted_lag] =
            compute_auto_correlation_24k(inverted_lag, pitch_buffer, vector_math);
        inverted_lags_index.push(inverted_lag);
    }
}

fn compute_pitch_period_48k_from_candidates(
    inverted_lags: &[usize],
    auto_correlation: &[f32; INITIAL_NUM_LAGS_24_KHZ],
    y_energy: &[f32; REFINE_NUM_LAGS_24_KHZ],
) -> i32 {
    let mut best_inverted_lag = 0usize;
    let mut best_numerator = -1.0f32;
    let mut best_denominator = 0.0f32;

    for &inverted_lag in inverted_lags {
        if auto_correlation[inverted_lag] > 0.0 {
            let numerator = auto_correlation[inverted_lag] * auto_correlation[inverted_lag];
            let denominator = y_energy[inverted_lag];
            if numerator * best_denominator > best_numerator * denominator {
                best_inverted_lag = inverted_lag;
                best_numerator = numerator;
                best_denominator = denominator;
            }
        }
    }

    if best_inverted_lag == 0 || best_inverted_lag >= INITIAL_NUM_LAGS_24_KHZ - 1 {
        return (best_inverted_lag as i32) * 2;
    }

    let offset = get_pitch_pseudo_interpolation_offset(
        auto_correlation[best_inverted_lag + 1],
        auto_correlation[best_inverted_lag],
        auto_correlation[best_inverted_lag - 1],
    );

    2 * best_inverted_lag as i32 + offset
}

fn get_alternative_pitch_period(pitch_period: i32, multiplier: i32, divisor: i32) -> i32 {
    (2 * multiplier * pitch_period + divisor) / (2 * divisor)
}

fn is_alternative_pitch_stronger_than_initial(
    last: PitchInfo,
    initial: PitchInfo,
    alternative: PitchInfo,
    period_divisor: i32,
) -> bool {
    let mut lower_threshold_term = 0.0f32;
    if (alternative.period - last.period).abs() <= 1 {
        lower_threshold_term = last.strength;
    } else if (alternative.period - last.period).abs() == 2
        && initial.period > INITIAL_PITCH_PERIOD_THRESHOLDS[(period_divisor - 2) as usize]
    {
        lower_threshold_term = 0.5 * last.strength;
    }

    let mut threshold = (0.7 * initial.strength - lower_threshold_term).max(0.3);
    if alternative.period < 3 * MIN_PITCH_24_KHZ as i32 {
        threshold = (0.85 * initial.strength - lower_threshold_term).max(0.4);
    } else if alternative.period < 2 * MIN_PITCH_24_KHZ as i32 {
        threshold = (0.9 * initial.strength - lower_threshold_term).max(0.5);
    }

    alternative.strength > threshold
}

/// Computes sum of squared samples for every sliding frame `y` in the pitch
/// buffer. `y_energy` indexes are inverted lags.
pub fn compute_sliding_frame_square_energies_24khz(
    pitch_buffer: &[f32; BUF_SIZE_24_KHZ],
    y_energy: &mut [f32; REFINE_NUM_LAGS_24_KHZ],
    cpu_features: AvailableCpuFeatures,
) {
    let vector_math = VectorMath::new(cpu_features);
    let frame_20ms = &pitch_buffer[0..FRAME_SIZE_20_MS_24_KHZ];
    let mut yy = vector_math.dot_product(frame_20ms, frame_20ms);
    y_energy[0] = yy;

    for inverted_lag in 0..MAX_PITCH_24_KHZ {
        yy -= pitch_buffer[inverted_lag] * pitch_buffer[inverted_lag];
        yy += pitch_buffer[inverted_lag + FRAME_SIZE_20_MS_24_KHZ]
            * pitch_buffer[inverted_lag + FRAME_SIZE_20_MS_24_KHZ];
        yy = yy.max(1.0);
        y_energy[inverted_lag + 1] = yy;
    }
}

/// Computes pitch-period candidates at 12 kHz from decimated pitch buffer and
/// auto-correlation values (inverted-lag indexed).
pub fn compute_pitch_period_12khz(
    pitch_buffer: &[f32; BUF_SIZE_12_KHZ],
    auto_correlation: &[f32; NUM_LAGS_12_KHZ],
    cpu_features: AvailableCpuFeatures,
) -> CandidatePitchPeriods {
    #[derive(Copy, Clone)]
    struct PitchCandidate {
        period_inverted_lag: i32,
        strength_numerator: f32,
        strength_denominator: f32,
    }

    impl PitchCandidate {
        fn has_stronger_pitch_than(&self, b: &PitchCandidate) -> bool {
            self.strength_numerator * b.strength_denominator
                > b.strength_numerator * self.strength_denominator
        }
    }

    let vector_math = VectorMath::new(cpu_features);
    let frame = &pitch_buffer[0..(FRAME_SIZE_20_MS_12_KHZ + 1)];
    let mut denominator = 1.0 + vector_math.dot_product(frame, frame);

    let mut best = PitchCandidate {
        period_inverted_lag: 0,
        strength_numerator: -1.0,
        strength_denominator: 0.0,
    };
    let mut second_best = PitchCandidate {
        period_inverted_lag: 1,
        strength_numerator: -1.0,
        strength_denominator: 0.0,
    };

    for inverted_lag in 0..NUM_LAGS_12_KHZ {
        if auto_correlation[inverted_lag] > 0.0 {
            let candidate = PitchCandidate {
                period_inverted_lag: inverted_lag as i32,
                strength_numerator: auto_correlation[inverted_lag] * auto_correlation[inverted_lag],
                strength_denominator: denominator,
            };
            if candidate.has_stronger_pitch_than(&second_best) {
                if candidate.has_stronger_pitch_than(&best) {
                    second_best = best;
                    best = candidate;
                } else {
                    second_best = candidate;
                }
            }
        }

        let y_old = pitch_buffer[inverted_lag];
        let y_new = pitch_buffer[inverted_lag + FRAME_SIZE_20_MS_12_KHZ];
        denominator -= y_old * y_old;
        denominator += y_new * y_new;
        denominator = denominator.max(0.0);
    }

    CandidatePitchPeriods {
        best: best.period_inverted_lag,
        second_best: second_best.period_inverted_lag,
    }
}

/// Refines pitch period at 48 kHz from 24 kHz buffer and 24 kHz candidates
/// (inverted-lag encoded).
pub fn compute_pitch_period_48khz(
    pitch_buffer: &[f32; BUF_SIZE_24_KHZ],
    y_energy: &[f32; REFINE_NUM_LAGS_24_KHZ],
    pitch_candidates: CandidatePitchPeriods,
    cpu_features: AvailableCpuFeatures,
) -> i32 {
    let mut auto_correlation = [0.0f32; INITIAL_NUM_LAGS_24_KHZ];
    let mut inverted_lags_index = Vec::<usize>::with_capacity(10);

    let best = pitch_candidates.best as usize;
    let second_best = pitch_candidates.second_best as usize;
    let swap = best > second_best;

    let r1 = create_inverted_lag_range(if swap { second_best } else { best });
    let r2 = create_inverted_lag_range(if swap { best } else { second_best });

    let vector_math = VectorMath::new(cpu_features);
    if r1.max + 1 >= r2.min {
        compute_auto_correlation_range(
            Range {
                min: r1.min,
                max: r2.max,
            },
            pitch_buffer,
            &mut auto_correlation,
            &mut inverted_lags_index,
            &vector_math,
        );
    } else {
        compute_auto_correlation_range(
            r1,
            pitch_buffer,
            &mut auto_correlation,
            &mut inverted_lags_index,
            &vector_math,
        );
        compute_auto_correlation_range(
            r2,
            pitch_buffer,
            &mut auto_correlation,
            &mut inverted_lags_index,
            &vector_math,
        );
    }

    compute_pitch_period_48k_from_candidates(&inverted_lags_index, &auto_correlation, y_energy)
}

/// Computes pitch period at 48 kHz searching an extended pitch range.
pub fn compute_extended_pitch_period_48khz(
    pitch_buffer: &[f32; BUF_SIZE_24_KHZ],
    y_energy: &[f32; REFINE_NUM_LAGS_24_KHZ],
    initial_pitch_period_48khz: i32,
    last_pitch_48khz: PitchInfo,
    cpu_features: AvailableCpuFeatures,
) -> PitchInfo {
    debug_assert!(initial_pitch_period_48khz >= MIN_PITCH_48_KHZ as i32);
    debug_assert!(initial_pitch_period_48khz <= MAX_PITCH_48_KHZ as i32);

    #[derive(Copy, Clone)]
    struct RefinedPitchCandidate {
        period: i32,
        strength: f32,
        xy: f32,
        y_energy: f32,
    }

    let x_energy = y_energy[MAX_PITCH_24_KHZ];
    let pitch_strength = |xy: f32, yy: f32| -> f32 { xy / (1.0 + x_energy * yy).sqrt() };
    let vector_math = VectorMath::new(cpu_features);

    let mut best_pitch = RefinedPitchCandidate {
        period: (initial_pitch_period_48khz / 2).min(MAX_PITCH_24_KHZ as i32 - 1),
        strength: 0.0,
        xy: 0.0,
        y_energy: 0.0,
    };
    best_pitch.xy = compute_auto_correlation_24k(
        (MAX_PITCH_24_KHZ as i32 - best_pitch.period) as usize,
        pitch_buffer,
        &vector_math,
    );
    best_pitch.y_energy = y_energy[(MAX_PITCH_24_KHZ as i32 - best_pitch.period) as usize];
    best_pitch.strength = pitch_strength(best_pitch.xy, best_pitch.y_energy);

    let initial_pitch = PitchInfo {
        period: best_pitch.period,
        strength: best_pitch.strength,
    };
    let last_pitch = PitchInfo {
        period: last_pitch_48khz.period / 2,
        strength: last_pitch_48khz.strength,
    };

    let max_period_divisor = (2 * initial_pitch.period) / (2 * MIN_PITCH_24_KHZ as i32 - 1);
    for period_divisor in 2..=max_period_divisor {
        let mut alternative_pitch = PitchInfo {
            period: get_alternative_pitch_period(initial_pitch.period, 1, period_divisor),
            strength: 0.0,
        };

        let mut dual_alternative_period = get_alternative_pitch_period(
            initial_pitch.period,
            SUB_HARMONIC_MULTIPLIERS[(period_divisor - 2) as usize],
            period_divisor,
        );
        if period_divisor == 2 && dual_alternative_period > MAX_PITCH_24_KHZ as i32 {
            dual_alternative_period = initial_pitch.period;
        }

        let xy_primary = compute_auto_correlation_24k(
            (MAX_PITCH_24_KHZ as i32 - alternative_pitch.period) as usize,
            pitch_buffer,
            &vector_math,
        );
        let xy_secondary = compute_auto_correlation_24k(
            (MAX_PITCH_24_KHZ as i32 - dual_alternative_period) as usize,
            pitch_buffer,
            &vector_math,
        );
        let xy = 0.5 * (xy_primary + xy_secondary);
        let yy = 0.5
            * (y_energy[(MAX_PITCH_24_KHZ as i32 - alternative_pitch.period) as usize]
                + y_energy[(MAX_PITCH_24_KHZ as i32 - dual_alternative_period) as usize]);

        alternative_pitch.strength = pitch_strength(xy, yy);

        if is_alternative_pitch_stronger_than_initial(
            last_pitch,
            initial_pitch,
            alternative_pitch,
            period_divisor,
        ) {
            best_pitch = RefinedPitchCandidate {
                period: alternative_pitch.period,
                strength: alternative_pitch.strength,
                xy,
                y_energy: yy,
            };
        }
    }

    best_pitch.xy = best_pitch.xy.max(0.0);
    let mut final_pitch_strength = if best_pitch.y_energy <= best_pitch.xy {
        1.0
    } else {
        best_pitch.xy / (best_pitch.y_energy + 1.0)
    };
    final_pitch_strength = final_pitch_strength.min(best_pitch.strength);

    let final_pitch_period_48khz = (MIN_PITCH_48_KHZ as i32).max(
        pitch_pseudo_interpolation_lag_pitch_buf(best_pitch.period, pitch_buffer, &vector_math),
    );

    PitchInfo {
        period: final_pitch_period_48khz,
        strength: final_pitch_strength,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_processing::agc2::cpu_features::{
        get_available_cpu_features, no_available_cpu_features,
    };
    use crate::audio_processing::agc2::rnn_vad::test_data::{PitchTestData, expect_near_absolute};

    fn cpu_features_to_test() -> Vec<AvailableCpuFeatures> {
        let mut out = vec![no_available_cpu_features()];
        let available = get_available_cpu_features();
        if available.avx2 {
            out.push(AvailableCpuFeatures::new(false, true, false));
        }
        if available.sse2 {
            out.push(AvailableCpuFeatures::new(true, false, false));
        }
        if available.neon {
            out.push(AvailableCpuFeatures::new(false, false, true));
        }
        out
    }

    fn make_synthetic_pitch_buffer() -> [f32; BUF_SIZE_24_KHZ] {
        let mut pitch = [0.0f32; BUF_SIZE_24_KHZ];
        for (i, v) in pitch.iter_mut().enumerate() {
            *v = (0.021 * i as f32).sin() + 0.31 * (0.046 * i as f32).cos();
        }
        pitch
    }

    #[test]
    fn decimate_2x_keeps_even_samples() {
        let mut src = [0.0f32; BUF_SIZE_24_KHZ];
        for (i, v) in src.iter_mut().enumerate() {
            *v = i as f32;
        }
        let mut dst = [0.0f32; BUF_SIZE_12_KHZ];
        decimate_2x(&src, &mut dst);

        for i in 0..BUF_SIZE_12_KHZ {
            assert_eq!(src[2 * i], dst[i]);
        }
    }

    #[test]
    fn compute_sliding_frame_square_energies_constant_signal() {
        let src = [1.0f32; BUF_SIZE_24_KHZ];
        let mut y_energy = [0.0f32; REFINE_NUM_LAGS_24_KHZ];
        compute_sliding_frame_square_energies_24khz(
            &src,
            &mut y_energy,
            no_available_cpu_features(),
        );

        let expected = FRAME_SIZE_20_MS_24_KHZ as f32;
        for &v in &y_energy {
            assert!((v - expected).abs() < 1e-4, "v={v}, expected={expected}");
        }
    }

    #[test]
    fn compute_sliding_frame_square_energies_24khz_within_tolerance_fixture() {
        let cpu = get_available_cpu_features();
        let test_data = PitchTestData::new();
        let mut computed_output = [0.0f32; REFINE_NUM_LAGS_24_KHZ];
        compute_sliding_frame_square_energies_24khz(
            test_data.pitch_buffer_24khz_view(),
            &mut computed_output,
            cpu,
        );

        expect_near_absolute(
            test_data.square_energies_24khz_view(),
            &computed_output,
            1e-3,
        );
    }

    #[test]
    fn compute_pitch_period_48khz_order_does_not_matter_parametrized() {
        let mut pitch = [0.0f32; BUF_SIZE_24_KHZ];
        for (i, v) in pitch.iter_mut().enumerate() {
            *v = (0.023 * i as f32).sin() + 0.25 * (0.051 * i as f32).cos();
        }
        let mut y_energy = [0.0f32; REFINE_NUM_LAGS_24_KHZ];
        let candidate_sets = [
            CandidatePitchPeriods {
                best: 0,
                second_best: 2,
            },
            CandidatePitchPeriods {
                best: 260,
                second_best: 284,
            },
            CandidatePitchPeriods {
                best: 280,
                second_best: 284,
            },
            CandidatePitchPeriods {
                best: INITIAL_NUM_LAGS_24_KHZ as i32 - 2,
                second_best: INITIAL_NUM_LAGS_24_KHZ as i32 - 1,
            },
        ];

        for cpu in cpu_features_to_test() {
            compute_sliding_frame_square_energies_24khz(&pitch, &mut y_energy, cpu);
            for candidates in candidate_sets {
                let p1 = compute_pitch_period_48khz(&pitch, &y_energy, candidates, cpu);
                let p2 = compute_pitch_period_48khz(
                    &pitch,
                    &y_energy,
                    CandidatePitchPeriods {
                        best: candidates.second_best,
                        second_best: candidates.best,
                    },
                    cpu,
                );
                assert_eq!(
                    p1, p2,
                    "cpu={cpu:?}, candidates=({}, {})",
                    candidates.best, candidates.second_best
                );
            }
        }
    }

    #[test]
    fn compute_extended_pitch_period_48khz_output_is_valid() {
        let pitch = make_synthetic_pitch_buffer();
        let mut y_energy = [0.0f32; REFINE_NUM_LAGS_24_KHZ];
        let cpu = no_available_cpu_features();
        compute_sliding_frame_square_energies_24khz(&pitch, &mut y_energy, cpu);

        let out = compute_extended_pitch_period_48khz(
            &pitch,
            &y_energy,
            560,
            PitchInfo {
                period: 180,
                strength: 0.35,
            },
            cpu,
        );

        assert!(out.period >= MIN_PITCH_48_KHZ as i32);
        assert!(out.period <= MAX_PITCH_48_KHZ as i32);
        assert!(out.strength.is_finite());
    }

    #[test]
    fn compute_extended_pitch_period_48khz_parameter_sweep() {
        let pitch = make_synthetic_pitch_buffer();
        let mut y_energy = [0.0f32; REFINE_NUM_LAGS_24_KHZ];

        let test_pitch_periods_low = 3 * MIN_PITCH_48_KHZ as i32 / 2;
        let test_pitch_periods_high = (3 * MIN_PITCH_48_KHZ as i32 + MAX_PITCH_48_KHZ as i32) / 2;
        let last_periods = [test_pitch_periods_low, test_pitch_periods_high];
        let last_strengths = [0.35f32, 0.75f32];
        let initial_periods = [test_pitch_periods_low, test_pitch_periods_high];

        for cpu in cpu_features_to_test() {
            compute_sliding_frame_square_energies_24khz(&pitch, &mut y_energy, cpu);

            for &initial_pitch_period in &initial_periods {
                for &last_pitch_period in &last_periods {
                    for &last_pitch_strength in &last_strengths {
                        let last = PitchInfo {
                            period: last_pitch_period,
                            strength: last_pitch_strength,
                        };
                        let out1 = compute_extended_pitch_period_48khz(
                            &pitch,
                            &y_energy,
                            initial_pitch_period,
                            last,
                            cpu,
                        );
                        let out2 = compute_extended_pitch_period_48khz(
                            &pitch,
                            &y_energy,
                            initial_pitch_period,
                            last,
                            cpu,
                        );

                        assert_eq!(
                            out1.period, out2.period,
                            "period mismatch, cpu={cpu:?}, initial={initial_pitch_period}, last={last_pitch_period}/{last_pitch_strength}"
                        );
                        assert!(
                            (out1.strength - out2.strength).abs() < 1e-6,
                            "strength mismatch, cpu={cpu:?}, initial={initial_pitch_period}, last={last_pitch_period}/{last_pitch_strength}, out1={}, out2={}",
                            out1.strength,
                            out2.strength
                        );
                        assert!(out1.period >= MIN_PITCH_48_KHZ as i32);
                        assert!(out1.period <= MAX_PITCH_48_KHZ as i32);
                        assert!(out1.strength.is_finite());
                    }
                }
            }
        }
    }

    #[test]
    fn compute_pitch_period_12khz_returns_valid_candidates() {
        let mut pitch_24 = [0.0f32; BUF_SIZE_24_KHZ];
        for (i, v) in pitch_24.iter_mut().enumerate() {
            *v = (0.019 * i as f32).sin() + 0.15 * (0.072 * i as f32).cos();
        }
        let mut pitch_12 = [0.0f32; BUF_SIZE_12_KHZ];
        decimate_2x(&pitch_24, &mut pitch_12);

        let mut auto_corr = [0.0f32; NUM_LAGS_12_KHZ];
        let x_start = BUF_SIZE_12_KHZ - FRAME_SIZE_20_MS_12_KHZ;
        for lag_inv in 0..NUM_LAGS_12_KHZ {
            let mut acc = 0.0f32;
            for n in 0..FRAME_SIZE_20_MS_12_KHZ {
                acc += pitch_12[lag_inv + n] * pitch_12[x_start + n];
            }
            auto_corr[lag_inv] = acc;
        }

        let out = compute_pitch_period_12khz(&pitch_12, &auto_corr, no_available_cpu_features());
        assert!(out.best >= 0 && out.best < NUM_LAGS_12_KHZ as i32);
        assert!(out.second_best >= 0 && out.second_best < NUM_LAGS_12_KHZ as i32);
        assert_ne!(out.best, out.second_best);
    }

    #[test]
    fn compute_pitch_period_12khz_bit_exactness_fixture() {
        let cpu = get_available_cpu_features();
        let test_data = PitchTestData::new();
        let mut pitch_buf_decimated = [0.0f32; BUF_SIZE_12_KHZ];
        decimate_2x(
            test_data.pitch_buffer_24khz_view(),
            &mut pitch_buf_decimated,
        );

        let pitch_candidates = compute_pitch_period_12khz(
            &pitch_buf_decimated,
            test_data.auto_correlation_12khz_view(),
            cpu,
        );
        assert_eq!(140, pitch_candidates.best);
        assert_eq!(142, pitch_candidates.second_best);
    }

    #[test]
    fn compute_pitch_period_48khz_bit_exactness_fixture() {
        let cpu = get_available_cpu_features();
        let test_data = PitchTestData::new();
        let mut y_energy = [0.0f32; REFINE_NUM_LAGS_24_KHZ];
        compute_sliding_frame_square_energies_24khz(
            test_data.pitch_buffer_24khz_view(),
            &mut y_energy,
            cpu,
        );

        assert_eq!(
            560,
            compute_pitch_period_48khz(
                test_data.pitch_buffer_24khz_view(),
                &y_energy,
                CandidatePitchPeriods {
                    best: 280,
                    second_best: 284,
                },
                cpu,
            )
        );
        assert_eq!(
            568,
            compute_pitch_period_48khz(
                test_data.pitch_buffer_24khz_view(),
                &y_energy,
                CandidatePitchPeriods {
                    best: 260,
                    second_best: 284,
                },
                cpu,
            )
        );
    }

    #[test]
    fn compute_extended_pitch_period_48khz_fixture_cases() {
        let test_pitch_periods_low = 3 * MIN_PITCH_48_KHZ as i32 / 2;
        let test_pitch_periods_high = (3 * MIN_PITCH_48_KHZ as i32 + MAX_PITCH_48_KHZ as i32) / 2;
        let test_pitch_strength_low = 0.35f32;
        let test_pitch_strength_high = 0.75f32;

        let cpu_set = [no_available_cpu_features(), get_available_cpu_features()];
        let test_data = PitchTestData::new();

        for &cpu in &cpu_set {
            let mut y_energy = [0.0f32; REFINE_NUM_LAGS_24_KHZ];
            compute_sliding_frame_square_energies_24khz(
                test_data.pitch_buffer_24khz_view(),
                &mut y_energy,
                cpu,
            );

            for &last_pitch_period in &[test_pitch_periods_low, test_pitch_periods_high] {
                for &last_pitch_strength in &[test_pitch_strength_low, test_pitch_strength_high] {
                    let out_low = compute_extended_pitch_period_48khz(
                        test_data.pitch_buffer_24khz_view(),
                        &y_energy,
                        test_pitch_periods_low,
                        PitchInfo {
                            period: last_pitch_period,
                            strength: last_pitch_strength,
                        },
                        cpu,
                    );
                    assert_eq!(91, out_low.period);
                    assert!((out_low.strength - (-0.0188608)).abs() < 1e-6);

                    let out_high = compute_extended_pitch_period_48khz(
                        test_data.pitch_buffer_24khz_view(),
                        &y_energy,
                        test_pitch_periods_high,
                        PitchInfo {
                            period: last_pitch_period,
                            strength: last_pitch_strength,
                        },
                        cpu,
                    );
                    assert_eq!(475, out_high.period);
                    assert!((out_high.strength - (-0.0904344)).abs() < 1e-6);
                }
            }
        }
    }
}
