#[cfg(target_arch = "aarch64")]
use crate::audio_processing::aec3::aec3_common::detect_neon;
use crate::audio_processing::aec3::aec3_common::{Aec3Optimization, BLOCK_SIZE};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::audio_processing::aec3::aec3_common::{detect_avx2, detect_sse2};
use crate::audio_processing::aec3::downsampled_render_buffer::DownsampledRenderBuffer;
use crate::audio_processing::logging::apm_data_dumper::{ApmDataDumper, DiagnosticLevel};
use std::cmp::Ordering;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vld1q_f32, vmulq_f32, vst1q_f32};
#[cfg(target_arch = "x86")]
use std::arch::x86::{
    _mm_add_ps, _mm_loadu_ps, _mm_mul_ps, _mm_set1_ps, _mm_setzero_ps, _mm_storeu_ps,
    _mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_setzero_ps,
    _mm256_storeu_ps,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _mm_add_ps, _mm_loadu_ps, _mm_mul_ps, _mm_set1_ps, _mm_setzero_ps, _mm_storeu_ps,
    _mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_setzero_ps,
    _mm256_storeu_ps,
};

const SATURATION_LIMIT: f32 = 32_000.0;

/// Subsample rate of the accumulated error. The scalar core depends on this
/// being 4.
const ACCUMULATED_ERROR_SUB_SAMPLE_RATE: usize = 4;

/// Stores properties for the lag estimate corresponding to a particular signal shift.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LagEstimate {
    pub lag: usize,
    /// Lag of the earliest reflection that already explains the capture signal.
    /// Equal to `lag` when pre-echo detection is off or still warming up.
    pub pre_echo_lag: usize,
}

impl LagEstimate {
    pub fn new(lag: usize, pre_echo_lag: usize) -> Self {
        Self { lag, pre_echo_lag }
    }
}

pub struct MatchedFilter {
    data_dumper: ApmDataDumper,
    optimization: Aec3Optimization,
    sub_block_size: usize,
    filter_intra_lag_shift: usize,
    filters: Vec<Vec<f32>>,
    /// Per-filter error of the filter truncated to every
    /// `ACCUMULATED_ERROR_SUB_SAMPLE_RATE` taps. Empty unless pre-echo
    /// detection is enabled.
    accumulated_error: Vec<Vec<f32>>,
    instantaneous_accumulated_error: Vec<f32>,
    /// Linearises the circular render buffer for the accumulated-error cores.
    scratch_memory: Vec<f32>,
    reported_lag_estimate: Option<LagEstimate>,
    winner_lag: Option<usize>,
    last_detected_best_lag_filter: i32,
    number_pre_echo_updates: i32,
    excitation_limit: f32,
    smoothing_fast: f32,
    smoothing_slow: f32,
    matching_filter_threshold: f32,
    detect_pre_echo: bool,
}

impl MatchedFilter {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        data_dumper: ApmDataDumper,
        optimization: Aec3Optimization,
        sub_block_size: usize,
        window_size_sub_blocks: usize,
        num_matched_filters: usize,
        alignment_shift_sub_blocks: usize,
        excitation_limit: f32,
        smoothing_fast: f32,
        smoothing_slow: f32,
        matching_filter_threshold: f32,
        detect_pre_echo: bool,
    ) -> Self {
        assert!(
            window_size_sub_blocks > 0,
            "window_size_sub_blocks must be positive"
        );
        assert!(sub_block_size > 0, "sub_block_size must be positive");
        assert!(
            sub_block_size % 4 == 0,
            "sub_block_size must be divisible by 4"
        );
        assert!(
            BLOCK_SIZE % sub_block_size == 0,
            "sub_block_size must divide BLOCK_SIZE"
        );

        let filter_length = window_size_sub_blocks * sub_block_size;
        let filters = (0..num_matched_filters)
            .map(|_| vec![0.0; filter_length])
            .collect::<Vec<_>>();
        let accumulated_error_length = filter_length / ACCUMULATED_ERROR_SUB_SAMPLE_RATE;
        let (accumulated_error, instantaneous_accumulated_error, scratch_memory) =
            if detect_pre_echo {
                (
                    vec![vec![1.0; accumulated_error_length]; num_matched_filters],
                    vec![0.0; accumulated_error_length],
                    vec![0.0; filter_length],
                )
            } else {
                (Vec::new(), Vec::new(), Vec::new())
            };

        Self {
            data_dumper,
            optimization,
            sub_block_size,
            filter_intra_lag_shift: alignment_shift_sub_blocks * sub_block_size,
            filters,
            accumulated_error,
            instantaneous_accumulated_error,
            scratch_memory,
            reported_lag_estimate: None,
            winner_lag: None,
            last_detected_best_lag_filter: -1,
            number_pre_echo_updates: 0,
            excitation_limit,
            smoothing_fast,
            smoothing_slow,
            matching_filter_threshold,
            detect_pre_echo,
        }
    }

    /// The best lag estimate of the last block, if one was found.
    pub fn best_lag_estimate(&self) -> Option<LagEstimate> {
        self.reported_lag_estimate
    }

    pub fn reset(&mut self, full_reset: bool) {
        for filter in &mut self.filters {
            filter.fill(0.0);
        }
        self.winner_lag = None;
        self.reported_lag_estimate = None;
        if full_reset {
            for error in &mut self.accumulated_error {
                error.fill(1.0);
            }
            self.number_pre_echo_updates = 0;
        }
    }

    pub fn update(
        &mut self,
        render_buffer: &DownsampledRenderBuffer,
        capture: &[f32],
        use_slow_smoothing: bool,
    ) {
        if self.filters.is_empty() {
            return;
        }
        let smoothing = if use_slow_smoothing {
            self.smoothing_slow
        } else {
            self.smoothing_fast
        };
        assert_eq!(self.sub_block_size, capture.len());
        let render_samples = &render_buffer.buffer;
        assert!(!render_samples.is_empty());

        let x2_sum_threshold =
            self.filters[0].len() as f32 * self.excitation_limit * self.excitation_limit;
        let error_sum_anchor = capture.iter().map(|v| v * v).sum::<f32>();
        let buffer_size = render_samples.len();
        let mut alignment_shift = 0usize;

        let mut winner_error_sum = error_sum_anchor;
        self.winner_lag = None;
        self.reported_lag_estimate = None;
        let mut winner_index: i32 = -1;
        let mut previous_lag_estimate: Option<usize> = None;

        for index in 0..self.filters.len() {
            let mut start = render_buffer.read + alignment_shift + self.sub_block_size - 1;
            start %= buffer_size;

            // Only the filter that won the previous block accumulates the
            // truncated-filter error, so the scalar core is enough here.
            let compute_pre_echo =
                self.detect_pre_echo && index as i32 == self.last_detected_best_lag_filter;

            let filter = &mut self.filters[index];
            let result = if compute_pre_echo {
                let accumulated = &mut self.instantaneous_accumulated_error;
                let scratch = &mut self.scratch_memory;
                match self.optimization {
                    Aec3Optimization::Avx2 => matched_filter_core_accumulated_error_avx2(
                        start,
                        x2_sum_threshold,
                        smoothing,
                        render_samples,
                        capture,
                        filter,
                        accumulated,
                        scratch,
                    ),
                    Aec3Optimization::Sse2 => matched_filter_core_accumulated_error_sse2(
                        start,
                        x2_sum_threshold,
                        smoothing,
                        render_samples,
                        capture,
                        filter,
                        accumulated,
                        scratch,
                    ),
                    Aec3Optimization::Neon => matched_filter_core_accumulated_error_neon(
                        start,
                        x2_sum_threshold,
                        smoothing,
                        render_samples,
                        capture,
                        filter,
                        accumulated,
                        scratch,
                    ),
                    Aec3Optimization::None => matched_filter_core_with_accumulated_error(
                        start,
                        x2_sum_threshold,
                        smoothing,
                        render_samples,
                        capture,
                        filter,
                        accumulated,
                    ),
                }
            } else {
                match self.optimization {
                    Aec3Optimization::Avx2 => matched_filter_core_avx2(
                        start,
                        x2_sum_threshold,
                        smoothing,
                        render_samples,
                        capture,
                        filter,
                    ),
                    Aec3Optimization::Sse2 => matched_filter_core_sse2(
                        start,
                        x2_sum_threshold,
                        smoothing,
                        render_samples,
                        capture,
                        filter,
                    ),
                    Aec3Optimization::Neon => matched_filter_core_neon(
                        start,
                        x2_sum_threshold,
                        smoothing,
                        render_samples,
                        capture,
                        filter,
                    ),
                    Aec3Optimization::None => matched_filter_core(
                        start,
                        x2_sum_threshold,
                        smoothing,
                        render_samples,
                        capture,
                        filter,
                    ),
                }
            };

            let peak_index = Self::detect_peak(filter);
            let reliable = peak_index > 2
                && peak_index + 10 < filter.len()
                && result.error_sum < self.matching_filter_threshold * error_sum_anchor;
            let lag = peak_index + alignment_shift;

            if result.filters_updated && reliable && result.error_sum < winner_error_sum {
                winner_error_sum = result.error_sum;
                winner_index = index as i32;
                // When two filters report the same lag (the overlap region),
                // take the earlier one so there is room to search for
                // pre-echoes ahead of it.
                if previous_lag_estimate == Some(lag) {
                    self.winner_lag = previous_lag_estimate;
                    winner_index = index as i32 - 1;
                } else {
                    self.winner_lag = Some(lag);
                }
            }
            previous_lag_estimate = Some(lag);

            self.data_dumper.dump_raw_f32_slice(
                DiagnosticLevel::DeepDebug,
                &format!("aec3_correlator_{}_h", index),
                &self.filters[index],
            );

            alignment_shift += self.filter_intra_lag_shift;
        }

        if winner_index != -1 {
            let winner_lag = self
                .winner_lag
                .expect("a winner lag is set with a winner index");
            let mut estimate = LagEstimate::new(winner_lag, winner_lag);
            if self.detect_pre_echo && self.last_detected_best_lag_filter == winner_index {
                const ENERGY_THRESHOLD: f32 = 1.0;
                if error_sum_anchor > ENERGY_THRESHOLD {
                    update_accumulated_error(
                        &self.instantaneous_accumulated_error,
                        &mut self.accumulated_error[winner_index as usize],
                        1.0 / error_sum_anchor,
                    );
                    self.number_pre_echo_updates += 1;
                }
                if self.number_pre_echo_updates >= 50 {
                    estimate.pre_echo_lag = compute_pre_echo_lag(
                        &self.accumulated_error[winner_index as usize],
                        winner_lag,
                        winner_index as usize * self.filter_intra_lag_shift,
                    );
                }
            }
            self.reported_lag_estimate = Some(estimate);
            self.last_detected_best_lag_filter = winner_index;
        }
    }

    pub fn max_filter_lag(&self) -> usize {
        if self.filters.is_empty() {
            0
        } else {
            self.filters.len() * self.filter_intra_lag_shift + self.filters[0].len()
        }
    }

    pub fn log_filter_properties(
        &self,
        sample_rate_hz: i32,
        shift: usize,
        downsampling_factor: usize,
    ) {
        let mut alignment_shift = 0usize;
        let samples_per_ms = 16; // 16 kHz reference clock.
        for (index, filter) in self.filters.iter().enumerate() {
            let start = (alignment_shift * downsampling_factor) as isize;
            let end = ((alignment_shift + filter.len()) * downsampling_factor) as isize;
            let start_ms = (start - shift as isize) / samples_per_ms;
            let end_ms = (end - shift as isize) / samples_per_ms;
            log::trace!(
                "Matched filter {}: start={} ms, end={} ms @ {} Hz",
                index,
                start_ms,
                end_ms,
                sample_rate_hz
            );
            alignment_shift += self.filter_intra_lag_shift;
        }
    }

    fn detect_peak(filter: &[f32]) -> usize {
        filter
            .iter()
            .enumerate()
            .max_by(|a, b| match a.1.abs().partial_cmp(&b.1.abs()) {
                Some(order) => order,
                None => Ordering::Equal,
            })
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }
}

/// Smooths the truncated-filter error towards the newest observation, dropping
/// immediately when the new value is lower.
fn update_accumulated_error(
    instantaneous_accumulated_error: &[f32],
    accumulated_error: &mut [f32],
    one_over_error_sum_anchor: f32,
) {
    const SMOOTH_CONSTANT_INCREASES: f32 = 0.015;
    for (accumulated, &instantaneous) in accumulated_error
        .iter_mut()
        .zip(instantaneous_accumulated_error.iter())
    {
        let error_norm = instantaneous * one_over_error_sum_anchor;
        if error_norm < *accumulated {
            *accumulated = error_norm;
        } else {
            *accumulated += SMOOTH_CONSTANT_INCREASES * (error_norm - *accumulated);
        }
    }
}

/// Walks back from the winning lag towards the start of the filter and returns
/// the earliest truncation that still explains the capture signal.
fn compute_pre_echo_lag(
    accumulated_error: &[f32],
    lag: usize,
    alignment_shift_winner: usize,
) -> usize {
    const PRE_ECHO_THRESHOLD: f32 = 0.5;
    debug_assert!(lag >= alignment_shift_winner);
    let mut pre_echo_lag_estimate = lag - alignment_shift_winner;
    let maximum_pre_echo_lag =
        (pre_echo_lag_estimate / ACCUMULATED_ERROR_SUB_SAMPLE_RATE).min(accumulated_error.len());
    for k in (0..maximum_pre_echo_lag).rev() {
        if accumulated_error[k] > PRE_ECHO_THRESHOLD {
            break;
        }
        pre_echo_lag_estimate = (k + 1) * ACCUMULATED_ERROR_SUB_SAMPLE_RATE - 1;
    }
    pre_echo_lag_estimate + alignment_shift_winner
}

struct MatchedFilterCoreResult {
    filters_updated: bool,
    error_sum: f32,
}

fn matched_filter_core(
    mut x_start_index: usize,
    x2_sum_threshold: f32,
    smoothing: f32,
    x: &[f32],
    y: &[f32],
    h: &mut [f32],
) -> MatchedFilterCoreResult {
    assert!(!x.is_empty());
    assert!(!h.is_empty());
    assert!(x_start_index < x.len());

    let mut filters_updated = false;
    let mut error_sum = 0.0f32;

    for &y_sample in y {
        let mut x2_sum = 0.0f32;
        let mut s = 0.0f32;
        let mut x_index = x_start_index;
        for &h_k in h.iter() {
            let x_k = x[x_index];
            x2_sum += x_k * x_k;
            s += h_k * x_k;
            x_index = if x_index + 1 < x.len() {
                x_index + 1
            } else {
                0
            };
        }

        let error = y_sample - s;
        let saturation = y_sample >= SATURATION_LIMIT || y_sample <= -SATURATION_LIMIT;
        error_sum += error * error;

        if x2_sum > x2_sum_threshold && !saturation {
            let alpha = smoothing * error / x2_sum;
            let mut x_index = x_start_index;
            for h_k in h.iter_mut() {
                *h_k += alpha * x[x_index];
                x_index = if x_index + 1 < x.len() {
                    x_index + 1
                } else {
                    0
                };
            }
            filters_updated = true;
        }

        x_start_index = if x_start_index > 0 {
            x_start_index - 1
        } else {
            x.len() - 1
        };
    }

    MatchedFilterCoreResult {
        filters_updated,
        error_sum,
    }
}

/// Scalar core that additionally records, for every
/// `ACCUMULATED_ERROR_SUB_SAMPLE_RATE` taps, the error of the filter truncated
/// to that many taps.
fn matched_filter_core_with_accumulated_error(
    mut x_start_index: usize,
    x2_sum_threshold: f32,
    smoothing: f32,
    x: &[f32],
    y: &[f32],
    h: &mut [f32],
    accumulated_error: &mut [f32],
) -> MatchedFilterCoreResult {
    assert!(!x.is_empty());
    assert!(!h.is_empty());
    assert!(x_start_index < x.len());

    let mut filters_updated = false;
    let mut error_sum = 0.0f32;
    accumulated_error.fill(0.0);

    for &y_sample in y {
        let mut x2_sum = 0.0f32;
        let mut s = 0.0f32;
        let mut x_index = x_start_index;
        for (k, &h_k) in h.iter().enumerate() {
            let x_k = x[x_index];
            x2_sum += x_k * x_k;
            s += h_k * x_k;
            x_index = if x_index + 1 < x.len() {
                x_index + 1
            } else {
                0
            };
            if (k + 1) % ACCUMULATED_ERROR_SUB_SAMPLE_RATE == 0 {
                let partial_error = y_sample - s;
                accumulated_error[k / ACCUMULATED_ERROR_SUB_SAMPLE_RATE] +=
                    partial_error * partial_error;
            }
        }

        let error = y_sample - s;
        let saturation = y_sample >= SATURATION_LIMIT || y_sample <= -SATURATION_LIMIT;
        error_sum += error * error;

        if x2_sum > x2_sum_threshold && !saturation {
            let alpha = smoothing * error / x2_sum;
            let mut x_index = x_start_index;
            for h_k in h.iter_mut() {
                *h_k += alpha * x[x_index];
                x_index = if x_index + 1 < x.len() {
                    x_index + 1
                } else {
                    0
                };
            }
            filters_updated = true;
        }

        x_start_index = if x_start_index > 0 {
            x_start_index - 1
        } else {
            x.len() - 1
        };
    }

    MatchedFilterCoreResult {
        filters_updated,
        error_sum,
    }
}

/// Copies `h_len` samples starting at `x_start_index` out of the circular
/// buffer `x` into `scratch`, so the accumulated-error cores can walk the taps
/// contiguously. Returns the slice to read from.
fn linearise_render<'a>(
    x: &'a [f32],
    x_start_index: usize,
    h_len: usize,
    scratch: &'a mut [f32],
) -> &'a [f32] {
    let chunk1 = h_len.min(x.len() - x_start_index);
    if chunk1 == h_len {
        return &x[x_start_index..x_start_index + h_len];
    }
    let chunk2 = h_len - chunk1;
    scratch[..chunk1].copy_from_slice(&x[x_start_index..x_start_index + chunk1]);
    scratch[chunk1..h_len].copy_from_slice(&x[..chunk2]);
    &scratch[..h_len]
}

/// Scalar fallback shared by the SIMD accumulated-error cores for the taps that
/// do not fill a whole vector.
#[inline(always)]
fn accumulate_tail(
    x_p: &[f32],
    h_p: &[f32],
    y_sample: f32,
    from: usize,
    accumulated_error: &mut [f32],
    x2_sum: &mut f32,
    s_acum: &mut f32,
) {
    for k in from..h_p.len() {
        let x_k = x_p[k];
        *x2_sum += x_k * x_k;
        *s_acum += h_p[k] * x_k;
        if (k + 1) % ACCUMULATED_ERROR_SUB_SAMPLE_RATE == 0 {
            let e = *s_acum - y_sample;
            accumulated_error[k / ACCUMULATED_ERROR_SUB_SAMPLE_RATE] += e * e;
        }
    }
}

/// NLMS filter update over a linearised render slice.
#[inline(always)]
fn nlms_update(h: &mut [f32], x_p: &[f32], alpha: f32) {
    for (h_k, &x_k) in h.iter_mut().zip(x_p.iter()) {
        *h_k += alpha * x_k;
    }
}

#[allow(clippy::too_many_arguments)]
fn matched_filter_core_accumulated_error_avx2(
    x_start_index: usize,
    x2_sum_threshold: f32,
    smoothing: f32,
    x: &[f32],
    y: &[f32],
    h: &mut [f32],
    accumulated_error: &mut [f32],
    scratch: &mut [f32],
) -> MatchedFilterCoreResult {
    let _ = &scratch;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if detect_avx2() {
            // The accumulated error is summed every four taps, so the 128-bit
            // path already matches the required granularity.
            return unsafe {
                matched_filter_core_accumulated_error_sse2_impl(
                    x_start_index,
                    x2_sum_threshold,
                    smoothing,
                    x,
                    y,
                    h,
                    accumulated_error,
                    scratch,
                )
            };
        }
    }
    matched_filter_core_with_accumulated_error(
        x_start_index,
        x2_sum_threshold,
        smoothing,
        x,
        y,
        h,
        accumulated_error,
    )
}

#[allow(clippy::too_many_arguments)]
fn matched_filter_core_accumulated_error_sse2(
    x_start_index: usize,
    x2_sum_threshold: f32,
    smoothing: f32,
    x: &[f32],
    y: &[f32],
    h: &mut [f32],
    accumulated_error: &mut [f32],
    scratch: &mut [f32],
) -> MatchedFilterCoreResult {
    let _ = &scratch;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if detect_sse2() {
            return unsafe {
                matched_filter_core_accumulated_error_sse2_impl(
                    x_start_index,
                    x2_sum_threshold,
                    smoothing,
                    x,
                    y,
                    h,
                    accumulated_error,
                    scratch,
                )
            };
        }
    }
    matched_filter_core_with_accumulated_error(
        x_start_index,
        x2_sum_threshold,
        smoothing,
        x,
        y,
        h,
        accumulated_error,
    )
}

#[allow(clippy::too_many_arguments)]
fn matched_filter_core_accumulated_error_neon(
    x_start_index: usize,
    x2_sum_threshold: f32,
    smoothing: f32,
    x: &[f32],
    y: &[f32],
    h: &mut [f32],
    accumulated_error: &mut [f32],
    scratch: &mut [f32],
) -> MatchedFilterCoreResult {
    let _ = &scratch;
    #[cfg(target_arch = "aarch64")]
    {
        if detect_neon() {
            return unsafe {
                matched_filter_core_accumulated_error_neon_impl(
                    x_start_index,
                    x2_sum_threshold,
                    smoothing,
                    x,
                    y,
                    h,
                    accumulated_error,
                    scratch,
                )
            };
        }
    }
    matched_filter_core_with_accumulated_error(
        x_start_index,
        x2_sum_threshold,
        smoothing,
        x,
        y,
        h,
        accumulated_error,
    )
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
#[allow(clippy::too_many_arguments)]
unsafe fn matched_filter_core_accumulated_error_sse2_impl(
    mut x_start_index: usize,
    x2_sum_threshold: f32,
    smoothing: f32,
    x: &[f32],
    y: &[f32],
    h: &mut [f32],
    accumulated_error: &mut [f32],
    scratch: &mut [f32],
) -> MatchedFilterCoreResult {
    let mut filters_updated = false;
    let mut error_sum = 0.0f32;
    let h_size = h.len();
    let vector_limit = h_size & !3;
    accumulated_error.fill(0.0);

    for &y_sample in y {
        let x_p = linearise_render(x, x_start_index, h_size, scratch);
        let mut x2_sum_vec = _mm_setzero_ps();
        let mut x2_sum = 0.0f32;
        let mut s_acum = 0.0f32;

        let mut k = 0usize;
        while k < vector_limit {
            let (x_k, h_k) = unsafe {
                (
                    _mm_loadu_ps(x_p.as_ptr().add(k)),
                    _mm_loadu_ps(h.as_ptr().add(k)),
                )
            };
            x2_sum_vec = _mm_add_ps(x2_sum_vec, _mm_mul_ps(x_k, x_k));
            let mut products = [0.0f32; 4];
            unsafe { _mm_storeu_ps(products.as_mut_ptr(), _mm_mul_ps(h_k, x_k)) };
            s_acum += products[0] + products[1] + products[2] + products[3];
            let e = s_acum - y_sample;
            accumulated_error[k / ACCUMULATED_ERROR_SUB_SAMPLE_RATE] += e * e;
            k += 4;
        }
        let mut vec_sum = [0.0f32; 4];
        unsafe { _mm_storeu_ps(vec_sum.as_mut_ptr(), x2_sum_vec) };
        x2_sum += vec_sum.iter().sum::<f32>();
        accumulate_tail(
            x_p,
            h,
            y_sample,
            vector_limit,
            accumulated_error,
            &mut x2_sum,
            &mut s_acum,
        );

        let error = y_sample - s_acum;
        let saturation = y_sample >= SATURATION_LIMIT || y_sample <= -SATURATION_LIMIT;
        error_sum += error * error;

        if x2_sum > x2_sum_threshold && !saturation {
            let alpha = smoothing * error / x2_sum;
            let x_p = linearise_render(x, x_start_index, h_size, scratch);
            nlms_update(h, x_p, alpha);
            filters_updated = true;
        }

        x_start_index = if x_start_index > 0 {
            x_start_index - 1
        } else {
            x.len() - 1
        };
    }

    MatchedFilterCoreResult {
        filters_updated,
        error_sum,
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn matched_filter_core_accumulated_error_neon_impl(
    mut x_start_index: usize,
    x2_sum_threshold: f32,
    smoothing: f32,
    x: &[f32],
    y: &[f32],
    h: &mut [f32],
    accumulated_error: &mut [f32],
    scratch: &mut [f32],
) -> MatchedFilterCoreResult {
    let mut filters_updated = false;
    let mut error_sum = 0.0f32;
    let h_size = h.len();
    let vector_limit = h_size & !3;
    accumulated_error.fill(0.0);

    for &y_sample in y {
        let x_p = linearise_render(x, x_start_index, h_size, scratch);
        let mut x2_sum_vec = vdupq_n_f32(0.0);
        let mut x2_sum = 0.0f32;
        let mut s_acum = 0.0f32;

        let mut k = 0usize;
        while k < vector_limit {
            let (x_k, h_k) =
                unsafe { (vld1q_f32(x_p.as_ptr().add(k)), vld1q_f32(h.as_ptr().add(k))) };
            x2_sum_vec = vaddq_f32(x2_sum_vec, vmulq_f32(x_k, x_k));
            s_acum += vaddvq_f32(vmulq_f32(h_k, x_k));
            let e = s_acum - y_sample;
            accumulated_error[k / ACCUMULATED_ERROR_SUB_SAMPLE_RATE] += e * e;
            k += 4;
        }
        x2_sum += vaddvq_f32(x2_sum_vec);
        accumulate_tail(
            x_p,
            h,
            y_sample,
            vector_limit,
            accumulated_error,
            &mut x2_sum,
            &mut s_acum,
        );

        let error = y_sample - s_acum;
        let saturation = y_sample >= SATURATION_LIMIT || y_sample <= -SATURATION_LIMIT;
        error_sum += error * error;

        if x2_sum > x2_sum_threshold && !saturation {
            let alpha = smoothing * error / x2_sum;
            let x_p = linearise_render(x, x_start_index, h_size, scratch);
            nlms_update(h, x_p, alpha);
            filters_updated = true;
        }

        x_start_index = if x_start_index > 0 {
            x_start_index - 1
        } else {
            x.len() - 1
        };
    }

    MatchedFilterCoreResult {
        filters_updated,
        error_sum,
    }
}

fn matched_filter_core_avx2(
    x_start_index: usize,
    x2_sum_threshold: f32,
    smoothing: f32,
    x: &[f32],
    y: &[f32],
    h: &mut [f32],
) -> MatchedFilterCoreResult {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if detect_avx2() {
        unsafe {
            return matched_filter_core_avx2_impl(
                x_start_index,
                x2_sum_threshold,
                smoothing,
                x,
                y,
                h,
            );
        }
    }
    matched_filter_core(x_start_index, x2_sum_threshold, smoothing, x, y, h)
}

fn matched_filter_core_sse2(
    x_start_index: usize,
    x2_sum_threshold: f32,
    smoothing: f32,
    x: &[f32],
    y: &[f32],
    h: &mut [f32],
) -> MatchedFilterCoreResult {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if detect_sse2() {
        unsafe {
            return matched_filter_core_sse2_impl(
                x_start_index,
                x2_sum_threshold,
                smoothing,
                x,
                y,
                h,
            );
        }
    }
    matched_filter_core(x_start_index, x2_sum_threshold, smoothing, x, y, h)
}

fn matched_filter_core_neon(
    x_start_index: usize,
    x2_sum_threshold: f32,
    smoothing: f32,
    x: &[f32],
    y: &[f32],
    h: &mut [f32],
) -> MatchedFilterCoreResult {
    #[cfg(target_arch = "aarch64")]
    if detect_neon() {
        unsafe {
            return matched_filter_core_neon_impl(
                x_start_index,
                x2_sum_threshold,
                smoothing,
                x,
                y,
                h,
            );
        }
    }
    matched_filter_core(x_start_index, x2_sum_threshold, smoothing, x, y, h)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2")]
unsafe fn matched_filter_core_avx2_impl(
    mut x_start_index: usize,
    x2_sum_threshold: f32,
    smoothing: f32,
    x: &[f32],
    y: &[f32],
    h: &mut [f32],
) -> MatchedFilterCoreResult {
    let mut filters_updated = false;
    let mut error_sum = 0.0f32;
    let x_size = x.len();
    let h_size = h.len();

    for &y_sample in y {
        let chunk1 = h_size.min(x_size - x_start_index);
        let chunk2 = h_size - chunk1;
        let mut x_ptr = x[x_start_index..].as_ptr();
        let mut h_ptr = h.as_ptr();
        let mut x2_sum_vec = _mm256_setzero_ps();
        let mut s_vec = _mm256_setzero_ps();
        let mut x2_sum = 0.0f32;
        let mut s = 0.0f32;

        for limit in [chunk1, chunk2] {
            let vector_limit = limit & !7;
            let mut processed = 0usize;
            while processed < vector_limit {
                let x_k = _mm256_loadu_ps(x_ptr.add(processed));
                let h_k = _mm256_loadu_ps(h_ptr.add(processed));
                x2_sum_vec = _mm256_add_ps(x2_sum_vec, _mm256_mul_ps(x_k, x_k));
                s_vec = _mm256_add_ps(s_vec, _mm256_mul_ps(h_k, x_k));
                processed += 8;
            }
            while processed < limit {
                let x_k = *x_ptr.add(processed);
                x2_sum += x_k * x_k;
                s += *h_ptr.add(processed) * x_k;
                processed += 1;
            }
            x_ptr = x.as_ptr();
            h_ptr = h_ptr.add(limit);
        }

        let mut vec_sum = [0.0f32; 8];
        _mm256_storeu_ps(vec_sum.as_mut_ptr(), x2_sum_vec);
        x2_sum += vec_sum.iter().sum::<f32>();
        _mm256_storeu_ps(vec_sum.as_mut_ptr(), s_vec);
        s += vec_sum.iter().sum::<f32>();

        let error = y_sample - s;
        let saturation = y_sample >= SATURATION_LIMIT || y_sample <= -SATURATION_LIMIT;
        error_sum += error * error;

        if x2_sum > x2_sum_threshold && !saturation {
            let alpha = smoothing * error / x2_sum;
            let alpha_vec = _mm256_set1_ps(alpha);
            let mut x_ptr = x[x_start_index..].as_ptr();
            let mut h_ptr = h.as_mut_ptr();
            for limit in [chunk1, chunk2] {
                let vector_limit = limit & !7;
                let mut processed = 0usize;
                while processed < vector_limit {
                    let h_k = _mm256_loadu_ps(h_ptr.add(processed));
                    let x_k = _mm256_loadu_ps(x_ptr.add(processed));
                    let update = _mm256_add_ps(h_k, _mm256_mul_ps(alpha_vec, x_k));
                    _mm256_storeu_ps(h_ptr.add(processed), update);
                    processed += 8;
                }
                while processed < limit {
                    *h_ptr.add(processed) += alpha * *x_ptr.add(processed);
                    processed += 1;
                }
                x_ptr = x.as_ptr();
                h_ptr = h_ptr.add(limit);
            }
            filters_updated = true;
        }

        x_start_index = if x_start_index > 0 {
            x_start_index - 1
        } else {
            x_size - 1
        };
    }

    MatchedFilterCoreResult {
        filters_updated,
        error_sum,
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse2")]
unsafe fn matched_filter_core_sse2_impl(
    mut x_start_index: usize,
    x2_sum_threshold: f32,
    smoothing: f32,
    x: &[f32],
    y: &[f32],
    h: &mut [f32],
) -> MatchedFilterCoreResult {
    let mut filters_updated = false;
    let mut error_sum = 0.0f32;
    let x_size = x.len();
    let h_size = h.len();

    for &y_sample in y {
        let chunk1 = h_size.min(x_size - x_start_index);
        let chunk2 = h_size - chunk1;
        let mut x_ptr = x[x_start_index..].as_ptr();
        let mut h_ptr = h.as_ptr();
        let mut x2_sum_vec = _mm_setzero_ps();
        let mut s_vec = _mm_setzero_ps();
        let mut x2_sum = 0.0f32;
        let mut s = 0.0f32;

        for limit in [chunk1, chunk2] {
            let vector_limit = limit & !3;
            let mut processed = 0usize;
            while processed < vector_limit {
                let x_k = _mm_loadu_ps(x_ptr.add(processed));
                let h_k = _mm_loadu_ps(h_ptr.add(processed));
                x2_sum_vec = _mm_add_ps(x2_sum_vec, _mm_mul_ps(x_k, x_k));
                s_vec = _mm_add_ps(s_vec, _mm_mul_ps(h_k, x_k));
                processed += 4;
            }
            while processed < limit {
                let x_k = *x_ptr.add(processed);
                x2_sum += x_k * x_k;
                s += *h_ptr.add(processed) * x_k;
                processed += 1;
            }
            x_ptr = x.as_ptr();
            h_ptr = h_ptr.add(limit);
        }

        let mut vec_sum = [0.0f32; 4];
        _mm_storeu_ps(vec_sum.as_mut_ptr(), x2_sum_vec);
        x2_sum += vec_sum.iter().sum::<f32>();
        _mm_storeu_ps(vec_sum.as_mut_ptr(), s_vec);
        s += vec_sum.iter().sum::<f32>();

        let error = y_sample - s;
        let saturation = y_sample >= SATURATION_LIMIT || y_sample <= -SATURATION_LIMIT;
        error_sum += error * error;

        if x2_sum > x2_sum_threshold && !saturation {
            let alpha = smoothing * error / x2_sum;
            let alpha_vec = _mm_set1_ps(alpha);
            let mut x_ptr = x[x_start_index..].as_ptr();
            let mut h_ptr = h.as_mut_ptr();
            for limit in [chunk1, chunk2] {
                let vector_limit = limit & !3;
                let mut processed = 0usize;
                while processed < vector_limit {
                    let h_k = _mm_loadu_ps(h_ptr.add(processed));
                    let x_k = _mm_loadu_ps(x_ptr.add(processed));
                    let update = _mm_add_ps(h_k, _mm_mul_ps(alpha_vec, x_k));
                    _mm_storeu_ps(h_ptr.add(processed), update);
                    processed += 4;
                }
                while processed < limit {
                    *h_ptr.add(processed) += alpha * *x_ptr.add(processed);
                    processed += 1;
                }
                x_ptr = x.as_ptr();
                h_ptr = h_ptr.add(limit);
            }
            filters_updated = true;
        }

        x_start_index = if x_start_index > 0 {
            x_start_index - 1
        } else {
            x_size - 1
        };
    }

    MatchedFilterCoreResult {
        filters_updated,
        error_sum,
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn matched_filter_core_neon_impl(
    mut x_start_index: usize,
    x2_sum_threshold: f32,
    smoothing: f32,
    x: &[f32],
    y: &[f32],
    h: &mut [f32],
) -> MatchedFilterCoreResult {
    let mut filters_updated = false;
    let mut error_sum = 0.0f32;
    let x_size = x.len();
    let h_size = h.len();

    for &y_sample in y {
        let chunk1 = h_size.min(x_size - x_start_index);
        let chunk2 = h_size - chunk1;
        let mut x_ptr = x[x_start_index..].as_ptr();
        let mut h_ptr = h.as_ptr();
        let mut x2_sum_vec = vdupq_n_f32(0.0);
        let mut s_vec = vdupq_n_f32(0.0);
        let mut x2_sum = 0.0f32;
        let mut s = 0.0f32;

        for limit in [chunk1, chunk2] {
            let vector_limit = limit & !3;
            let mut processed = 0usize;
            while processed < vector_limit {
                let x_k = vld1q_f32(x_ptr.add(processed));
                let h_k = vld1q_f32(h_ptr.add(processed));
                x2_sum_vec = vaddq_f32(x2_sum_vec, vmulq_f32(x_k, x_k));
                s_vec = vaddq_f32(s_vec, vmulq_f32(h_k, x_k));
                processed += 4;
            }
            while processed < limit {
                let x_k = *x_ptr.add(processed);
                x2_sum += x_k * x_k;
                s += *h_ptr.add(processed) * x_k;
                processed += 1;
            }
            x_ptr = x.as_ptr();
            h_ptr = h_ptr.add(limit);
        }

        let mut vec_sum = [0.0f32; 4];
        vst1q_f32(vec_sum.as_mut_ptr(), x2_sum_vec);
        x2_sum += vec_sum.iter().sum::<f32>();
        vst1q_f32(vec_sum.as_mut_ptr(), s_vec);
        s += vec_sum.iter().sum::<f32>();

        let error = y_sample - s;
        let saturation = y_sample >= SATURATION_LIMIT || y_sample <= -SATURATION_LIMIT;
        error_sum += error * error;

        if x2_sum > x2_sum_threshold && !saturation {
            let alpha = smoothing * error / x2_sum;
            let alpha_vec = vdupq_n_f32(alpha);
            let mut x_ptr = x[x_start_index..].as_ptr();
            let mut h_ptr = h.as_mut_ptr();
            for limit in [chunk1, chunk2] {
                let vector_limit = limit & !3;
                let mut processed = 0usize;
                while processed < vector_limit {
                    let h_k = vld1q_f32(h_ptr.add(processed));
                    let x_k = vld1q_f32(x_ptr.add(processed));
                    let update = vaddq_f32(h_k, vmulq_f32(alpha_vec, x_k));
                    vst1q_f32(h_ptr.add(processed), update);
                    processed += 4;
                }
                while processed < limit {
                    *h_ptr.add(processed) += alpha * *x_ptr.add(processed);
                    processed += 1;
                }
                x_ptr = x.as_ptr();
                h_ptr = h_ptr.add(limit);
            }
            filters_updated = true;
        }

        x_start_index = if x_start_index > 0 {
            x_start_index - 1
        } else {
            x_size - 1
        };
    }

    MatchedFilterCoreResult {
        filters_updated,
        error_sum,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::config::EchoCanceller3Config;
    use crate::audio_processing::aec3::aec3_common::{
        MATCHED_FILTER_ALIGNMENT_SHIFT_SIZE_SUB_BLOCKS, MATCHED_FILTER_WINDOW_SIZE_SUB_BLOCKS,
        detect_optimization,
    };
    use crate::audio_processing::aec3::block::Block;
    use crate::audio_processing::aec3::downsampled_render_buffer::DownsampledRenderBuffer;
    use crate::test_support::echo_canceller_test_tools::randomize_sample_vector_with_amplitude;
    use crate::test_support::random::Random;

    const BUFFER_BLOCKS: usize = 128;
    const DOWN_SAMPLING_FACTORS: [usize; 3] = [2, 4, 8];
    const NON_SATURATING_AMPLITUDE: f32 = 30_000.0;
    const REFERENCE_NUM_MATCHED_FILTERS: usize = 10;

    fn init_render_buffer(
        sub_block_size: usize,
        rng: &mut Random,
        amplitude: f32,
    ) -> DownsampledRenderBuffer {
        let len = sub_block_size * BUFFER_BLOCKS;
        let mut buffer = DownsampledRenderBuffer::new(len);
        randomize_sample_vector_with_amplitude(rng, &mut buffer.buffer, amplitude);
        buffer
    }

    fn set_block_start(
        buffer: &mut DownsampledRenderBuffer,
        block_start: usize,
        sub_block_size: usize,
    ) {
        let offset = buffer.offset_index(block_start, -((sub_block_size - 1) as isize));
        buffer.read = offset;
    }

    fn fill_capture_from_signal(
        signal: &[f32],
        block_start: usize,
        delay: usize,
        sub_block_size: usize,
        capture: &mut [f32],
    ) {
        let size = signal.len();
        for i in 0..sub_block_size {
            let idx = (size + block_start + delay - i) % size;
            capture[i] = signal[idx];
        }
    }

    fn assert_slice_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&lhs, &rhs)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (lhs - rhs).abs() <= tolerance,
                "mismatch at index {index}: got {lhs}, expected {rhs}"
            );
        }
    }

    #[test]
    fn matched_filter_core_matches_scalar_across_optimizations() {
        let x = (0..192)
            .map(|i| ((i as f32 * 0.17).sin() * 48.0) + ((i as f32 * 0.11).cos() * 12.0))
            .collect::<Vec<_>>();
        let y = (0..24)
            .map(|i| ((i as f32 * 0.29).cos() * 20.0) + 5.0)
            .collect::<Vec<_>>();
        let initial_h = (0..48)
            .map(|i| ((i as f32 * 0.07).sin() * 0.05) - 0.02)
            .collect::<Vec<_>>();
        let x_start_index = x.len() - 9;
        let x2_sum_threshold = 100.0;
        let smoothing = 0.2;

        let mut expected_h = initial_h.clone();
        let expected = matched_filter_core(
            x_start_index,
            x2_sum_threshold,
            smoothing,
            &x,
            &y,
            &mut expected_h,
        );

        for optimization in [
            Aec3Optimization::Sse2,
            Aec3Optimization::Avx2,
            Aec3Optimization::Neon,
        ] {
            let mut actual_h = initial_h.clone();
            let actual = match optimization {
                Aec3Optimization::Sse2 => matched_filter_core_sse2(
                    x_start_index,
                    x2_sum_threshold,
                    smoothing,
                    &x,
                    &y,
                    &mut actual_h,
                ),
                Aec3Optimization::Avx2 => matched_filter_core_avx2(
                    x_start_index,
                    x2_sum_threshold,
                    smoothing,
                    &x,
                    &y,
                    &mut actual_h,
                ),
                Aec3Optimization::Neon => matched_filter_core_neon(
                    x_start_index,
                    x2_sum_threshold,
                    smoothing,
                    &x,
                    &y,
                    &mut actual_h,
                ),
                Aec3Optimization::None => unreachable!(),
            };

            assert_eq!(
                expected.filters_updated, actual.filters_updated,
                "update flag mismatch for {optimization:?}"
            );
            assert!(
                (expected.error_sum - actual.error_sum).abs() <= 1e-2,
                "error sum mismatch for {optimization:?}: got {}, expected {}",
                actual.error_sum,
                expected.error_sum
            );
            assert_slice_close(&actual_h, &expected_h, 1e-4);
        }
    }

    /// The SIMD accumulated-error cores must agree with the scalar one, both in
    /// the filter update and in the per-truncation error that drives pre-echo
    /// detection.
    #[test]
    fn accumulated_error_cores_match_scalar_across_optimizations() {
        let x = (0..192)
            .map(|i| ((i as f32 * 0.17).sin() * 48.0) + ((i as f32 * 0.11).cos() * 12.0))
            .collect::<Vec<_>>();
        let y = (0..24)
            .map(|i| ((i as f32 * 0.29).cos() * 20.0) + 5.0)
            .collect::<Vec<_>>();
        let initial_h = (0..48)
            .map(|i| ((i as f32 * 0.07).sin() * 0.05) - 0.02)
            .collect::<Vec<_>>();
        // Starts near the end of the buffer so the render read wraps.
        let x_start_index = x.len() - 9;
        let x2_sum_threshold = 100.0;
        let smoothing = 0.2;
        let accumulated_len = initial_h.len() / ACCUMULATED_ERROR_SUB_SAMPLE_RATE;

        let mut expected_h = initial_h.clone();
        let mut expected_error = vec![0.0f32; accumulated_len];
        let expected = matched_filter_core_with_accumulated_error(
            x_start_index,
            x2_sum_threshold,
            smoothing,
            &x,
            &y,
            &mut expected_h,
            &mut expected_error,
        );

        for optimization in [
            Aec3Optimization::Sse2,
            Aec3Optimization::Avx2,
            Aec3Optimization::Neon,
        ] {
            let mut actual_h = initial_h.clone();
            let mut actual_error = vec![0.0f32; accumulated_len];
            let mut scratch = vec![0.0f32; initial_h.len()];
            let actual = match optimization {
                Aec3Optimization::Sse2 => matched_filter_core_accumulated_error_sse2(
                    x_start_index,
                    x2_sum_threshold,
                    smoothing,
                    &x,
                    &y,
                    &mut actual_h,
                    &mut actual_error,
                    &mut scratch,
                ),
                Aec3Optimization::Avx2 => matched_filter_core_accumulated_error_avx2(
                    x_start_index,
                    x2_sum_threshold,
                    smoothing,
                    &x,
                    &y,
                    &mut actual_h,
                    &mut actual_error,
                    &mut scratch,
                ),
                Aec3Optimization::Neon => matched_filter_core_accumulated_error_neon(
                    x_start_index,
                    x2_sum_threshold,
                    smoothing,
                    &x,
                    &y,
                    &mut actual_h,
                    &mut actual_error,
                    &mut scratch,
                ),
                Aec3Optimization::None => unreachable!(),
            };

            assert_eq!(
                expected.filters_updated, actual.filters_updated,
                "update flag mismatch for {optimization:?}"
            );
            assert!(
                (expected.error_sum - actual.error_sum).abs() <= 1e-2,
                "error sum mismatch for {optimization:?}: got {}, expected {}",
                actual.error_sum,
                expected.error_sum
            );
            assert_slice_close(&actual_h, &expected_h, 1e-4);
            for (k, (a, e)) in actual_error.iter().zip(expected_error.iter()).enumerate() {
                let tolerance = 1e-3 * e.abs().max(1.0);
                assert!(
                    (a - e).abs() <= tolerance,
                    "accumulated error[{k}] mismatch for {optimization:?}: got {a}, expected {e}"
                );
            }
        }
    }

    #[test]
    fn lag_estimation_detects_known_delay() {
        let mut rng = Random::new(42);
        let config = EchoCanceller3Config::default();
        for &down_sampling_factor in &DOWN_SAMPLING_FACTORS {
            let sub_block_size = BLOCK_SIZE / down_sampling_factor;
            // Mirror the C++ reference test more closely by using the
            // RenderDelayBuffer + Decimator pipeline. This exercises the
            // full render->delay->downsample->matched-filter flow.
            let num_channels = 1usize;
            let num_bands = crate::audio_processing::aec3::aec3_common::num_bands_for_rate(48_000);

            // Prepare storage for render and capture blocks.
            let mut render = Block::new(num_bands, num_channels);
            let mut capture = vec![vec![0.0f32; BLOCK_SIZE]; 1];

            for &delay_samples in &[5usize, 64, 150, 200, 800, 1000] {
                let mut cfg = config.clone();
                cfg.delay.down_sampling_factor = down_sampling_factor;
                cfg.delay.num_filters = REFERENCE_NUM_MATCHED_FILTERS;

                let mut capture_decimator =
                    crate::audio_processing::aec3::decimator::Decimator::new(down_sampling_factor);
                let mut signal_delay_buffer =
                    crate::test_support::echo_canceller_test_tools::DelayBuffer::new(
                        down_sampling_factor * delay_samples,
                    );

                let mut filter = MatchedFilter::new(
                    ApmDataDumper::new_unique(),
                    detect_optimization(),
                    sub_block_size,
                    MATCHED_FILTER_WINDOW_SIZE_SUB_BLOCKS,
                    REFERENCE_NUM_MATCHED_FILTERS,
                    MATCHED_FILTER_ALIGNMENT_SHIFT_SIZE_SUB_BLOCKS,
                    150.0,
                    cfg.delay.delay_estimate_smoothing,
                    cfg.delay.delay_estimate_smoothing_delay_found,
                    cfg.delay.delay_candidate_detection_threshold,
                    /*detect_pre_echo=*/ false,
                );

                let mut render_delay_buffer =
                    crate::audio_processing::aec3::render_delay_buffer::RenderDelayBuffer::new(
                        cfg.clone(),
                        48_000,
                        num_channels,
                    );

                // Number of iterations as in the reference: 600 + delay / sub_block_size
                let iterations = 600 + delay_samples / sub_block_size;

                let mut downsampled_capture = vec![0.0f32; sub_block_size];

                for k in 0..iterations {
                    // Randomize render for each band/channel.
                    for band in 0..num_bands {
                        for ch in 0..num_channels {
                            randomize_sample_vector_with_amplitude(
                                &mut rng,
                                render.view_mut(band, ch),
                                NON_SATURATING_AMPLITUDE,
                            );
                        }
                    }

                    // Delay render into capture (only use band 0, channel 0 like the ref).
                    let source = *render.view(0, 0);
                    signal_delay_buffer.delay(&source, &mut capture[0]);

                    // Insert render block into RenderDelayBuffer and prepare capture.
                    render_delay_buffer.insert(&render);
                    if k == 0 {
                        render_delay_buffer.reset();
                    }
                    render_delay_buffer.prepare_capture_processing();

                    // Downsample capture and update matched filter.
                    capture_decimator.decimate(&capture[0], &mut downsampled_capture);
                    filter.update(
                        render_delay_buffer.downsampled_render_buffer(),
                        &downsampled_capture,
                        /*use_slow_smoothing=*/ false,
                    );
                }

                let lag_estimate = filter
                    .best_lag_estimate()
                    .expect("a lag estimate should be found");
                assert_eq!(delay_samples, lag_estimate.lag);
                assert_eq!(delay_samples, lag_estimate.pre_echo_lag);
            }
        }
    }

    /// A capture signal built from an echo at 50 ms plus a weaker pre-echo at
    /// 20 ms. With detection on, the reported pre-echo lag should land on the
    /// early reflection; with it off, on the strongest peak.
    /// Not part of the suite. Run with:
    /// `cargo test --release --lib pre_echo_cost -- --ignored --nocapture`
    #[test]
    #[ignore = "benchmark"]
    fn pre_echo_cost() {
        use std::time::Instant;

        let down_sampling_factor = 4usize;
        let sub_block_size = BLOCK_SIZE / down_sampling_factor;
        let filter_len = MATCHED_FILTER_WINDOW_SIZE_SUB_BLOCKS * sub_block_size;
        const ITERATIONS: usize = 250 * 60; // one minute of blocks

        let mut rng = Random::new(42);
        let mut x = vec![0.0f32; sub_block_size * BUFFER_BLOCKS];
        randomize_sample_vector_with_amplitude(&mut rng, &mut x, NON_SATURATING_AMPLITUDE);
        let mut y = vec![0.0f32; sub_block_size];
        randomize_sample_vector_with_amplitude(&mut rng, &mut y, NON_SATURATING_AMPLITUDE);
        let mut accumulated = vec![0.0f32; filter_len / ACCUMULATED_ERROR_SUB_SAMPLE_RATE];

        // Sweep the whole render buffer so the circular wrap is exercised at a
        // realistic rate.
        let starts: Vec<usize> = (0..ITERATIONS)
            .map(|k| (k * 37) % (sub_block_size * BUFFER_BLOCKS))
            .collect();
        let bench = |name: &str, mut f: Box<dyn FnMut(usize)>| {
            let start = Instant::now();
            for k in 0..ITERATIONS {
                f(starts[k]);
            }
            let elapsed = start.elapsed();
            println!(
                "{name}: {:.4} us/call",
                elapsed.as_secs_f64() * 1e6 / ITERATIONS as f64
            );
        };

        {
            let mut h = vec![0.0f32; filter_len];
            let x = x.clone();
            let y = y.clone();
            bench(
                "simd core            ",
                Box::new(move |start| {
                    matched_filter_core_neon(start, 1.0, 0.7, &x, &y, &mut h);
                }),
            );
        }
        {
            let mut h = vec![0.0f32; filter_len];
            let x = x.clone();
            let y = y.clone();
            bench(
                "scalar core          ",
                Box::new(move |start| {
                    matched_filter_core(start, 1.0, 0.7, &x, &y, &mut h);
                }),
            );
        }
        {
            let mut h = vec![0.0f32; filter_len];
            let x = x.clone();
            let y = y.clone();
            let mut acc = accumulated.clone();
            bench(
                "scalar + accum error ",
                Box::new(move |start| {
                    matched_filter_core_with_accumulated_error(
                        start, 1.0, 0.7, &x, &y, &mut h, &mut acc,
                    );
                }),
            );
        }
        {
            let mut h = vec![0.0f32; filter_len];
            let x = x.clone();
            let y = y.clone();
            let mut scratch = vec![0.0f32; filter_len];
            bench(
                "simd + accum error   ",
                Box::new(move |start| {
                    matched_filter_core_accumulated_error_neon(
                        start,
                        1.0,
                        0.7,
                        &x,
                        &y,
                        &mut h,
                        &mut accumulated,
                        &mut scratch,
                    );
                }),
            );
        }
        println!("optimization = {:?}", detect_optimization());

        // End to end: the full matched filter over a correlated signal, so a
        // winner is found and the accumulated-error path actually runs.
        let num_bands = crate::audio_processing::aec3::aec3_common::num_bands_for_rate(48_000);
        for &detect_pre_echo in &[false, true] {
            let mut rng = Random::new(42);
            let mut cfg = EchoCanceller3Config::default();
            cfg.delay.down_sampling_factor = down_sampling_factor;
            let mut render = Block::new(num_bands, 1);
            let mut capture_fullband = vec![0.0f32; BLOCK_SIZE];
            let mut downsampled_capture = vec![0.0f32; sub_block_size];
            let mut decimator =
                crate::audio_processing::aec3::decimator::Decimator::new(down_sampling_factor);
            let mut delay_buffer = crate::test_support::echo_canceller_test_tools::DelayBuffer::new(
                down_sampling_factor * 50,
            );
            let mut filter = MatchedFilter::new(
                ApmDataDumper::new_unique(),
                detect_optimization(),
                sub_block_size,
                MATCHED_FILTER_WINDOW_SIZE_SUB_BLOCKS,
                cfg.delay.num_filters,
                MATCHED_FILTER_ALIGNMENT_SHIFT_SIZE_SUB_BLOCKS,
                150.0,
                cfg.delay.delay_estimate_smoothing,
                cfg.delay.delay_estimate_smoothing_delay_found,
                cfg.delay.delay_candidate_detection_threshold,
                detect_pre_echo,
            );
            let mut render_delay_buffer =
                crate::audio_processing::aec3::render_delay_buffer::RenderDelayBuffer::new(
                    cfg.clone(),
                    48_000,
                    1,
                );

            const E2E_BLOCKS: usize = 250 * 20;
            let mut elapsed = std::time::Duration::ZERO;
            for k in 0..E2E_BLOCKS {
                for band in 0..num_bands {
                    randomize_sample_vector_with_amplitude(
                        &mut rng,
                        render.view_mut(band, 0),
                        NON_SATURATING_AMPLITUDE,
                    );
                }
                let source = *render.view(0, 0);
                delay_buffer.delay(&source, &mut capture_fullband);
                render_delay_buffer.insert(&render);
                if k == 0 {
                    render_delay_buffer.reset();
                }
                render_delay_buffer.prepare_capture_processing();
                decimator.decimate(&capture_fullband, &mut downsampled_capture);
                let start = Instant::now();
                filter.update(
                    render_delay_buffer.downsampled_render_buffer(),
                    &downsampled_capture,
                    false,
                );
                elapsed += start.elapsed();
            }
            assert!(
                filter.best_lag_estimate().is_some(),
                "the benchmark must find a winner or it measures nothing"
            );
            println!(
                "update() detect_pre_echo={detect_pre_echo}: {:.3} us/block ({:.3}% of a 4 ms block)",
                elapsed.as_secs_f64() * 1e6 / E2E_BLOCKS as f64,
                elapsed.as_secs_f64() * 1e3 / E2E_BLOCKS as f64 / 4.0 * 100.0
            );
        }
    }

    #[test]
    fn pre_echo_estimation_finds_the_early_reflection() {
        let mut rng = Random::new(42);
        let num_channels = 1usize;
        let num_bands = crate::audio_processing::aec3::aec3_common::num_bands_for_rate(48_000);

        for &down_sampling_factor in &DOWN_SAMPLING_FACTORS {
            for &detect_pre_echo in &[false, true] {
                let sub_block_size = BLOCK_SIZE / down_sampling_factor;
                let pre_echo_delay_samples = 20 * 16_000 / 1000 / down_sampling_factor;
                let echo_delay_samples = 50 * 16_000 / 1000 / down_sampling_factor;

                let mut cfg = EchoCanceller3Config::default();
                cfg.delay.down_sampling_factor = down_sampling_factor;
                cfg.delay.num_filters = REFERENCE_NUM_MATCHED_FILTERS;

                let mut render = Block::new(num_bands, num_channels);
                let mut capture = vec![0.0f32; BLOCK_SIZE];
                let mut capture_with_pre_echo = vec![0.0f32; BLOCK_SIZE];

                let mut capture_decimator =
                    crate::audio_processing::aec3::decimator::Decimator::new(down_sampling_factor);
                let mut signal_echo_delay_buffer =
                    crate::test_support::echo_canceller_test_tools::DelayBuffer::new(
                        down_sampling_factor * echo_delay_samples,
                    );
                let mut signal_pre_echo_delay_buffer =
                    crate::test_support::echo_canceller_test_tools::DelayBuffer::new(
                        down_sampling_factor * pre_echo_delay_samples,
                    );

                let mut filter = MatchedFilter::new(
                    ApmDataDumper::new_unique(),
                    detect_optimization(),
                    sub_block_size,
                    MATCHED_FILTER_WINDOW_SIZE_SUB_BLOCKS,
                    REFERENCE_NUM_MATCHED_FILTERS,
                    MATCHED_FILTER_ALIGNMENT_SHIFT_SIZE_SUB_BLOCKS,
                    150.0,
                    cfg.delay.delay_estimate_smoothing,
                    cfg.delay.delay_estimate_smoothing_delay_found,
                    cfg.delay.delay_candidate_detection_threshold,
                    detect_pre_echo,
                );

                let mut render_delay_buffer =
                    crate::audio_processing::aec3::render_delay_buffer::RenderDelayBuffer::new(
                        cfg.clone(),
                        48_000,
                        num_channels,
                    );

                let mut downsampled_capture = vec![0.0f32; sub_block_size];
                let iterations = 600 + echo_delay_samples / sub_block_size;
                for k in 0..iterations {
                    for band in 0..num_bands {
                        for ch in 0..num_channels {
                            randomize_sample_vector_with_amplitude(
                                &mut rng,
                                render.view_mut(band, ch),
                                NON_SATURATING_AMPLITUDE,
                            );
                        }
                    }

                    let source = *render.view(0, 0);
                    signal_echo_delay_buffer.delay(&source, &mut capture);
                    signal_pre_echo_delay_buffer.delay(&source, &mut capture_with_pre_echo);
                    const GAIN_PRE_ECHO: f32 = 0.8;
                    for (dst, &src) in capture.iter_mut().zip(capture_with_pre_echo.iter()) {
                        *dst += GAIN_PRE_ECHO * src;
                    }

                    render_delay_buffer.insert(&render);
                    if k == 0 {
                        render_delay_buffer.reset();
                    }
                    render_delay_buffer.prepare_capture_processing();
                    capture_decimator.decimate(&capture, &mut downsampled_capture);
                    filter.update(
                        render_delay_buffer.downsampled_render_buffer(),
                        &downsampled_capture,
                        /*use_slow_smoothing=*/ false,
                    );
                }

                let lag_estimate = filter
                    .best_lag_estimate()
                    .expect("a lag estimate should be found");
                assert_eq!(
                    echo_delay_samples, lag_estimate.lag,
                    "dsf {down_sampling_factor}: strongest peak"
                );
                if detect_pre_echo {
                    // The pre-echo lag is estimated in a subsampled domain, so a
                    // larger error is allowed.
                    assert!(
                        lag_estimate.pre_echo_lag.abs_diff(pre_echo_delay_samples) <= 4,
                        "dsf {down_sampling_factor}: pre-echo lag {} should be near {}",
                        lag_estimate.pre_echo_lag,
                        pre_echo_delay_samples
                    );
                } else {
                    assert_eq!(
                        echo_delay_samples, lag_estimate.pre_echo_lag,
                        "dsf {down_sampling_factor}: pre-echo falls back to the peak"
                    );
                }
            }
        }
    }

    #[test]
    fn lag_not_reliable_for_uncorrelated_signals() {
        let mut rng = Random::new(1337);
        let config = EchoCanceller3Config::default();
        for &down_sampling_factor in &DOWN_SAMPLING_FACTORS {
            let sub_block_size = BLOCK_SIZE / down_sampling_factor;
            let mut buffer = init_render_buffer(sub_block_size, &mut rng, NON_SATURATING_AMPLITUDE);
            let mut capture = vec![0.0f32; sub_block_size];
            let mut filter = MatchedFilter::new(
                ApmDataDumper::new_unique(),
                detect_optimization(),
                sub_block_size,
                MATCHED_FILTER_WINDOW_SIZE_SUB_BLOCKS,
                3,
                MATCHED_FILTER_ALIGNMENT_SHIFT_SIZE_SUB_BLOCKS,
                150.0,
                config.delay.delay_estimate_smoothing,
                config.delay.delay_estimate_smoothing_delay_found,
                config.delay.delay_candidate_detection_threshold,
                /*detect_pre_echo=*/ false,
            );

            for _ in 0..200 {
                let block_start =
                    rng.rand_u32((buffer.buffer.len() - sub_block_size) as u32) as usize;
                set_block_start(&mut buffer, block_start, sub_block_size);
                randomize_sample_vector_with_amplitude(
                    &mut rng,
                    &mut capture,
                    NON_SATURATING_AMPLITUDE,
                );
                filter.update(&buffer, &capture, /*use_slow_smoothing=*/ false);
            }

            assert!(filter.best_lag_estimate().is_none());
        }
    }

    #[test]
    fn lag_not_updated_for_low_level_render() {
        let mut rng = Random::new(7);
        let config = EchoCanceller3Config::default();
        for &down_sampling_factor in &DOWN_SAMPLING_FACTORS {
            let sub_block_size = BLOCK_SIZE / down_sampling_factor;
            let mut buffer = init_render_buffer(sub_block_size, &mut rng, 149.0);
            let mut capture = vec![0.0f32; sub_block_size];
            let mut filter = MatchedFilter::new(
                ApmDataDumper::new_unique(),
                detect_optimization(),
                sub_block_size,
                MATCHED_FILTER_WINDOW_SIZE_SUB_BLOCKS,
                2,
                MATCHED_FILTER_ALIGNMENT_SHIFT_SIZE_SUB_BLOCKS,
                150.0,
                config.delay.delay_estimate_smoothing,
                config.delay.delay_estimate_smoothing_delay_found,
                config.delay.delay_candidate_detection_threshold,
                /*detect_pre_echo=*/ false,
            );

            for _ in 0..100 {
                let block_start =
                    rng.rand_u32((buffer.buffer.len() - sub_block_size) as u32) as usize;
                set_block_start(&mut buffer, block_start, sub_block_size);
                fill_capture_from_signal(
                    &buffer.buffer,
                    block_start,
                    sub_block_size,
                    sub_block_size,
                    &mut capture,
                );
                filter.update(&buffer, &capture, /*use_slow_smoothing=*/ false);
            }

            assert!(filter.best_lag_estimate().is_none());
        }
    }
}
