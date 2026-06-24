#[cfg(target_arch = "aarch64")]
use crate::audio_processing::aec3::aec3_common::detect_neon;
use crate::audio_processing::aec3::aec3_common::{Aec3Optimization, BLOCK_SIZE};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::audio_processing::aec3::aec3_common::{detect_avx2, detect_sse2};
use crate::audio_processing::aec3::downsampled_render_buffer::DownsampledRenderBuffer;
use crate::audio_processing::logging::apm_data_dumper::{ApmDataDumper, DiagnosticLevel};
use std::cmp::Ordering;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{vaddq_f32, vdupq_n_f32, vld1q_f32, vmulq_f32, vst1q_f32};
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

/// Stores properties for the lag estimate corresponding to a particular signal shift.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LagEstimate {
    pub accuracy: f32,
    pub reliable: bool,
    pub lag: usize,
    pub updated: bool,
}

impl LagEstimate {
    pub fn new(accuracy: f32, reliable: bool, lag: usize, updated: bool) -> Self {
        Self {
            accuracy,
            reliable,
            lag,
            updated,
        }
    }
}

pub struct MatchedFilter {
    data_dumper: ApmDataDumper,
    optimization: Aec3Optimization,
    sub_block_size: usize,
    filter_intra_lag_shift: usize,
    filters: Vec<Vec<f32>>,
    lag_estimates: Vec<LagEstimate>,
    excitation_limit: f32,
    smoothing: f32,
    matching_filter_threshold: f32,
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
        smoothing: f32,
        matching_filter_threshold: f32,
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
        let lag_estimates = vec![LagEstimate::default(); num_matched_filters];

        Self {
            data_dumper,
            optimization,
            sub_block_size,
            filter_intra_lag_shift: alignment_shift_sub_blocks * sub_block_size,
            filters,
            lag_estimates,
            excitation_limit,
            smoothing,
            matching_filter_threshold,
        }
    }

    pub fn reset(&mut self) {
        for filter in &mut self.filters {
            filter.fill(0.0);
        }
        for estimate in &mut self.lag_estimates {
            *estimate = LagEstimate::default();
        }
    }

    pub fn update(&mut self, render_buffer: &DownsampledRenderBuffer, capture: &[f32]) {
        if self.filters.is_empty() {
            return;
        }
        assert_eq!(self.sub_block_size, capture.len());
        let render_samples = &render_buffer.buffer;
        assert!(!render_samples.is_empty());

        let x2_sum_threshold =
            self.filters[0].len() as f32 * self.excitation_limit * self.excitation_limit;
        let error_sum_anchor = capture.iter().map(|v| v * v).sum::<f32>();
        let buffer_size = render_samples.len();
        let mut alignment_shift = 0usize;

        for (index, filter) in self.filters.iter_mut().enumerate() {
            let mut start = render_buffer.read + alignment_shift + self.sub_block_size - 1;
            start %= buffer_size;

            let result = match self.optimization {
                Aec3Optimization::Avx2 => matched_filter_core_avx2(
                    start,
                    x2_sum_threshold,
                    self.smoothing,
                    render_samples,
                    capture,
                    filter,
                ),
                Aec3Optimization::Sse2 => matched_filter_core_sse2(
                    start,
                    x2_sum_threshold,
                    self.smoothing,
                    render_samples,
                    capture,
                    filter,
                ),
                Aec3Optimization::Neon => matched_filter_core_neon(
                    start,
                    x2_sum_threshold,
                    self.smoothing,
                    render_samples,
                    capture,
                    filter,
                ),
                Aec3Optimization::None => matched_filter_core(
                    start,
                    x2_sum_threshold,
                    self.smoothing,
                    render_samples,
                    capture,
                    filter,
                ),
            };

            let peak_index = Self::detect_peak(filter);
            let absolute_lag = peak_index + alignment_shift;
            let reliable = peak_index > 2
                && peak_index + 10 < filter.len()
                && result.error_sum < self.matching_filter_threshold * error_sum_anchor;

            self.lag_estimates[index] = LagEstimate::new(
                error_sum_anchor - result.error_sum,
                reliable,
                absolute_lag,
                result.filters_updated,
            );

            self.data_dumper.dump_raw_f32_slice(
                DiagnosticLevel::DeepDebug,
                &format!("aec3_correlator_{}_h", index),
                filter,
            );

            alignment_shift += self.filter_intra_lag_shift;
        }
    }

    pub fn lag_estimates(&self) -> &[LagEstimate] {
        &self.lag_estimates
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
            let mut render = vec![vec![vec![0.0f32; BLOCK_SIZE]; num_channels]; num_bands];
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
                    cfg.delay.delay_candidate_detection_threshold,
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
                                &mut render[band][ch],
                                NON_SATURATING_AMPLITUDE,
                            );
                        }
                    }

                    // Delay render into capture (only use band 0, channel 0 like the ref).
                    signal_delay_buffer.delay(&render[0][0], &mut capture[0]);

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
                    );
                }

                let lag_estimates = filter.lag_estimates();

                // Determine expected most accurate index using same logic as ref.
                let mut expected_most_accurate: Option<usize> = None;
                let mut alignment_shift_sub_blocks = 0usize;
                for k in 0..cfg.delay.num_filters {
                    if (alignment_shift_sub_blocks + 3 * MATCHED_FILTER_WINDOW_SIZE_SUB_BLOCKS / 4)
                        * sub_block_size
                        > delay_samples
                    {
                        expected_most_accurate = Some(if k > 0 { k - 1 } else { 0 });
                        break;
                    }
                    alignment_shift_sub_blocks += MATCHED_FILTER_ALIGNMENT_SHIFT_SIZE_SUB_BLOCKS;
                }
                let expected = expected_most_accurate
                    .unwrap_or_else(|| cfg.delay.num_filters.saturating_sub(1));

                // Check accuracy ordering / reliability similar to C++ test.
                for k in 0..lag_estimates.len() {
                    if k != expected && k != expected + 1 {
                        assert!(
                            lag_estimates[expected].accuracy > lag_estimates[k].accuracy
                                || !lag_estimates[k].reliable
                                || !lag_estimates[expected].reliable
                        );
                    }
                }

                for le in lag_estimates.iter() {
                    assert!(le.updated, "all lag estimates should be updated");
                }

                // At least one of the expected or its neighbor should be reliable
                // (matches the C++ reference assertions).
                let next_idx = std::cmp::min(expected + 1, lag_estimates.len() - 1);
                assert!(
                    lag_estimates[expected].reliable || lag_estimates[next_idx].reliable,
                    "expected most-accurate or neighbor must be reliable"
                );

                if lag_estimates[expected].reliable {
                    assert_eq!(delay_samples, lag_estimates[expected].lag);
                } else {
                    assert_eq!(delay_samples, lag_estimates[next_idx].lag);
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
                config.delay.delay_candidate_detection_threshold,
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
                filter.update(&buffer, &capture);
            }

            assert!(
                filter
                    .lag_estimates()
                    .iter()
                    .all(|estimate| !estimate.reliable)
            );
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
                config.delay.delay_candidate_detection_threshold,
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
                filter.update(&buffer, &capture);
            }

            assert!(
                filter
                    .lag_estimates()
                    .iter()
                    .all(|estimate| !estimate.reliable && !estimate.updated)
            );
        }
    }

    #[test]
    fn number_of_lag_estimates_matches_configuration() {
        let config = EchoCanceller3Config::default();
        for &down_sampling_factor in &DOWN_SAMPLING_FACTORS {
            let sub_block_size = BLOCK_SIZE / down_sampling_factor;
            for num_filters in 0..10usize {
                let filter = MatchedFilter::new(
                    ApmDataDumper::new_unique(),
                    detect_optimization(),
                    sub_block_size,
                    32,
                    num_filters,
                    1,
                    150.0,
                    config.delay.delay_estimate_smoothing,
                    config.delay.delay_candidate_detection_threshold,
                );
                assert_eq!(num_filters, filter.lag_estimates().len());
            }
        }
    }
}
