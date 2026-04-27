//! Vector math helpers for RNN-VAD.

use crate::audio_processing::agc2::cpu_features::AvailableCpuFeatures;

/// Provides mathematical operations over vectors with architecture-dispatch
/// structure equivalent to the WebRTC implementation.
#[derive(Debug, Copy, Clone)]
pub struct VectorMath {
    cpu_features: AvailableCpuFeatures,
}

impl VectorMath {
    pub fn new(cpu_features: AvailableCpuFeatures) -> Self {
        Self { cpu_features }
    }

    /// Computes the dot product between two equally sized vectors.
    pub fn dot_product(&self, x: &[f32], y: &[f32]) -> f32 {
        assert_eq!(x.len(), y.len());

        if self.cpu_features.avx2 {
            return self.dot_product_avx2(x, y);
        }
        if self.cpu_features.sse2 {
            return self.dot_product_sse2(x, y);
        }
        if self.cpu_features.neon {
            return self.dot_product_neon(x, y);
        }
        self.dot_product_scalar(x, y)
    }

    fn dot_product_scalar(&self, x: &[f32], y: &[f32]) -> f32 {
        x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
    }

    fn dot_product_avx2(&self, x: &[f32], y: &[f32]) -> f32 {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            dot_product_avx2_impl(x, y)
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            self.dot_product_scalar(x, y)
        }
    }

    fn dot_product_sse2(&self, x: &[f32], y: &[f32]) -> f32 {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            dot_product_sse2_impl(x, y)
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            self.dot_product_scalar(x, y)
        }
    }

    fn dot_product_neon(&self, x: &[f32], y: &[f32]) -> f32 {
        #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
        unsafe {
            dot_product_neon_impl(x, y)
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
        {
            self.dot_product_scalar(x, y)
        }
    }
}

#[cfg(target_arch = "x86")]
use std::arch::x86::{
    __m128, _mm_add_ps, _mm_cvtss_f32, _mm_loadu_ps, _mm_movehl_ps, _mm_mul_ps, _mm_setzero_ps,
    _mm_shuffle_ps, _mm256_add_ps, _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_loadu_ps,
    _mm256_mul_ps, _mm256_setzero_ps,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m128, _mm_add_ps, _mm_cvtss_f32, _mm_loadu_ps, _mm_movehl_ps, _mm_mul_ps, _mm_setzero_ps,
    _mm_shuffle_ps, _mm256_add_ps, _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_loadu_ps,
    _mm256_mul_ps, _mm256_setzero_ps,
};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn reduce_m128(sum: __m128) -> f32 {
    let mut low = sum;
    let mut high = _mm_movehl_ps(sum, sum);
    low = _mm_add_ps(low, high);
    high = _mm_shuffle_ps(low, low, 1);
    low = _mm_add_ps(low, high);
    _mm_cvtss_f32(low)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2")]
unsafe fn dot_product_avx2_impl(x: &[f32], y: &[f32]) -> f32 {
    let mut accumulator = _mm256_setzero_ps();
    let len = x.len();
    let incomplete_block_index = len & !7;

    let mut i = 0usize;
    while i < incomplete_block_index {
        let x_i = _mm256_loadu_ps(x.as_ptr().add(i));
        let y_i = _mm256_loadu_ps(y.as_ptr().add(i));
        accumulator = _mm256_add_ps(accumulator, _mm256_mul_ps(x_i, y_i));
        i += 8;
    }

    let high = _mm256_extractf128_ps(accumulator, 1);
    let low = _mm256_castps256_ps128(accumulator);
    let mut dot_product = reduce_m128(_mm_add_ps(low, high));
    while i < len {
        dot_product += x[i] * y[i];
        i += 1;
    }
    dot_product
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse2")]
unsafe fn dot_product_sse2_impl(x: &[f32], y: &[f32]) -> f32 {
    let mut accumulator = _mm_setzero_ps();
    let len = x.len();
    let incomplete_block_index = len & !3;

    let mut i = 0usize;
    while i < incomplete_block_index {
        let x_i = _mm_loadu_ps(x.as_ptr().add(i));
        let y_i = _mm_loadu_ps(y.as_ptr().add(i));
        accumulator = _mm_add_ps(accumulator, _mm_mul_ps(x_i, y_i));
        i += 4;
    }

    let mut dot_product = reduce_m128(accumulator);
    while i < len {
        dot_product += x[i] * y[i];
        i += 1;
    }
    dot_product
}

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    float32x4_t, vadd_f32, vaddq_f32, vdupq_n_f32, vget_high_f32, vget_low_f32, vld1q_f32,
    vmulq_f32, vpadd_f32, vst1_f32,
};
#[cfg(target_arch = "arm")]
use std::arch::arm::{
    float32x4_t, vadd_f32, vaddq_f32, vdupq_n_f32, vget_high_f32, vget_low_f32, vld1q_f32,
    vmulq_f32, vpadd_f32, vst1_f32,
};

#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn reduce_f32x4(sum: float32x4_t) -> f32 {
    let pairwise = vadd_f32(vget_low_f32(sum), vget_high_f32(sum));
    let reduced = vpadd_f32(pairwise, pairwise);
    let mut result = [0.0f32; 2];
    vst1_f32(result.as_mut_ptr(), reduced);
    result[0]
}

#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn dot_product_neon_impl(x: &[f32], y: &[f32]) -> f32 {
    let mut accumulator = vdupq_n_f32(0.0);
    let len = x.len();
    let incomplete_block_index = len & !3;

    let mut i = 0usize;
    while i < incomplete_block_index {
        let x_i = vld1q_f32(x.as_ptr().add(i));
        let y_i = vld1q_f32(y.as_ptr().add(i));
        accumulator = vaddq_f32(accumulator, vmulq_f32(x_i, y_i));
        i += 4;
    }

    let mut dot_product = reduce_f32x4(accumulator);
    while i < len {
        dot_product += x[i] * y[i];
        i += 1;
    }
    dot_product
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_processing::agc2::cpu_features::get_available_cpu_features;

    const X: [f32; 19] = [
        0.31593041,
        0.9350786,
        -0.25252445,
        -0.86956251,
        -0.9673632,
        0.54571901,
        -0.72504495,
        -0.79509912,
        -0.25525012,
        -0.73340473,
        0.15747377,
        -0.04370565,
        0.76135145,
        -0.57239645,
        0.68616848,
        0.3740298,
        0.34710799,
        -0.92207423,
        0.10738454,
    ];
    const SIZE_OF_X_SUBSPAN: usize = 16;
    const ENERGY_OF_X: f32 = 7.315563958160327;
    const ENERGY_OF_X_SUBSPAN: f32 = 6.333327669592963;

    fn get_cpu_features_to_test() -> Vec<AvailableCpuFeatures> {
        let mut v = vec![AvailableCpuFeatures::new(false, false, false)];
        let available = get_available_cpu_features();

        if available.avx2 {
            v.push(AvailableCpuFeatures::new(false, true, false));
        }
        if available.sse2 {
            v.push(AvailableCpuFeatures::new(true, false, false));
        }
        if available.neon {
            v.push(AvailableCpuFeatures::new(false, false, true));
        }
        v
    }

    #[test]
    fn test_dot_product() {
        for cpu_features in get_cpu_features_to_test() {
            let vector_math = VectorMath::new(cpu_features);
            let energy = vector_math.dot_product(&X, &X);
            let energy_subspan =
                vector_math.dot_product(&X[..SIZE_OF_X_SUBSPAN], &X[..SIZE_OF_X_SUBSPAN]);

            assert!(
                (energy - ENERGY_OF_X).abs() < 1e-6,
                "cpu_features={cpu_features}, got={energy}, expected={ENERGY_OF_X}"
            );
            assert!(
                (energy_subspan - ENERGY_OF_X_SUBSPAN).abs() < 1e-6,
                "cpu_features={cpu_features}, got={energy_subspan}, expected={ENERGY_OF_X_SUBSPAN}"
            );
        }
    }
}
