use crate::audio_processing::aec3::aec3_common::Aec3Optimization;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::audio_processing::aec3::aec3_common::{detect_avx2, detect_sse2};
#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
use crate::audio_processing::aec3::aec3_common::detect_neon;

#[derive(Debug, Copy, Clone)]
pub(crate) struct VectorMath {
    optimization: Aec3Optimization,
}

impl VectorMath {
    pub(crate) fn new(optimization: Aec3Optimization) -> Self {
        Self { optimization }
    }

    pub(crate) fn sqrt(&self, x: &mut [f32]) {
        match self.optimization {
            Aec3Optimization::Avx2 => self.sqrt_avx2(x),
            Aec3Optimization::Sse2 => self.sqrt_sse2(x),
            Aec3Optimization::Neon => self.sqrt_neon(x),
            Aec3Optimization::None => sqrt_scalar(x),
        }
    }

    pub(crate) fn multiply(&self, x: &[f32], y: &[f32], z: &mut [f32]) {
        assert_eq!(x.len(), y.len());
        assert_eq!(x.len(), z.len());
        match self.optimization {
            Aec3Optimization::Avx2 => self.multiply_avx2(x, y, z),
            Aec3Optimization::Sse2 => self.multiply_sse2(x, y, z),
            Aec3Optimization::Neon => self.multiply_neon(x, y, z),
            Aec3Optimization::None => multiply_scalar(x, y, z),
        }
    }

    pub(crate) fn accumulate(&self, x: &[f32], z: &mut [f32]) {
        assert_eq!(x.len(), z.len());
        match self.optimization {
            Aec3Optimization::Avx2 => self.accumulate_avx2(x, z),
            Aec3Optimization::Sse2 => self.accumulate_sse2(x, z),
            Aec3Optimization::Neon => self.accumulate_neon(x, z),
            Aec3Optimization::None => accumulate_scalar(x, z),
        }
    }

    fn sqrt_avx2(&self, x: &mut [f32]) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if detect_avx2() {
            unsafe {
                sqrt_avx2_impl(x);
            }
            return;
        }
        sqrt_scalar(x);
    }

    fn sqrt_sse2(&self, x: &mut [f32]) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if detect_sse2() {
            unsafe {
                sqrt_sse2_impl(x);
            }
            return;
        }
        sqrt_scalar(x);
    }

    fn sqrt_neon(&self, x: &mut [f32]) {
        #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
        if detect_neon() {
            unsafe {
                sqrt_neon_impl(x);
            }
            return;
        }
        sqrt_scalar(x);
    }

    fn multiply_avx2(&self, x: &[f32], y: &[f32], z: &mut [f32]) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if detect_avx2() {
            unsafe {
                multiply_avx2_impl(x, y, z);
            }
            return;
        }
        multiply_scalar(x, y, z);
    }

    fn multiply_sse2(&self, x: &[f32], y: &[f32], z: &mut [f32]) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if detect_sse2() {
            unsafe {
                multiply_sse2_impl(x, y, z);
            }
            return;
        }
        multiply_scalar(x, y, z);
    }

    fn multiply_neon(&self, x: &[f32], y: &[f32], z: &mut [f32]) {
        #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
        if detect_neon() {
            unsafe {
                multiply_neon_impl(x, y, z);
            }
            return;
        }
        multiply_scalar(x, y, z);
    }

    fn accumulate_avx2(&self, x: &[f32], z: &mut [f32]) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if detect_avx2() {
            unsafe {
                accumulate_avx2_impl(x, z);
            }
            return;
        }
        accumulate_scalar(x, z);
    }

    fn accumulate_sse2(&self, x: &[f32], z: &mut [f32]) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if detect_sse2() {
            unsafe {
                accumulate_sse2_impl(x, z);
            }
            return;
        }
        accumulate_scalar(x, z);
    }

    fn accumulate_neon(&self, x: &[f32], z: &mut [f32]) {
        #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
        if detect_neon() {
            unsafe {
                accumulate_neon_impl(x, z);
            }
            return;
        }
        accumulate_scalar(x, z);
    }
}

fn sqrt_scalar(x: &mut [f32]) {
    for sample in x {
        *sample = sample.sqrt();
    }
}

fn multiply_scalar(x: &[f32], y: &[f32], z: &mut [f32]) {
    for ((dst, &lhs), &rhs) in z.iter_mut().zip(x.iter()).zip(y.iter()) {
        *dst = lhs * rhs;
    }
}

fn accumulate_scalar(x: &[f32], z: &mut [f32]) {
    for (dst, &src) in z.iter_mut().zip(x.iter()) {
        *dst += src;
    }
}

#[cfg(target_arch = "x86")]
use std::arch::x86::{
    _mm_add_ps, _mm_loadu_ps, _mm_mul_ps, _mm_sqrt_ps, _mm_storeu_ps, _mm256_add_ps,
    _mm256_loadu_ps, _mm256_mul_ps, _mm256_sqrt_ps, _mm256_storeu_ps,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _mm_add_ps, _mm_loadu_ps, _mm_mul_ps, _mm_sqrt_ps, _mm_storeu_ps, _mm256_add_ps,
    _mm256_loadu_ps, _mm256_mul_ps, _mm256_sqrt_ps, _mm256_storeu_ps,
};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2")]
unsafe fn sqrt_avx2_impl(x: &mut [f32]) {
    let vector_limit = x.len() & !7;
    let mut j = 0usize;
    while j < vector_limit {
        let g = _mm256_sqrt_ps(_mm256_loadu_ps(x.as_ptr().add(j)));
        _mm256_storeu_ps(x.as_mut_ptr().add(j), g);
        j += 8;
    }
    for sample in &mut x[j..] {
        *sample = sample.sqrt();
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse2")]
unsafe fn sqrt_sse2_impl(x: &mut [f32]) {
    let vector_limit = x.len() & !3;
    let mut j = 0usize;
    while j < vector_limit {
        let g = _mm_sqrt_ps(_mm_loadu_ps(x.as_ptr().add(j)));
        _mm_storeu_ps(x.as_mut_ptr().add(j), g);
        j += 4;
    }
    for sample in &mut x[j..] {
        *sample = sample.sqrt();
    }
}

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    vaddq_f32, vld1q_f32, vmulq_f32, vsqrtq_f32, vst1q_f32,
};
#[cfg(target_arch = "arm")]
use std::arch::arm::{vaddq_f32, vld1q_f32, vmulq_f32, vst1q_f32};

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn sqrt_neon_impl(x: &mut [f32]) {
    let vector_limit = x.len() & !3;
    let mut j = 0usize;
    while j < vector_limit {
        let g = vsqrtq_f32(vld1q_f32(x.as_ptr().add(j)));
        vst1q_f32(x.as_mut_ptr().add(j), g);
        j += 4;
    }
    for sample in &mut x[j..] {
        *sample = sample.sqrt();
    }
}

#[cfg(target_arch = "arm")]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn sqrt_neon_impl(x: &mut [f32]) {
    sqrt_scalar(x);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2")]
unsafe fn multiply_avx2_impl(x: &[f32], y: &[f32], z: &mut [f32]) {
    let vector_limit = x.len() & !7;
    let mut j = 0usize;
    while j < vector_limit {
        let z_j = _mm256_mul_ps(
            _mm256_loadu_ps(x.as_ptr().add(j)),
            _mm256_loadu_ps(y.as_ptr().add(j)),
        );
        _mm256_storeu_ps(z.as_mut_ptr().add(j), z_j);
        j += 8;
    }
    multiply_scalar(&x[j..], &y[j..], &mut z[j..]);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse2")]
unsafe fn multiply_sse2_impl(x: &[f32], y: &[f32], z: &mut [f32]) {
    let vector_limit = x.len() & !3;
    let mut j = 0usize;
    while j < vector_limit {
        let z_j = _mm_mul_ps(_mm_loadu_ps(x.as_ptr().add(j)), _mm_loadu_ps(y.as_ptr().add(j)));
        _mm_storeu_ps(z.as_mut_ptr().add(j), z_j);
        j += 4;
    }
    multiply_scalar(&x[j..], &y[j..], &mut z[j..]);
}

#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn multiply_neon_impl(x: &[f32], y: &[f32], z: &mut [f32]) {
    let vector_limit = x.len() & !3;
    let mut j = 0usize;
    while j < vector_limit {
        let z_j = vmulq_f32(vld1q_f32(x.as_ptr().add(j)), vld1q_f32(y.as_ptr().add(j)));
        vst1q_f32(z.as_mut_ptr().add(j), z_j);
        j += 4;
    }
    multiply_scalar(&x[j..], &y[j..], &mut z[j..]);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2")]
unsafe fn accumulate_avx2_impl(x: &[f32], z: &mut [f32]) {
    let vector_limit = x.len() & !7;
    let mut j = 0usize;
    while j < vector_limit {
        let z_j = _mm256_add_ps(
            _mm256_loadu_ps(z.as_ptr().add(j)),
            _mm256_loadu_ps(x.as_ptr().add(j)),
        );
        _mm256_storeu_ps(z.as_mut_ptr().add(j), z_j);
        j += 8;
    }
    accumulate_scalar(&x[j..], &mut z[j..]);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse2")]
unsafe fn accumulate_sse2_impl(x: &[f32], z: &mut [f32]) {
    let vector_limit = x.len() & !3;
    let mut j = 0usize;
    while j < vector_limit {
        let z_j = _mm_add_ps(_mm_loadu_ps(z.as_ptr().add(j)), _mm_loadu_ps(x.as_ptr().add(j)));
        _mm_storeu_ps(z.as_mut_ptr().add(j), z_j);
        j += 4;
    }
    accumulate_scalar(&x[j..], &mut z[j..]);
}

#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn accumulate_neon_impl(x: &[f32], z: &mut [f32]) {
    let vector_limit = x.len() & !3;
    let mut j = 0usize;
    while j < vector_limit {
        let z_j = vaddq_f32(vld1q_f32(z.as_ptr().add(j)), vld1q_f32(x.as_ptr().add(j)));
        vst1q_f32(z.as_mut_ptr().add(j), z_j);
        j += 4;
    }
    accumulate_scalar(&x[j..], &mut z[j..]);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn optimizations_to_test() -> [Aec3Optimization; 4] {
        [
            Aec3Optimization::None,
            Aec3Optimization::Sse2,
            Aec3Optimization::Avx2,
            Aec3Optimization::Neon,
        ]
    }

    #[test]
    fn sqrt_matches_scalar() {
        let input = [
            0.0f32, 0.01, 0.25, 0.5, 1.0, 2.0, 3.5, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0,
        ];
        let mut expected = input;
        sqrt_scalar(&mut expected);

        for optimization in optimizations_to_test() {
            let mut actual = input;
            VectorMath::new(optimization).sqrt(&mut actual);
            for (&lhs, &rhs) in actual.iter().zip(expected.iter()) {
                assert!((lhs - rhs).abs() < 1e-5, "optimization={optimization:?}");
            }
        }
    }

    #[test]
    fn multiply_matches_scalar() {
        let x = [1.0f32, -2.0, 3.5, 0.5, 2.0, -4.0, 7.0, 8.0, 0.25];
        let y = [2.0f32, 4.0, -1.0, 0.0, -2.0, 0.5, 3.0, -1.5, 6.0];
        let mut expected = [0.0f32; 9];
        multiply_scalar(&x, &y, &mut expected);

        for optimization in optimizations_to_test() {
            let mut actual = [0.0f32; 9];
            VectorMath::new(optimization).multiply(&x, &y, &mut actual);
            assert_eq!(actual, expected, "optimization={optimization:?}");
        }
    }

    #[test]
    fn accumulate_matches_scalar() {
        let x = [1.0f32, -2.0, 3.5, 0.5, 2.0, -4.0, 7.0, 8.0, 0.25];
        let mut expected = [0.5f32, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        accumulate_scalar(&x, &mut expected);

        for optimization in optimizations_to_test() {
            let mut actual = [0.5f32; 9];
            VectorMath::new(optimization).accumulate(&x, &mut actual);
            assert_eq!(actual, expected, "optimization={optimization:?}");
        }
    }
}
