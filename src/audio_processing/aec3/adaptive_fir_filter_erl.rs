use crate::audio_processing::aec3::aec3_common::{
    Aec3Optimization, FFT_LENGTH_BY_2, FFT_LENGTH_BY_2_PLUS_1,
};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::audio_processing::aec3::aec3_common::{detect_avx2, detect_sse2};
#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
use crate::audio_processing::aec3::aec3_common::detect_neon;

/// Reference implementation for accumulating an echo return loss estimate from
/// frequency responses per partition.
pub fn erl_computer(h2: &[[f32; FFT_LENGTH_BY_2_PLUS_1]], erl: &mut [f32; FFT_LENGTH_BY_2_PLUS_1]) {
    assert_eq!(erl.len(), FFT_LENGTH_BY_2_PLUS_1);
    erl.fill(0.0);
    for partition in h2 {
        for (dst, &value) in erl.iter_mut().zip(partition.iter()) {
            *dst += value;
        }
    }
}

/// Selects the optimization specific implementation for computing the ERL.
pub fn compute_erl(
    optimization: Aec3Optimization,
    h2: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
    erl: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
) {
    match optimization {
        Aec3Optimization::Avx2 => compute_erl_avx2(h2, erl),
        Aec3Optimization::Sse2 => compute_erl_sse2(h2, erl),
        Aec3Optimization::Neon => compute_erl_neon(h2, erl),
        Aec3Optimization::None => erl_computer(h2, erl),
    }
}

fn compute_erl_avx2(
    h2: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
    erl: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if detect_avx2() {
        unsafe {
            compute_erl_avx2_impl(h2, erl);
        }
        return;
    }
    erl_computer(h2, erl);
}

fn compute_erl_sse2(
    h2: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
    erl: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if detect_sse2() {
        unsafe {
            compute_erl_sse2_impl(h2, erl);
        }
        return;
    }
    erl_computer(h2, erl);
}

fn compute_erl_neon(
    h2: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
    erl: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
) {
    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    if detect_neon() {
        unsafe {
            compute_erl_neon_impl(h2, erl);
        }
        return;
    }
    erl_computer(h2, erl);
}

#[cfg(target_arch = "x86")]
use std::arch::x86::{
    _mm_add_ps, _mm_loadu_ps, _mm_storeu_ps, _mm256_add_ps, _mm256_loadu_ps, _mm256_storeu_ps,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _mm_add_ps, _mm_loadu_ps, _mm_storeu_ps, _mm256_add_ps, _mm256_loadu_ps, _mm256_storeu_ps,
};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2")]
unsafe fn compute_erl_avx2_impl(
    h2: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
    erl: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
) {
    erl.fill(0.0);
    for partition in h2 {
        let mut k = 0usize;
        while k < FFT_LENGTH_BY_2 {
            let sum = _mm256_add_ps(
                _mm256_loadu_ps(erl.as_ptr().add(k)),
                _mm256_loadu_ps(partition.as_ptr().add(k)),
            );
            _mm256_storeu_ps(erl.as_mut_ptr().add(k), sum);
            k += 8;
        }
        erl[FFT_LENGTH_BY_2] += partition[FFT_LENGTH_BY_2];
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse2")]
unsafe fn compute_erl_sse2_impl(
    h2: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
    erl: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
) {
    erl.fill(0.0);
    for partition in h2 {
        let mut k = 0usize;
        while k < FFT_LENGTH_BY_2 {
            let sum = _mm_add_ps(
                _mm_loadu_ps(erl.as_ptr().add(k)),
                _mm_loadu_ps(partition.as_ptr().add(k)),
            );
            _mm_storeu_ps(erl.as_mut_ptr().add(k), sum);
            k += 4;
        }
        erl[FFT_LENGTH_BY_2] += partition[FFT_LENGTH_BY_2];
    }
}

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{vaddq_f32, vld1q_f32, vst1q_f32};
#[cfg(target_arch = "arm")]
use std::arch::arm::{vaddq_f32, vld1q_f32, vst1q_f32};

#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn compute_erl_neon_impl(
    h2: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
    erl: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
) {
    erl.fill(0.0);
    for partition in h2 {
        let mut k = 0usize;
        while k < FFT_LENGTH_BY_2 {
            let sum = vaddq_f32(vld1q_f32(erl.as_ptr().add(k)), vld1q_f32(partition.as_ptr().add(k)));
            vst1q_f32(erl.as_mut_ptr().add(k), sum);
            k += 4;
        }
        erl[FFT_LENGTH_BY_2] += partition[FFT_LENGTH_BY_2];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erl_accumulates_energy() {
        let mut h2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; 3];
        for (p, partition) in h2.iter_mut().enumerate() {
            for (k, value) in partition.iter_mut().enumerate() {
                *value = p as f32 + k as f32 * 0.1;
            }
        }
        let mut erl = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        compute_erl(Aec3Optimization::None, &h2, &mut erl);
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            let expected = h2[0][k] + h2[1][k] + h2[2][k];
            assert!((erl[k] - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn erl_matches_scalar_across_optimizations() {
        let mut h2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; 4];
        for (p, partition) in h2.iter_mut().enumerate() {
            for (k, value) in partition.iter_mut().enumerate() {
                *value = p as f32 * 0.75 + k as f32 * 0.125;
            }
        }

        let mut expected = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        compute_erl(Aec3Optimization::None, &h2, &mut expected);

        for optimization in [
            Aec3Optimization::Sse2,
            Aec3Optimization::Avx2,
            Aec3Optimization::Neon,
        ] {
            let mut actual = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
            compute_erl(optimization, &h2, &mut actual);
            assert_eq!(expected, actual, "optimization={optimization:?}");
        }
    }
}
