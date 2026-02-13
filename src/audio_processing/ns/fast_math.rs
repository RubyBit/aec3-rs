//! Fast math helpers used by the NS pipeline.

use std::f32::consts::{LN_2, LOG10_E};

#[inline]
fn fast_log2f(input: f32) -> f32 {
    debug_assert!(input > 0.0);
    let bits = input.to_bits() as f32;
    bits * 1.192_092_9e-7 - 126.942_695
}

#[inline]
pub fn sqrt_fast_approximation(value: f32) -> f32 {
    value.sqrt()
}

#[inline]
pub fn pow2_approximation(power: f32) -> f32 {
    2.0f32.powf(power)
}

#[inline]
pub fn pow_approximation(x: f32, power: f32) -> f32 {
    pow2_approximation(power * fast_log2f(x))
}

#[inline]
pub fn log_approximation(x: f32) -> f32 {
    fast_log2f(x) * LN_2
}

pub fn log_approximation_slice(x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), y.len());
    for (src, dst) in x.iter().zip(y.iter_mut()) {
        *dst = log_approximation(*src);
    }
}

#[inline]
pub fn exp_approximation(x: f32) -> f32 {
    pow_approximation(10.0, x * LOG10_E)
}

pub fn exp_approximation_slice(x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), y.len());
    for (src, dst) in x.iter().zip(y.iter_mut()) {
        *dst = exp_approximation(*src);
    }
}

pub fn exp_approximation_sign_flip_slice(x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), y.len());
    for (src, dst) in x.iter().zip(y.iter_mut()) {
        *dst = exp_approximation(-*src);
    }
}
