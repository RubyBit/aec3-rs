use crate::audio_processing::aec3::aec3_common::{Aec3Optimization, FFT_LENGTH_BY_2_PLUS_1};

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
        Aec3Optimization::Sse2 | Aec3Optimization::Neon | Aec3Optimization::None => {
            erl_computer(h2, erl)
        }
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
}
