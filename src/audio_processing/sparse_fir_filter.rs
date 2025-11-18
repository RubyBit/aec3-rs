//! Sparse finite impulse response filter implementation.

#[derive(Debug, Clone)]
pub struct SparseFIRFilter {
    sparsity: usize,
    offset: usize,
    coeffs: Vec<f32>,
    state: Vec<f32>,
}

impl SparseFIRFilter {
    pub fn new(nonzero_coeffs: &[f32], sparsity: usize, offset: usize) -> Self {
        assert!(!nonzero_coeffs.is_empty());
        assert!(sparsity >= 1);
        let state_len = if nonzero_coeffs.len() > 1 {
            sparsity * (nonzero_coeffs.len() - 1) + offset
        } else {
            offset
        };
        Self {
            sparsity,
            offset,
            coeffs: nonzero_coeffs.to_vec(),
            state: vec![0.0; state_len],
        }
    }

    pub fn filter(&mut self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), output.len());
        for (i, out_sample) in output.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            let mut tap = 0;
            while tap < self.coeffs.len() {
                let idx = tap * self.sparsity + self.offset;
                if i >= idx {
                    acc += input[i - idx] * self.coeffs[tap];
                } else {
                    let state_index = i + (self.coeffs.len() - tap - 1) * self.sparsity;
                    if state_index < self.state.len() {
                        acc += self.state[state_index] * self.coeffs[tap];
                    }
                }
                tap += 1;
            }
            *out_sample = acc;
        }

        if self.state.is_empty() {
            return;
        }
        let state_len = self.state.len();
        if input.len() >= state_len {
            self.state
                .copy_from_slice(&input[input.len() - state_len..input.len()]);
        } else {
            self.state.copy_within(input.len()..state_len, 0);
            let tail_start = state_len - input.len();
            self.state[tail_start..].copy_from_slice(input);
        }
    }
}
