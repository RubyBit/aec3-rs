//! Buffer for pair-wise symmetric comparisons among ring-buffer items.

/// Data structure that buffers pair-wise comparison values among `S` items,
/// updating when the oldest item is replaced.
#[derive(Debug, Clone)]
pub struct SymmetricMatrixBuffer<T, const S: usize>
where
    T: Copy + Default,
{
    // Encoded upper-right triangular matrix (excluding diagonal) in a square
    // storage to allow one contiguous move on push.
    buf: Vec<T>,
}

impl<T, const S: usize> Default for SymmetricMatrixBuffer<T, S>
where
    T: Copy + Default,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const S: usize> SymmetricMatrixBuffer<T, S>
where
    T: Copy + Default,
{
    pub fn new() -> Self {
        assert!(S > 2, "S must be > 2");
        Self {
            buf: vec![T::default(); (S - 1) * (S - 1)],
        }
    }

    /// Sets the buffer values to zero/default.
    pub fn reset(&mut self) {
        self.buf.fill(T::default());
    }

    /// Pushes comparison values between newest and older elements.
    ///
    /// `values[0]` corresponds to comparison between delay 0 and delay 1,
    /// `values[S-2]` to comparison between delay 0 and delay S-1.
    pub fn push(&mut self, values: &[T]) {
        assert_eq!(values.len(), S - 1);

        // Move lower-right (S-2)x(S-2) block one row up and one column left.
        if self.buf.len() > S {
            let len = self.buf.len();
            self.buf.copy_within(S..len, 0);
        }

        // Copy new values into the last column in reversed row order.
        for (i, value) in values.iter().copied().enumerate() {
            let index = (S - 1 - i) * (S - 1) - 1;
            self.buf[index] = value;
        }
    }

    /// Reads the comparison value between items with delays `delay1` and
    /// `delay2`. Delays must be distinct and in `[0, S-1]`.
    pub fn get_value(&self, delay1: usize, delay2: usize) -> T {
        assert!(delay1 < S && delay2 < S);
        assert_ne!(delay1, delay2, "The diagonal cannot be accessed.");

        let mut row = S - 1 - delay1;
        let mut col = S - 1 - delay2;
        if row > col {
            std::mem::swap(&mut row, &mut col);
        }

        assert!(row < S - 1);
        assert!(col >= 1 && col < S);
        let index = row * (S - 1) + (col - 1);
        self.buf[index]
    }
}

#[cfg(test)]
mod tests {
    use super::SymmetricMatrixBuffer;
    use crate::audio_processing::agc2::rnn_vad::ring_buffer::RingBuffer;

    fn check_symmetry<const S: usize>(sym_matrix_buf: &SymmetricMatrixBuffer<(i32, i32), S>) {
        for row in 0..(S - 1) {
            for col in (row + 1)..S {
                assert_eq!(
                    sym_matrix_buf.get_value(row, col),
                    sym_matrix_buf.get_value(col, row)
                );
            }
        }
    }

    fn check_pairs_with_value_exist<const S: usize>(
        sym_matrix_buf: &SymmetricMatrixBuffer<(i32, i32), S>,
        value: i32,
    ) -> bool {
        for row in 0..(S - 1) {
            for col in (row + 1)..S {
                let p = sym_matrix_buf.get_value(row, col);
                if p.0 == value || p.1 == value {
                    return true;
                }
            }
        }
        false
    }

    #[test]
    fn symmetric_matrix_buffer_use_case() {
        const RING_BUF_SIZE: usize = 10;
        let mut ring_buf = RingBuffer::<i32, 1, RING_BUF_SIZE>::new();
        let mut sym_matrix_buf = SymmetricMatrixBuffer::<(i32, i32), RING_BUF_SIZE>::new();

        for t in 1..=100 {
            let t_removed = ring_buf.get_array_view(RING_BUF_SIZE - 1)[0];
            ring_buf.push(&[t]);

            assert_eq!(t, ring_buf.get_array_view(0)[0]);

            let mut new_comparisons = [(0i32, 0i32); RING_BUF_SIZE - 1];
            for i in 0..(RING_BUF_SIZE - 1) {
                let delay = i + 1;
                let t_prev = ring_buf.get_array_view(delay)[0];
                assert_eq!(std::cmp::max(0, t - delay as i32), t_prev);
                new_comparisons[i] = (t_prev, t);
            }

            sym_matrix_buf.push(&new_comparisons);

            check_symmetry(&sym_matrix_buf);

            for delay1 in 0..(RING_BUF_SIZE - 1) {
                for delay2 in (delay1 + 1)..RING_BUF_SIZE {
                    let t1 = ring_buf.get_array_view(delay1)[0];
                    let t2 = ring_buf.get_array_view(delay2)[0];
                    assert!(t2 <= t1);
                    let p = sym_matrix_buf.get_value(delay1, delay2);
                    assert_eq!(p.0, t2);
                    assert_eq!(p.1, t1);
                }
            }

            for delay in 1..RING_BUF_SIZE {
                let t_prev = ring_buf.get_array_view(delay)[0];
                assert!(check_pairs_with_value_exist(&sym_matrix_buf, t_prev));
            }

            if t > (RING_BUF_SIZE as i32 - 1) {
                assert!(!check_pairs_with_value_exist(&sym_matrix_buf, t_removed));
            }
        }
    }
}
