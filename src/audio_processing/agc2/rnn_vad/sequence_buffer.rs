//! Linear sequence buffer for fixed-size chunk pushes.

/// Linear buffer that supports pushing fixed-size sequential chunks and reading
/// contiguous views over the buffer.
#[derive(Debug, Clone)]
pub struct SequenceBuffer<T, const S: usize, const N: usize, const M: usize = N>
where
    T: Copy + Default,
{
    buffer: [T; S],
}

impl<T, const S: usize, const N: usize, const M: usize> Default for SequenceBuffer<T, S, N, M>
where
    T: Copy + Default,
{
    fn default() -> Self {
        let mut seq = Self {
            buffer: [T::default(); S],
        };
        seq.reset();
        seq
    }
}

impl<T, const S: usize, const N: usize, const M: usize> SequenceBuffer<T, S, N, M>
where
    T: Copy + Default,
{
    pub fn new() -> Self {
        assert!(N <= S, "N must be <= S");
        assert!(M <= S, "M must be <= S");
        Self::default()
    }

    pub const fn size(&self) -> usize {
        S
    }

    pub const fn chunks_size(&self) -> usize {
        N
    }

    /// Sets all sequence-buffer values to zero/default.
    pub fn reset(&mut self) {
        self.buffer.fill(T::default());
    }

    /// Returns a view on the whole buffer.
    pub fn get_buffer_view(&self) -> &[T; S] {
        &self.buffer
    }

    /// Returns a view on the `M` most recent values of the buffer.
    pub fn get_most_recent_values_view(&self) -> &[T] {
        &self.buffer[S - M..S]
    }

    /// Shifts the buffer left by `N` items and appends `new_values` at the end.
    pub fn push(&mut self, new_values: &[T; N]) {
        if S > N {
            self.buffer.copy_within(N..S, 0);
        }
        self.buffer[S - N..S].copy_from_slice(new_values);
    }
}

#[cfg(test)]
mod tests {
    use super::SequenceBuffer;

    fn test_sequence_buffer_push_op_u8<const S: usize, const N: usize>() {
        let mut seq_buf = SequenceBuffer::<u8, S, N>::new();

        let mut chunk = [0u8; N];
        chunk.fill(1);
        seq_buf.push(&chunk);

        let required_push_ops = if S % N != 0 { S / N + 1 } else { S / N };
        for _ in 0..(required_push_ops - 1) {
            chunk.fill(0);
            seq_buf.push(&chunk);
            let max = seq_buf
                .get_buffer_view()
                .iter()
                .copied()
                .max()
                .expect("buffer must be non-empty");
            assert_eq!(1, max);
        }

        chunk.fill(0);
        seq_buf.push(&chunk);
        let max = seq_buf
            .get_buffer_view()
            .iter()
            .copied()
            .max()
            .expect("buffer must be non-empty");
        assert_eq!(0, max);

        if S > N {
            for (i, c) in chunk.iter_mut().enumerate() {
                *c = (i + 1) as u8;
            }
            seq_buf.push(&chunk);
            let last = chunk[N - 1];
            for (i, c) in chunk.iter_mut().enumerate() {
                *c = last + i as u8 + 1;
            }
            seq_buf.push(&chunk);
            assert_eq!(last, seq_buf.get_buffer_view()[S - N - 1]);
        }
    }

    fn test_sequence_buffer_push_op_i32<const S: usize, const N: usize>() {
        let mut seq_buf = SequenceBuffer::<i32, S, N>::new();

        let mut chunk = [0i32; N];
        chunk.fill(1);
        seq_buf.push(&chunk);

        let required_push_ops = if S % N != 0 { S / N + 1 } else { S / N };
        for _ in 0..(required_push_ops - 1) {
            chunk.fill(0);
            seq_buf.push(&chunk);
            let max = seq_buf
                .get_buffer_view()
                .iter()
                .copied()
                .max()
                .expect("buffer must be non-empty");
            assert_eq!(1, max);
        }

        chunk.fill(0);
        seq_buf.push(&chunk);
        let max = seq_buf
            .get_buffer_view()
            .iter()
            .copied()
            .max()
            .expect("buffer must be non-empty");
        assert_eq!(0, max);

        if S > N {
            for (i, c) in chunk.iter_mut().enumerate() {
                *c = (i + 1) as i32;
            }
            seq_buf.push(&chunk);
            let last = chunk[N - 1];
            for (i, c) in chunk.iter_mut().enumerate() {
                *c = last + i as i32 + 1;
            }
            seq_buf.push(&chunk);
            assert_eq!(last, seq_buf.get_buffer_view()[S - N - 1]);
        }
    }

    fn test_sequence_buffer_push_op_f32<const S: usize, const N: usize>() {
        let mut seq_buf = SequenceBuffer::<f32, S, N>::new();

        let mut chunk = [0.0f32; N];
        chunk.fill(1.0);
        seq_buf.push(&chunk);

        let required_push_ops = if S % N != 0 { S / N + 1 } else { S / N };
        for _ in 0..(required_push_ops - 1) {
            chunk.fill(0.0);
            seq_buf.push(&chunk);
            let max = seq_buf
                .get_buffer_view()
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            assert_eq!(1.0, max);
        }

        chunk.fill(0.0);
        seq_buf.push(&chunk);
        let max = seq_buf
            .get_buffer_view()
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(0.0, max);

        if S > N {
            for (i, c) in chunk.iter_mut().enumerate() {
                *c = (i + 1) as f32;
            }
            seq_buf.push(&chunk);
            let last = chunk[N - 1];
            for (i, c) in chunk.iter_mut().enumerate() {
                *c = last + i as f32 + 1.0;
            }
            seq_buf.push(&chunk);
            assert_eq!(last, seq_buf.get_buffer_view()[S - N - 1]);
        }
    }

    #[test]
    fn sequence_buffer_getters() {
        const BUFFER_SIZE: usize = 8;
        const CHUNK_SIZE: usize = 8;

        let mut seq_buf = SequenceBuffer::<i32, BUFFER_SIZE, CHUNK_SIZE>::new();
        assert_eq!(BUFFER_SIZE, seq_buf.size());
        assert_eq!(CHUNK_SIZE, seq_buf.chunks_size());

        let seq_buf_view = seq_buf.get_buffer_view();
        assert_eq!(0, seq_buf_view[0]);
        assert_eq!(0, seq_buf_view[seq_buf_view.len() - 1]);

        let chunk = [10, 20, 30, 40, 50, 60, 70, 80];
        seq_buf.push(&chunk);

        assert_eq!(10, *seq_buf.get_buffer_view().first().unwrap());
        assert_eq!(80, *seq_buf.get_buffer_view().last().unwrap());
    }

    #[test]
    fn sequence_buffer_push_ops_unsigned() {
        test_sequence_buffer_push_op_u8::<32, 8>();
        test_sequence_buffer_push_op_u8::<32, 16>();
        test_sequence_buffer_push_op_u8::<32, 32>();
        test_sequence_buffer_push_op_u8::<23, 7>();
    }

    #[test]
    fn sequence_buffer_push_ops_signed() {
        test_sequence_buffer_push_op_i32::<32, 8>();
        test_sequence_buffer_push_op_i32::<32, 16>();
        test_sequence_buffer_push_op_i32::<32, 32>();
        test_sequence_buffer_push_op_i32::<23, 7>();
    }

    #[test]
    fn sequence_buffer_push_ops_floating() {
        test_sequence_buffer_push_op_f32::<32, 8>();
        test_sequence_buffer_push_op_f32::<32, 16>();
        test_sequence_buffer_push_op_f32::<32, 32>();
        test_sequence_buffer_push_op_f32::<23, 7>();
    }
}
