//! Ring buffer utility for fixed-size arrays.

/// Ring buffer for `N` arrays of type `T`, each with size `S`.
#[derive(Debug, Clone)]
pub struct RingBuffer<T, const S: usize, const N: usize>
where
    T: Copy + Default,
{
    /// Index of the least recently pushed sub-array.
    tail: usize,
    buffer: [[T; S]; N],
}

impl<T, const S: usize, const N: usize> Default for RingBuffer<T, S, N>
where
    T: Copy + Default,
{
    fn default() -> Self {
        Self {
            tail: 0,
            buffer: [[T::default(); S]; N],
        }
    }
}

impl<T, const S: usize, const N: usize> RingBuffer<T, S, N>
where
    T: Copy + Default,
{
    pub fn new() -> Self {
        assert!(S > 0, "S must be > 0");
        assert!(N > 0, "N must be > 0");
        Self::default()
    }

    /// Sets the ring buffer values to zero/default.
    pub fn reset(&mut self) {
        self.buffer = [[T::default(); S]; N];
    }

    /// Replaces the least recently pushed array with `new_values`.
    pub fn push(&mut self, new_values: &[T; S]) {
        self.buffer[self.tail] = *new_values;
        self.tail += 1;
        if self.tail == N {
            self.tail = 0;
        }
    }

    /// Returns the array for a given delay.
    ///
    /// Delay 0 returns the most recently pushed array, delay N-1 the least
    /// recently pushed array.
    pub fn get_array_view(&self, delay: usize) -> &[T; S] {
        assert!(delay < N);
        let offset = (self.tail + N - 1 - delay) % N;
        &self.buffer[offset]
    }
}

#[cfg(test)]
mod tests {
    use super::RingBuffer;

    fn test_ring_buffer_u8<const S: usize, const N: usize>() {
        let mut ring_buf = RingBuffer::<u8, S, N>::new();

        let mut pushed_array = [0u8; S];
        ring_buf.push(&pushed_array);
        assert_eq!(ring_buf.get_array_view(0), &pushed_array);

        for v in 1..=(N as u8) {
            pushed_array = [v; S];
            ring_buf.push(&pushed_array);
            assert_eq!(ring_buf.get_array_view(0), &pushed_array);

            if N > 1 {
                let prev_pushed_array = [v - 1; S];
                assert_eq!(ring_buf.get_array_view(1), &prev_pushed_array);
            }
        }

        for delay in 2..N {
            let expected_value = (N - delay) as u8;
            let expected = [expected_value; S];
            assert_eq!(ring_buf.get_array_view(delay), &expected);
        }
    }

    fn test_ring_buffer_i32<const S: usize, const N: usize>() {
        let mut ring_buf = RingBuffer::<i32, S, N>::new();

        let mut pushed_array = [0i32; S];
        ring_buf.push(&pushed_array);
        assert_eq!(ring_buf.get_array_view(0), &pushed_array);

        for v in 1..=(N as i32) {
            pushed_array = [v; S];
            ring_buf.push(&pushed_array);
            assert_eq!(ring_buf.get_array_view(0), &pushed_array);

            if N > 1 {
                let prev_pushed_array = [v - 1; S];
                assert_eq!(ring_buf.get_array_view(1), &prev_pushed_array);
            }
        }

        for delay in 2..N {
            let expected_value = (N - delay) as i32;
            let expected = [expected_value; S];
            assert_eq!(ring_buf.get_array_view(delay), &expected);
        }
    }

    fn test_ring_buffer_f32<const S: usize, const N: usize>() {
        let mut ring_buf = RingBuffer::<f32, S, N>::new();

        let mut pushed_array = [0.0f32; S];
        ring_buf.push(&pushed_array);
        assert_eq!(ring_buf.get_array_view(0), &pushed_array);

        for v in 1..=(N as i32) {
            let value = v as f32;
            pushed_array = [value; S];
            ring_buf.push(&pushed_array);
            assert_eq!(ring_buf.get_array_view(0), &pushed_array);

            if N > 1 {
                let prev_pushed_array = [(v - 1) as f32; S];
                assert_eq!(ring_buf.get_array_view(1), &prev_pushed_array);
            }
        }

        for delay in 2..N {
            let expected_value = (N - delay) as f32;
            let expected = [expected_value; S];
            assert_eq!(ring_buf.get_array_view(delay), &expected);
        }
    }

    #[test]
    fn ring_buffer_array_views() {
        const S: usize = 3;
        const N: usize = 4;
        let mut ring_buf = RingBuffer::<i32, S, N>::new();
        let pushed_array = [1i32; S];

        for _ in 0..=N {
            for i in 0..N {
                let view_i = ring_buf.get_array_view(i);
                for j in (i + 1)..N {
                    let view_j = ring_buf.get_array_view(j);
                    assert_ne!(view_i.as_ptr(), view_j.as_ptr());
                }
            }
            ring_buf.push(&pushed_array);
        }
    }

    #[test]
    fn ring_buffer_unsigned() {
        test_ring_buffer_u8::<1, 1>();
        test_ring_buffer_u8::<2, 5>();
        test_ring_buffer_u8::<5, 2>();
        test_ring_buffer_u8::<5, 5>();
    }

    #[test]
    fn ring_buffer_signed() {
        test_ring_buffer_i32::<1, 1>();
        test_ring_buffer_i32::<2, 5>();
        test_ring_buffer_i32::<5, 2>();
        test_ring_buffer_i32::<5, 5>();
    }

    #[test]
    fn ring_buffer_floating() {
        test_ring_buffer_f32::<1, 1>();
        test_ring_buffer_f32::<2, 5>();
        test_ring_buffer_f32::<5, 2>();
        test_ring_buffer_f32::<5, 5>();
    }
}
