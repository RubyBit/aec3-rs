use crate::test_support::random::Random;

/// Randomizes the samples in `v` using the same scaling as WebRTC's
/// `RandomizeSampleVector` helper.
pub fn randomize_sample_vector(rng: &mut Random, v: &mut [f32]) {
    randomize_sample_vector_with_amplitude(rng, v, 32767.0);
}

/// Randomizes the samples in `v` with a configurable amplitude, mirroring the
/// behavior of the reference helper in `echo_canceller_test_tools.cc`.
pub fn randomize_sample_vector_with_amplitude(rng: &mut Random, v: &mut [f32], amplitude: f32) {
    for sample in v.iter_mut() {
        let draw: f32 = rng.rand_float();
        *sample = 2.0 * amplitude * draw - amplitude;
    }
}

/// Delay buffer that mimics `DelayBuffer<T>` from the WebRTC test helpers.
#[derive(Clone, Debug)]
pub struct DelayBuffer<T> {
    buffer: Vec<T>,
    next_insert_index: usize,
}

impl<T: Copy + Default> DelayBuffer<T> {
    /// Creates a new delay buffer capable of delaying `delay_samples` elements.
    pub fn new(delay_samples: usize) -> Self {
        Self {
            buffer: vec![T::default(); delay_samples],
            next_insert_index: 0,
        }
    }

    /// Delays the vector `x` and writes the delayed samples into `x_delayed`.
    ///
    /// When the configured delay is zero, the input is copied directly to the
    /// output, matching the behavior of the C++ helper.
    pub fn delay(&mut self, x: &[T], x_delayed: &mut [T]) {
        assert_eq!(x.len(), x_delayed.len());
        if self.buffer.is_empty() {
            x_delayed.copy_from_slice(x);
            return;
        }

        for (idx, &sample) in x.iter().enumerate() {
            let delayed = self.buffer[self.next_insert_index];
            x_delayed[idx] = delayed;
            self.buffer[self.next_insert_index] = sample;
            self.next_insert_index += 1;
            if self.next_insert_index == self.buffer.len() {
                self.next_insert_index = 0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn randomize_sample_vector_uses_full_range() {
        // Match the original test seed used with `StdRng::seed_from_u64(1234)`.
        let mut rng = Random::new(1234);
        let mut samples = [0.0f32; 8];
        randomize_sample_vector_with_amplitude(&mut rng, &mut samples, 1.0);
        assert!(samples.iter().any(|s| *s > 0.5));
        assert!(samples.iter().any(|s| *s < -0.5));
    }

    #[test]
    fn delay_buffer_zero_delay_copies_input() {
        let mut delay: DelayBuffer<f32> = DelayBuffer::new(0);
        let input = [1.0f32, 2.0, 3.0];
        let mut output = [0.0f32; 3];
        delay.delay(&input, &mut output);
        assert_eq!(input, output);
    }

    #[test]
    fn delay_buffer_applies_delay() {
        let mut delay: DelayBuffer<i32> = DelayBuffer::new(3);
        let input = [1, 2, 3, 4, 5, 6];
        let mut aggregated = Vec::new();
        for chunk in input.chunks(2) {
            let mut delayed = [0; 2];
            delay.delay(chunk, &mut delayed);
            aggregated.extend_from_slice(&delayed);
        }
        assert_eq!(vec![0, 0, 0, 1, 2, 3], aggregated);
    }
}
