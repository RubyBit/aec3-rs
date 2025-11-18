use crate::audio_processing::utility::cascaded_biquad_filter::{
    BiQuadCoefficients, CascadedBiQuadFilter,
};

/// High-pass filter wrapper that applies a 2nd-order Butterworth high-pass
/// filter tuned for supported sample rates.
pub struct HighPassFilter {
    sample_rate_hz: i32,
    filters: Vec<CascadedBiQuadFilter>,
}

const K_HIGH_PASS_16K: BiQuadCoefficients = BiQuadCoefficients {
    b: [0.97261f32, -1.94523f32, 0.97261f32],
    a: [-1.94448f32, 0.94598f32],
};

const K_HIGH_PASS_32K: BiQuadCoefficients = BiQuadCoefficients {
    b: [0.98621f32, -1.97242f32, 0.98621f32],
    a: [-1.97223f32, 0.97261f32],
};

const K_HIGH_PASS_48K: BiQuadCoefficients = BiQuadCoefficients {
    b: [0.99079f32, -1.98157f32, 0.99079f32],
    a: [-1.98149f32, 0.98166f32],
};

fn choose_coefficients(sample_rate_hz: i32) -> BiQuadCoefficients {
    match sample_rate_hz {
        16_000 => K_HIGH_PASS_16K,
        32_000 => K_HIGH_PASS_32K,
        48_000 => K_HIGH_PASS_48K,
        _ => panic!("Unsupported sample rate {}", sample_rate_hz),
    }
}

impl HighPassFilter {
    pub fn new(sample_rate_hz: i32, num_channels: usize) -> Self {
        let coeffs = choose_coefficients(sample_rate_hz);
        let mut filters = Vec::with_capacity(num_channels);
        for _ in 0..num_channels {
            filters.push(CascadedBiQuadFilter::with_coefficients(coeffs, 1));
        }
        Self {
            sample_rate_hz,
            filters,
        }
    }

    /// Process multi-channel audio in-place. Each element of `audio` is a
    /// channel buffer (frame samples).
    pub fn process(&mut self, audio: &mut [Vec<f32>]) {
        assert_eq!(self.filters.len(), audio.len());
        for (ch, filter) in self.filters.iter_mut().enumerate() {
            filter.process_in_place(&mut audio[ch]);
        }
    }

    pub fn reset(&mut self) {
        for f in &mut self.filters {
            f.reset();
        }
    }

    pub fn reset_channels(&mut self, num_channels: usize) {
        let old = self.filters.len();
        if num_channels < old {
            self.filters.truncate(num_channels);
        } else if num_channels > old {
            let coeffs = choose_coefficients(self.sample_rate_hz);
            for _ in old..num_channels {
                self.filters
                    .push(CascadedBiQuadFilter::with_coefficients(coeffs, 1));
            }
        }
        // Reset existing filters
        for f in &mut self.filters {
            f.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_does_not_panic_and_preserves_length() {
        let mut filter = HighPassFilter::new(16_000, 2);
        let mut audio = vec![vec![0.0f32; 160], vec![1.0f32; 160]];
        filter.process(&mut audio);
        assert_eq!(audio[0].len(), 160);
        assert_eq!(audio[1].len(), 160);
    }

    #[test]
    fn reset_channels_increases_and_decreases() {
        let mut filter = HighPassFilter::new(16_000, 1);
        filter.reset_channels(3);
        assert_eq!(filter.filters.len(), 3);
        filter.reset_channels(1);
        assert_eq!(filter.filters.len(), 1);
    }
}
