//! AGC2 test helper mirroring WebRTC `VectorFloatFrame`.

/// Multi-channel float frame with convenience methods for deinterleaved access.
pub struct VectorFloatFrame {
    channels: Vec<Vec<f32>>,
    samples_per_channel: usize,
}

impl VectorFloatFrame {
    pub fn new(num_channels: usize, samples_per_channel: usize, start_value: f32) -> Self {
        assert!(num_channels > 0);
        assert!(samples_per_channel > 0);
        let channels = (0..num_channels)
            .map(|_| vec![start_value; samples_per_channel])
            .collect();
        Self {
            channels,
            samples_per_channel,
        }
    }

    pub fn num_channels(&self) -> usize {
        self.channels.len()
    }

    pub fn samples_per_channel(&self) -> usize {
        self.samples_per_channel
    }

    /// Returns a deinterleaved mutable view equivalent to C++ `DeinterleavedView<float>`.
    pub fn view(&mut self) -> Vec<&mut [f32]> {
        self.channels
            .iter_mut()
            .map(Vec::as_mut_slice)
            .collect::<Vec<_>>()
    }

    /// Returns a deinterleaved immutable view equivalent to C++ `DeinterleavedView<const float>`.
    pub fn view_const(&self) -> Vec<&[f32]> {
        self.channels
            .iter()
            .map(Vec::as_slice)
            .collect::<Vec<_>>()
    }

    pub fn sample(&self, channel: usize, index: usize) -> f32 {
        self.channels[channel][index]
    }
}
