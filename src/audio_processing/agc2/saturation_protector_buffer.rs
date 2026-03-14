//! Fixed-size ring buffer used by AGC2 saturation protector.

use crate::audio_processing::agc2::agc2_common::SATURATION_PROTECTOR_BUFFER_SIZE;

/// A circular buffer with fixed maximum capacity.
///
/// New values are pushed at the back. When full, the oldest value is dropped.
#[derive(Clone)]
pub struct SaturationProtectorBuffer {
    data: [f32; SATURATION_PROTECTOR_BUFFER_SIZE],
    next: usize,
    size: usize,
}

impl SaturationProtectorBuffer {
    pub const K_BUFFER_SIZE: usize = SATURATION_PROTECTOR_BUFFER_SIZE;

    pub fn new() -> Self {
        Self {
            data: [0.0; Self::K_BUFFER_SIZE],
            next: 0,
            size: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        Self::K_BUFFER_SIZE
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn reset(&mut self) {
        self.next = 0;
        self.size = 0;
    }

    /// Returns the oldest element currently stored.
    pub fn front(&self) -> Option<f32> {
        if self.size == 0 {
            return None;
        }
        Some(self.data[self.front_index()])
    }

    /// Pushes one value, overwriting the oldest.
    pub fn push_back(&mut self, value: f32) {
        self.data[self.next] = value;
        self.next += 1;
        if self.next == Self::K_BUFFER_SIZE {
            self.next = 0;
        }
        if self.size < Self::K_BUFFER_SIZE {
            self.size += 1;
        }
    }

    fn front_index(&self) -> usize {
        if self.size == Self::K_BUFFER_SIZE {
            self.next
        } else {
            0
        }
    }
}

impl Default for SaturationProtectorBuffer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init() {
        let b = SaturationProtectorBuffer::new();
        assert_eq!(b.size(), 0);
        assert_eq!(b.front(), None);
    }

    #[test]
    fn push_back() {
        let mut b = SaturationProtectorBuffer::new();
        const VALUE: f32 = 123.0;
        b.push_back(VALUE);
        assert_eq!(b.size(), 1);
        assert_eq!(b.front(), Some(VALUE));
    }

    #[test]
    fn reset() {
        let mut b = SaturationProtectorBuffer::new();
        b.push_back(123.0);
        b.reset();
        assert_eq!(b.size(), 0);
        assert_eq!(b.front(), None);
    }

    #[test]
    fn front_until_buffer_is_full() {
        let mut b = SaturationProtectorBuffer::new();
        const VALUE: f32 = 123.0;
        b.push_back(VALUE);
        for i in 1..b.capacity() {
            assert_eq!(b.front(), Some(VALUE));
            b.push_back(VALUE + i as f32);
        }
    }

    #[test]
    fn front_is_delayed() {
        let mut b = SaturationProtectorBuffer::new();

        // Fill the buffer.
        for i in 0..b.capacity() {
            b.push_back(i as f32);
        }

        // Ring buffer should now behave as a shift register with delay=capacity.
        for i in b.capacity()..(2 * b.capacity() + 1) {
            assert_eq!(b.front(), Some((i - b.capacity()) as f32));
            b.push_back(i as f32);
        }
    }
}
