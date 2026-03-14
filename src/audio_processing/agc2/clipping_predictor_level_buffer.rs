//! Circular level buffer for clipping prediction mirrored from WebRTC AGC2.

/// A circular buffer to store frame-wise levels for clipping prediction.
///
/// The implementation is intentionally simple and not optimized for large
/// capacities, matching the C++ reference behavior.
pub struct ClippingPredictorLevelBuffer {
    tail: i32,
    size: i32,
    data: Vec<Level>,
}

/// Frame-level statistics used by the clipping predictor.
#[derive(Clone, Copy, Debug)]
pub struct Level {
    pub average: f32,
    pub max: f32,
}

impl PartialEq for Level {
    fn eq(&self, other: &Self) -> bool {
        const EPSILON: f32 = 1e-6;
        (self.average - other.average).abs() < EPSILON && (self.max - other.max).abs() < EPSILON
    }
}

impl ClippingPredictorLevelBuffer {
    /// Recommended maximum capacity. Larger values are allowed but not optimized.
    pub const MAX_CAPACITY: i32 = 100;

    /// Creates a buffer with capacity `max(1, capacity)`.
    pub fn new(capacity: i32) -> Self {
        let clamped_capacity = capacity.max(1) as usize;
        let data = vec![
            Level {
                average: 0.0,
                max: 0.0,
            };
            clamped_capacity
        ];
        Self {
            tail: -1,
            size: 0,
            data,
        }
    }

    pub fn reset(&mut self) {
        self.tail = -1;
        self.size = 0;
    }

    pub fn size(&self) -> i32 {
        self.size
    }

    pub fn capacity(&self) -> i32 {
        self.data.len() as i32
    }

    /// Adds `level`; when full, replaces the least recently pushed item.
    pub fn push(&mut self, level: Level) {
        self.tail += 1;
        if self.tail == self.capacity() {
            self.tail = 0;
        }
        if self.size < self.capacity() {
            self.size += 1;
        }
        self.data[self.tail as usize] = level;
    }

    /// Returns metrics over `num_items` recent items with `delay`.
    ///
    /// `delay=0` means latest item.
    pub fn compute_partial_metrics(&self, delay: i32, num_items: i32) -> Option<Level> {
        assert!(delay >= 0);
        assert!(delay < self.capacity());
        assert!(num_items > 0);
        assert!(num_items <= self.capacity());
        assert!(delay + num_items <= self.capacity());

        if delay + num_items > self.size() {
            return None;
        }

        let mut sum = 0.0f32;
        let mut max_v = 0.0f32;

        let upper = num_items.min(self.size());
        for i in 0..upper {
            let mut idx = self.tail - delay - i;
            if idx < 0 {
                idx += self.capacity();
            }
            let lvl = self.data[idx as usize];
            sum += lvl.average;
            max_v = max_v.max(lvl.max);
        }

        Some(Level {
            average: sum / num_items as f32,
            max: max_v,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_optional_level_eq(actual: Option<Level>, expected: Level) {
        let got = actual.expect("expected value");
        assert_eq!(got, expected);
    }

    #[test]
    fn check_empty_buffer_size() {
        for &capacity in &[-1, 0, 1, 123] {
            let buffer = ClippingPredictorLevelBuffer::new(capacity);
            assert_eq!(buffer.capacity(), capacity.max(1));
            assert_eq!(buffer.size(), 0);
        }
    }

    #[test]
    fn check_half_empty_buffer_size() {
        for &capacity in &[-1, 0, 1, 123] {
            let mut buffer = ClippingPredictorLevelBuffer::new(capacity);
            for _ in 0..buffer.capacity() / 2 {
                buffer.push(Level {
                    average: 2.0,
                    max: 4.0,
                });
            }
            assert_eq!(buffer.capacity(), capacity.max(1));
            assert_eq!(buffer.size(), capacity.max(1) / 2);
        }
    }

    #[test]
    fn check_full_buffer_size() {
        for &capacity in &[-1, 0, 1, 123] {
            let mut buffer = ClippingPredictorLevelBuffer::new(capacity);
            for _ in 0..buffer.capacity() {
                buffer.push(Level {
                    average: 2.0,
                    max: 4.0,
                });
            }
            assert_eq!(buffer.capacity(), capacity.max(1));
            assert_eq!(buffer.size(), capacity.max(1));
        }
    }

    #[test]
    fn check_large_buffer_size() {
        for &capacity in &[-1, 0, 1, 123] {
            let mut buffer = ClippingPredictorLevelBuffer::new(capacity);
            for _ in 0..2 * buffer.capacity() {
                buffer.push(Level {
                    average: 2.0,
                    max: 4.0,
                });
            }
            assert_eq!(buffer.capacity(), capacity.max(1));
            assert_eq!(buffer.size(), capacity.max(1));
        }
    }

    #[test]
    fn check_size_after_reset() {
        for &capacity in &[-1, 0, 1, 123] {
            let mut buffer = ClippingPredictorLevelBuffer::new(capacity);
            buffer.push(Level {
                average: 1.0,
                max: 1.0,
            });
            buffer.push(Level {
                average: 1.0,
                max: 1.0,
            });
            buffer.reset();
            assert_eq!(buffer.capacity(), capacity.max(1));
            assert_eq!(buffer.size(), 0);
            buffer.push(Level {
                average: 1.0,
                max: 1.0,
            });
            assert_eq!(buffer.capacity(), capacity.max(1));
            assert_eq!(buffer.size(), 1);
        }
    }

    #[test]
    fn check_metrics_after_full_buffer() {
        let mut buffer = ClippingPredictorLevelBuffer::new(2);
        buffer.push(Level {
            average: 1.0,
            max: 2.0,
        });
        buffer.push(Level {
            average: 3.0,
            max: 6.0,
        });
        assert_optional_level_eq(
            buffer.compute_partial_metrics(0, 1),
            Level {
                average: 3.0,
                max: 6.0,
            },
        );
        assert_optional_level_eq(
            buffer.compute_partial_metrics(1, 1),
            Level {
                average: 1.0,
                max: 2.0,
            },
        );
        assert_optional_level_eq(
            buffer.compute_partial_metrics(0, 2),
            Level {
                average: 2.0,
                max: 6.0,
            },
        );
    }

    #[test]
    fn check_metrics_after_push_beyond_capacity() {
        let mut buffer = ClippingPredictorLevelBuffer::new(2);
        buffer.push(Level {
            average: 1.0,
            max: 1.0,
        });
        buffer.push(Level {
            average: 3.0,
            max: 6.0,
        });
        buffer.push(Level {
            average: 5.0,
            max: 10.0,
        });
        buffer.push(Level {
            average: 7.0,
            max: 14.0,
        });
        buffer.push(Level {
            average: 6.0,
            max: 12.0,
        });
        assert_optional_level_eq(
            buffer.compute_partial_metrics(0, 1),
            Level {
                average: 6.0,
                max: 12.0,
            },
        );
        assert_optional_level_eq(
            buffer.compute_partial_metrics(1, 1),
            Level {
                average: 7.0,
                max: 14.0,
            },
        );
        assert_optional_level_eq(
            buffer.compute_partial_metrics(0, 2),
            Level {
                average: 6.5,
                max: 14.0,
            },
        );
    }

    #[test]
    fn check_metrics_after_too_few_items() {
        let mut buffer = ClippingPredictorLevelBuffer::new(4);
        buffer.push(Level {
            average: 1.0,
            max: 2.0,
        });
        buffer.push(Level {
            average: 3.0,
            max: 6.0,
        });
        assert_eq!(buffer.compute_partial_metrics(0, 3), None);
        assert_eq!(buffer.compute_partial_metrics(2, 1), None);
    }

    #[test]
    fn check_metrics_after_reset() {
        let mut buffer = ClippingPredictorLevelBuffer::new(2);
        buffer.push(Level {
            average: 1.0,
            max: 2.0,
        });
        buffer.reset();
        buffer.push(Level {
            average: 5.0,
            max: 10.0,
        });
        buffer.push(Level {
            average: 7.0,
            max: 14.0,
        });
        assert_optional_level_eq(
            buffer.compute_partial_metrics(0, 1),
            Level {
                average: 7.0,
                max: 14.0,
            },
        );
        assert_optional_level_eq(
            buffer.compute_partial_metrics(0, 2),
            Level {
                average: 6.0,
                max: 14.0,
            },
        );
        assert_optional_level_eq(
            buffer.compute_partial_metrics(1, 1),
            Level {
                average: 5.0,
                max: 10.0,
            },
        );
    }
}
