//! Circular speech-probability buffer mirrored from WebRTC AGC2.

const ACTIVITY_THRESHOLD: f32 = 0.9;
const NUM_ANALYSIS_FRAMES: usize = 100;
// We use 12 in AGC2 adaptive digital, but with slightly different logic.
const TRANSIENT_WIDTH_THRESHOLD: i32 = 7;

/// Circular buffer that stores speech probabilities for a segment
/// and estimates whether the segment is active.
pub struct SpeechProbabilityBuffer {
    low_probability_threshold: f32,
    /// Sum of probabilities stored in `probabilities`.
    sum_probabilities: f32,
    /// Circular buffer for probabilities.
    probabilities: Vec<f32>,
    /// Current circular index where newest data is written.
    buffer_index: usize,
    /// Indicates if buffer is full.
    buffer_is_full: bool,
    num_high_probability_observations: i32,
}

impl SpeechProbabilityBuffer {
    /// `low_probability_threshold` must be in `[0.0, 1.0]`.
    pub fn new(low_probability_threshold: f32) -> Self {
        assert!((0.0..=1.0).contains(&low_probability_threshold));
        let probabilities = vec![0.0; NUM_ANALYSIS_FRAMES];
        assert!(!probabilities.is_empty());
        Self {
            low_probability_threshold,
            sum_probabilities: 0.0,
            probabilities,
            buffer_index: 0,
            buffer_is_full: false,
            num_high_probability_observations: 0,
        }
    }

    /// Adds a probability to the buffer and updates the running sum.
    pub fn update(&mut self, mut probability: f32) {
        // Remove oldest entry if circular buffer is full.
        if self.buffer_is_full {
            let oldest_probability = self.probabilities[self.buffer_index];
            self.sum_probabilities -= oldest_probability;
        }

        // Check for transients.
        if probability <= self.low_probability_threshold {
            // Set low probability to zero.
            probability = 0.0;

            // Check if this has been a transient.
            if self.num_high_probability_observations <= TRANSIENT_WIDTH_THRESHOLD {
                self.remove_transient();
            }
            self.num_high_probability_observations = 0;
        } else if self.num_high_probability_observations <= TRANSIENT_WIDTH_THRESHOLD {
            self.num_high_probability_observations += 1;
        }

        // Update circular buffer and sum.
        self.probabilities[self.buffer_index] = probability;
        self.sum_probabilities += probability;

        // Increment circular index and handle wrap-around.
        self.buffer_index += 1;
        if self.buffer_index >= NUM_ANALYSIS_FRAMES {
            self.buffer_index = 0;
            self.buffer_is_full = true;
        }
    }

    /// Resets histogram and forgets past.
    pub fn reset(&mut self) {
        self.sum_probabilities = 0.0;
        self.buffer_index = 0;
        self.buffer_is_full = false;
        self.num_high_probability_observations = 0;
    }

    /// Returns true if segment is active.
    pub fn is_active_segment(&self) -> bool {
        if !self.buffer_is_full {
            return false;
        }
        if self.sum_probabilities < ACTIVITY_THRESHOLD * NUM_ANALYSIS_FRAMES as f32 {
            return false;
        }
        true
    }

    fn remove_transient(&mut self) {
        // Don't expect to be here if region is longer than transient width
        // or there has not been any transient.
        assert!(self.num_high_probability_observations <= TRANSIENT_WIDTH_THRESHOLD);

        // Replace previously added probabilities with zero.
        let mut index = if self.buffer_index > 0 {
            self.buffer_index - 1
        } else {
            NUM_ANALYSIS_FRAMES - 1
        };

        while self.num_high_probability_observations > 0 {
            self.num_high_probability_observations -= 1;
            self.sum_probabilities -= self.probabilities[index];
            self.probabilities[index] = 0.0;

            index = if index > 0 {
                index - 1
            } else {
                NUM_ANALYSIS_FRAMES - 1
            };
        }
    }

    #[cfg(test)]
    fn get_sum_probabilities(&self) -> f32 {
        self.sum_probabilities
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ABS_ERROR: f32 = 0.001;
    const ACTIVITY_THRESHOLD: f32 = 0.9;
    const LOW_PROBABILITY_THRESHOLD: f32 = 0.2;
    const NUM_ANALYSIS_FRAMES: usize = 100;

    #[test]
    fn check_sum_after_initialization() {
        let buffer = SpeechProbabilityBuffer::new(LOW_PROBABILITY_THRESHOLD);
        assert_eq!(buffer.get_sum_probabilities(), 0.0);
    }

    #[test]
    fn check_sum_after_update() {
        let mut buffer = SpeechProbabilityBuffer::new(LOW_PROBABILITY_THRESHOLD);

        buffer.update(0.7);
        assert!((buffer.get_sum_probabilities() - 0.7).abs() < ABS_ERROR);

        buffer.update(0.6);
        assert!((buffer.get_sum_probabilities() - 1.3).abs() < ABS_ERROR);

        for _ in 0..NUM_ANALYSIS_FRAMES - 1 {
            buffer.update(1.0);
        }

        assert!((buffer.get_sum_probabilities() - 99.6).abs() < ABS_ERROR);
    }

    #[test]
    fn check_sum_after_reset() {
        let mut buffer = SpeechProbabilityBuffer::new(LOW_PROBABILITY_THRESHOLD);

        buffer.update(0.7);
        buffer.update(0.6);
        buffer.update(0.3);

        assert!(buffer.get_sum_probabilities() > 0.0);

        buffer.reset();

        assert_eq!(buffer.get_sum_probabilities(), 0.0);
    }

    #[test]
    fn check_sum_after_transient_not_removed() {
        let mut buffer = SpeechProbabilityBuffer::new(LOW_PROBABILITY_THRESHOLD);

        for _ in 0..9 {
            buffer.update(1.0);
        }

        buffer.update(0.0);
        assert!((buffer.get_sum_probabilities() - 9.0).abs() < ABS_ERROR);

        buffer.update(0.0);
        assert!((buffer.get_sum_probabilities() - 9.0).abs() < ABS_ERROR);
    }

    #[test]
    fn check_sum_after_transient_removed() {
        let mut buffer = SpeechProbabilityBuffer::new(LOW_PROBABILITY_THRESHOLD);

        for _ in 0..6 {
            buffer.update(0.0);
        }
        for _ in 0..3 {
            buffer.update(1.0);
        }

        assert!((buffer.get_sum_probabilities() - 3.0).abs() < ABS_ERROR);

        buffer.update(0.0);
        assert!((buffer.get_sum_probabilities() - 0.0).abs() < ABS_ERROR);
    }

    #[test]
    fn check_segment_is_not_active_after_no_updates() {
        let buffer = SpeechProbabilityBuffer::new(LOW_PROBABILITY_THRESHOLD);
        assert!(!buffer.is_active_segment());
    }

    #[test]
    fn check_segment_is_active_changes_from_false_to_true() {
        let mut buffer = SpeechProbabilityBuffer::new(LOW_PROBABILITY_THRESHOLD);

        for _ in 0..NUM_ANALYSIS_FRAMES {
            buffer.update(0.0);
        }
        assert!(!buffer.is_active_segment());

        for _ in 0..(ACTIVITY_THRESHOLD * NUM_ANALYSIS_FRAMES as f32 - 1.0) as usize {
            buffer.update(1.0);
        }
        assert!(!buffer.is_active_segment());

        buffer.update(1.0);
        assert!(buffer.is_active_segment());
    }

    #[test]
    fn check_segment_is_active_changes_from_true_to_false() {
        let mut buffer = SpeechProbabilityBuffer::new(LOW_PROBABILITY_THRESHOLD);

        for _ in 0..NUM_ANALYSIS_FRAMES {
            buffer.update(1.0);
        }
        assert!(buffer.is_active_segment());

        for _ in 0..((1.0 - ACTIVITY_THRESHOLD) * NUM_ANALYSIS_FRAMES as f32 - 1.0) as usize {
            buffer.update(0.0);
        }
        assert!(buffer.is_active_segment());

        buffer.update(1.0);
        assert!(buffer.is_active_segment());
    }

    #[test]
    fn check_segment_is_active_after_updates_with_high_probabilities() {
        let mut buffer = SpeechProbabilityBuffer::new(LOW_PROBABILITY_THRESHOLD);

        for _ in 0..NUM_ANALYSIS_FRAMES - 1 {
            buffer.update(1.0);
        }
        assert!(!buffer.is_active_segment());

        buffer.update(1.0);
        assert!(buffer.is_active_segment());
    }

    #[test]
    fn check_segment_is_not_active_after_updates_with_low_probabilities() {
        let mut buffer = SpeechProbabilityBuffer::new(LOW_PROBABILITY_THRESHOLD);

        for _ in 0..NUM_ANALYSIS_FRAMES - 1 {
            buffer.update(0.3);
        }
        assert!(!buffer.is_active_segment());

        buffer.update(0.3);
        assert!(!buffer.is_active_segment());
    }

    #[test]
    fn check_segment_is_active_after_buffer_is_full() {
        let mut buffer = SpeechProbabilityBuffer::new(LOW_PROBABILITY_THRESHOLD);

        for _ in 0..NUM_ANALYSIS_FRAMES - 1 {
            buffer.update(1.0);
        }
        assert!(!buffer.is_active_segment());

        buffer.update(1.0);
        assert!(buffer.is_active_segment());

        buffer.update(1.0);
        assert!(buffer.is_active_segment());
    }

    #[test]
    fn check_segment_is_not_active_after_buffer_is_full() {
        let mut buffer = SpeechProbabilityBuffer::new(LOW_PROBABILITY_THRESHOLD);

        for _ in 0..NUM_ANALYSIS_FRAMES - 1 {
            buffer.update(0.29);
        }
        assert!(!buffer.is_active_segment());

        buffer.update(0.29);
        assert!(!buffer.is_active_segment());

        buffer.update(0.29);
        assert!(!buffer.is_active_segment());
    }

    #[test]
    fn check_segment_is_not_active_after_reset() {
        let mut buffer = SpeechProbabilityBuffer::new(LOW_PROBABILITY_THRESHOLD);

        for _ in 0..NUM_ANALYSIS_FRAMES {
            buffer.update(1.0);
        }
        assert!(buffer.is_active_segment());

        buffer.reset();
        assert!(!buffer.is_active_segment());
    }

    #[test]
    fn check_segment_is_not_active_after_transient_removed_after_few_updates() {
        let mut buffer = SpeechProbabilityBuffer::new(LOW_PROBABILITY_THRESHOLD);

        buffer.update(0.4);
        buffer.update(0.4);
        buffer.update(0.0);

        assert!(!buffer.is_active_segment());
    }

    #[test]
    fn check_segment_is_active_after_transient_not_removed() {
        let mut buffer = SpeechProbabilityBuffer::new(LOW_PROBABILITY_THRESHOLD);

        for _ in 0..NUM_ANALYSIS_FRAMES {
            buffer.update(1.0);
        }

        buffer.update(0.7);
        buffer.update(0.8);
        buffer.update(0.9);
        buffer.update(1.0);

        assert!(buffer.is_active_segment());

        buffer.update(0.0);
        assert!(buffer.is_active_segment());

        buffer.update(0.7);
        assert!(buffer.is_active_segment());
    }

    #[test]
    fn check_segment_is_not_active_after_transient_not_removed() {
        let mut buffer = SpeechProbabilityBuffer::new(LOW_PROBABILITY_THRESHOLD);

        for _ in 0..NUM_ANALYSIS_FRAMES {
            buffer.update(0.1);
        }

        buffer.update(0.7);
        buffer.update(0.8);
        buffer.update(0.9);
        buffer.update(1.0);

        assert!(!buffer.is_active_segment());

        buffer.update(0.0);
        assert!(!buffer.is_active_segment());

        buffer.update(0.7);
        assert!(!buffer.is_active_segment());
    }

    #[test]
    fn check_segment_is_not_active_after_transient_removed() {
        let mut buffer = SpeechProbabilityBuffer::new(LOW_PROBABILITY_THRESHOLD);

        for _ in 0..NUM_ANALYSIS_FRAMES {
            buffer.update(0.1);
        }

        buffer.update(0.7);
        buffer.update(0.8);
        buffer.update(0.9);
        buffer.update(1.0);

        assert!(!buffer.is_active_segment());

        buffer.update(0.0);
        assert!(!buffer.is_active_segment());

        buffer.update(0.7);
        assert!(!buffer.is_active_segment());
    }

    #[test]
    fn check_segment_is_active_after_transient_removed() {
        let mut buffer = SpeechProbabilityBuffer::new(LOW_PROBABILITY_THRESHOLD);

        for _ in 0..NUM_ANALYSIS_FRAMES {
            buffer.update(1.0);
        }

        buffer.update(0.7);
        buffer.update(0.8);
        buffer.update(0.9);
        buffer.update(1.0);

        assert!(buffer.is_active_segment());

        buffer.update(0.0);
        assert!(buffer.is_active_segment());

        buffer.update(0.7);
        assert!(buffer.is_active_segment());
    }
}
