/// Captures information about echo path changes between consecutive blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DelayAdjustment {
    None,
    BufferReadjustment,
    BufferFlush,
    DelayReset,
    NewDetectedDelay,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EchoPathVariability {
    pub gain_change: bool,
    pub delay_change: DelayAdjustment,
    pub clock_drift: bool,
}

impl EchoPathVariability {
    pub fn new(gain_change: bool, delay_change: DelayAdjustment, clock_drift: bool) -> Self {
        Self {
            gain_change,
            delay_change,
            clock_drift,
        }
    }

    pub fn audio_path_changed(&self) -> bool {
        self.gain_change || !matches!(self.delay_change, DelayAdjustment::None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_path_change_detection_matches_flags() {
        let variability = EchoPathVariability::new(false, DelayAdjustment::None, false);
        assert!(!variability.audio_path_changed());
        let variability = EchoPathVariability::new(true, DelayAdjustment::None, false);
        assert!(variability.audio_path_changed());
        let variability = EchoPathVariability::new(false, DelayAdjustment::BufferFlush, false);
        assert!(variability.audio_path_changed());
    }
}
