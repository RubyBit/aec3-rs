/// Stores properties of a delay estimate produced by the render delay controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DelayEstimateQuality {
    Coarse,
    Refined,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DelayEstimate {
    pub quality: DelayEstimateQuality,
    pub delay: usize,
    pub blocks_since_last_change: usize,
    pub blocks_since_last_update: usize,
}

impl DelayEstimate {
    pub fn new(quality: DelayEstimateQuality, delay: usize) -> Self {
        Self {
            quality,
            delay,
            blocks_since_last_change: 0,
            blocks_since_last_update: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initializes_with_zero_counters() {
        let estimate = DelayEstimate::new(DelayEstimateQuality::Coarse, 42);
        assert_eq!(DelayEstimateQuality::Coarse, estimate.quality);
        assert_eq!(42, estimate.delay);
        assert_eq!(0, estimate.blocks_since_last_change);
        assert_eq!(0, estimate.blocks_since_last_update);
    }
}
