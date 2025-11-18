/// Detects clock drift patterns in the estimated render/capture delay sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClockDriftLevel {
    None,
    Probable,
    Verified,
}

#[derive(Debug, Clone)]
pub struct ClockDriftDetector {
    delay_history: [i32; 3],
    level: ClockDriftLevel,
    stability_counter: usize,
}

impl ClockDriftDetector {
    pub fn new() -> Self {
        Self {
            delay_history: [0; 3],
            level: ClockDriftLevel::None,
            stability_counter: 0,
        }
    }

    pub fn level(&self) -> ClockDriftLevel {
        self.level
    }

    pub fn update(&mut self, delay_estimate: i32) {
        if delay_estimate == self.delay_history[0] {
            self.stability_counter += 1;
            if self.stability_counter > 7500 {
                self.level = ClockDriftLevel::None;
            }
            return;
        }

        self.stability_counter = 0;
        let d1 = self.delay_history[0] - delay_estimate;
        let d2 = self.delay_history[1] - delay_estimate;
        let d3 = self.delay_history[2] - delay_estimate;

        let probable_drift_up = (d1 == -1 && d2 == -2) || (d1 == -2 && d2 == -1);
        let drift_up = probable_drift_up && d3 == -3;

        let probable_drift_down = (d1 == 1 && d2 == 2) || (d1 == 2 && d2 == 1);
        let drift_down = probable_drift_down && d3 == 3;

        if drift_up || drift_down {
            self.level = ClockDriftLevel::Verified;
        } else if (probable_drift_up || probable_drift_down)
            && matches!(self.level, ClockDriftLevel::None)
        {
            self.level = ClockDriftLevel::Probable;
        }

        self.delay_history[2] = self.delay_history[1];
        self.delay_history[1] = self.delay_history[0];
        self.delay_history[0] = delay_estimate;
    }
}

impl Default for ClockDriftDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{ClockDriftDetector, ClockDriftLevel};

    #[test]
    fn detector_matches_reference_transitions() {
        let mut detector = ClockDriftDetector::new();
        assert_eq!(ClockDriftLevel::None, detector.level());

        for _ in 0..100 {
            detector.update(1000);
        }
        assert_eq!(ClockDriftLevel::None, detector.level());
        for _ in 0..100 {
            detector.update(1001);
        }
        assert_eq!(ClockDriftLevel::None, detector.level());
        for _ in 0..100 {
            detector.update(1002);
        }
        assert_eq!(ClockDriftLevel::Probable, detector.level());
        for _ in 0..100 {
            detector.update(1003);
        }
        assert_eq!(ClockDriftLevel::Verified, detector.level());

        for _ in 0..10_000 {
            detector.update(1003);
        }
        assert_eq!(ClockDriftLevel::None, detector.level());

        for _ in 0..100 {
            detector.update(1001);
        }
        for _ in 0..100 {
            detector.update(999);
        }
        assert_eq!(ClockDriftLevel::Probable, detector.level());
        for _ in 0..100 {
            detector.update(1000);
        }
        for _ in 0..100 {
            detector.update(998);
        }
        assert_eq!(ClockDriftLevel::Verified, detector.level());
    }
}
