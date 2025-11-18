use crate::api::config::DominantNearendDetection;
use crate::audio_processing::aec3::aec3_common::FFT_LENGTH_BY_2_PLUS_1;
use crate::audio_processing::aec3::nearend_detector::NearendDetector;

pub struct DominantNearendDetector {
    enr_threshold: f32,
    enr_exit_threshold: f32,
    snr_threshold: f32,
    hold_duration: i32,
    trigger_threshold: i32,
    use_during_initial_phase: bool,
    num_capture_channels: usize,
    nearend_state: bool,
    trigger_counters: Vec<i32>,
    hold_counters: Vec<i32>,
}

impl DominantNearendDetector {
    pub fn new(config: &DominantNearendDetection, num_capture_channels: usize) -> Self {
        assert!(num_capture_channels > 0);
        Self {
            enr_threshold: config.enr_threshold,
            enr_exit_threshold: config.enr_exit_threshold,
            snr_threshold: config.snr_threshold,
            hold_duration: config.hold_duration as i32,
            trigger_threshold: config.trigger_threshold as i32,
            use_during_initial_phase: config.use_during_initial_phase,
            num_capture_channels,
            nearend_state: false,
            trigger_counters: vec![0; num_capture_channels],
            hold_counters: vec![0; num_capture_channels],
        }
    }

    fn low_frequency_energy(spectrum: &[f32; FFT_LENGTH_BY_2_PLUS_1]) -> f32 {
        spectrum[1..16].iter().sum()
    }
}

impl NearendDetector for DominantNearendDetector {
    fn is_nearend_state(&self) -> bool {
        self.nearend_state
    }

    fn update(
        &mut self,
        nearend_spectrum: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        residual_echo_spectrum: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        comfort_noise_spectrum: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        initial_state: bool,
    ) {
        assert_eq!(nearend_spectrum.len(), self.num_capture_channels);
        assert_eq!(residual_echo_spectrum.len(), self.num_capture_channels);
        assert_eq!(comfort_noise_spectrum.len(), self.num_capture_channels);

        self.nearend_state = false;
        for ch in 0..self.num_capture_channels {
            let ne_sum = Self::low_frequency_energy(&nearend_spectrum[ch]);
            let echo_sum = Self::low_frequency_energy(&residual_echo_spectrum[ch]);
            let noise_sum = Self::low_frequency_energy(&comfort_noise_spectrum[ch]);

            if (!initial_state || self.use_during_initial_phase)
                && echo_sum < self.enr_threshold * ne_sum
                && ne_sum > self.snr_threshold * noise_sum
            {
                self.trigger_counters[ch] += 1;
                if self.trigger_counters[ch] >= self.trigger_threshold {
                    self.hold_counters[ch] = self.hold_duration;
                    self.trigger_counters[ch] = self.trigger_threshold;
                }
            } else {
                self.trigger_counters[ch] = (self.trigger_counters[ch] - 1).max(0);
            }

            if echo_sum > self.enr_exit_threshold * ne_sum
                && echo_sum > self.snr_threshold * noise_sum
            {
                self.hold_counters[ch] = 0;
            }

            self.hold_counters[ch] = (self.hold_counters[ch] - 1).max(0);
            self.nearend_state |= self.hold_counters[ch] > 0;
        }
    }
}
