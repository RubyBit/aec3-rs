use crate::api::config::SubbandNearendDetection;
use crate::audio_processing::aec3::aec3_common::FFT_LENGTH_BY_2_PLUS_1;
use crate::audio_processing::aec3::moving_average::MovingAverage;
use crate::audio_processing::aec3::nearend_detector::NearendDetector;

pub struct SubbandNearendDetector {
    config: SubbandNearendDetection,
    num_capture_channels: usize,
    nearend_smoothers: Vec<MovingAverage>,
    one_over_subband_length1: f32,
    one_over_subband_length2: f32,
    nearend_state: bool,
}

impl SubbandNearendDetector {
    pub fn new(config: &SubbandNearendDetection, num_capture_channels: usize) -> Self {
        assert!(num_capture_channels > 0);
        let config = config.clone();
        assert!(config.subband1.low <= config.subband1.high);
        assert!(config.subband2.low <= config.subband2.high);
        let nearend_smoothers = (0..num_capture_channels)
            .map(|_| MovingAverage::new(FFT_LENGTH_BY_2_PLUS_1, config.nearend_average_blocks))
            .collect();
        let len1 = (config.subband1.high - config.subband1.low + 1) as f32;
        let len2 = (config.subband2.high - config.subband2.low + 1) as f32;
        Self {
            config,
            num_capture_channels,
            nearend_smoothers,
            one_over_subband_length1: 1.0 / len1,
            one_over_subband_length2: 1.0 / len2,
            nearend_state: false,
        }
    }
}

impl NearendDetector for SubbandNearendDetector {
    fn is_nearend_state(&self) -> bool {
        self.nearend_state
    }

    fn update(
        &mut self,
        nearend_spectrum: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        residual_echo_spectrum: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        comfort_noise_spectrum: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        _initial_state: bool,
    ) {
        let _ = residual_echo_spectrum; // Not used in reference implementation.
        assert_eq!(nearend_spectrum.len(), self.num_capture_channels);
        assert_eq!(comfort_noise_spectrum.len(), self.num_capture_channels);

        self.nearend_state = false;
        for ch in 0..self.num_capture_channels {
            let mut smoothed_nearend = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
            self.nearend_smoothers[ch].average(&nearend_spectrum[ch], &mut smoothed_nearend);
            let noise = &comfort_noise_spectrum[ch];

            let noise_power: f32 = noise[self.config.subband1.low..=self.config.subband1.high]
                .iter()
                .sum::<f32>()
                * self.one_over_subband_length1;
            let nearend_power_subband1: f32 = smoothed_nearend
                [self.config.subband1.low..=self.config.subband1.high]
                .iter()
                .sum::<f32>()
                * self.one_over_subband_length1;
            let nearend_power_subband2: f32 = smoothed_nearend
                [self.config.subband2.low..=self.config.subband2.high]
                .iter()
                .sum::<f32>()
                * self.one_over_subband_length2;

            if nearend_power_subband1 < self.config.nearend_threshold * nearend_power_subband2
                && nearend_power_subband1 > self.config.snr_threshold * noise_power
            {
                self.nearend_state = true;
                break;
            }
        }
    }
}
