use crate::audio_processing::aec3::subtractor_output::SubtractorOutput;

#[derive(Default)]
pub(super) struct SaturationDetector {
    saturated_echo: bool,
}

impl SaturationDetector {
    pub(super) fn update(
        &mut self,
        x: &[Vec<f32>],
        saturated_capture: bool,
        usable_linear_estimate: bool,
        subtractor_output: &[SubtractorOutput],
        echo_path_gain: f32,
    ) {
        self.saturated_echo = false;
        if !saturated_capture {
            return;
        }

        if usable_linear_estimate {
            const SATURATION_THRESHOLD: f32 = 20_000.0;
            for output in subtractor_output {
                if output.s_main_max_abs > SATURATION_THRESHOLD
                    || output.s_shadow_max_abs > SATURATION_THRESHOLD
                {
                    self.saturated_echo = true;
                    break;
                }
            }
        } else {
            let mut max_sample = 0.0f32;
            for channel in x {
                for &sample in channel {
                    max_sample = max_sample.max(sample.abs());
                }
            }
            const MARGIN: f32 = 10.0;
            let peak_echo_amplitude = max_sample * echo_path_gain * MARGIN;
            if peak_echo_amplitude > 32_000.0 {
                self.saturated_echo = true;
            }
        }
    }

    pub(super) fn saturated_echo(&self) -> bool {
        self.saturated_echo
    }
}
