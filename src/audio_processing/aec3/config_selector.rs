//! Selection between the mono and the multichannel configuration.
//!
//! Ported from `modules/audio_processing/aec3/config_selector.{h,cc}`.

use crate::api::config::EchoCanceller3Config;

/// Validates that the mono and the multichannel configs have compatible fields.
///
/// The reference additionally compares `filter.high_pass_filter_echo_reference`,
/// which this crate does not implement.
fn compatible_configs(
    mono_config: &EchoCanceller3Config,
    multichannel_config: &EchoCanceller3Config,
) -> bool {
    if mono_config.delay.fixed_capture_delay_samples
        != multichannel_config.delay.fixed_capture_delay_samples
    {
        return false;
    }
    if mono_config.filter.export_linear_aec_output
        != multichannel_config.filter.export_linear_aec_output
    {
        return false;
    }
    if mono_config.multi_channel.detect_stereo_content
        != multichannel_config.multi_channel.detect_stereo_content
    {
        return false;
    }
    if mono_config
        .multi_channel
        .stereo_detection_timeout_threshold_seconds
        != multichannel_config
            .multi_channel
            .stereo_detection_timeout_threshold_seconds
    {
        return false;
    }
    true
}

/// Selects the config to use.
pub struct ConfigSelector {
    config: EchoCanceller3Config,
    multichannel_config: Option<EchoCanceller3Config>,
    multichannel_config_active: bool,
}

impl ConfigSelector {
    pub fn new(
        config: EchoCanceller3Config,
        multichannel_config: Option<EchoCanceller3Config>,
        num_render_input_channels: usize,
    ) -> Self {
        if let Some(multichannel_config) = multichannel_config.as_ref() {
            debug_assert!(compatible_configs(&config, multichannel_config));
        }

        let initial_multichannel_content =
            !config.multi_channel.detect_stereo_content && num_render_input_channels > 1;

        let mut selector = Self {
            config,
            multichannel_config,
            multichannel_config_active: false,
        };
        selector.update(initial_multichannel_content);
        selector
    }

    /// Updates the config selection based on the detection of multichannel
    /// content.
    pub fn update(&mut self, multichannel_content: bool) {
        self.multichannel_config_active =
            multichannel_content && self.multichannel_config.is_some();
    }

    pub fn active_config(&self) -> &EchoCanceller3Config {
        if self.multichannel_config_active {
            self.multichannel_config
                .as_ref()
                .expect("multichannel config is active only when present")
        } else {
            &self.config
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHANNEL_COUNTS: [usize; 3] = [1, 2, 8];

    #[test]
    fn mono_config_is_selected_when_no_multi_channel_config_present() {
        for num_channels in CHANNEL_COUNTS {
            for detect_stereo_content in [false, true] {
                let mut config = EchoCanceller3Config::default();
                config.multi_channel.detect_stereo_content = detect_stereo_content;
                let multichannel_config: Option<EchoCanceller3Config> = None;

                config.delay.default_delay += 1;
                let custom_delay_value_in_config = config.delay.default_delay;

                let mut cs = ConfigSelector::new(config, multichannel_config, num_channels);
                assert_eq!(
                    cs.active_config().delay.default_delay,
                    custom_delay_value_in_config
                );

                cs.update(false);
                assert_eq!(
                    cs.active_config().delay.default_delay,
                    custom_delay_value_in_config
                );

                cs.update(true);
                assert_eq!(
                    cs.active_config().delay.default_delay,
                    custom_delay_value_in_config
                );
            }
        }
    }

    #[test]
    fn correct_initial_config_is_selected() {
        for num_channels in CHANNEL_COUNTS {
            for detect_stereo_content in [false, true] {
                let mut config = EchoCanceller3Config::default();
                config.multi_channel.detect_stereo_content = detect_stereo_content;
                let mut multichannel_config = config.clone();

                config.delay.default_delay += 1;
                let custom_delay_value_in_config = config.delay.default_delay;
                multichannel_config.delay.default_delay += 2;
                let custom_delay_value_in_multichannel_config =
                    multichannel_config.delay.default_delay;

                let cs = ConfigSelector::new(config, Some(multichannel_config), num_channels);

                if num_channels == 1 || detect_stereo_content {
                    assert_eq!(
                        cs.active_config().delay.default_delay,
                        custom_delay_value_in_config
                    );
                } else {
                    assert_eq!(
                        cs.active_config().delay.default_delay,
                        custom_delay_value_in_multichannel_config
                    );
                }
            }
        }
    }

    #[test]
    fn correct_config_update_behavior() {
        for num_channels in CHANNEL_COUNTS {
            let mut config = EchoCanceller3Config::default();
            config.multi_channel.detect_stereo_content = true;
            let mut multichannel_config = config.clone();

            config.delay.default_delay += 1;
            let custom_delay_value_in_config = config.delay.default_delay;
            multichannel_config.delay.default_delay += 2;
            let custom_delay_value_in_multichannel_config = multichannel_config.delay.default_delay;

            let mut cs = ConfigSelector::new(config, Some(multichannel_config), num_channels);

            cs.update(false);
            assert_eq!(
                cs.active_config().delay.default_delay,
                custom_delay_value_in_config
            );

            if num_channels == 1 {
                cs.update(false);
                assert_eq!(
                    cs.active_config().delay.default_delay,
                    custom_delay_value_in_config
                );
            } else {
                cs.update(true);
                assert_eq!(
                    cs.active_config().delay.default_delay,
                    custom_delay_value_in_multichannel_config
                );
            }
        }
    }
}
