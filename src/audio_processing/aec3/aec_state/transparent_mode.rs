use crate::api::config::EchoCanceller3Config;
use crate::audio_processing::aec3::aec3_common::NUM_BLOCKS_PER_SECOND;

const BLOCKS_SINCE_CONVERGED_FILTER_INIT: usize = 10_000;
const BLOCKS_SINCE_CONSISTENT_ESTIMATE_INIT: usize = 10_000;

pub(super) enum TransparentMode {
    Disabled,
    Hmm(TransparentModeHmm),
    Legacy(TransparentModeLegacy),
}

impl TransparentMode {
    pub(super) fn new(config: &EchoCanceller3Config) -> Self {
        // Mirror `TransparentMode::Create` from WebRTC:
        // - disabled when bounded ERL is used
        // - disabled by a kill-switch (field trial in WebRTC; explicit config here)
        if config.ep_strength.bounded_erl || !config.transparent_mode.enabled {
            return Self::Disabled;
        }

        if config.transparent_mode.use_hmm {
            Self::Hmm(TransparentModeHmm::new())
        } else {
            Self::Legacy(TransparentModeLegacy::new(config))
        }
    }

    pub(super) fn reset(&mut self) {
        match self {
            TransparentMode::Disabled => {}
            TransparentMode::Hmm(hmm) => hmm.reset(),
            TransparentMode::Legacy(legacy) => legacy.reset(),
        }
    }

    pub(super) fn active(&self) -> bool {
        match self {
            TransparentMode::Disabled => false,
            TransparentMode::Hmm(hmm) => hmm.active(),
            TransparentMode::Legacy(legacy) => legacy.active(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn update(
        &mut self,
        filter_delay_blocks: i32,
        any_filter_consistent: bool,
        any_filter_converged: bool,
        any_coarse_filter_converged: bool,
        all_filters_diverged: bool,
        active_render: bool,
        saturated_capture: bool,
    ) {
        match self {
            TransparentMode::Disabled => {}
            TransparentMode::Hmm(hmm) => hmm.update(
                /* filter_delay_blocks */ filter_delay_blocks,
                /* any_filter_consistent */ any_filter_consistent,
                /* any_filter_converged */ any_filter_converged,
                any_coarse_filter_converged,
                /* all_filters_diverged */ all_filters_diverged,
                active_render,
                /* saturated_capture */ saturated_capture,
            ),
            TransparentMode::Legacy(legacy) => legacy.update(
                filter_delay_blocks,
                any_filter_consistent,
                any_filter_converged,
                /* any_coarse_filter_converged */ any_coarse_filter_converged,
                all_filters_diverged,
                active_render,
                saturated_capture,
            ),
        }
    }
}

pub(super) struct TransparentModeHmm {
    transparency_activated: bool,
    prob_transparent_state: f32,
}

impl TransparentModeHmm {
    const INITIAL_PROB_TRANSPARENT: f32 = 0.2;

    fn new() -> Self {
        let mut this = Self {
            transparency_activated: false,
            prob_transparent_state: Self::INITIAL_PROB_TRANSPARENT,
        };
        this.reset();
        this
    }

    fn reset(&mut self) {
        self.transparency_activated = false;
        self.prob_transparent_state = Self::INITIAL_PROB_TRANSPARENT;
    }

    fn active(&self) -> bool {
        self.transparency_activated
    }

    #[allow(clippy::too_many_arguments)]
    fn update(
        &mut self,
        _filter_delay_blocks: i32,
        _any_filter_consistent: bool,
        _any_filter_converged: bool,
        any_coarse_filter_converged: bool,
        _all_filters_diverged: bool,
        active_render: bool,
        _saturated_capture: bool,
    ) {
        // The HMM is only updated during active render.
        if !active_render {
            return;
        }

        // Probability of switching from one state to the other.
        const SWITCH: f32 = 0.000001;

        // Probability of observing converged coarse filters in states "normal"
        // and "transparent" during active render.
        const CONVERGED_NORMAL: f32 = 0.01;
        const CONVERGED_TRANSPARENT: f32 = 0.001;

        // Probability of transitioning to the transparent state from the normal
        // and transparent state respectively.
        const A0: f32 = SWITCH;
        const A1: f32 = 1.0 - SWITCH;

        // Observation probabilities (out = 0/1) in the normal and transparent
        // states respectively.
        const B00: f32 = 1.0 - CONVERGED_NORMAL;
        const B01: f32 = CONVERGED_NORMAL;
        const B10: f32 = 1.0 - CONVERGED_TRANSPARENT;
        const B11: f32 = CONVERGED_TRANSPARENT;

        let prob_transparent = self.prob_transparent_state;
        let prob_normal = 1.0 - prob_transparent;

        let prob_transition_transparent = prob_normal * A0 + prob_transparent * A1;
        let prob_transition_normal = 1.0 - prob_transition_transparent;

        let out = if any_coarse_filter_converged { 1 } else { 0 };
        let (b_normal, b_transparent) = if out == 1 { (B01, B11) } else { (B00, B10) };

        let prob_joint_normal = prob_transition_normal * b_normal;
        let prob_joint_transparent = prob_transition_transparent * b_transparent;
        let denom = prob_joint_normal + prob_joint_transparent;
        debug_assert!(denom > 0.0);
        self.prob_transparent_state = prob_joint_transparent / denom;

        // Transparent mode is only activated when its probability is high.
        // Dead zone between activation/deactivation thresholds.
        if self.prob_transparent_state > 0.95 {
            self.transparency_activated = true;
        } else if self.prob_transparent_state < 0.5 {
            self.transparency_activated = false;
        }
    }
}

pub(super) struct TransparentModeLegacy {
    linear_and_stable_echo_path: bool,
    capture_block_counter: usize,
    transparency_activated: bool,
    active_blocks_since_sane_filter: usize,
    sane_filter_observed: bool,
    finite_erl_recently_detected: bool,
    non_converged_sequence_size: usize,
    diverged_sequence_size: usize,
    active_non_converged_sequence_size: usize,
    num_converged_blocks: usize,
    recent_convergence_during_activity: bool,
    strong_not_saturated_render_blocks: usize,
}

impl TransparentModeLegacy {
    fn new(config: &EchoCanceller3Config) -> Self {
        Self {
            linear_and_stable_echo_path: config.echo_removal_control.linear_and_stable_echo_path,
            capture_block_counter: 0,
            transparency_activated: false,
            active_blocks_since_sane_filter: BLOCKS_SINCE_CONSISTENT_ESTIMATE_INIT,
            sane_filter_observed: false,
            finite_erl_recently_detected: false,
            non_converged_sequence_size: BLOCKS_SINCE_CONVERGED_FILTER_INIT,
            diverged_sequence_size: 0,
            active_non_converged_sequence_size: 0,
            num_converged_blocks: 0,
            recent_convergence_during_activity: false,
            strong_not_saturated_render_blocks: 0,
        }
    }

    fn reset(&mut self) {
        self.non_converged_sequence_size = BLOCKS_SINCE_CONVERGED_FILTER_INIT;
        self.diverged_sequence_size = 0;
        self.strong_not_saturated_render_blocks = 0;
        if self.linear_and_stable_echo_path {
            self.recent_convergence_during_activity = false;
        }
    }

    fn active(&self) -> bool {
        self.transparency_activated
    }

    #[allow(clippy::too_many_arguments)]
    fn update(
        &mut self,
        filter_delay_blocks: i32,
        any_filter_consistent: bool,
        any_filter_converged: bool,
        _any_coarse_filter_converged: bool,
        all_filters_diverged: bool,
        active_render: bool,
        saturated_capture: bool,
    ) {
        self.capture_block_counter += 1;
        if active_render && !saturated_capture {
            self.strong_not_saturated_render_blocks += 1;
        }

        if any_filter_consistent && filter_delay_blocks < 5 {
            self.sane_filter_observed = true;
            self.active_blocks_since_sane_filter = 0;
        } else if active_render {
            self.active_blocks_since_sane_filter += 1;
        }

        let sane_filter_recently_seen = if !self.sane_filter_observed {
            self.capture_block_counter <= 5 * NUM_BLOCKS_PER_SECOND
        } else {
            self.active_blocks_since_sane_filter <= 30 * NUM_BLOCKS_PER_SECOND
        };

        if any_filter_converged {
            self.recent_convergence_during_activity = true;
            self.active_non_converged_sequence_size = 0;
            self.non_converged_sequence_size = 0;
            self.num_converged_blocks += 1;
        } else {
            self.non_converged_sequence_size += 1;
            if self.non_converged_sequence_size > 20 * NUM_BLOCKS_PER_SECOND {
                self.num_converged_blocks = 0;
            }
            if active_render {
                self.active_non_converged_sequence_size += 1;
                if self.active_non_converged_sequence_size > 60 * NUM_BLOCKS_PER_SECOND {
                    self.recent_convergence_during_activity = false;
                }
            }
        }

        if !all_filters_diverged {
            self.diverged_sequence_size = 0;
        } else {
            self.diverged_sequence_size += 1;
            if self.diverged_sequence_size >= 60 {
                // TODO(peah): Change these lines to ensure proper triggering of usable filter. (straight from webrtc source!)
                self.non_converged_sequence_size = BLOCKS_SINCE_CONVERGED_FILTER_INIT;
            }
        }

        if self.active_non_converged_sequence_size > 60 * NUM_BLOCKS_PER_SECOND {
            self.finite_erl_recently_detected = false;
        }
        if self.num_converged_blocks > 50 {
            self.finite_erl_recently_detected = true;
        }

        if self.finite_erl_recently_detected {
            self.transparency_activated = false;
        } else if sane_filter_recently_seen && self.recent_convergence_during_activity {
            self.transparency_activated = false;
        } else {
            let filter_should_have_converged =
                self.strong_not_saturated_render_blocks > 6 * NUM_BLOCKS_PER_SECOND;
            self.transparency_activated = filter_should_have_converged;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::TransparentMode;
    use crate::api::config::EchoCanceller3Config;

    #[test]
    fn hmm_transparent_mode_toggles_as_reference() {
        let mut config = EchoCanceller3Config::default();
        config.transparent_mode.enabled = true;
        config.transparent_mode.use_hmm = true;
        config.ep_strength.bounded_erl = false;

        let mut mode = TransparentMode::new(&config);
        assert!(!mode.active());

        // With active render and no (relaxed) coarse convergence, probability of
        // transparent mode should increase and eventually activate.
        for _ in 0..1000 {
            mode.update(
                /* filter_delay_blocks */ 0, /* any_filter_consistent */ false,
                /* any_filter_converged */ false,
                /* any_coarse_filter_converged */ false,
                /* all_filters_diverged */ false, /* active_render */ true,
                /* saturated_capture */ false,
            );
        }
        assert!(mode.active());

        // A few converged-coarse observations should deactivate due to the
        // hysteresis thresholds.
        for _ in 0..10 {
            mode.update(0, false, false, true, false, true, false);
        }
        assert!(!mode.active());
    }

    #[test]
    fn transparent_mode_kill_switch_disables_classifier() {
        let mut config = EchoCanceller3Config::default();
        config.transparent_mode.enabled = false;
        config.transparent_mode.use_hmm = true;
        config.ep_strength.bounded_erl = false;

        let mut mode = TransparentMode::new(&config);
        for _ in 0..2000 {
            mode.update(0, false, false, false, false, true, false);
        }
        assert!(!mode.active());
    }
}
