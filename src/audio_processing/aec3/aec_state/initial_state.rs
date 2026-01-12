use crate::api::config::EchoCanceller3Config;
use crate::audio_processing::aec3::aec3_common::NUM_BLOCKS_PER_SECOND;

pub(super) struct InitialState {
    conservative_initial_phase: bool,
    initial_state_seconds: f32,
    transition_triggered: bool,
    initial_state_active: bool,
    strong_not_saturated_render_blocks: usize,
}

impl InitialState {
    pub(super) fn new(config: &EchoCanceller3Config) -> Self {
        let mut state = Self {
            conservative_initial_phase: config.filter.conservative_initial_phase,
            initial_state_seconds: config.filter.initial_state_seconds,
            transition_triggered: false,
            initial_state_active: true,
            strong_not_saturated_render_blocks: 0,
        };
        state.reset();
        state
    }

    pub(super) fn reset(&mut self) {
        self.transition_triggered = false;
        self.initial_state_active = true;
        self.strong_not_saturated_render_blocks = 0;
    }

    pub(super) fn update(&mut self, active_render: bool, saturated_capture: bool) {
        if active_render && !saturated_capture {
            self.strong_not_saturated_render_blocks += 1;
        }

        let prev_initial_state = self.initial_state_active;
        let limit = if self.conservative_initial_phase {
            5.0 * NUM_BLOCKS_PER_SECOND as f32
        } else {
            self.initial_state_seconds * NUM_BLOCKS_PER_SECOND as f32
        };
        self.initial_state_active = (self.strong_not_saturated_render_blocks as f32) < limit;
        self.transition_triggered = !self.initial_state_active && prev_initial_state;
    }

    pub(super) fn initial_state_active(&self) -> bool {
        self.initial_state_active
    }

    pub(super) fn transition_triggered(&self) -> bool {
        self.transition_triggered
    }
}
