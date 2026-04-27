//! Saturation protector for AGC2.

use crate::audio_processing::agc2::agc2_common::{
    FRAME_DURATION_MS, MIN_LEVEL_DBFS, VAD_CONFIDENCE_THRESHOLD,
};
use crate::audio_processing::agc2::saturation_protector_buffer::SaturationProtectorBuffer;

const PEAK_ENVELOPER_SUPER_FRAME_LENGTH_MS: i32 = 400;
const MIN_MARGIN_DB: f32 = 12.0;
const MAX_MARGIN_DB: f32 = 25.0;
const ATTACK: f32 = 0.998_849_369_936_505_2;
const DECAY: f32 = 0.999_769_767_998_156_5;

pub trait SaturationProtector {
    /// Returns the recommended headroom in dB.
    fn headroom_db(&self) -> f32;

    /// Analyzes one frame.
    fn analyze(&mut self, speech_probability: f32, peak_dbfs: f32, speech_level_dbfs: f32);

    /// Resets internal state.
    fn reset(&mut self);
}

#[derive(Clone)]
struct SaturationProtectorState {
    headroom_db: f32,
    peak_delay_buffer: SaturationProtectorBuffer,
    max_peaks_dbfs: f32,
    time_since_push_ms: i32,
}

fn reset_saturation_protector_state(
    initial_headroom_db: f32,
    state: &mut SaturationProtectorState,
) {
    state.headroom_db = initial_headroom_db;
    state.peak_delay_buffer.reset();
    state.max_peaks_dbfs = MIN_LEVEL_DBFS;
    state.time_since_push_ms = 0;
}

fn update_saturation_protector_state(
    peak_dbfs: f32,
    speech_level_dbfs: f32,
    state: &mut SaturationProtectorState,
) {
    // Get max peak over super frame.
    state.max_peaks_dbfs = state.max_peaks_dbfs.max(peak_dbfs);
    state.time_since_push_ms += FRAME_DURATION_MS as i32;
    if state.time_since_push_ms > PEAK_ENVELOPER_SUPER_FRAME_LENGTH_MS {
        state.peak_delay_buffer.push_back(state.max_peaks_dbfs);
        state.max_peaks_dbfs = MIN_LEVEL_DBFS;
        state.time_since_push_ms = 0;
    }

    // Update headroom comparing estimated speech level and delayed max speech peak.
    let delayed_peak_dbfs = state
        .peak_delay_buffer
        .front()
        .unwrap_or(state.max_peaks_dbfs);
    let difference_db = delayed_peak_dbfs - speech_level_dbfs;
    if difference_db > state.headroom_db {
        // Attack.
        state.headroom_db = state.headroom_db * ATTACK + difference_db * (1.0 - ATTACK);
    } else {
        // Decay.
        state.headroom_db = state.headroom_db * DECAY + difference_db * (1.0 - DECAY);
    }

    state.headroom_db = state.headroom_db.clamp(MIN_MARGIN_DB, MAX_MARGIN_DB);
}

pub struct SaturationProtectorImpl {
    initial_headroom_db: f32,
    adjacent_speech_frames_threshold: i32,
    num_adjacent_speech_frames: i32,
    headroom_db: f32,
    preliminary_state: SaturationProtectorState,
    reliable_state: SaturationProtectorState,
}

impl SaturationProtectorImpl {
    pub fn new(initial_headroom_db: f32, adjacent_speech_frames_threshold: i32) -> Self {
        let mut s = Self {
            initial_headroom_db,
            adjacent_speech_frames_threshold,
            num_adjacent_speech_frames: 0,
            headroom_db: initial_headroom_db,
            preliminary_state: SaturationProtectorState {
                headroom_db: initial_headroom_db,
                peak_delay_buffer: SaturationProtectorBuffer::new(),
                max_peaks_dbfs: MIN_LEVEL_DBFS,
                time_since_push_ms: 0,
            },
            reliable_state: SaturationProtectorState {
                headroom_db: initial_headroom_db,
                peak_delay_buffer: SaturationProtectorBuffer::new(),
                max_peaks_dbfs: MIN_LEVEL_DBFS,
                time_since_push_ms: 0,
            },
        };
        s.reset();
        s
    }
}

impl SaturationProtector for SaturationProtectorImpl {
    fn headroom_db(&self) -> f32 {
        self.headroom_db
    }

    fn analyze(&mut self, speech_probability: f32, peak_dbfs: f32, speech_level_dbfs: f32) {
        if speech_probability < VAD_CONFIDENCE_THRESHOLD {
            // Not a speech frame.
            if self.adjacent_speech_frames_threshold > 1 {
                // Need to decide whether to discard or confirm updates based on speech sequence length.
                if self.num_adjacent_speech_frames >= self.adjacent_speech_frames_threshold {
                    // First non-speech frame after long enough speech sequence.
                    self.reliable_state = self.preliminary_state.clone();
                } else if self.num_adjacent_speech_frames > 0 {
                    // First non-speech after too-short sequence.
                    self.preliminary_state = self.reliable_state.clone();
                }
            }
            self.num_adjacent_speech_frames = 0;
        } else {
            // Speech frame observed.
            self.num_adjacent_speech_frames += 1;

            // Update preliminary level estimate.
            update_saturation_protector_state(
                peak_dbfs,
                speech_level_dbfs,
                &mut self.preliminary_state,
            );

            if self.num_adjacent_speech_frames >= self.adjacent_speech_frames_threshold {
                // Preliminary state is now reliable.
                self.headroom_db = self.preliminary_state.headroom_db;
            }
        }
    }

    fn reset(&mut self) {
        self.num_adjacent_speech_frames = 0;
        self.headroom_db = self.initial_headroom_db;
        reset_saturation_protector_state(self.initial_headroom_db, &mut self.preliminary_state);
        reset_saturation_protector_state(self.initial_headroom_db, &mut self.reliable_state);
    }
}

pub fn create_saturation_protector(
    initial_headroom_db: f32,
    adjacent_speech_frames_threshold: i32,
) -> Box<dyn SaturationProtector> {
    Box::new(SaturationProtectorImpl::new(
        initial_headroom_db,
        adjacent_speech_frames_threshold,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_processing::agc2::agc2_common::FRAME_DURATION_MS;

    const INITIAL_HEADROOM_DB: f32 = 20.0;
    const NO_ADJACENT_SPEECH_FRAMES_REQUIRED: i32 = 1;
    const MAX_SPEECH_PROBABILITY: f32 = 1.0;

    // Calls analyze() repeatedly and returns largest headroom difference between consecutive calls.
    fn run_on_constant_level(
        num_iterations: i32,
        speech_probability: f32,
        peak_dbfs: f32,
        speech_level_dbfs: f32,
        saturation_protector: &mut dyn SaturationProtector,
    ) -> f32 {
        let mut last_headroom = saturation_protector.headroom_db();
        let mut max_difference: f32 = 0.0;
        for _ in 0..num_iterations {
            saturation_protector.analyze(speech_probability, peak_dbfs, speech_level_dbfs);
            let new_headroom = saturation_protector.headroom_db();
            max_difference = max_difference.max((new_headroom - last_headroom).abs());
            last_headroom = new_headroom;
        }
        max_difference
    }

    #[test]
    fn reset() {
        let mut saturation_protector =
            create_saturation_protector(INITIAL_HEADROOM_DB, NO_ADJACENT_SPEECH_FRAMES_REQUIRED);
        let initial_headroom_db = saturation_protector.headroom_db();
        run_on_constant_level(
            10,
            MAX_SPEECH_PROBABILITY,
            0.0,
            -10.0,
            saturation_protector.as_mut(),
        );
        assert_ne!(initial_headroom_db, saturation_protector.headroom_db());
        saturation_protector.reset();
        assert_eq!(initial_headroom_db, saturation_protector.headroom_db());
    }

    #[test]
    fn estimates_crest_ratio() {
        const NUM_ITERATIONS: i32 = 2000;
        const PEAK_LEVEL_DBFS: f32 = -20.0;
        const CREST_FACTOR_DB: f32 = INITIAL_HEADROOM_DB + 1.0;
        const SPEECH_LEVEL_DBFS: f32 = PEAK_LEVEL_DBFS - CREST_FACTOR_DB;
        let max_difference_db = 0.5 * (INITIAL_HEADROOM_DB - CREST_FACTOR_DB).abs();

        let mut saturation_protector =
            create_saturation_protector(INITIAL_HEADROOM_DB, NO_ADJACENT_SPEECH_FRAMES_REQUIRED);
        run_on_constant_level(
            NUM_ITERATIONS,
            MAX_SPEECH_PROBABILITY,
            PEAK_LEVEL_DBFS,
            SPEECH_LEVEL_DBFS,
            saturation_protector.as_mut(),
        );
        assert!((saturation_protector.headroom_db() - CREST_FACTOR_DB).abs() <= max_difference_db);
    }

    #[test]
    fn change_slowly() {
        const NUM_ITERATIONS: i32 = 1000;
        const PEAK_LEVEL_DBFS: f32 = -20.0;
        const CREST_FACTOR_DB: f32 = INITIAL_HEADROOM_DB - 5.0;
        const OTHER_CREST_FACTOR_DB: f32 = INITIAL_HEADROOM_DB;
        const SPEECH_LEVEL_DBFS: f32 = PEAK_LEVEL_DBFS - CREST_FACTOR_DB;
        const OTHER_SPEECH_LEVEL_DBFS: f32 = PEAK_LEVEL_DBFS - OTHER_CREST_FACTOR_DB;

        let mut saturation_protector =
            create_saturation_protector(INITIAL_HEADROOM_DB, NO_ADJACENT_SPEECH_FRAMES_REQUIRED);

        let mut max_difference_db = run_on_constant_level(
            NUM_ITERATIONS,
            MAX_SPEECH_PROBABILITY,
            PEAK_LEVEL_DBFS,
            SPEECH_LEVEL_DBFS,
            saturation_protector.as_mut(),
        );
        max_difference_db = max_difference_db.max(run_on_constant_level(
            NUM_ITERATIONS,
            MAX_SPEECH_PROBABILITY,
            PEAK_LEVEL_DBFS,
            OTHER_SPEECH_LEVEL_DBFS,
            saturation_protector.as_mut(),
        ));

        const MAX_CHANGE_SPEED_DB_PER_SECOND: f32 = 0.5; // 1 dB / 2 seconds.
        assert!(
            max_difference_db <= MAX_CHANGE_SPEED_DB_PER_SECOND / 1000.0 * FRAME_DURATION_MS as f32
        );
    }

    #[test]
    fn do_not_adapt_to_short_speech_segments() {
        for adjacent_speech_frames_threshold in [2, 9, 17] {
            let mut saturation_protector =
                create_saturation_protector(INITIAL_HEADROOM_DB, adjacent_speech_frames_threshold);
            let initial_headroom_db = saturation_protector.headroom_db();
            run_on_constant_level(
                adjacent_speech_frames_threshold - 1,
                MAX_SPEECH_PROBABILITY,
                0.0,
                -10.0,
                saturation_protector.as_mut(),
            );
            assert_eq!(initial_headroom_db, saturation_protector.headroom_db());
        }
    }

    #[test]
    fn adapt_to_enough_speech_segments() {
        for adjacent_speech_frames_threshold in [2, 9, 17] {
            let mut saturation_protector =
                create_saturation_protector(INITIAL_HEADROOM_DB, adjacent_speech_frames_threshold);
            let initial_headroom_db = saturation_protector.headroom_db();
            run_on_constant_level(
                adjacent_speech_frames_threshold + 1,
                MAX_SPEECH_PROBABILITY,
                0.0,
                -10.0,
                saturation_protector.as_mut(),
            );
            assert_ne!(initial_headroom_db, saturation_protector.headroom_db());
        }
    }
}
