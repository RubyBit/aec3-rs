//! Wrapper around a mono VAD with first-channel analysis, resampling, and periodic reset.

use crate::audio_processing::agc2::cpu_features::AvailableCpuFeatures;
use crate::audio_processing::agc2::rnn_vad::common::{FEATURE_VECTOR_SIZE, FRAME_SIZE_10_MS_24_KHZ};
use crate::audio_processing::agc2::rnn_vad::features_extraction::FeaturesExtractor;
use crate::audio_processing::agc2::rnn_vad::rnn::RnnVad;
use crate::audio_processing::resampler::push_sinc_resampler::PushSincResampler;

const FRAME_DURATION_MS: usize = 10;
const NUM_FRAMES_PER_SECOND: usize = 100;
const VAD_RESET_PERIOD_MS: usize = 1500;

/// Single-channel VAD interface.
pub trait MonoVad {
    /// Returns input sample rate expected by [`Self::analyze`].
    fn sample_rate_hz(&self) -> usize;
    /// Resets the internal state.
    fn reset(&mut self);
    /// Analyzes one 10 ms mono frame and returns a speech probability.
    fn analyze(&mut self, frame: &[f32]) -> f32;
}

struct MonoVadImpl {
    features_extractor: FeaturesExtractor,
    rnn_vad: RnnVad,
}

impl MonoVadImpl {
    fn new(cpu_features: AvailableCpuFeatures) -> Self {
        Self {
            features_extractor: FeaturesExtractor::new(cpu_features),
            rnn_vad: RnnVad::new(cpu_features),
        }
    }
}

impl MonoVad for MonoVadImpl {
    fn sample_rate_hz(&self) -> usize {
        24_000
    }

    fn reset(&mut self) {
        self.rnn_vad.reset();
    }

    fn analyze(&mut self, frame: &[f32]) -> f32 {
        assert_eq!(frame.len(), FRAME_SIZE_10_MS_24_KHZ);
        let mut feature_vector = [0.0f32; FEATURE_VECTOR_SIZE];
        let frame_array: &[f32; FRAME_SIZE_10_MS_24_KHZ] = frame
            .try_into()
            .expect("24 kHz 10 ms frame has fixed size");
        let is_silence = self
            .features_extractor
            .check_silence_compute_features(frame_array, &mut feature_vector);
        self.rnn_vad
            .compute_vad_probability(&feature_vector, is_silence)
    }
}

/// Wraps a mono VAD to analyze the first channel of deinterleaved 10 ms frames.
///
/// The wrapper handles:
/// - input-to-VAD sample-rate conversion,
/// - periodic VAD reset,
/// - one-time VAD reset on construction.
pub struct VoiceActivityDetectorWrapper {
    vad_reset_period_frames: usize,
    frame_size: usize,
    time_to_vad_reset: usize,
    vad: Box<dyn MonoVad>,
    resampled_buffer: Vec<f32>,
    resampler: PushSincResampler,
}

impl VoiceActivityDetectorWrapper {
    /// Creates a wrapper using the default reset period and RNN-VAD backend.
    pub fn new(cpu_features: AvailableCpuFeatures, sample_rate_hz: usize) -> Self {
        Self::with_vad_reset_period(VAD_RESET_PERIOD_MS, cpu_features, sample_rate_hz)
    }

    /// Creates a wrapper with custom periodic reset and default RNN-VAD backend.
    pub fn with_vad_reset_period(
        vad_reset_period_ms: usize,
        cpu_features: AvailableCpuFeatures,
        sample_rate_hz: usize,
    ) -> Self {
        Self::with_custom_vad(
            vad_reset_period_ms,
            Box::new(MonoVadImpl::new(cpu_features)),
            sample_rate_hz,
        )
    }

    /// Creates a wrapper with a custom injected mono VAD.
    pub fn with_custom_vad(
        vad_reset_period_ms: usize,
        mut vad: Box<dyn MonoVad>,
        sample_rate_hz: usize,
    ) -> Self {
        assert_eq!(vad_reset_period_ms % FRAME_DURATION_MS, 0);
        let vad_reset_period_frames = vad_reset_period_ms / FRAME_DURATION_MS;
        assert!(vad_reset_period_frames > 1);

        assert_eq!(sample_rate_hz % NUM_FRAMES_PER_SECOND, 0);
        let frame_size = sample_rate_hz / NUM_FRAMES_PER_SECOND;

        let vad_sample_rate_hz = vad.sample_rate_hz();
        assert_eq!(vad_sample_rate_hz % NUM_FRAMES_PER_SECOND, 0);
        let vad_frame_size = vad_sample_rate_hz / NUM_FRAMES_PER_SECOND;

        vad.reset();

        Self {
            vad_reset_period_frames,
            frame_size,
            time_to_vad_reset: vad_reset_period_frames,
            vad,
            resampled_buffer: vec![0.0; vad_frame_size],
            resampler: PushSincResampler::new(frame_size, vad_frame_size),
        }
    }

    /// Analyzes the first channel in a deinterleaved frame.
    ///
    /// `frame[0]` must contain exactly 10 ms of samples at the wrapper input sample rate.
    pub fn analyze(&mut self, frame: &[&[f32]]) -> f32 {
        self.time_to_vad_reset = self
            .time_to_vad_reset
            .checked_sub(1)
            .expect("time_to_vad_reset must be positive");
        if self.time_to_vad_reset == 0 {
            self.vad.reset();
            self.time_to_vad_reset = self.vad_reset_period_frames;
        }

        let first_channel = frame.first().expect("frame must contain at least one channel");
        assert_eq!(first_channel.len(), self.frame_size);
        self.resampler
            .resample_f32(first_channel, &mut self.resampled_buffer);
        self.vad.analyze(&self.resampled_buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    const NUM_FRAMES_PER_SECOND: usize = 100;
    const SAMPLE_RATE_8_KHZ: usize = 8000;
    const NO_VAD_PERIODIC_RESET_MS: usize =
        FRAME_DURATION_MS * (usize::MAX / FRAME_DURATION_MS);

    #[derive(Default)]
    struct MockVadState {
        sample_rate_calls: usize,
        reset_calls: usize,
        analyze_calls: usize,
        analyzed_frame_sizes: Vec<usize>,
        next_probability_index: usize,
    }

    struct MockVad {
        sample_rate_hz: usize,
        probabilities: Vec<f32>,
        state: Rc<RefCell<MockVadState>>,
    }

    impl MonoVad for MockVad {
        fn sample_rate_hz(&self) -> usize {
            let mut s = self.state.borrow_mut();
            s.sample_rate_calls += 1;
            self.sample_rate_hz
        }

        fn reset(&mut self) {
            let mut s = self.state.borrow_mut();
            s.reset_calls += 1;
        }

        fn analyze(&mut self, frame: &[f32]) -> f32 {
            let mut s = self.state.borrow_mut();
            s.analyze_calls += 1;
            s.analyzed_frame_sizes.push(frame.len());
            assert!(!self.probabilities.is_empty());
            let value = self.probabilities[s.next_probability_index % self.probabilities.len()];
            s.next_probability_index += 1;
            value
        }
    }

    fn create_mock_vad_wrapper(
        vad_reset_period_ms: usize,
        input_sample_rate_hz: usize,
        vad_sample_rate_hz: usize,
        speech_probabilities: Vec<f32>,
    ) -> (VoiceActivityDetectorWrapper, Rc<RefCell<MockVadState>>) {
        let state = Rc::new(RefCell::new(MockVadState::default()));
        let vad = MockVad {
            sample_rate_hz: vad_sample_rate_hz,
            probabilities: speech_probabilities,
            state: state.clone(),
        };
        (
            VoiceActivityDetectorWrapper::with_custom_vad(
                vad_reset_period_ms,
                Box::new(vad),
                input_sample_rate_hz,
            ),
            state,
        )
    }

    #[test]
    fn ctor_and_init_read_sample_rate() {
        let (_wrapper, state) = create_mock_vad_wrapper(
            NO_VAD_PERIODIC_RESET_MS,
            SAMPLE_RATE_8_KHZ,
            SAMPLE_RATE_8_KHZ,
            vec![1.0],
        );
        let s = state.borrow();
        assert!(s.sample_rate_calls >= 1);
        assert_eq!(1, s.reset_calls);
    }

    #[test]
    fn check_speech_probabilities() {
        let speech_probabilities = vec![
            0.709f32, 0.484, 0.882, 0.167, 0.44, 0.525, 0.858, 0.314, 0.653, 0.965, 0.413, 0.0,
        ];
        let (mut wrapper, _state) = create_mock_vad_wrapper(
            NO_VAD_PERIODIC_RESET_MS,
            SAMPLE_RATE_8_KHZ,
            SAMPLE_RATE_8_KHZ,
            speech_probabilities.clone(),
        );

        let samples = vec![0.0f32; SAMPLE_RATE_8_KHZ / NUM_FRAMES_PER_SECOND];
        let frame = [samples.as_slice()];
        for expected in speech_probabilities {
            let prob = wrapper.analyze(&frame);
            assert_eq!(expected, prob);
        }
    }

    #[test]
    fn vad_no_periodic_reset() {
        let (mut wrapper, state) = create_mock_vad_wrapper(
            NO_VAD_PERIODIC_RESET_MS,
            SAMPLE_RATE_8_KHZ,
            SAMPLE_RATE_8_KHZ,
            vec![1.0],
        );
        let samples = vec![0.0f32; SAMPLE_RATE_8_KHZ / NUM_FRAMES_PER_SECOND];
        let frame = [samples.as_slice()];
        for _ in 0..19 {
            wrapper.analyze(&frame);
        }
        assert_eq!(1, state.borrow().reset_calls);
    }

    #[test]
    fn vad_periodic_reset() {
        for &num_frames in &[1usize, 19, 123] {
            for &vad_reset_period_frames in &[2usize, 5, 20, 53] {
                let (mut wrapper, state) = create_mock_vad_wrapper(
                    vad_reset_period_frames * FRAME_DURATION_MS,
                    SAMPLE_RATE_8_KHZ,
                    SAMPLE_RATE_8_KHZ,
                    vec![1.0],
                );
                let samples = vec![0.0f32; SAMPLE_RATE_8_KHZ / NUM_FRAMES_PER_SECOND];
                let frame = [samples.as_slice()];
                for _ in 0..num_frames {
                    wrapper.analyze(&frame);
                }
                let expected_reset_calls = 1 + num_frames / vad_reset_period_frames;
                assert_eq!(
                    expected_reset_calls,
                    state.borrow().reset_calls,
                    "num_frames={}, period={}",
                    num_frames,
                    vad_reset_period_frames
                );
            }
        }
    }

    #[test]
    fn check_resampled_frame_size() {
        for &input_sample_rate_hz in &[8000usize, 16000, 44100, 48000] {
            for &vad_sample_rate_hz in &[6000usize, 8000, 12000, 16000, 24000] {
                let (mut wrapper, state) = create_mock_vad_wrapper(
                    NO_VAD_PERIODIC_RESET_MS,
                    input_sample_rate_hz,
                    vad_sample_rate_hz,
                    vec![1.0],
                );
                let samples = vec![0.0f32; input_sample_rate_hz / NUM_FRAMES_PER_SECOND];
                let frame = [samples.as_slice()];
                wrapper.analyze(&frame);
                let analyzed_frame_size = *state
                    .borrow()
                    .analyzed_frame_sizes
                    .first()
                    .expect("analyze should be called once");
                assert_eq!(vad_sample_rate_hz / NUM_FRAMES_PER_SECOND, analyzed_frame_size);
            }
        }
    }
}
