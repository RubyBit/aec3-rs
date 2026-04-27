//! Noise level estimator mirrored from WebRTC AGC2 noise floor estimator.

use crate::audio_processing::agc2::agc2_common::MIN_LEVEL_DBFS;

const FRAMES_PER_SECOND: usize = 100;

pub trait NoiseLevelEstimator {
    /// Analyzes a 10ms deinterleaved frame and returns the noise level in dBFS.
    fn analyze(&mut self, frame: &[&[f32]]) -> f32;
}

pub fn create_noise_floor_estimator() -> Box<dyn NoiseLevelEstimator> {
    Box::new(NoiseFloorEstimator::new())
}

fn frame_energy(audio: &[&[f32]]) -> f32 {
    let mut energy = 0.0f32;
    for &ch in audio {
        let channel_energy = ch.iter().fold(0.0f32, |acc, &b| acc + b * b);
        energy = energy.max(channel_energy);
    }
    energy
}

fn energy_to_dbfs(signal_energy: f32, num_samples: usize) -> f32 {
    assert!(signal_energy >= 0.0);
    let rms_square = signal_energy / num_samples as f32;
    if rms_square <= 1.0 {
        return MIN_LEVEL_DBFS;
    }
    10.0 * rms_square.log10() + MIN_LEVEL_DBFS
}

// Instant decay, slow attack.
fn smooth_noise_floor_estimate(current_estimate: f32, new_estimate: f32) -> f32 {
    const ATTACK: f32 = 0.5;
    if current_estimate < new_estimate {
        return ATTACK * new_estimate + (1.0 - ATTACK) * current_estimate;
    }
    new_estimate
}

pub struct NoiseFloorEstimator {
    sample_rate_hz: usize,
    min_noise_energy: f32,
    first_period: bool,
    preliminary_noise_energy_set: bool,
    preliminary_noise_energy: f32,
    noise_energy: f32,
    counter: i32,
}

impl NoiseFloorEstimator {
    // Update noise floor every 5 seconds.
    pub const UPDATE_PERIOD_NUM_FRAMES: i32 = 500;

    pub fn new() -> Self {
        let mut s = Self {
            sample_rate_hz: 0,
            min_noise_energy: 0.0,
            first_period: true,
            preliminary_noise_energy_set: false,
            preliminary_noise_energy: 0.0,
            noise_energy: 0.0,
            counter: Self::UPDATE_PERIOD_NUM_FRAMES,
        };
        // Initially assume 48kHz.
        s.initialize(48_000);
        s
    }

    fn initialize(&mut self, sample_rate_hz: usize) {
        self.sample_rate_hz = sample_rate_hz;
        self.first_period = true;
        self.preliminary_noise_energy_set = false;
        // Initialize minimum noise energy to -84 dBFS.
        self.min_noise_energy = sample_rate_hz as f32 * 2.0 * 2.0 / FRAMES_PER_SECOND as f32;
        self.preliminary_noise_energy = self.min_noise_energy;
        self.noise_energy = self.min_noise_energy;
        self.counter = Self::UPDATE_PERIOD_NUM_FRAMES;
    }
}

impl Default for NoiseFloorEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl NoiseLevelEstimator for NoiseFloorEstimator {
    fn analyze(&mut self, frame: &[&[f32]]) -> f32 {
        assert!(!frame.is_empty());
        let samples_per_channel = frame[0].len();
        assert!(samples_per_channel > 0);
        for &ch in frame {
            assert_eq!(ch.len(), samples_per_channel);
        }

        // Detect sample-rate changes.
        let sample_rate_hz = samples_per_channel * FRAMES_PER_SECOND;
        if sample_rate_hz != self.sample_rate_hz {
            self.initialize(sample_rate_hz);
        }

        let frame_energy = frame_energy(frame);
        if frame_energy <= self.min_noise_energy {
            return energy_to_dbfs(self.noise_energy, samples_per_channel);
        }

        if self.preliminary_noise_energy_set {
            self.preliminary_noise_energy = self.preliminary_noise_energy.min(frame_energy);
        } else {
            self.preliminary_noise_energy = frame_energy;
            self.preliminary_noise_energy_set = true;
        }

        if self.counter == 0 {
            self.first_period = false;
            self.noise_energy =
                smooth_noise_floor_estimate(self.noise_energy, self.preliminary_noise_energy);
            self.counter = Self::UPDATE_PERIOD_NUM_FRAMES;
            self.preliminary_noise_energy_set = false;
        } else if self.first_period {
            self.noise_energy = self.preliminary_noise_energy;
            self.counter -= 1;
        } else {
            self.noise_energy = self.noise_energy.min(self.preliminary_noise_energy);
            self.counter -= 1;
        }

        energy_to_dbfs(self.noise_energy, samples_per_channel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_processing::agc2::vector_float_frame::VectorFloatFrame;
    use crate::test_support::random::Random;
    use std::f32::consts::PI;

    const NUM_ITERATIONS: usize = 200;
    const MAX_S16: f32 = i16::MAX as f32;
    const MIN_S16: f32 = i16::MIN as f32;

    fn run_estimator<F: FnMut() -> f32>(
        mut sample_generator: F,
        estimator: &mut dyn NoiseLevelEstimator,
        sample_rate_hz: usize,
    ) -> f32 {
        let samples_per_channel = sample_rate_hz / FRAMES_PER_SECOND;
        let mut signal = VectorFloatFrame::new(1, samples_per_channel, 0.0);
        for _ in 0..NUM_ITERATIONS {
            let mut frame_view = signal.view();
            for j in 0..samples_per_channel {
                frame_view[0][j] = sample_generator();
            }
            let frame_const = signal.view_const();
            let _ = estimator.analyze(&frame_const);
        }
        let frame_const = signal.view_const();
        estimator.analyze(&frame_const)
    }

    struct WhiteNoiseGenerator {
        rng: Random,
        min_amplitude: i32,
        max_amplitude: i32,
    }

    impl WhiteNoiseGenerator {
        fn new(min_amplitude: f32, max_amplitude: f32) -> Self {
            let min_amplitude = min_amplitude as i32;
            let max_amplitude = max_amplitude as i32;
            assert!(min_amplitude < max_amplitude);
            assert!(MIN_S16 as i32 <= min_amplitude);
            assert!(max_amplitude <= MAX_S16 as i32);
            Self {
                rng: Random::new(42),
                min_amplitude,
                max_amplitude,
            }
        }

        fn next(&mut self) -> f32 {
            self.rng
                .rand_i32_range(self.min_amplitude, self.max_amplitude) as f32
        }
    }

    struct SineGenerator {
        amplitude: f32,
        frequency_hz: f32,
        sample_rate_hz: f32,
        x_radians: f32,
    }

    impl SineGenerator {
        fn new(amplitude: f32, frequency_hz: f32, sample_rate_hz: usize) -> Self {
            assert!(amplitude > 0.0);
            assert!(amplitude <= MAX_S16);
            Self {
                amplitude,
                frequency_hz,
                sample_rate_hz: sample_rate_hz as f32,
                x_radians: 0.0,
            }
        }

        fn next(&mut self) -> f32 {
            self.x_radians += self.frequency_hz / self.sample_rate_hz * 2.0 * PI;
            if self.x_radians >= 2.0 * PI {
                self.x_radians -= 2.0 * PI;
            }
            self.amplitude * self.x_radians.sin()
        }
    }

    struct PulseGenerator {
        pulse_amplitude: f32,
        no_pulse_amplitude: f32,
        samples_period: i32,
        sample_counter: i32,
    }

    impl PulseGenerator {
        fn new(
            pulse_amplitude: f32,
            no_pulse_amplitude: f32,
            frequency_hz: f32,
            sample_rate_hz: usize,
        ) -> Self {
            assert!(pulse_amplitude >= MIN_S16);
            assert!(pulse_amplitude <= MAX_S16);
            assert!(no_pulse_amplitude > MIN_S16);
            assert!(no_pulse_amplitude <= MAX_S16);
            assert!(sample_rate_hz as f32 > frequency_hz);
            Self {
                pulse_amplitude,
                no_pulse_amplitude,
                samples_period: (sample_rate_hz as f32 / frequency_hz) as i32,
                sample_counter: 0,
            }
        }

        fn next(&mut self) -> f32 {
            self.sample_counter += 1;
            if self.sample_counter >= self.samples_period {
                self.sample_counter -= self.samples_period;
            }
            if self.sample_counter == 0 {
                self.pulse_amplitude
            } else {
                self.no_pulse_amplitude
            }
        }
    }

    #[test]
    fn noise_floor_estimator_with_random_noise() {
        for sample_rate_hz in [8000usize, 16000, 32000, 48000] {
            let mut estimator = create_noise_floor_estimator();
            let mut generator = WhiteNoiseGenerator::new(MIN_S16, MAX_S16);
            let noise_level_dbfs =
                run_estimator(|| generator.next(), estimator.as_mut(), sample_rate_hz);
            assert!((noise_level_dbfs - (-5.5)).abs() <= 0.5);
        }
    }

    #[test]
    fn noise_floor_estimator_with_sine_tone() {
        for sample_rate_hz in [8000usize, 16000, 32000, 48000] {
            let mut estimator = create_noise_floor_estimator();
            let mut generator = SineGenerator::new(MAX_S16, 600.0, sample_rate_hz);
            let noise_level_dbfs =
                run_estimator(|| generator.next(), estimator.as_mut(), sample_rate_hz);
            assert!((noise_level_dbfs - (-3.0)).abs() <= 0.1);
        }
    }

    #[test]
    fn noise_floor_estimator_with_pulse_tone() {
        for sample_rate_hz in [8000usize, 16000, 32000, 48000] {
            let mut estimator = create_noise_floor_estimator();
            const NO_PULSE_AMPLITUDE: f32 = 10.0;
            let mut generator =
                PulseGenerator::new(MAX_S16, NO_PULSE_AMPLITUDE, 20.0, sample_rate_hz);
            let noise_level_dbfs =
                run_estimator(|| generator.next(), estimator.as_mut(), sample_rate_hz);
            let expected_noise_floor_dbfs = 20.0 * (NO_PULSE_AMPLITUDE / MAX_S16).log10();
            assert!((noise_level_dbfs - expected_noise_floor_dbfs).abs() <= 0.5);
        }
    }
}
