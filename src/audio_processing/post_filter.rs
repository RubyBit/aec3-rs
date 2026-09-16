//! Post-processing filter for the fullband capture signal, compensating for
//! artefacts introduced by prior processing.
//!
//! Ported from `reference_aec_cpp/modules/audio_processing/post_filter.{h,cc}`.

use crate::audio_processing::utility::cascaded_biquad_filter::{
    BiQuadCoefficients, CascadedBiQuadFilter,
};

/// Removes frequencies above 19.5 kHz. Designed with:
///
/// ```text
/// sos = signal.iirdesign(
///     19200 * 2 / 48000, 19500 * 2 / 48000,
///     3, 20, ftype='cheby2', output="sos")
/// ```
const POST_FILTER_COEFFICIENTS_48K: [BiQuadCoefficients; 4] = [
    BiQuadCoefficients {
        b: [0.561_421_56, 1.114_999_31, 0.561_421_56],
        a: [1.579_142_49, 0.633_794_96],
    },
    BiQuadCoefficients {
        b: [1.0, 1.889_441_7, 1.0],
        a: [1.551_300_66, 0.687_087_19],
    },
    BiQuadCoefficients {
        b: [1.0, 1.760_573_1, 1.0],
        a: [1.530_013_28, 0.785_912_24],
    },
    BiQuadCoefficients {
        b: [1.0, 1.674_485_35, 1.0],
        a: [1.565_066_7, 0.920_965_76],
    },
];

/// Fullband post-processing filter.
pub struct PostFilter {
    filters: Vec<CascadedBiQuadFilter>,
}

impl PostFilter {
    /// Returns `None` below 48 kHz, where the filter passband already extends
    /// past Nyquist.
    pub fn create_if_needed(sample_rate_hz: i32, num_channels: usize) -> Option<Self> {
        if sample_rate_hz != 48_000 {
            return None;
        }

        Some(Self::new(&POST_FILTER_COEFFICIENTS_48K, num_channels))
    }

    fn new(coefficients: &[BiQuadCoefficients], num_channels: usize) -> Self {
        debug_assert!(!coefficients.is_empty());
        Self {
            filters: (0..num_channels)
                .map(|_| CascadedBiQuadFilter::from_coefficients(coefficients))
                .collect(),
        }
    }

    pub fn num_channels(&self) -> usize {
        self.filters.len()
    }

    /// Filters in place; each element of `audio` is one channel of fullband
    /// samples.
    pub fn process(&mut self, audio: &mut [Vec<f32>]) {
        assert_eq!(self.filters.len(), audio.len());
        for (filter, channel) in self.filters.iter_mut().zip(audio.iter_mut()) {
            filter.process_in_place(channel);
        }
    }

    pub fn reset(&mut self) {
        for filter in &mut self.filters {
            filter.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE_HZ: f32 = 48_000.0;

    /// Output amplitude of a sine at `frequency_hz`, measured past the
    /// transient.
    fn steady_state_amplitude(frequency_hz: f32) -> f32 {
        let mut filter = PostFilter::create_if_needed(48_000, 1).expect("filter at 48 kHz");
        let num_samples = 4800;
        let mut audio = vec![
            (0..num_samples)
                .map(|n| {
                    (2.0 * std::f32::consts::PI * frequency_hz * n as f32 / SAMPLE_RATE_HZ).sin()
                })
                .collect::<Vec<f32>>(),
        ];

        filter.process(&mut audio);

        audio[0][num_samples / 2..]
            .iter()
            .fold(0.0f32, |peak, sample| peak.max(sample.abs()))
    }

    #[test]
    fn not_created_below_48k() {
        assert!(PostFilter::create_if_needed(16_000, 1).is_none());
        assert!(PostFilter::create_if_needed(32_000, 1).is_none());
        assert!(PostFilter::create_if_needed(44_100, 1).is_none());
        assert!(PostFilter::create_if_needed(48_000, 2).is_some());
    }

    #[test]
    fn passband_is_preserved() {
        // Unity gain to within the 3 dB design ripple.
        for frequency_hz in [100.0, 1_000.0, 8_000.0, 16_000.0] {
            let amplitude = steady_state_amplitude(frequency_hz);
            assert!(
                (0.708..=1.05).contains(&amplitude),
                "{frequency_hz} Hz attenuated to {amplitude}"
            );
        }
    }

    #[test]
    fn stopband_is_attenuated() {
        // The design asks for 20 dB of stopband attenuation above 19.5 kHz.
        for frequency_hz in [19_500.0, 20_500.0, 22_000.0] {
            let amplitude = steady_state_amplitude(frequency_hz);
            assert!(
                amplitude < 0.1,
                "{frequency_hz} Hz only attenuated to {amplitude}"
            );
        }
    }

    #[test]
    fn channels_are_filtered_independently() {
        let mut filter = PostFilter::create_if_needed(48_000, 2).expect("filter at 48 kHz");
        assert_eq!(2, filter.num_channels());

        let mut audio = vec![vec![1.0f32; 480], vec![0.0f32; 480]];
        filter.process(&mut audio);

        assert!(audio[0].iter().any(|sample| *sample != 0.0));
        assert!(audio[1].iter().all(|sample| *sample == 0.0));
    }

    #[test]
    fn reset_clears_filter_state() {
        let mut filter = PostFilter::create_if_needed(48_000, 1).expect("filter at 48 kHz");

        // Short blocks, so the impulse response is still ringing in the next one.
        let mut impulse = vec![vec![0.0f32; 8]];
        impulse[0][0] = 1.0;
        filter.process(&mut impulse);

        let mut silence = vec![vec![0.0f32; 8]];
        filter.process(&mut silence);
        assert!(silence[0].iter().any(|sample| sample.abs() > 1e-6));

        filter.reset();
        let mut after_reset = vec![vec![0.0f32; 8]];
        filter.process(&mut after_reset);
        assert!(after_reset[0].iter().all(|sample| *sample == 0.0));
    }
}
