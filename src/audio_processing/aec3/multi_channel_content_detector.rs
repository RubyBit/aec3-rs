//! Detection of multichannel render content.
//!
//! Ported from `modules/audio_processing/aec3/multi_channel_content_detector.{h,cc}`.

const NUM_FRAMES_PER_SECOND: i32 = 100;

/// In order to avoid logging metrics for very short lifetimes that are unlikely
/// to reflect real calls and that may dilute the "real" data, logging is
/// limited to lifetimes of at least 5 seconds.
const MIN_NUMBER_OF_FRAMES_REQUIRED_TO_LOG_METRICS: i32 = 500;

/// Continuous metrics are logged every 10 seconds.
const FRAMES_PER_10_SECONDS: i32 = 1000;

/// Compares the left and right channels in the render `frame` to determine
/// whether the signal is a proper stereo signal. To allow for differences
/// introduced by hardware drivers, a threshold `detection_threshold` is used
/// for the detection.
fn has_stereo_content(frame: &[Vec<Vec<f32>>], detection_threshold: f32) -> bool {
    if frame[0].len() < 2 {
        return false;
    }

    frame.iter().any(|band| {
        band[0]
            .iter()
            .zip(band[1].iter())
            .any(|(left, right)| (left - right).abs() > detection_threshold)
    })
}

/// Tracks metrics for the amount of multichannel content detected.
///
/// The reference emits these through `RTC_HISTOGRAM_BOOLEAN`. This crate has no
/// histogram facility, so — following the convention already used by
/// [`BlockProcessorMetrics`](super::block_processor_metrics::BlockProcessorMetrics)
/// — the state machine is ported verbatim and the values that would have been
/// logged are recorded for inspection instead.
pub struct MetricsLogger {
    frame_counter: i32,

    /// Counts the number of frames of persistent multichannel audio observed
    /// during the current metrics collection interval.
    persistent_multichannel_frame_counter: i32,

    /// Indicates whether persistent multichannel content has ever been
    /// detected.
    any_multichannel_content_detected: bool,

    /// Values logged to
    /// `WebRTC.Audio.EchoCanceller.ProcessingPersistentMultichannelContent`.
    processing_persistent_multichannel_content: Vec<bool>,
}

impl MetricsLogger {
    fn new() -> Self {
        Self {
            frame_counter: 0,
            persistent_multichannel_frame_counter: 0,
            any_multichannel_content_detected: false,
            processing_persistent_multichannel_content: Vec::new(),
        }
    }

    fn update(&mut self, persistent_multichannel_content_detected: bool) {
        self.frame_counter += 1;
        if persistent_multichannel_content_detected {
            self.any_multichannel_content_detected = true;
            self.persistent_multichannel_frame_counter += 1;
        }

        if self.frame_counter < MIN_NUMBER_OF_FRAMES_REQUIRED_TO_LOG_METRICS {
            return;
        }
        if self.frame_counter % FRAMES_PER_10_SECONDS != 0 {
            return;
        }
        let mostly_multichannel_last_10_seconds =
            self.persistent_multichannel_frame_counter >= FRAMES_PER_10_SECONDS / 2;
        self.processing_persistent_multichannel_content
            .push(mostly_multichannel_last_10_seconds);

        self.persistent_multichannel_frame_counter = 0;
    }

    /// The periodic samples logged so far.
    pub fn processing_persistent_multichannel_content(&self) -> &[bool] {
        &self.processing_persistent_multichannel_content
    }

    /// The value the reference logs to
    /// `WebRTC.Audio.EchoCanceller.PersistentMultichannelContentEverDetected`
    /// from its destructor, or `None` when the lifetime was too short to log.
    pub fn persistent_multichannel_content_ever_detected(&self) -> Option<bool> {
        if self.frame_counter < MIN_NUMBER_OF_FRAMES_REQUIRED_TO_LOG_METRICS {
            return None;
        }
        Some(self.any_multichannel_content_detected)
    }
}

/// Analyzes audio content to determine whether the contained audio is proper
/// multichannel, or only upmixed mono. To allow for differences introduced by
/// hardware drivers, a threshold `detection_threshold` is used for the
/// detection.
pub struct MultiChannelContentDetector {
    detect_stereo_content: bool,
    detection_threshold: f32,
    detection_timeout_threshold_frames: Option<i32>,
    stereo_detection_hysteresis_frames: i32,

    /// Collects metrics on the amount of multichannel content detected. Only
    /// created if `num_render_input_channels > 1` and `detect_stereo_content`
    /// is true.
    metrics_logger: Option<MetricsLogger>,

    persistent_multichannel_content_detected: bool,
    temporary_multichannel_content_detected: bool,
    frames_since_stereo_detected_last: i64,
    consecutive_frames_with_stereo: i64,
}

impl MultiChannelContentDetector {
    /// If `stereo_detection_timeout_threshold_seconds` <= 0, no timeout is
    /// applied: once multichannel is detected, the detector remains in that
    /// state for its lifetime.
    pub fn new(
        detect_stereo_content: bool,
        num_render_input_channels: usize,
        detection_threshold: f32,
        stereo_detection_timeout_threshold_seconds: i32,
        stereo_detection_hysteresis_seconds: f32,
    ) -> Self {
        Self {
            detect_stereo_content,
            detection_threshold,
            detection_timeout_threshold_frames: if stereo_detection_timeout_threshold_seconds > 0 {
                Some(stereo_detection_timeout_threshold_seconds * NUM_FRAMES_PER_SECOND)
            } else {
                None
            },
            stereo_detection_hysteresis_frames: (stereo_detection_hysteresis_seconds
                * NUM_FRAMES_PER_SECOND as f32)
                as i32,
            metrics_logger: (detect_stereo_content && num_render_input_channels > 1)
                .then(MetricsLogger::new),
            persistent_multichannel_content_detected: !detect_stereo_content
                && num_render_input_channels > 1,
            temporary_multichannel_content_detected: false,
            frames_since_stereo_detected_last: 0,
            consecutive_frames_with_stereo: 0,
        }
    }

    /// Compares the left and right channels in the render `frame` to determine
    /// whether the signal is a proper multichannel signal. Returns a bool
    /// indicating whether a change in the proper multichannel content was
    /// detected.
    pub fn update_detection(&mut self, frame: &[Vec<Vec<f32>>]) -> bool {
        if !self.detect_stereo_content {
            debug_assert_eq!(
                frame[0].len() > 1,
                self.persistent_multichannel_content_detected
            );
            return false;
        }

        let previous_persistent_multichannel_content_detected =
            self.persistent_multichannel_content_detected;
        let stereo_detected_in_frame = has_stereo_content(frame, self.detection_threshold);

        self.consecutive_frames_with_stereo = if stereo_detected_in_frame {
            self.consecutive_frames_with_stereo + 1
        } else {
            0
        };
        self.frames_since_stereo_detected_last = if stereo_detected_in_frame {
            0
        } else {
            self.frames_since_stereo_detected_last + 1
        };

        // Detect persistent multichannel content.
        if self.consecutive_frames_with_stereo > self.stereo_detection_hysteresis_frames as i64 {
            self.persistent_multichannel_content_detected = true;
        }
        if let Some(timeout) = self.detection_timeout_threshold_frames
            && self.frames_since_stereo_detected_last >= timeout as i64
        {
            self.persistent_multichannel_content_detected = false;
        }

        // Detect temporary multichannel content.
        self.temporary_multichannel_content_detected =
            if self.persistent_multichannel_content_detected {
                false
            } else {
                stereo_detected_in_frame
            };

        if let Some(metrics_logger) = self.metrics_logger.as_mut() {
            metrics_logger.update(self.persistent_multichannel_content_detected);
        }

        previous_persistent_multichannel_content_detected
            != self.persistent_multichannel_content_detected
    }

    pub fn is_proper_multi_channel_content_detected(&self) -> bool {
        self.persistent_multichannel_content_detected
    }

    pub fn is_temporary_multi_channel_content_detected(&self) -> bool {
        self.temporary_multichannel_content_detected
    }

    /// The metrics logger, present only when metrics are collected.
    pub fn metrics_logger(&self) -> Option<&MetricsLogger> {
        self.metrics_logger.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(num_channels: usize, values: &[f32]) -> Vec<Vec<Vec<f32>>> {
        assert_eq!(values.len(), num_channels);
        vec![values.iter().map(|&v| vec![v; 160]).collect()]
    }

    fn true_stereo_frame() -> Vec<Vec<Vec<f32>>> {
        frame(2, &[100.0, 101.0])
    }

    fn fake_stereo_frame() -> Vec<Vec<Vec<f32>>> {
        frame(2, &[100.0, 100.0])
    }

    #[test]
    fn handling_of_mono() {
        let mc = MultiChannelContentDetector::new(true, 1, 0.0, 0, 0.0);
        assert!(!mc.is_proper_multi_channel_content_detected());
    }

    #[test]
    fn handling_of_mono_and_detection_off() {
        let mc = MultiChannelContentDetector::new(false, 1, 0.0, 0, 0.0);
        assert!(!mc.is_proper_multi_channel_content_detected());
    }

    #[test]
    fn handling_of_detection_off() {
        let mut mc = MultiChannelContentDetector::new(false, 2, 0.0, 0, 0.0);
        assert!(mc.is_proper_multi_channel_content_detected());

        let frame = frame(2, &[100.0, 101.0]);

        assert!(!mc.update_detection(&frame));
        assert!(mc.is_proper_multi_channel_content_detected());

        assert!(!mc.update_detection(&frame));
    }

    #[test]
    fn initial_detection_of_stereo() {
        let mc = MultiChannelContentDetector::new(true, 2, 0.0, 0, 0.0);
        assert!(!mc.is_proper_multi_channel_content_detected());
    }

    #[test]
    fn detection_when_fake_stereo() {
        let mut mc = MultiChannelContentDetector::new(true, 2, 0.0, 0, 0.0);
        let frame = fake_stereo_frame();
        assert!(!mc.update_detection(&frame));
        assert!(!mc.is_proper_multi_channel_content_detected());

        assert!(!mc.update_detection(&frame));
    }

    #[test]
    fn detection_when_stereo() {
        let mut mc = MultiChannelContentDetector::new(true, 2, 0.0, 0, 0.0);
        let frame = true_stereo_frame();
        assert!(mc.update_detection(&frame));
        assert!(mc.is_proper_multi_channel_content_detected());

        assert!(!mc.update_detection(&frame));
    }

    #[test]
    fn detection_when_stereo_after_a_while() {
        let mut mc = MultiChannelContentDetector::new(true, 2, 0.0, 0, 0.0);

        let frame = fake_stereo_frame();
        assert!(!mc.update_detection(&frame));
        assert!(!mc.is_proper_multi_channel_content_detected());

        assert!(!mc.update_detection(&frame));

        let frame = true_stereo_frame();

        assert!(mc.update_detection(&frame));
        assert!(mc.is_proper_multi_channel_content_detected());

        assert!(!mc.update_detection(&frame));
    }

    #[test]
    fn detection_with_stereo_below_threshold() {
        const THRESHOLD: f32 = 1.0;
        let mut mc = MultiChannelContentDetector::new(true, 2, THRESHOLD, 0, 0.0);
        let frame = frame(2, &[100.0, 100.0 + THRESHOLD]);

        assert!(!mc.update_detection(&frame));
        assert!(!mc.is_proper_multi_channel_content_detected());

        assert!(!mc.update_detection(&frame));
    }

    #[test]
    fn detection_with_stereo_above_threshold() {
        const THRESHOLD: f32 = 1.0;
        let mut mc = MultiChannelContentDetector::new(true, 2, THRESHOLD, 0, 0.0);
        let frame = frame(2, &[100.0, 100.0 + THRESHOLD + 0.1]);

        assert!(mc.update_detection(&frame));
        assert!(mc.is_proper_multi_channel_content_detected());

        assert!(!mc.update_detection(&frame));
    }

    /// `TimeOutBehaviorForNonTrueStereo`, over the reference's parameter grid.
    #[test]
    fn time_out_behavior_for_non_true_stereo() {
        for detect_stereo_content in [false, true] {
            for stereo_detection_timeout_threshold_seconds in [0, 1, 10] {
                let stereo_detection_timeout_threshold_frames =
                    stereo_detection_timeout_threshold_seconds * NUM_FRAMES_PER_SECOND;

                let mut mc = MultiChannelContentDetector::new(
                    detect_stereo_content,
                    2,
                    0.0,
                    stereo_detection_timeout_threshold_seconds,
                    0.0,
                );
                let true_stereo_frame = true_stereo_frame();
                let fake_stereo_frame = fake_stereo_frame();

                // Pass fake stereo frames and verify the content detection.
                for _ in 0..10 {
                    assert!(!mc.update_detection(&fake_stereo_frame));
                    assert_eq!(
                        mc.is_proper_multi_channel_content_detected(),
                        !detect_stereo_content
                    );
                }

                // Pass a true stereo frame and verify that it is properly
                // detected.
                assert_eq!(mc.update_detection(&true_stereo_frame), detect_stereo_content);
                assert!(mc.is_proper_multi_channel_content_detected());

                // Pass fake stereo frames until any timeouts are about to occur.
                for _ in 0..stereo_detection_timeout_threshold_frames - 1 {
                    assert!(!mc.update_detection(&fake_stereo_frame));
                    assert!(mc.is_proper_multi_channel_content_detected());
                }

                // Pass a fake stereo frame and verify that any timeouts
                // properly occur.
                let times_out =
                    detect_stereo_content && stereo_detection_timeout_threshold_frames > 0;
                assert_eq!(mc.update_detection(&fake_stereo_frame), times_out);
                assert_eq!(mc.is_proper_multi_channel_content_detected(), !times_out);

                // Pass fake stereo frames and verify the behavior after any
                // timeout.
                for _ in 0..10 {
                    assert!(!mc.update_detection(&fake_stereo_frame));
                    assert_eq!(mc.is_proper_multi_channel_content_detected(), !times_out);
                }
            }
        }
    }

    /// `PeriodBeforeStereoDetectionIsTriggered`, over the reference's parameter
    /// grid.
    ///
    /// The reference parameterises the hysteresis over `{0.0f, 0.1f, 0.2f}` but
    /// then binds it to an `int`, so every case truncates to zero and the test
    /// returns before reaching the hysteresis assertions. That truncation is
    /// reproduced here so the port stays faithful; the hysteresis path proper
    /// is covered by [`hysteresis_delays_persistent_detection`], which is not
    /// part of the reference suite.
    #[test]
    fn period_before_stereo_detection_is_triggered() {
        for detect_stereo_content in [false, true] {
            for stereo_detection_hysteresis_seconds_param in [0.0f32, 0.1, 0.2] {
                let stereo_detection_hysteresis_seconds =
                    stereo_detection_hysteresis_seconds_param as i32;
                let stereo_detection_hysteresis_frames =
                    stereo_detection_hysteresis_seconds * NUM_FRAMES_PER_SECOND;

                let mut mc = MultiChannelContentDetector::new(
                    detect_stereo_content,
                    2,
                    0.0,
                    0,
                    stereo_detection_hysteresis_seconds as f32,
                );
                let true_stereo_frame = true_stereo_frame();
                let fake_stereo_frame = fake_stereo_frame();

                // Pass fake stereo frames and verify the content detection.
                for _ in 0..10 {
                    assert!(!mc.update_detection(&fake_stereo_frame));
                    assert_eq!(
                        mc.is_proper_multi_channel_content_detected(),
                        !detect_stereo_content
                    );
                    assert!(!mc.is_temporary_multi_channel_content_detected());
                }

                // Pass two true stereo frames and verify that they are properly
                // detected.
                assert!(
                    stereo_detection_hysteresis_frames > 2
                        || stereo_detection_hysteresis_frames == 0
                );
                for k in 0..2 {
                    if detect_stereo_content {
                        if stereo_detection_hysteresis_seconds == 0 {
                            assert_eq!(mc.update_detection(&true_stereo_frame), k == 0);
                            assert!(mc.is_proper_multi_channel_content_detected());
                            assert!(!mc.is_temporary_multi_channel_content_detected());
                        } else {
                            assert!(!mc.update_detection(&true_stereo_frame));
                            assert!(!mc.is_proper_multi_channel_content_detected());
                            assert!(mc.is_temporary_multi_channel_content_detected());
                        }
                    } else {
                        assert!(!mc.update_detection(&true_stereo_frame));
                        assert!(mc.is_proper_multi_channel_content_detected());
                        assert!(!mc.is_temporary_multi_channel_content_detected());
                    }
                }

                if stereo_detection_hysteresis_seconds == 0 {
                    continue;
                }

                // Pass true stereo frames until any timeouts are about to occur.
                for _ in 0..stereo_detection_hysteresis_frames - 3 {
                    assert!(!mc.update_detection(&true_stereo_frame));
                    assert_eq!(
                        mc.is_proper_multi_channel_content_detected(),
                        !detect_stereo_content
                    );
                    assert_eq!(
                        mc.is_temporary_multi_channel_content_detected(),
                        detect_stereo_content
                    );
                }

                // Pass a true stereo frame and verify that it is properly
                // detected.
                assert_eq!(mc.update_detection(&true_stereo_frame), detect_stereo_content);
                assert!(mc.is_proper_multi_channel_content_detected());
                assert!(!mc.is_temporary_multi_channel_content_detected());

                // Pass an additional true stereo frame and verify that it is
                // properly detected.
                assert!(!mc.update_detection(&true_stereo_frame));
                assert!(mc.is_proper_multi_channel_content_detected());
                assert!(!mc.is_temporary_multi_channel_content_detected());

                // Pass a fake stereo frame and verify that it is properly
                // detected.
                assert!(!mc.update_detection(&fake_stereo_frame));
                assert!(mc.is_proper_multi_channel_content_detected());
                assert!(!mc.is_temporary_multi_channel_content_detected());
            }
        }
    }

    /// Not part of the reference suite: covers the hysteresis path that
    /// [`period_before_stereo_detection_is_triggered`] cannot reach because of
    /// the reference's float-to-int truncation.
    #[test]
    fn hysteresis_delays_persistent_detection() {
        let hysteresis_seconds = 0.1f32;
        let hysteresis_frames = (hysteresis_seconds * NUM_FRAMES_PER_SECOND as f32) as i32;
        assert_eq!(hysteresis_frames, 10);

        let mut mc = MultiChannelContentDetector::new(true, 2, 0.0, 0, hysteresis_seconds);
        let true_stereo_frame = true_stereo_frame();

        // Persistent detection only triggers once the stereo run exceeds the
        // hysteresis window; until then the content is reported as temporary.
        for _ in 0..hysteresis_frames {
            assert!(!mc.update_detection(&true_stereo_frame));
            assert!(!mc.is_proper_multi_channel_content_detected());
            assert!(mc.is_temporary_multi_channel_content_detected());
        }

        assert!(mc.update_detection(&true_stereo_frame));
        assert!(mc.is_proper_multi_channel_content_detected());
        assert!(!mc.is_temporary_multi_channel_content_detected());
    }

    /// `ReportsNoMetrics`: no metrics are collected when the reference audio is
    /// single channel, or when dynamic detection is disabled.
    #[test]
    fn reports_no_metrics() {
        for (detect_stereo_content, channel_count) in [(false, 2), (true, 1)] {
            let audio_frame = frame(channel_count, &vec![100.0; channel_count]);
            let mut mc = MultiChannelContentDetector::new(
                detect_stereo_content,
                channel_count,
                0.0,
                1,
                0.0,
            );
            for _ in 0..20 * NUM_FRAMES_PER_SECOND {
                mc.update_detection(&audio_frame);
            }
            assert!(mc.metrics_logger().is_none());
        }
    }

    /// `ReportsNoMetricsForShortLifetime`: after 3 seconds, nothing is reported.
    #[test]
    fn reports_no_metrics_for_short_lifetime() {
        const TOO_FEW_FRAMES_TO_LOG_METRICS: i32 = 3 * NUM_FRAMES_PER_SECOND;
        let audio_frame = frame(2, &[100.0, 100.0]);
        let mut mc = MultiChannelContentDetector::new(true, 2, 0.0, 1, 0.0);
        for _ in 0..TOO_FEW_FRAMES_TO_LOG_METRICS {
            mc.update_detection(&audio_frame);
        }
        let logger = mc.metrics_logger().expect("metrics are collected");
        assert!(logger.processing_persistent_multichannel_content().is_empty());
        assert_eq!(logger.persistent_multichannel_content_ever_detected(), None);
    }

    /// `ReportsMetrics`: after 25 seconds, metrics are reported.
    #[test]
    fn reports_metrics() {
        let true_stereo_frame = true_stereo_frame();
        let fake_stereo_frame = fake_stereo_frame();

        let mut mc = MultiChannelContentDetector::new(true, 2, 0.0, 1, 0.0);
        for _ in 0..10 * NUM_FRAMES_PER_SECOND {
            mc.update_detection(&true_stereo_frame);
        }
        for _ in 0..15 * NUM_FRAMES_PER_SECOND {
            mc.update_detection(&fake_stereo_frame);
        }

        // After 10 seconds of true stereo and the remainder fake stereo, we
        // expect one lifetime metric sample (multichannel detected) and two
        // periodic samples (one multichannel, one mono).
        let logger = mc.metrics_logger().expect("metrics are collected");
        assert_eq!(
            logger.persistent_multichannel_content_ever_detected(),
            Some(true)
        );

        let periodic = logger.processing_persistent_multichannel_content();
        assert_eq!(periodic.len(), 2);
        assert_eq!(periodic.iter().filter(|&&v| v).count(), 1);
        assert_eq!(periodic.iter().filter(|&&v| !v).count(), 1);
    }
}
