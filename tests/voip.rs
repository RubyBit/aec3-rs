use aec3::audio_processing::agc2::input_volume_controller::Config as InputVolumeControllerConfig;
use aec3::audio_processing::gain_controller2::GainController2Config;
use aec3::audio_processing::ns::{NsConfig, SuppressionLevel};
use aec3::voip::{VoipAec3, VoipAec3Error};

#[test]
fn voip_wrapper_processes_frame() {
    let sample_rate_hz = 16_000;
    let channels = 1usize;
    let mut pipeline = VoipAec3::builder(sample_rate_hz, channels, channels)
        .initial_delay_ms(0)
        .build()
        .expect("failed to build pipeline");

    let capture_frame_samples = pipeline.capture_frame_samples();
    let render_frame_samples = pipeline.render_frame_samples();
    let capture_frame_len = capture_frame_samples * channels;
    let render_frame_len = render_frame_samples * channels;

    let mut render = vec![0.0f32; render_frame_len];
    let mut capture = vec![0.0f32; capture_frame_len];
    for i in 0..capture_frame_samples {
        let t = i as f32 / 40.0;
        render[i] = (t).sin() * 0.8;
        capture[i] = (t * 2.0).cos() * 0.2 + render[i] * 0.5;
    }

    let mut output = vec![0.0f32; capture_frame_len];
    let metrics = pipeline
        .process(&capture, Some(&render), false, &mut output)
        .expect("processing should succeed");

    assert_ne!(capture, output, "AEC output should differ from raw capture");
    assert!(metrics.delay_ms >= 0);
}

#[test]
fn voip_wrapper_validates_frame_sizes() {
    let sample_rate_hz = 32_000;
    let channels = 2usize;
    let mut pipeline = VoipAec3::builder(sample_rate_hz, channels, channels)
        .build()
        .expect("failed to build pipeline");

    let capture_frame_samples = pipeline.capture_frame_samples();
    let render_frame_samples = pipeline.render_frame_samples();
    let capture_frame_len = capture_frame_samples * channels;
    let render_frame_len = render_frame_samples * channels;
    let capture = vec![0.0f32; capture_frame_len - 1];
    let render = vec![0.0f32; render_frame_len];
    let mut output = vec![0.0f32; capture_frame_len];
    let err = pipeline
        .process(&capture, Some(&render), false, &mut output)
        .unwrap_err();
    assert!(matches!(err, VoipAec3Error::CaptureFrameSize { .. }));
}

#[test]
fn voip_wrapper_supports_various_sample_rates() {
    let rates = [16_000, 32_000, 44_100, 48_000];
    
    for &rate in &rates {
        let channels = 1usize;
        let mut pipeline = VoipAec3::builder(rate, channels, channels)
            .build()
            .expect(&format!("failed to build pipeline for rate {}", rate));

        let capture_frame_samples = pipeline.capture_frame_samples();
        let render_frame_samples = pipeline.render_frame_samples();
        
        // Basic check that frame sizes make sense (10ms)
        let expected_samples = rate as usize / 100;
        assert_eq!(capture_frame_samples, expected_samples, "Incorrect capture frame size for rate {}", rate);
        assert_eq!(render_frame_samples, expected_samples, "Incorrect render frame size for rate {}", rate);

        let capture = vec![0.0f32; capture_frame_samples * channels];
        let render = vec![0.0f32; render_frame_samples * channels];
        let mut output = vec![0.0f32; capture_frame_samples * channels];

        pipeline
            .process(&capture, Some(&render), false, &mut output)
            .expect(&format!("processing failed for rate {}", rate));
    }
}

#[test]
fn voip_wrapper_supports_mixed_sample_rates() {
    let capture_rate = 48_000;
    let render_rate = 16_000;
    let channels = 1usize;
    
    let mut pipeline = VoipAec3::builder(capture_rate, channels, channels)
        .render_sample_rate_hz(render_rate)
        .build()
        .expect("failed to build pipeline with mixed rates");

    let capture_frame_samples = pipeline.capture_frame_samples();
    let render_frame_samples = pipeline.render_frame_samples();

    assert_eq!(capture_frame_samples, 480);
    assert_eq!(render_frame_samples, 160);

    let capture = vec![0.0f32; capture_frame_samples * channels];
    let render = vec![0.0f32; render_frame_samples * channels];
    let mut output = vec![0.0f32; capture_frame_samples * channels];

    pipeline
        .process(&capture, Some(&render), false, &mut output)
        .expect("processing failed for mixed rates");
}

#[test]
fn voip_wrapper_supports_noise_suppression_toggle_and_config() {
    let sample_rate_hz = 16_000;
    let channels = 1usize;

    let ns_config = NsConfig {
        target_level: SuppressionLevel::K18dB,
        analyze_linear_aec_output_when_available: false,
    };

    let mut pipeline = VoipAec3::builder(sample_rate_hz, channels, channels)
        .enable_noise_suppression(true)
        .noise_suppression_config(ns_config)
        .build()
        .expect("failed to build pipeline with noise suppression");

    let capture_frame_samples = pipeline.capture_frame_samples();
    let render_frame_samples = pipeline.render_frame_samples();
    let capture = vec![0.0f32; capture_frame_samples * channels];
    let render = vec![0.0f32; render_frame_samples * channels];
    let mut output = vec![0.0f32; capture_frame_samples * channels];

    pipeline
        .process(&capture, Some(&render), false, &mut output)
        .expect("processing should succeed with noise suppression enabled");
}

#[test]
fn voip_wrapper_supports_ns_analysis_on_linear_aec_output_when_available() {
    let sample_rate_hz = 48_000;
    let channels = 1usize;

    let ns_config = NsConfig {
        target_level: SuppressionLevel::K12dB,
        analyze_linear_aec_output_when_available: true,
    };

    let mut pipeline = VoipAec3::builder(sample_rate_hz, channels, channels)
        .enable_noise_suppression(true)
        .noise_suppression_config(ns_config)
        .build()
        .expect("failed to build pipeline with linear-output NS analysis");

    let capture_frame_samples = pipeline.capture_frame_samples();
    let render_frame_samples = pipeline.render_frame_samples();

    let mut capture = vec![0.0f32; capture_frame_samples * channels];
    let mut render = vec![0.0f32; render_frame_samples * channels];
    for i in 0..capture_frame_samples {
        let t = i as f32 / 50.0;
        capture[i] = (t * 1.7).sin() * 1200.0;
        render[i] = (t * 1.1).cos() * 700.0;
    }

    let mut output = vec![0.0f32; capture_frame_samples * channels];
    pipeline
        .process(&capture, Some(&render), false, &mut output)
        .expect("processing should succeed with linear-output NS analysis enabled");
}

#[test]
fn voip_wrapper_supports_gain_controller2_toggle_and_config() {
    let sample_rate_hz = 16_000;
    let channels = 1usize;

    let mut gc2_config = GainController2Config::default();
    gc2_config.fixed_digital.gain_db = 6.0;

    let mut pipeline = VoipAec3::builder(sample_rate_hz, channels, channels)
        .enable_gain_controller2(true)
        .gain_controller2_config(gc2_config)
        .input_volume_controller_config(InputVolumeControllerConfig::default())
        .build()
        .expect("failed to build pipeline with gain controller2");

    let capture_frame_samples = pipeline.capture_frame_samples();
    let render_frame_samples = pipeline.render_frame_samples();
    let capture = vec![1000.0f32; capture_frame_samples * channels];
    let render = vec![0.0f32; render_frame_samples * channels];
    let mut output = vec![0.0f32; capture_frame_samples * channels];

    pipeline
        .process(&capture, Some(&render), false, &mut output)
        .expect("processing should succeed with gain controller2 enabled");
}

#[test]
fn voip_wrapper_validates_applied_input_volume_for_gain_controller2() {
    let sample_rate_hz = 16_000;
    let channels = 1usize;

    let mut pipeline = VoipAec3::builder(sample_rate_hz, channels, channels)
        .enable_gain_controller2(true)
        .build()
        .expect("failed to build pipeline with gain controller2");

    let err = pipeline.set_applied_input_volume(300).unwrap_err();
    assert!(matches!(
        err,
        VoipAec3Error::InvalidAppliedInputVolume { actual: 300 }
    ));
}

#[test]
fn voip_wrapper_fixed_gain_control_changes_output_level() {
    let sample_rate_hz = 16_000;
    let channels = 1usize;

    let mut disabled_adaptive = GainController2Config::default();
    disabled_adaptive.adaptive_digital.enabled = false;
    disabled_adaptive.input_volume_controller.enabled = false;
    disabled_adaptive.fixed_digital.gain_db = 0.0;

    let mut pipeline_no_gain = VoipAec3::builder(sample_rate_hz, channels, channels)
        .enable_gain_controller2(true)
        .gain_controller2_config(disabled_adaptive)
        .build()
        .expect("failed to build no-gain pipeline");

    let mut pipeline_with_gain = VoipAec3::builder(sample_rate_hz, channels, channels)
        .enable_gain_controller2(true)
        .gain_controller2_config(disabled_adaptive)
        .build()
        .expect("failed to build gain pipeline");
    pipeline_with_gain.set_fixed_gain_db(20.0);

    let capture_frame_samples = pipeline_no_gain.capture_frame_samples();
    let capture = vec![0.01f32; capture_frame_samples * channels];
    let mut out_no_gain = vec![0.0f32; capture_frame_samples * channels];
    let mut out_with_gain = vec![0.0f32; capture_frame_samples * channels];

    for _ in 0..5 {
        pipeline_no_gain
            .process(&capture, None, false, &mut out_no_gain)
            .expect("processing should succeed");
        pipeline_with_gain
            .process(&capture, None, false, &mut out_with_gain)
            .expect("processing should succeed");
    }

    let no_gain_avg_abs =
        out_no_gain.iter().map(|v| v.abs()).sum::<f32>() / out_no_gain.len() as f32;
    let with_gain_avg_abs =
        out_with_gain.iter().map(|v| v.abs()).sum::<f32>() / out_with_gain.len() as f32;

    assert!(
        with_gain_avg_abs > no_gain_avg_abs,
        "expected fixed gain to increase output amplitude"
    );
}

#[test]
fn voip_wrapper_reports_recommended_input_volume_with_gain_controller2() {
    let sample_rate_hz = 16_000;
    let channels = 1usize;

    let mut gc2_config = GainController2Config::default();
    gc2_config.adaptive_digital.enabled = true;
    gc2_config.input_volume_controller.enabled = true;

    let mut pipeline = VoipAec3::builder(sample_rate_hz, channels, channels)
        .enable_gain_controller2(true)
        .gain_controller2_config(gc2_config)
        .build()
        .expect("failed to build pipeline with gain controller2");

    let capture_frame_samples = pipeline.capture_frame_samples();
    let capture = vec![3000.0f32; capture_frame_samples * channels];
    let mut output = vec![0.0f32; capture_frame_samples * channels];

    pipeline
        .set_applied_input_volume(128)
        .expect("valid applied input volume should be accepted");

    for _ in 0..10 {
        pipeline
            .process(&capture, None, false, &mut output)
            .expect("processing should succeed");
        if pipeline.recommended_input_volume().is_some() {
            break;
        }
    }

    assert!(
        pipeline.recommended_input_volume().is_some(),
        "expected gain controller2 to provide a recommended input volume"
    );
}

#[test]
fn voip_wrapper_does_not_recommend_input_volume_without_applied_input_volume() {
    let sample_rate_hz = 16_000;
    let channels = 1usize;

    let mut gc2_config = GainController2Config::default();
    gc2_config.adaptive_digital.enabled = true;
    gc2_config.input_volume_controller.enabled = true;

    let mut pipeline = VoipAec3::builder(sample_rate_hz, channels, channels)
        .enable_gain_controller2(true)
        .gain_controller2_config(gc2_config)
        .build()
        .expect("failed to build pipeline with gain controller2");

    let capture_frame_samples = pipeline.capture_frame_samples();
    let capture = vec![3000.0f32; capture_frame_samples * channels];
    let mut output = vec![0.0f32; capture_frame_samples * channels];

    for _ in 0..10 {
        pipeline
            .process(&capture, None, false, &mut output)
            .expect("processing should succeed");
    }

    assert_eq!(
        pipeline.recommended_input_volume(),
        None,
        "recommended input volume should remain None when applied input volume was never provided"
    );
}

#[test]
fn voip_wrapper_supports_noise_suppression_with_gain_controller2() {
    let sample_rate_hz = 16_000;
    let channels = 1usize;

    let mut pipeline = VoipAec3::builder(sample_rate_hz, channels, channels)
        .enable_noise_suppression(true)
        .noise_suppression_config(NsConfig::default())
        .enable_gain_controller2(true)
        .build()
        .expect("failed to build pipeline with NS + gain controller2");

    let capture_frame_samples = pipeline.capture_frame_samples();
    let render_frame_samples = pipeline.render_frame_samples();
    let capture = vec![500.0f32; capture_frame_samples * channels];
    let render = vec![100.0f32; render_frame_samples * channels];
    let mut output = vec![0.0f32; capture_frame_samples * channels];

    pipeline
        .process(&capture, Some(&render), false, &mut output)
        .expect("processing should succeed with NS + gain controller2");
}
