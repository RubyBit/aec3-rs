use crate::api::config::EchoCanceller3Config;
use crate::audio_processing::aec3::aec3_common::{
    BLOCK_SIZE, FFT_LENGTH_BY_2, FFT_LENGTH_BY_2_PLUS_1, NUM_BLOCKS_PER_SECOND,
};
use crate::audio_processing::aec3::delay_estimate::DelayEstimate;
use crate::audio_processing::aec3::echo_audibility::EchoAudibility;
use crate::audio_processing::aec3::echo_path_variability::{DelayAdjustment, EchoPathVariability};
use crate::audio_processing::aec3::echo_remover_metrics::EchoRemoverMetricsAecState;
use crate::audio_processing::aec3::erl_estimator::ErlEstimator;
use crate::audio_processing::aec3::erle_estimator::ErleEstimator;
use crate::audio_processing::aec3::filter_analyzer::FilterAnalyzer;
use crate::audio_processing::aec3::render_buffer::RenderBuffer;
use crate::audio_processing::aec3::reverb_model::ReverbModel;
use crate::audio_processing::aec3::reverb_model_estimator::ReverbModelEstimator;
use crate::audio_processing::aec3::spectrum_buffer::SpectrumBuffer;
use crate::audio_processing::aec3::subtractor_output::SubtractorOutput;
use crate::audio_processing::aec3::subtractor_output_analyzer::SubtractorOutputAnalyzer;
use crate::audio_processing::logging::apm_data_dumper::ApmDataDumper;

const BLOCKS_SINCE_CONVERGED_FILTER_INIT: usize = 10_000;
const BLOCKS_SINCE_CONSISTENT_ESTIMATE_INIT: usize = 10_000;

fn compute_avg_render_reverb(
    spectrum_buffer: &SpectrumBuffer,
    delay_blocks: i32,
    reverb_decay: f32,
    reverb_model: &mut ReverbModel,
    reverb_power_spectrum: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
) {
    let num_render_channels = spectrum_buffer.buffer[0].len();
    let idx_at_delay = spectrum_buffer.offset_index(spectrum_buffer.read, delay_blocks as isize);
    let idx_past = spectrum_buffer.inc_index(idx_at_delay);

    let mut x2_data = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
    let averaged_power = if num_render_channels > 1 {
        let average_channels = |idx: usize, dst: &mut [f32; FFT_LENGTH_BY_2_PLUS_1]| {
            dst.fill(0.0);
            for ch in 0..num_render_channels {
                for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                    dst[k] += spectrum_buffer.buffer[idx][ch][k];
                }
            }
            let normalizer = 1.0 / num_render_channels as f32;
            for value in dst.iter_mut() {
                *value *= normalizer;
            }
        };
        average_channels(idx_past, &mut x2_data);
        reverb_model.update_reverb_no_freq_shaping(&x2_data, 1.0, reverb_decay);
        average_channels(idx_at_delay, &mut x2_data);
        &x2_data
    } else {
        reverb_model.update_reverb_no_freq_shaping(
            &spectrum_buffer.buffer[idx_past][0],
            1.0,
            reverb_decay,
        );
        &spectrum_buffer.buffer[idx_at_delay][0]
    };

    let reverb = reverb_model.reverb();
    for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
        reverb_power_spectrum[k] = averaged_power[k] + reverb[k];
    }
}

pub struct AecState {
    data_dumper: ApmDataDumper,
    config: EchoCanceller3Config,
    num_capture_channels: usize,
    initial_state: InitialState,
    delay_state: FilterDelay,
    transparent_state: TransparentMode,
    filter_quality_state: FilteringQualityAnalyzer,
    erl_estimator: ErlEstimator,
    erle_estimator: ErleEstimator,
    filter_analyzer: FilterAnalyzer,
    echo_audibility: EchoAudibility,
    reverb_model_estimator: ReverbModelEstimator,
    avg_render_reverb: ReverbModel,
    subtractor_output_analyzer: SubtractorOutputAnalyzer,
    saturation_detector: SaturationDetector,
    strong_not_saturated_render_blocks: usize,
    blocks_with_active_render: usize,
    capture_signal_saturation: bool,
}

impl AecState {
    pub fn new(config: EchoCanceller3Config, num_capture_channels: usize) -> Self {
        assert!(num_capture_channels > 0);
        let data_dumper = ApmDataDumper::new_unique();
        let initial_state = InitialState::new(&config);
        let delay_state = FilterDelay::new(&config, num_capture_channels);
        let transparent_state = TransparentMode::new(&config);
        let filter_quality_state = FilteringQualityAnalyzer::new(&config, num_capture_channels);
        let erl_estimator = ErlEstimator::new(2 * NUM_BLOCKS_PER_SECOND);
        let erle_estimator =
            ErleEstimator::new(2 * NUM_BLOCKS_PER_SECOND, &config, num_capture_channels);
        let filter_analyzer = FilterAnalyzer::new(&config, num_capture_channels);
        let echo_audibility =
            EchoAudibility::new(config.echo_audibility.use_stationarity_properties_at_init);
        let reverb_model_estimator = ReverbModelEstimator::new(&config, num_capture_channels);
        let avg_render_reverb = ReverbModel::new();
        let subtractor_output_analyzer = SubtractorOutputAnalyzer::new(num_capture_channels);
        let saturation_detector = SaturationDetector::default();

        Self {
            data_dumper,
            config,
            num_capture_channels,
            initial_state,
            delay_state,
            transparent_state,
            filter_quality_state,
            erl_estimator,
            erle_estimator,
            filter_analyzer,
            echo_audibility,
            reverb_model_estimator,
            avg_render_reverb,
            subtractor_output_analyzer,
            saturation_detector,
            strong_not_saturated_render_blocks: 0,
            blocks_with_active_render: 0,
            capture_signal_saturation: false,
        }
    }

    pub fn usable_linear_estimate(&self) -> bool {
        self.filter_quality_state.linear_filter_usable() && self.config.filter.use_linear_filter
    }

    pub fn use_linear_filter_output(&self) -> bool {
        self.usable_linear_estimate()
    }

    pub fn active_render(&self) -> bool {
        self.blocks_with_active_render > 200
    }

    pub fn get_residual_echo_scaling(&self, residual_scaling: &mut [f32]) {
        let limit_blocks = if self.config.filter.conservative_initial_phase {
            1.5 * NUM_BLOCKS_PER_SECOND as f32
        } else {
            0.8 * NUM_BLOCKS_PER_SECOND as f32
        };
        let filter_has_had_time_to_converge =
            self.strong_not_saturated_render_blocks as f32 >= limit_blocks;
        self.echo_audibility
            .get_residual_echo_scaling(filter_has_had_time_to_converge, residual_scaling);
    }

    pub fn use_stationarity_properties(&self) -> bool {
        self.config.echo_audibility.use_stationarity_properties
    }

    pub fn erle(&self) -> &[[f32; FFT_LENGTH_BY_2_PLUS_1]] {
        self.erle_estimator.erle()
    }

    pub fn erle_uncertainty(&self) -> Option<f32> {
        if self.saturated_echo() {
            Some(1.0)
        } else {
            None
        }
    }

    pub fn fullband_erle_log2(&self) -> f32 {
        self.erle_estimator.fullband_erle_log2()
    }

    pub fn erl(&self) -> &[f32; FFT_LENGTH_BY_2_PLUS_1] {
        self.erl_estimator.erl()
    }

    pub fn erl_time_domain(&self) -> f32 {
        self.erl_estimator.erl_time_domain()
    }

    pub fn min_direct_path_filter_delay(&self) -> i32 {
        self.delay_state.min_direct_path_filter_delay()
    }

    pub fn saturated_capture(&self) -> bool {
        self.capture_signal_saturation
    }

    pub fn saturated_echo(&self) -> bool {
        self.saturation_detector.saturated_echo()
    }

    pub fn update_capture_saturation(&mut self, capture_signal_saturation: bool) {
        self.capture_signal_saturation = capture_signal_saturation;
    }

    pub fn transparent_mode(&self) -> bool {
        self.transparent_state.active()
    }

    pub fn handle_echo_path_change(&mut self, variability: &EchoPathVariability) {
        if !matches!(variability.delay_change, DelayAdjustment::None) {
            self.full_reset();
        } else if variability.gain_change {
            self.erle_estimator.reset(false);
        }
        self.subtractor_output_analyzer.handle_echo_path_change();
    }

    fn full_reset(&mut self) {
        self.filter_analyzer.reset();
        self.capture_signal_saturation = false;
        self.strong_not_saturated_render_blocks = 0;
        self.blocks_with_active_render = 0;
        self.initial_state.reset();
        self.transparent_state.reset();
        self.erle_estimator.reset(true);
        self.erl_estimator.reset();
        self.filter_quality_state.reset();
    }

    pub fn reverb_decay(&self) -> f32 {
        self.reverb_model_estimator.reverb_decay()
    }

    pub fn get_reverb_frequency_response(&self) -> &[f32; FFT_LENGTH_BY_2_PLUS_1] {
        self.reverb_model_estimator.get_reverb_frequency_response()
    }

    pub fn transition_triggered(&self) -> bool {
        self.initial_state.transition_triggered()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        external_delay: Option<DelayEstimate>,
        adaptive_filter_frequency_responses: &[Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>],
        adaptive_filter_impulse_responses: &[Vec<f32>],
        render_buffer: &RenderBuffer<'_>,
        e2_main: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        y2: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        subtractor_output: &[SubtractorOutput],
    ) {
        assert_eq!(self.num_capture_channels, e2_main.len());
        assert_eq!(self.num_capture_channels, y2.len());
        assert_eq!(self.num_capture_channels, subtractor_output.len());
        assert_eq!(
            self.num_capture_channels,
            adaptive_filter_frequency_responses.len()
        );
        assert_eq!(
            self.num_capture_channels,
            adaptive_filter_impulse_responses.len()
        );

        let mut any_filter_converged = false;
        let mut all_filters_diverged = false;
        self.subtractor_output_analyzer.update(
            subtractor_output,
            &mut any_filter_converged,
            &mut all_filters_diverged,
        );

        let mut any_filter_consistent = false;
        let mut max_echo_path_gain = 0.0f32;
        self.filter_analyzer.update(
            adaptive_filter_impulse_responses,
            render_buffer,
            &mut any_filter_consistent,
            &mut max_echo_path_gain,
        );

        if self.config.filter.use_linear_filter {
            self.delay_state.update(
                self.filter_analyzer.filter_delays_blocks(),
                external_delay,
                self.strong_not_saturated_render_blocks,
            );
        }

        let min_delay = self.delay_state.min_direct_path_filter_delay();
        let aligned_render_block = &render_buffer.block(-(min_delay as isize))[0];

        let limit = self.config.render_levels.active_render_limit
            * self.config.render_levels.active_render_limit
            * FFT_LENGTH_BY_2 as f32;
        let mut active_render = false;
        for channel in aligned_render_block {
            let energy: f32 = channel.iter().map(|sample| sample * sample).sum();
            if energy > limit {
                active_render = true;
                break;
            }
        }
        if active_render {
            self.blocks_with_active_render = self.blocks_with_active_render.saturating_add(1);
        }
        if active_render && !self.saturated_capture() {
            self.strong_not_saturated_render_blocks =
                self.strong_not_saturated_render_blocks.saturating_add(1);
        }

        let mut avg_render_spectrum_with_reverb = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        compute_avg_render_reverb(
            render_buffer.spectrum_buffer(),
            min_delay,
            self.reverb_decay(),
            &mut self.avg_render_reverb,
            &mut avg_render_spectrum_with_reverb,
        );

        if self.use_stationarity_properties() {
            self.echo_audibility.update(
                render_buffer,
                self.avg_render_reverb.reverb(),
                min_delay,
                self.delay_state.external_delay_reported(),
            );
        }

        if self.initial_state.transition_triggered() {
            self.erle_estimator.reset(false);
        }

        self.erle_estimator.update(
            render_buffer,
            adaptive_filter_frequency_responses,
            &avg_render_spectrum_with_reverb,
            y2,
            e2_main,
            self.subtractor_output_analyzer.converged_filters(),
        );

        self.erl_estimator.update(
            self.subtractor_output_analyzer.converged_filters(),
            render_buffer.spectrum(min_delay as isize),
            y2,
        );

        self.saturation_detector.update(
            aligned_render_block,
            self.saturated_capture(),
            self.usable_linear_estimate(),
            subtractor_output,
            max_echo_path_gain,
        );

        self.initial_state
            .update(active_render, self.saturated_capture());

        self.transparent_state.update(
            min_delay,
            any_filter_consistent,
            any_filter_converged,
            all_filters_diverged,
            active_render,
            self.saturated_capture(),
        );

        self.filter_quality_state.update(
            active_render,
            self.transparent_mode(),
            self.saturated_capture(),
            external_delay,
            any_filter_converged,
        );

        let stationary_block =
            self.use_stationarity_properties() && self.echo_audibility.is_block_stationary();

        self.reverb_model_estimator.update(
            self.filter_analyzer.adjusted_filters(),
            adaptive_filter_frequency_responses,
            self.erle_estimator.get_inst_linear_quality_estimates(),
            self.delay_state.direct_path_filter_delays(),
            self.filter_quality_state.usable_linear_filter_outputs(),
            stationary_block,
        );

        self.erle_estimator.dump(&self.data_dumper);
        self.reverb_model_estimator.dump(&self.data_dumper);
        self.data_dumper.dump_raw_f32_slice("aec3_erl", self.erl());
        self.data_dumper
            .dump_raw_f32("aec3_erl_time_domain", self.erl_time_domain());
        self.data_dumper
            .dump_raw_f32_slice("aec3_erle", &self.erle()[0]);
        self.data_dumper.dump_raw_f32(
            "aec3_usable_linear_estimate",
            if self.usable_linear_estimate() {
                1.0
            } else {
                0.0
            },
        );
        self.data_dumper.dump_raw_f32(
            "aec3_transparent_mode",
            if self.transparent_mode() { 1.0 } else { 0.0 },
        );
        self.data_dumper.dump_raw_i32(
            "aec3_filter_delay",
            self.filter_analyzer.min_filter_delay_blocks(),
        );
        self.data_dumper.dump_raw_f32(
            "aec3_any_filter_consistent",
            if any_filter_consistent { 1.0 } else { 0.0 },
        );
        self.data_dumper.dump_raw_f32(
            "aec3_initial_state",
            if self.initial_state.initial_state_active() {
                1.0
            } else {
                0.0
            },
        );
        self.data_dumper.dump_raw_f32(
            "aec3_capture_saturation",
            if self.saturated_capture() { 1.0 } else { 0.0 },
        );
        self.data_dumper.dump_raw_f32(
            "aec3_echo_saturation",
            if self.saturated_echo() { 1.0 } else { 0.0 },
        );
        self.data_dumper.dump_raw_f32(
            "aec3_any_filter_converged",
            if any_filter_converged { 1.0 } else { 0.0 },
        );
        self.data_dumper.dump_raw_f32(
            "aec3_all_filters_diverged",
            if all_filters_diverged { 1.0 } else { 0.0 },
        );
        self.data_dumper.dump_raw_f32(
            "aec3_external_delay_avaliable",
            if external_delay.is_some() { 1.0 } else { 0.0 },
        );
        self.data_dumper.dump_raw_f32_slice(
            "aec3_filter_tail_freq_resp_est",
            self.get_reverb_frequency_response(),
        );
    }

    pub fn filter_length_blocks(&self) -> usize {
        self.filter_analyzer.filter_length_blocks()
    }
}

impl EchoRemoverMetricsAecState for AecState {
    fn erl(&self) -> &[f32; FFT_LENGTH_BY_2_PLUS_1] {
        self.erl_estimator.erl()
    }

    fn erl_time_domain(&self) -> f32 {
        self.erl_estimator.erl_time_domain()
    }

    fn erle(&self) -> &[f32; FFT_LENGTH_BY_2_PLUS_1] {
        &self.erle_estimator.erle()[0]
    }

    fn fullband_erle_log2(&self) -> f32 {
        self.erle_estimator.fullband_erle_log2()
    }

    fn active_render(&self) -> bool {
        self.active_render()
    }

    fn saturated_capture(&self) -> bool {
        self.saturated_capture()
    }

    fn usable_linear_estimate(&self) -> bool {
        self.usable_linear_estimate()
    }

    fn min_direct_path_filter_delay(&self) -> i32 {
        self.min_direct_path_filter_delay()
    }
}

struct InitialState {
    conservative_initial_phase: bool,
    initial_state_seconds: f32,
    transition_triggered: bool,
    initial_state_active: bool,
    strong_not_saturated_render_blocks: usize,
}

impl InitialState {
    fn new(config: &EchoCanceller3Config) -> Self {
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

    fn reset(&mut self) {
        self.transition_triggered = false;
        self.initial_state_active = true;
        self.strong_not_saturated_render_blocks = 0;
    }

    fn update(&mut self, active_render: bool, saturated_capture: bool) {
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

    fn initial_state_active(&self) -> bool {
        self.initial_state_active
    }

    fn transition_triggered(&self) -> bool {
        self.transition_triggered
    }
}

struct FilterDelay {
    delay_headroom_samples: usize,
    external_delay_reported: bool,
    filter_delays_blocks: Vec<i32>,
    min_filter_delay: i32,
    external_delay: Option<DelayEstimate>,
}

impl FilterDelay {
    fn new(config: &EchoCanceller3Config, num_capture_channels: usize) -> Self {
        Self {
            delay_headroom_samples: config.delay.delay_headroom_samples,
            external_delay_reported: false,
            filter_delays_blocks: vec![0; num_capture_channels],
            min_filter_delay: 0,
            external_delay: None,
        }
    }

    fn external_delay_reported(&self) -> bool {
        self.external_delay_reported
    }

    fn direct_path_filter_delays(&self) -> &[i32] {
        &self.filter_delays_blocks
    }

    fn min_direct_path_filter_delay(&self) -> i32 {
        self.min_filter_delay
    }

    fn update(
        &mut self,
        analyzer_filter_delay_estimates_blocks: &[i32],
        external_delay: Option<DelayEstimate>,
        blocks_with_proper_filter_adaptation: usize,
    ) {
        if let Some(delay) = external_delay {
            if self
                .external_delay
                .map(|stored| stored.delay != delay.delay)
                .unwrap_or(true)
            {
                self.external_delay = Some(delay);
                self.external_delay_reported = true;
            }
        }

        let delay_estimator_may_not_have_converged =
            blocks_with_proper_filter_adaptation < 2 * NUM_BLOCKS_PER_SECOND;
        if delay_estimator_may_not_have_converged && self.external_delay.is_some() {
            let delay_guess = (self.delay_headroom_samples / BLOCK_SIZE) as i32;
            self.filter_delays_blocks.fill(delay_guess);
        } else {
            assert_eq!(
                self.filter_delays_blocks.len(),
                analyzer_filter_delay_estimates_blocks.len()
            );
            self.filter_delays_blocks
                .copy_from_slice(analyzer_filter_delay_estimates_blocks);
        }

        self.min_filter_delay = *self.filter_delays_blocks.iter().min().unwrap_or(&0);
    }
}

struct TransparentMode {
    bounded_erl: bool,
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

impl TransparentMode {
    fn new(config: &EchoCanceller3Config) -> Self {
        Self {
            bounded_erl: config.ep_strength.bounded_erl,
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
                self.non_converged_sequence_size = BLOCKS_SINCE_CONVERGED_FILTER_INIT;
            }
        }

        if self.active_non_converged_sequence_size > 60 * NUM_BLOCKS_PER_SECOND {
            self.finite_erl_recently_detected = false;
        }

        if self.num_converged_blocks > 50 {
            self.finite_erl_recently_detected = true;
        }

        if self.bounded_erl {
            self.transparency_activated = false;
        } else if self.finite_erl_recently_detected {
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

struct FilteringQualityAnalyzer {
    use_linear_filter: bool,
    overall_usable_linear_estimates: bool,
    filter_update_blocks_since_reset: usize,
    filter_update_blocks_since_start: usize,
    convergence_seen: bool,
    usable_linear_filter_estimates: Vec<bool>,
}

impl FilteringQualityAnalyzer {
    fn new(config: &EchoCanceller3Config, num_capture_channels: usize) -> Self {
        Self {
            use_linear_filter: config.filter.use_linear_filter,
            overall_usable_linear_estimates: false,
            filter_update_blocks_since_reset: 0,
            filter_update_blocks_since_start: 0,
            convergence_seen: false,
            usable_linear_filter_estimates: vec![false; num_capture_channels],
        }
    }

    fn reset(&mut self) {
        self.usable_linear_filter_estimates.fill(false);
        self.overall_usable_linear_estimates = false;
        self.filter_update_blocks_since_reset = 0;
    }

    fn update(
        &mut self,
        active_render: bool,
        transparent_mode: bool,
        saturated_capture: bool,
        external_delay: Option<DelayEstimate>,
        any_filter_converged: bool,
    ) {
        let filter_update = active_render && !saturated_capture;
        if filter_update {
            self.filter_update_blocks_since_reset += 1;
            self.filter_update_blocks_since_start += 1;
        }

        if any_filter_converged {
            self.convergence_seen = true;
        }

        let sufficient_data_to_converge_at_startup =
            self.filter_update_blocks_since_start as f32 > 0.4 * NUM_BLOCKS_PER_SECOND as f32;
        let sufficient_data_to_converge_at_reset = sufficient_data_to_converge_at_startup
            && self.filter_update_blocks_since_reset as f32 > 0.2 * NUM_BLOCKS_PER_SECOND as f32;

        self.overall_usable_linear_estimates =
            sufficient_data_to_converge_at_startup && sufficient_data_to_converge_at_reset;
        self.overall_usable_linear_estimates &= external_delay.is_some() || self.convergence_seen;
        self.overall_usable_linear_estimates &= !transparent_mode;

        if self.use_linear_filter {
            self.usable_linear_filter_estimates
                .fill(self.overall_usable_linear_estimates);
        }
    }

    fn linear_filter_usable(&self) -> bool {
        self.overall_usable_linear_estimates
    }

    fn usable_linear_filter_outputs(&self) -> &[bool] {
        &self.usable_linear_filter_estimates
    }
}

#[derive(Default)]
struct SaturationDetector {
    saturated_echo: bool,
}

impl SaturationDetector {
    fn update(
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

    fn saturated_echo(&self) -> bool {
        self.saturated_echo
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_processing::aec3::aec3_common::{
        BLOCK_SIZE, get_time_domain_length, num_bands_for_rate,
    };
    use crate::audio_processing::aec3::delay_estimate::{DelayEstimate, DelayEstimateQuality};
    use crate::audio_processing::aec3::echo_path_variability::EchoPathVariability;
    use crate::audio_processing::aec3::render_delay_buffer::RenderDelayBuffer;

    const SAMPLE_RATE_HZ: i32 = 48_000;

    fn run_normal_usage_test(num_render_channels: usize, num_capture_channels: usize) {
        let config = EchoCanceller3Config::default();
        let mut state = AecState::new(config.clone(), num_capture_channels);
        let delay_estimate = Some(DelayEstimate::new(DelayEstimateQuality::Refined, 10));
        let mut render_delay_buffer =
            RenderDelayBuffer::new(config.clone(), SAMPLE_RATE_HZ, num_render_channels);

        let num_bands = num_bands_for_rate(SAMPLE_RATE_HZ);
        let mut x = vec![vec![vec![0.0f32; BLOCK_SIZE]; num_render_channels]; num_bands];
        let mut y = vec![[0.0f32; BLOCK_SIZE]; num_capture_channels];
        let mut subtractor_output = vec![SubtractorOutput::new(); num_capture_channels];
        let mut e2_main = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels];
        let mut y2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels];
        for ch in 0..num_capture_channels {
            subtractor_output[ch].s_main.fill(100.0);
            subtractor_output[ch].e_main.fill(100.0);
            y[ch].fill(1_000.0);
        }

        let impulse_response_len = get_time_domain_length(config.filter.main.length_blocks);
        let impulse_response = vec![vec![0.0f32; impulse_response_len]; num_capture_channels];

        let mut filter_frequency_response =
            vec![vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; 10]; num_capture_channels];
        for channel in &mut filter_frequency_response {
            for block in channel {
                block.fill(0.01);
            }
        }
        filter_frequency_response[0][2].fill(100.0);
        filter_frequency_response[0][2][0] = 1.0;

        for band in &mut x {
            for channel in band {
                channel.fill(101.0);
            }
        }

        for _ in 0..3000 {
            render_delay_buffer.insert(&x);
            render_delay_buffer.prepare_capture_processing();
            for ch in 0..num_capture_channels {
                subtractor_output[ch].compute_metrics(&y[ch]);
            }
            let render_buffer = render_delay_buffer.render_buffer();
            state.update(
                delay_estimate,
                &filter_frequency_response,
                &impulse_response,
                &render_buffer,
                &e2_main,
                &y2,
                &subtractor_output,
            );
        }
        assert!(state.usable_linear_estimate());

        for ch in 0..num_capture_channels {
            subtractor_output[ch].compute_metrics(&y[ch]);
        }
        state.handle_echo_path_change(&EchoPathVariability::new(
            false,
            DelayAdjustment::BufferReadjustment,
            false,
        ));
        let render_buffer = render_delay_buffer.render_buffer();
        state.update(
            delay_estimate,
            &filter_frequency_response,
            &impulse_response,
            &render_buffer,
            &e2_main,
            &y2,
            &subtractor_output,
        );
        assert!(!state.usable_linear_estimate());

        for ch in 0..num_render_channels {
            x[0][ch].fill(101.0);
        }
        render_delay_buffer.insert(&x);
        render_delay_buffer.prepare_capture_processing();
        for ch in 0..num_capture_channels {
            subtractor_output[ch].compute_metrics(&y[ch]);
        }
        state.handle_echo_path_change(&EchoPathVariability::new(
            true,
            DelayAdjustment::NewDetectedDelay,
            false,
        ));
        let render_buffer = render_delay_buffer.render_buffer();
        state.update(
            delay_estimate,
            &filter_frequency_response,
            &impulse_response,
            &render_buffer,
            &e2_main,
            &y2,
            &subtractor_output,
        );
        assert!(!state.active_render());

        for _ in 0..1000 {
            render_delay_buffer.insert(&x);
            render_delay_buffer.prepare_capture_processing();
            for ch in 0..num_capture_channels {
                subtractor_output[ch].compute_metrics(&y[ch]);
            }
            let render_buffer = render_delay_buffer.render_buffer();
            state.update(
                delay_estimate,
                &filter_frequency_response,
                &impulse_response,
                &render_buffer,
                &e2_main,
                &y2,
                &subtractor_output,
            );
        }
        assert!(state.active_render());

        for band in &mut x {
            for channel in band {
                channel.fill(0.0);
            }
        }
        for ch in 0..num_render_channels {
            x[0][ch][0] = 5000.0;
        }

        let fft_buffer_len = {
            let render_buffer = render_delay_buffer.render_buffer();
            render_buffer.fft_buffer().len()
        };
        for k in 0..fft_buffer_len {
            render_delay_buffer.insert(&x);
            if k == 0 {
                render_delay_buffer.reset();
            }
            render_delay_buffer.prepare_capture_processing();
        }

        for channel in &mut y2 {
            channel.fill(10.0 * 10_000.0 * 10_000.0);
        }
        for _ in 0..1000 {
            for ch in 0..num_capture_channels {
                subtractor_output[ch].compute_metrics(&y[ch]);
            }
            let render_buffer = render_delay_buffer.render_buffer();
            state.update(
                delay_estimate,
                &filter_frequency_response,
                &impulse_response,
                &render_buffer,
                &e2_main,
                &y2,
                &subtractor_output,
            );
        }

        assert!(state.usable_linear_estimate());
        let erl = state.erl();
        assert_eq!(erl[0], erl[1]);
        for k in 1..erl.len() - 1 {
            let expected = if k % 2 == 0 { 10.0 } else { 1000.0 };
            assert!((erl[k] - expected).abs() < 0.1);
        }
        assert_eq!(erl[erl.len() - 2], erl[erl.len() - 1]);

        for channel in &mut e2_main {
            channel.fill(1.0 * 10_000.0 * 10_000.0);
        }
        for channel in &mut y2 {
            channel.fill(10.0 * e2_main[0][0]);
        }

        for _ in 0..1000 {
            for ch in 0..num_capture_channels {
                subtractor_output[ch].compute_metrics(&y[ch]);
            }
            let render_buffer = render_delay_buffer.render_buffer();
            state.update(
                delay_estimate,
                &filter_frequency_response,
                &impulse_response,
                &render_buffer,
                &e2_main,
                &y2,
                &subtractor_output,
            );
        }

        assert!(state.usable_linear_estimate());
        {
            let erle = state.erle()[0];
            assert_eq!(erle[0], erle[1]);
            const LOW_FREQUENCY_LIMIT: usize = 32;
            for k in (2..LOW_FREQUENCY_LIMIT).step_by(2) {
                assert!((erle[k] - 4.0).abs() < 0.1);
            }
            for k in (LOW_FREQUENCY_LIMIT..erle.len() - 1).step_by(2) {
                assert!((erle[k] - 1.5).abs() < 0.1);
            }
            assert_eq!(erle[erle.len() - 2], erle[erle.len() - 1]);
        }

        for channel in &mut e2_main {
            channel.fill(1.0 * 10_000.0 * 10_000.0);
        }
        for channel in &mut y2 {
            channel.fill(5.0 * e2_main[0][0]);
        }

        for _ in 0..1000 {
            for ch in 0..num_capture_channels {
                subtractor_output[ch].compute_metrics(&y[ch]);
            }
            let render_buffer = render_delay_buffer.render_buffer();
            state.update(
                delay_estimate,
                &filter_frequency_response,
                &impulse_response,
                &render_buffer,
                &e2_main,
                &y2,
                &subtractor_output,
            );
        }

        assert!(state.usable_linear_estimate());
        {
            let erle = state.erle()[0];
            assert_eq!(erle[0], erle[1]);
            const LOW_FREQUENCY_LIMIT: usize = 32;
            for k in 1..LOW_FREQUENCY_LIMIT {
                let expected = if k % 2 == 0 { 4.0 } else { 1.0 };
                assert!((erle[k] - expected).abs() < 0.1);
            }
            for k in LOW_FREQUENCY_LIMIT..erle.len() - 1 {
                let expected = if k % 2 == 0 { 1.5 } else { 1.0 };
                assert!((erle[k] - expected).abs() < 0.1);
            }
            assert_eq!(erle[erle.len() - 2], erle[erle.len() - 1]);
        }
    }

    #[test]
    fn normal_usage_matches_reference_behavior() {
        for &num_render_channels in &[1, 2, 8] {
            for &num_capture_channels in &[1, 2, 8] {
                run_normal_usage_test(num_render_channels, num_capture_channels);
            }
        }
    }

    #[test]
    fn converged_filter_delay_processed_without_panics() {
        const FILTER_LENGTH_BLOCKS: usize = 10;
        const NUM_CAPTURE_CHANNELS: usize = 1;
        let config = EchoCanceller3Config::default();
        let mut state = AecState::new(config.clone(), NUM_CAPTURE_CHANNELS);
        let render_delay_buffer = RenderDelayBuffer::new(config.clone(), SAMPLE_RATE_HZ, 1);
        let delay_estimate: Option<DelayEstimate> = None;
        let e2_main = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; NUM_CAPTURE_CHANNELS];
        let y2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; NUM_CAPTURE_CHANNELS];
        let mut subtractor_output = vec![SubtractorOutput::new(); NUM_CAPTURE_CHANNELS];
        for output in &mut subtractor_output {
            output.s_main.fill(100.0);
        }
        let y = [0.0f32; BLOCK_SIZE];

        let mut frequency_response =
            vec![vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; FILTER_LENGTH_BLOCKS]];
        for channel in &mut frequency_response {
            for block in channel {
                block.fill(0.01);
            }
        }

        let impulse_response_len = get_time_domain_length(config.filter.main.length_blocks);
        let mut impulse_response = vec![vec![0.0f32; impulse_response_len]; NUM_CAPTURE_CHANNELS];

        let echo_path_variability = EchoPathVariability::new(false, DelayAdjustment::None, false);

        for k in 0..FILTER_LENGTH_BLOCKS {
            for ir in &mut impulse_response {
                ir.fill(0.0);
                ir[k * BLOCK_SIZE + 1] = 1.0;
            }

            state.handle_echo_path_change(&echo_path_variability);
            subtractor_output[0].compute_metrics(&y);
            let render_buffer = render_delay_buffer.render_buffer();
            state.update(
                delay_estimate,
                &frequency_response,
                &impulse_response,
                &render_buffer,
                &e2_main,
                &y2,
                &subtractor_output,
            );
        }
    }
}
