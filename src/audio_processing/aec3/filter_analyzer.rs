use std::cmp::{max, min};

use crate::api::config::EchoCanceller3Config;
use crate::audio_processing::logging::apm_data_dumper::ApmDataDumper;

use super::aec3_common::{
    BLOCK_SIZE, BLOCK_SIZE_LOG2, FFT_LENGTH_BY_2, NUM_BLOCKS_PER_SECOND, get_time_domain_length,
};
use super::render_buffer::RenderBuffer;

pub struct FilterAnalyzer {
    data_dumper: ApmDataDumper,
    bounded_erl: bool,
    default_gain: f32,
    h_highpass: Vec<Vec<f32>>,
    blocks_since_reset: usize,
    region: FilterRegion,
    filter_analysis_states: Vec<FilterAnalysisState>,
    filter_delays_blocks: Vec<i32>,
    min_filter_delay_blocks: i32,
}

impl FilterAnalyzer {
    pub fn new(config: &EchoCanceller3Config, num_capture_channels: usize) -> Self {
        assert!(num_capture_channels > 0);
        let mut h_highpass = Vec::with_capacity(num_capture_channels);
        let filter_length = get_time_domain_length(config.filter.main.length_blocks);
        for _ in 0..num_capture_channels {
            h_highpass.push(vec![0.0f32; filter_length]);
        }

        let data_dumper = ApmDataDumper::new_unique();

        let filter_analysis_states = (0..num_capture_channels)
            .map(|_| FilterAnalysisState::new(config))
            .collect();

        let filter_delays_blocks = vec![0; num_capture_channels];

        let mut analyzer = Self {
            data_dumper,
            bounded_erl: config.ep_strength.bounded_erl,
            default_gain: config.ep_strength.default_gain,
            h_highpass,
            blocks_since_reset: 0,
            region: FilterRegion::default(),
            filter_analysis_states,
            filter_delays_blocks,
            min_filter_delay_blocks: 0,
        };
        analyzer.reset();
        analyzer
    }

    pub fn reset(&mut self) {
        self.blocks_since_reset = 0;
        self.region.reset();
        for state in &mut self.filter_analysis_states {
            state.peak_index = 0;
            state.gain = self.default_gain;
            state.consistent_estimate = false;
            state.consistent_filter_detector.reset();
        }
        self.filter_delays_blocks.fill(0);
        self.min_filter_delay_blocks = 0;
    }

    pub fn update(
        &mut self,
        filters_time_domain: &[Vec<f32>],
        render_buffer: &RenderBuffer,
        any_filter_consistent: &mut bool,
        max_echo_path_gain: &mut f32,
    ) {
        assert_eq!(filters_time_domain.len(), self.filter_analysis_states.len());
        assert_eq!(filters_time_domain.len(), self.h_highpass.len());

        self.blocks_since_reset += 1;
        self.set_region_to_analyze(filters_time_domain[0].len());
        self.analyze_region(filters_time_domain, render_buffer);

        let mut any_consistent = self.filter_analysis_states[0].consistent_estimate;
        let mut gain = self.filter_analysis_states[0].gain;
        let mut min_delay = self.filter_delays_blocks[0];
        for ch in 1..self.filter_analysis_states.len() {
            let state = &self.filter_analysis_states[ch];
            let delay = self.filter_delays_blocks[ch];
            any_consistent |= state.consistent_estimate;
            gain = gain.max(state.gain);
            min_delay = min(min_delay, delay);
        }
        self.min_filter_delay_blocks = min_delay;
        *any_filter_consistent = any_consistent;
        *max_echo_path_gain = gain;
    }

    pub fn filter_delays_blocks(&self) -> &[i32] {
        &self.filter_delays_blocks
    }

    pub fn min_filter_delay_blocks(&self) -> i32 {
        self.min_filter_delay_blocks
    }

    pub fn filter_length_blocks(&self) -> usize {
        self.filter_analysis_states[0].filter_length_blocks
    }

    pub fn adjusted_filters(&self) -> &[Vec<f32>] {
        &self.h_highpass
    }

    pub fn set_region_to_analyze(&mut self, filter_size: usize) {
        self.region.advance(
            filter_size,
            BLOCK_SIZE * FilterRegion::NUMBER_BLOCKS_TO_UPDATE,
        );
    }

    fn analyze_region(&mut self, filters_time_domain: &[Vec<f32>], render_buffer: &RenderBuffer) {
        self.pre_process_filters(filters_time_domain);
        self.data_dumper
            .dump_raw_f32_slice("aec3_linear_filter_processed_td", &self.h_highpass[0]);

        for (ch, ((state, h_hp), filter)) in self
            .filter_analysis_states
            .iter_mut()
            .zip(self.h_highpass.iter_mut())
            .zip(filters_time_domain.iter())
            .enumerate()
        {
            let start = self.region.start_sample;
            let end = self.region.end_sample;
            debug_assert!(end < h_hp.len());
            debug_assert!(start <= end);

            state.peak_index = find_peak_index(h_hp, state.peak_index, start, end);
            self.filter_delays_blocks[ch] = (state.peak_index >> BLOCK_SIZE_LOG2) as i32;
            Self::update_filter_gain(self.bounded_erl, self.blocks_since_reset, h_hp, state);
            state.filter_length_blocks = filter.len() / BLOCK_SIZE;
            let block_offset = -(self.filter_delays_blocks[ch] as isize);
            let render_block = &render_buffer.block(block_offset)[0];
            state.consistent_estimate = state.consistent_filter_detector.detect(
                h_hp,
                &self.region,
                render_block,
                state.peak_index,
                self.filter_delays_blocks[ch],
            );
        }
    }

    fn update_filter_gain(
        bounded_erl: bool,
        blocks_since_reset: usize,
        filter_time_domain: &[f32],
        state: &mut FilterAnalysisState,
    ) {
        let sufficient_time_to_converge = blocks_since_reset > 5 * NUM_BLOCKS_PER_SECOND;
        if sufficient_time_to_converge && state.consistent_estimate {
            state.gain = filter_time_domain[state.peak_index].abs();
        } else if state.gain != 0.0 {
            state.gain = state.gain.max(filter_time_domain[state.peak_index].abs());
        }
        if bounded_erl && state.gain != 0.0 {
            state.gain = state.gain.max(0.01);
        }
    }

    fn pre_process_filters(&mut self, filters_time_domain: &[Vec<f32>]) {
        const H: [f32; 3] = [0.7929742, -0.36072128, -0.47047766];
        for (h_hp, filter) in self.h_highpass.iter_mut().zip(filters_time_domain.iter()) {
            let start = self.region.start_sample;
            let end = self.region.end_sample;
            assert!(end < filter.len());
            if h_hp.len() < filter.len() {
                h_hp.resize(filter.len(), 0.0);
            }
            for sample in &mut h_hp[start..=end] {
                *sample = 0.0;
            }
            let begin = max(H.len() - 1, start);
            for k in begin..=end {
                let mut acc = 0.0f32;
                for (j, &h) in H.iter().enumerate() {
                    acc += filter[k - j] * h;
                }
                h_hp[k] = acc;
            }
        }
    }
}

fn find_peak_index(filter_td: &[f32], mut peak_index: usize, start: usize, end: usize) -> usize {
    let mut max_value = filter_td[peak_index] * filter_td[peak_index];
    for k in start..=end {
        let tmp = filter_td[k] * filter_td[k];
        if tmp > max_value {
            max_value = tmp;
            peak_index = k;
        }
    }
    peak_index
}

#[derive(Default)]
struct FilterRegion {
    start_sample: usize,
    end_sample: usize,
}

impl FilterRegion {
    const NUMBER_BLOCKS_TO_UPDATE: usize = 1;

    fn reset(&mut self) {
        self.start_sample = 0;
        self.end_sample = 0;
    }

    fn advance(&mut self, filter_size: usize, span: usize) {
        if filter_size == 0 {
            self.reset();
            return;
        }
        self.start_sample = if self.end_sample >= filter_size - 1 {
            0
        } else {
            self.end_sample + 1
        };
        let last = min(self.start_sample + span - 1, filter_size - 1);
        self.end_sample = last;
        debug_assert!(self.start_sample <= self.end_sample);
        debug_assert!(self.end_sample < filter_size);
    }
}

struct FilterAnalysisState {
    gain: f32,
    peak_index: usize,
    filter_length_blocks: usize,
    consistent_estimate: bool,
    consistent_filter_detector: ConsistentFilterDetector,
}

impl FilterAnalysisState {
    fn new(config: &EchoCanceller3Config) -> Self {
        Self {
            gain: 0.0,
            peak_index: 0,
            filter_length_blocks: config.filter.main_initial.length_blocks,
            consistent_estimate: false,
            consistent_filter_detector: ConsistentFilterDetector::new(config),
        }
    }
}

struct ConsistentFilterDetector {
    significant_peak: bool,
    filter_floor_accum: f32,
    filter_secondary_peak: f32,
    filter_floor_low_limit: usize,
    filter_floor_high_limit: usize,
    active_render_threshold: f32,
    consistent_estimate_counter: usize,
    consistent_delay_reference: i32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::config::EchoCanceller3Config;
    use crate::audio_processing::aec3::aec3_common::BLOCK_SIZE;
    use crate::audio_processing::aec3::block_buffer::BlockBuffer;
    use crate::audio_processing::aec3::fft_buffer::FftBuffer;
    use crate::audio_processing::aec3::render_buffer::RenderBuffer;
    use crate::audio_processing::aec3::spectrum_buffer::SpectrumBuffer;

    #[test]
    fn filter_analyzer_handles_filter_resize() {
        let config = EchoCanceller3Config::default();
        let mut analyzer = FilterAnalyzer::new(&config, 1);
        let block_buffer = BlockBuffer::new(8, 1, 1, BLOCK_SIZE);
        let spectrum_buffer = SpectrumBuffer::new(8, 1);
        let fft_buffer = FftBuffer::new(8, 1);
        let render_buffer = RenderBuffer::new(&block_buffer, &spectrum_buffer, &fft_buffer, false);

        let mut filters = vec![vec![0.0f32; BLOCK_SIZE * 2]];
        let mut consistent = false;
        let mut gain = 0.0;
        analyzer.update(&filters, &render_buffer, &mut consistent, &mut gain);
        assert_eq!(
            filters[0].len() / BLOCK_SIZE,
            analyzer.filter_length_blocks()
        );

        filters[0].resize(BLOCK_SIZE * 3, 0.0);
        analyzer.update(&filters, &render_buffer, &mut consistent, &mut gain);
        assert_eq!(
            filters[0].len() / BLOCK_SIZE,
            analyzer.filter_length_blocks()
        );

        filters[0].resize(BLOCK_SIZE, 0.0);
        analyzer.update(&filters, &render_buffer, &mut consistent, &mut gain);
        assert_eq!(
            filters[0].len() / BLOCK_SIZE,
            analyzer.filter_length_blocks()
        );
    }
}

impl ConsistentFilterDetector {
    fn new(config: &EchoCanceller3Config) -> Self {
        let threshold = config.render_levels.active_render_limit
            * config.render_levels.active_render_limit
            * FFT_LENGTH_BY_2 as f32;
        Self {
            significant_peak: false,
            filter_floor_accum: 0.0,
            filter_secondary_peak: 0.0,
            filter_floor_low_limit: 0,
            filter_floor_high_limit: 0,
            active_render_threshold: threshold,
            consistent_estimate_counter: 0,
            consistent_delay_reference: -10,
        }
    }

    fn reset(&mut self) {
        self.significant_peak = false;
        self.filter_floor_accum = 0.0;
        self.filter_secondary_peak = 0.0;
        self.filter_floor_low_limit = 0;
        self.filter_floor_high_limit = 0;
        self.consistent_estimate_counter = 0;
        self.consistent_delay_reference = -10;
    }

    fn detect(
        &mut self,
        filter_to_analyze: &[f32],
        region: &FilterRegion,
        x_block: &[Vec<f32>],
        peak_index: usize,
        delay_blocks: i32,
    ) -> bool {
        if region.start_sample == 0 {
            self.filter_floor_accum = 0.0;
            self.filter_secondary_peak = 0.0;
            self.filter_floor_low_limit = if peak_index < 64 { 0 } else { peak_index - 64 };
            let high_limit_cap = filter_to_analyze.len().saturating_sub(129);
            self.filter_floor_high_limit = if peak_index > high_limit_cap {
                0
            } else {
                peak_index + 128
            };
        }

        let low_stop = min(region.end_sample + 1, self.filter_floor_low_limit);
        if region.start_sample < low_stop {
            for k in region.start_sample..low_stop {
                let abs_h = filter_to_analyze[k].abs();
                self.filter_floor_accum += abs_h;
                self.filter_secondary_peak = self.filter_secondary_peak.max(abs_h);
            }
        }

        let high_start = max(self.filter_floor_high_limit, region.start_sample);
        if high_start <= region.end_sample {
            for k in high_start..=region.end_sample {
                let abs_h = filter_to_analyze[k].abs();
                self.filter_floor_accum += abs_h;
                self.filter_secondary_peak = self.filter_secondary_peak.max(abs_h);
            }
        }

        if region.end_sample == filter_to_analyze.len() - 1 {
            let denom = (self.filter_floor_low_limit + filter_to_analyze.len()
                - self.filter_floor_high_limit) as f32;
            let filter_floor = if denom > 0.0 {
                self.filter_floor_accum / denom
            } else {
                0.0
            };
            let abs_peak = filter_to_analyze[peak_index].abs();
            self.significant_peak =
                abs_peak > 10.0 * filter_floor && abs_peak > 2.0 * self.filter_secondary_peak;
        }

        if self.significant_peak {
            let mut active_render_block = false;
            for channel in x_block {
                let energy = channel.iter().map(|x| x * x).sum::<f32>();
                if energy > self.active_render_threshold {
                    active_render_block = true;
                    break;
                }
            }

            if self.consistent_delay_reference == delay_blocks {
                if active_render_block {
                    self.consistent_estimate_counter += 1;
                }
            } else {
                self.consistent_estimate_counter = 0;
                self.consistent_delay_reference = delay_blocks;
            }
        }

        self.consistent_estimate_counter as f32 > 1.5 * NUM_BLOCKS_PER_SECOND as f32
    }
}
