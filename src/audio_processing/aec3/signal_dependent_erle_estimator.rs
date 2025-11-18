use crate::api::config::EchoCanceller3Config;
use crate::audio_processing::aec3::aec3_common::{
    BLOCK_SIZE, FFT_LENGTH_BY_2, FFT_LENGTH_BY_2_PLUS_1,
};
use crate::audio_processing::aec3::render_buffer::RenderBuffer;
use crate::audio_processing::logging::apm_data_dumper::ApmDataDumper;

const SUBBANDS: usize = 6;
const BAND_BOUNDARIES: [usize; SUBBANDS + 1] = [1, 8, 16, 24, 32, 48, FFT_LENGTH_BY_2_PLUS_1];
const X2_BAND_ENERGY_THRESHOLD: f32 = 44_015_068.0;
const SMOOTH_DECREASE: f32 = 0.1;
const SMOOTH_INCREASE: f32 = SMOOTH_DECREASE / 2.0;
const MIN_UPDATES_FOR_CORRECTION: usize = 50;

fn form_subband_map() -> [usize; FFT_LENGTH_BY_2_PLUS_1] {
    let mut map = [0usize; FFT_LENGTH_BY_2_PLUS_1];
    let mut subband = 1usize;
    for (k, value) in map.iter_mut().enumerate() {
        while subband < BAND_BOUNDARIES.len() && k >= BAND_BOUNDARIES[subband] {
            subband += 1;
        }
        *value = (subband - 1).min(SUBBANDS - 1);
    }
    map
}

fn define_filter_section_sizes(
    delay_headroom_blocks: usize,
    num_blocks: usize,
    num_sections: usize,
) -> Vec<usize> {
    let filter_length_blocks = num_blocks.saturating_sub(delay_headroom_blocks);
    let mut section_sizes = vec![0usize; num_sections];
    if num_sections == 0 {
        return section_sizes;
    }
    let mut remaining_blocks = filter_length_blocks;
    let mut remaining_sections = num_sections;
    let mut estimator_size = 2usize;
    let mut idx = 0usize;
    while remaining_sections > 1 && remaining_blocks > estimator_size * remaining_sections {
        section_sizes[idx] = estimator_size;
        remaining_blocks -= estimator_size;
        remaining_sections -= 1;
        estimator_size *= 2;
        idx += 1;
        if idx == section_sizes.len() {
            break;
        }
    }

    if remaining_sections == 0 {
        return section_sizes;
    }

    let last_group_size = if remaining_sections > 0 {
        remaining_blocks / remaining_sections
    } else {
        0
    };

    while idx < section_sizes.len() {
        section_sizes[idx] = last_group_size;
        idx += 1;
    }

    if remaining_sections > 0 {
        let used_blocks = last_group_size * remaining_sections;
        if let Some(last) = section_sizes.last_mut() {
            *last += remaining_blocks.saturating_sub(used_blocks);
        }
    }

    section_sizes
}

fn set_sections_boundaries(
    delay_headroom_blocks: usize,
    num_blocks: usize,
    num_sections: usize,
) -> Vec<usize> {
    let mut boundaries = vec![0usize; num_sections + 1];
    if num_sections == 0 {
        return boundaries;
    }
    if num_sections == 1 {
        boundaries[0] = 0;
        boundaries[1] = num_blocks;
        return boundaries;
    }

    let section_sizes =
        define_filter_section_sizes(delay_headroom_blocks, num_blocks, num_sections);
    boundaries[0] = delay_headroom_blocks;
    let mut idx = 0usize;
    let mut current_size = 0usize;
    for k in delay_headroom_blocks..num_blocks {
        current_size += 1;
        if current_size >= section_sizes[idx] {
            idx += 1;
            if idx >= section_sizes.len() {
                break;
            }
            boundaries[idx] = k + 1;
            current_size = 0;
        }
    }
    boundaries[section_sizes.len()] = num_blocks;
    boundaries
}

fn set_max_erle_subbands(
    max_erle_l: f32,
    max_erle_h: f32,
    limit_subband_l: usize,
) -> [f32; SUBBANDS] {
    let mut max_erle = [max_erle_h; SUBBANDS];
    for value in max_erle.iter_mut().take(limit_subband_l) {
        *value = max_erle_l;
    }
    max_erle
}

pub struct SignalDependentErleEstimator {
    min_erle: f32,
    num_sections: usize,
    band_to_subband: [usize; FFT_LENGTH_BY_2_PLUS_1],
    max_erle: [f32; SUBBANDS],
    section_boundaries_blocks: Vec<usize>,
    erle: Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
    s2_section_accum: Vec<Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>>,
    erle_estimators: Vec<Vec<[f32; SUBBANDS]>>,
    erle_ref: Vec<[f32; SUBBANDS]>,
    correction_factors: Vec<Vec<[f32; SUBBANDS]>>,
    num_updates: Vec<[usize; SUBBANDS]>,
    n_active_sections: Vec<[usize; FFT_LENGTH_BY_2_PLUS_1]>,
}

impl SignalDependentErleEstimator {
    pub fn new(config: &EchoCanceller3Config, num_capture_channels: usize) -> Self {
        assert!(config.erle.num_sections >= 1);
        let band_to_subband = form_subband_map();
        let limit_subband_l = band_to_subband[FFT_LENGTH_BY_2 / 2];
        let num_sections = config.erle.num_sections;
        let mut instance = Self {
            min_erle: config.erle.min,
            num_sections,
            band_to_subband,
            max_erle: set_max_erle_subbands(config.erle.max_l, config.erle.max_h, limit_subband_l),
            section_boundaries_blocks: set_sections_boundaries(
                config.delay.delay_headroom_samples / BLOCK_SIZE,
                config.filter.main.length_blocks,
                num_sections,
            ),
            erle: vec![[0.0; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels],
            s2_section_accum: vec![
                vec![[0.0; FFT_LENGTH_BY_2_PLUS_1]; num_sections];
                num_capture_channels
            ],
            erle_estimators: vec![vec![[0.0; SUBBANDS]; num_sections]; num_capture_channels],
            erle_ref: vec![[0.0; SUBBANDS]; num_capture_channels],
            correction_factors: vec![vec![[1.0; SUBBANDS]; num_sections]; num_capture_channels],
            num_updates: vec![[0; SUBBANDS]; num_capture_channels],
            n_active_sections: vec![[0; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels],
        };
        instance.reset();
        instance
    }

    pub fn reset(&mut self) {
        for ch in 0..self.erle.len() {
            self.erle[ch].fill(self.min_erle);
            for estimator in &mut self.erle_estimators[ch] {
                estimator.fill(self.min_erle);
            }
            self.erle_ref[ch].fill(self.min_erle);
            for factor in &mut self.correction_factors[ch] {
                factor.fill(1.0);
            }
            self.num_updates[ch].fill(0);
            self.n_active_sections[ch].fill(0);
        }
    }

    pub fn erle(&self) -> &[[f32; FFT_LENGTH_BY_2_PLUS_1]] {
        &self.erle
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        render_buffer: &RenderBuffer<'_>,
        filter_frequency_responses: &[Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>],
        x2: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        y2: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        e2: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        average_erle: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        converged_filters: &[bool],
    ) {
        if self.num_sections <= 1 {
            return;
        }
        assert_eq!(filter_frequency_responses.len(), y2.len());
        assert_eq!(average_erle.len(), y2.len());
        assert_eq!(converged_filters.len(), y2.len());

        self.compute_number_of_active_filter_sections(render_buffer, filter_frequency_responses);
        self.update_correction_factors(x2, y2, e2, converged_filters);

        for ch in 0..self.erle.len() {
            for k in 0..FFT_LENGTH_BY_2 {
                let section_idx = self.n_active_sections[ch][k].min(self.num_sections - 1);
                let subband = self.band_to_subband[k];
                let correction = self.correction_factors[ch][section_idx][subband];
                self.erle[ch][k] =
                    (average_erle[ch][k] * correction).clamp(self.min_erle, self.max_erle[subband]);
            }
            self.erle[ch][FFT_LENGTH_BY_2] = self.erle[ch][FFT_LENGTH_BY_2 - 1];
        }
    }

    pub fn dump(&self, dumper: &ApmDataDumper) {
        if let Some(channel_estimators) = self.erle_estimators.first() {
            for estimator in channel_estimators {
                dumper.dump_raw_f32_slice("aec3_all_erle", estimator);
            }
        }
        if let Some(erle_ref) = self.erle_ref.first() {
            dumper.dump_raw_f32_slice("aec3_ref_erle", erle_ref);
        }
        if let Some(factors) = self.correction_factors.first() {
            for factor in factors {
                dumper.dump_raw_f32_slice("aec3_erle_correction_factor", factor);
            }
        }
    }

    fn compute_number_of_active_filter_sections(
        &mut self,
        render_buffer: &RenderBuffer<'_>,
        filter_frequency_responses: &[Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>],
    ) {
        self.compute_echo_estimate_per_filter_section(render_buffer, filter_frequency_responses);
        self.compute_active_filter_sections();
    }

    fn compute_echo_estimate_per_filter_section(
        &mut self,
        render_buffer: &RenderBuffer<'_>,
        filter_frequency_responses: &[Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>],
    ) {
        let spectrum_buffer = render_buffer.spectrum_buffer();
        let num_render_channels = spectrum_buffer.buffer[0].len();
        let inv_num_render_channels = if num_render_channels > 0 {
            1.0 / num_render_channels as f32
        } else {
            0.0
        };

        for capture_ch in 0..self.s2_section_accum.len() {
            let mut idx_render = render_buffer.position();
            let offset0 = self.section_boundaries_blocks[0] as isize;
            idx_render = spectrum_buffer.offset_index(idx_render, offset0);

            for section in 0..self.num_sections {
                let mut x2_section = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
                let mut h2_section = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
                let start_block = self.section_boundaries_blocks[section];
                let block_limit = self.section_boundaries_blocks[section + 1]
                    .min(filter_frequency_responses[capture_ch].len());

                for block in start_block..block_limit {
                    for channel in 0..num_render_channels {
                        for (dst, &value) in x2_section
                            .iter_mut()
                            .zip(spectrum_buffer.buffer[idx_render][channel].iter())
                        {
                            *dst += value * inv_num_render_channels;
                        }
                    }
                    for (dst, &value) in h2_section
                        .iter_mut()
                        .zip(filter_frequency_responses[capture_ch][block].iter())
                    {
                        *dst += value;
                    }
                    idx_render = spectrum_buffer.inc_index(idx_render);
                }

                for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                    self.s2_section_accum[capture_ch][section][k] = x2_section[k] * h2_section[k];
                }
            }

            for section in 1..self.num_sections {
                for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                    self.s2_section_accum[capture_ch][section][k] +=
                        self.s2_section_accum[capture_ch][section - 1][k];
                }
            }
        }
    }

    fn compute_active_filter_sections(&mut self) {
        for ch in 0..self.n_active_sections.len() {
            self.n_active_sections[ch].fill(0);
            for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                let mut section = self.num_sections;
                let target = 0.9 * self.s2_section_accum[ch][self.num_sections - 1][k];
                while section > 0 && self.s2_section_accum[ch][section - 1][k] >= target {
                    section -= 1;
                    self.n_active_sections[ch][k] = section;
                }
            }
        }
    }

    fn update_correction_factors(
        &mut self,
        x2: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        y2: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        e2: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        converged_filters: &[bool],
    ) {
        for ch in 0..converged_filters.len() {
            if !converged_filters[ch] {
                continue;
            }

            let x2_subbands = self.aggregate_subband_powers(x2);
            let e2_subbands = self.aggregate_subband_powers(&e2[ch]);
            let y2_subbands = self.aggregate_subband_powers(&y2[ch]);
            let idx_subbands = self.active_sections_per_subband(ch);

            let mut new_erle = [0.0f32; SUBBANDS];
            let mut is_updated = [false; SUBBANDS];

            for subband in 0..SUBBANDS {
                if x2_subbands[subband] > X2_BAND_ENERGY_THRESHOLD && e2_subbands[subband] > 0.0 {
                    new_erle[subband] = y2_subbands[subband] / e2_subbands[subband];
                    is_updated[subband] = true;
                    self.num_updates[ch][subband] += 1;
                }
            }

            for subband in 0..SUBBANDS {
                let idx = idx_subbands[subband].min(self.num_sections - 1);
                let alpha = if new_erle[subband] > self.erle_estimators[ch][idx][subband] {
                    SMOOTH_INCREASE
                } else {
                    SMOOTH_DECREASE
                };
                let alpha = if is_updated[subband] { alpha } else { 0.0 };
                self.erle_estimators[ch][idx][subband] = (self.erle_estimators[ch][idx][subband]
                    + alpha * (new_erle[subband] - self.erle_estimators[ch][idx][subband]))
                    .clamp(self.min_erle, self.max_erle[subband]);
            }

            for subband in 0..SUBBANDS {
                let alpha = if new_erle[subband] > self.erle_ref[ch][subband] {
                    SMOOTH_INCREASE
                } else {
                    SMOOTH_DECREASE
                };
                let alpha = if is_updated[subband] { alpha } else { 0.0 };
                self.erle_ref[ch][subband] = (self.erle_ref[ch][subband]
                    + alpha * (new_erle[subband] - self.erle_ref[ch][subband]))
                    .clamp(self.min_erle, self.max_erle[subband]);
            }

            for subband in 0..SUBBANDS {
                if is_updated[subband] && self.num_updates[ch][subband] > MIN_UPDATES_FOR_CORRECTION
                {
                    let idx = idx_subbands[subband].min(self.num_sections - 1);
                    if self.erle_ref[ch][subband] > 0.0 {
                        let new_factor =
                            self.erle_estimators[ch][idx][subband] / self.erle_ref[ch][subband];
                        self.correction_factors[ch][idx][subband] +=
                            0.1 * (new_factor - self.correction_factors[ch][idx][subband]);
                    }
                }
            }
        }
    }

    fn aggregate_subband_powers(
        &self,
        spectrum: &[f32; FFT_LENGTH_BY_2_PLUS_1],
    ) -> [f32; SUBBANDS] {
        let mut result = [0.0f32; SUBBANDS];
        for subband in 0..SUBBANDS {
            let start = BAND_BOUNDARIES[subband];
            let end = BAND_BOUNDARIES[subband + 1].min(spectrum.len());
            for k in start..end {
                result[subband] += spectrum[k];
            }
        }
        result
    }

    fn active_sections_per_subband(&self, ch: usize) -> [usize; SUBBANDS] {
        let mut idx_subbands = [0usize; SUBBANDS];
        for subband in 0..SUBBANDS {
            let start = BAND_BOUNDARIES[subband];
            let end = BAND_BOUNDARIES[subband + 1].min(self.n_active_sections[ch].len());
            let mut min_section = self.num_sections - 1;
            for &section in &self.n_active_sections[ch][start..end] {
                min_section = min_section.min(section);
            }
            idx_subbands[subband] = min_section;
        }
        idx_subbands
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_processing::aec3::aec3_common::num_bands_for_rate;
    use crate::audio_processing::aec3::render_delay_buffer::RenderDelayBuffer;

    const SAMPLE_RATE_HZ: i32 = 16_000;

    const ACTIVE_FRAME: [f32; BLOCK_SIZE] = [
        7459.88, 17209.6, 17383.0, 20768.9, 16816.7, 18386.3, 4492.83, 9675.85, 6665.52, 14808.6,
        9342.3, 7483.28, 19261.7, 4145.98, 1622.18, 13475.2, 7166.32, 6856.61, 21937.0, 7263.14,
        9569.07, 14919.0, 8413.32, 7551.89, 7848.65, 6011.27, 13080.6, 15865.2, 12656.0, 17459.6,
        4263.93, 4503.03, 9311.79, 21095.8, 12657.9, 13906.6, 19267.2, 11338.1, 16828.9, 11501.6,
        11405.0, 15031.4, 14541.6, 19765.5, 18346.3, 19350.2, 3157.47, 18095.8, 1743.68, 21328.2,
        19727.5, 7295.16, 10332.4, 11055.5, 20107.4, 14708.4, 12416.2, 16434.0, 2454.69, 9840.8,
        6867.23, 1615.75, 6059.9, 8394.19,
    ];

    struct TestInputs {
        render_delay_buffer: RenderDelayBuffer,
        x2: [f32; FFT_LENGTH_BY_2_PLUS_1],
        y2: Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
        e2: Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
        h2: Vec<Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>>,
        x: Vec<Vec<Vec<f32>>>,
        converged_filters: Vec<bool>,
        toggle: i32,
    }

    impl TestInputs {
        fn new(
            cfg: &EchoCanceller3Config,
            num_render_channels: usize,
            num_capture_channels: usize,
        ) -> Self {
            let mut render_delay_buffer =
                RenderDelayBuffer::new(cfg.clone(), SAMPLE_RATE_HZ, num_render_channels);
            render_delay_buffer.align_from_delay(4);
            let mut h2 = vec![
                vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; cfg.filter.main.length_blocks];
                num_capture_channels
            ];
            for block in &mut h2[0] {
                block.fill(1.0);
            }
            let num_bands = num_bands_for_rate(SAMPLE_RATE_HZ) as usize;
            let x = vec![vec![vec![0.0f32; BLOCK_SIZE]; num_render_channels]; num_bands];
            Self {
                render_delay_buffer,
                x2: [0.0; FFT_LENGTH_BY_2_PLUS_1],
                y2: vec![[0.0; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels],
                e2: vec![[0.0; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels],
                h2,
                x,
                converged_filters: vec![true; num_capture_channels],
                toggle: 0,
            }
        }

        fn update(&mut self) {
            if self.toggle % 2 == 0 {
                for band in &mut self.x {
                    for channel in band {
                        channel.fill(0.0);
                    }
                }
            } else {
                for band in &mut self.x {
                    for channel in band {
                        channel.copy_from_slice(&ACTIVE_FRAME);
                    }
                }
            }
            self.toggle += 1;
            self.render_delay_buffer.insert(&self.x);
            self.render_delay_buffer.prepare_capture_processing();
            self.update_current_power_spectra();
        }

        fn render_buffer(&self) -> RenderBuffer<'_> {
            self.render_delay_buffer.render_buffer()
        }

        fn x2(&self) -> &[f32; FFT_LENGTH_BY_2_PLUS_1] {
            &self.x2
        }

        fn y2(&self) -> &[[f32; FFT_LENGTH_BY_2_PLUS_1]] {
            &self.y2
        }

        fn e2(&self) -> &[[f32; FFT_LENGTH_BY_2_PLUS_1]] {
            &self.e2
        }

        fn h2(&self) -> &[Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>] {
            &self.h2
        }

        fn converged_filters(&self) -> &[bool] {
            &self.converged_filters
        }

        fn update_current_power_spectra(&mut self) {
            let render_buffer = self.render_delay_buffer.render_buffer();
            let spectrum_buffer = render_buffer.spectrum_buffer();
            let idx = render_buffer.position();
            let prev_idx = spectrum_buffer.offset_index(idx, 1);
            let current = &spectrum_buffer.buffer[idx][0];
            let previous = &spectrum_buffer.buffer[prev_idx][0];
            self.x2.copy_from_slice(current);
            for ch in 0..self.y2.len() {
                for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                    self.e2[ch][k] = 0.01 * previous[k];
                    self.y2[ch][k] = current[k] + self.e2[ch][k];
                }
            }
        }
    }

    #[test]
    fn sweep_settings() {
        for &num_render_channels in &[1, 2, 4] {
            for &num_capture_channels in &[1, 2, 4] {
                let base_cfg = EchoCanceller3Config::default();
                let max_length_blocks = 50usize;
                let mut blocks = 1usize;
                while blocks < max_length_blocks {
                    for delay_headroom in 0..5usize {
                        for num_sections in 2..max_length_blocks {
                            let mut cfg = base_cfg.clone();
                            cfg.filter.main.length_blocks = blocks;
                            cfg.filter.main_initial.length_blocks =
                                cfg.filter.main_initial.length_blocks.min(blocks);
                            cfg.delay.delay_headroom_samples = delay_headroom * BLOCK_SIZE;
                            cfg.erle.num_sections = num_sections;
                            if cfg.validate() {
                                let cfg_for_run = cfg.clone();
                                let mut estimator = SignalDependentErleEstimator::new(
                                    &cfg_for_run,
                                    num_capture_channels,
                                );
                                let average_erle = vec![
                                    [cfg_for_run.erle.max_l;
                                        FFT_LENGTH_BY_2_PLUS_1];
                                    num_capture_channels
                                ];
                                let mut inputs = TestInputs::new(
                                    &cfg_for_run,
                                    num_render_channels,
                                    num_capture_channels,
                                );
                                for _ in 0..10 {
                                    inputs.update();
                                    let render_buffer = inputs.render_buffer();
                                    estimator.update(
                                        &render_buffer,
                                        inputs.h2(),
                                        inputs.x2(),
                                        inputs.y2(),
                                        inputs.e2(),
                                        &average_erle,
                                        inputs.converged_filters(),
                                    );
                                }
                                drop(average_erle);
                            }
                        }
                    }
                    blocks += 10;
                }
            }
        }
    }

    #[test]
    fn longer_run() {
        for &num_render_channels in &[1, 2, 4] {
            for &num_capture_channels in &[1, 2, 4] {
                let mut cfg = EchoCanceller3Config::default();
                cfg.filter.main.length_blocks = 2;
                cfg.filter.main_initial.length_blocks = 1;
                cfg.delay.delay_headroom_samples = 0;
                cfg.delay.hysteresis_limit_blocks = 0;
                cfg.erle.num_sections = 2;
                assert!(cfg.validate());
                let cfg_for_run = cfg.clone();
                let mut estimator =
                    SignalDependentErleEstimator::new(&cfg_for_run, num_capture_channels);
                let average_erle =
                    vec![[cfg_for_run.erle.max_l; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels];
                let mut inputs =
                    TestInputs::new(&cfg_for_run, num_render_channels, num_capture_channels);
                for _ in 0..200 {
                    inputs.update();
                    let render_buffer = inputs.render_buffer();
                    estimator.update(
                        &render_buffer,
                        inputs.h2(),
                        inputs.x2(),
                        inputs.y2(),
                        inputs.e2(),
                        &average_erle,
                        inputs.converged_filters(),
                    );
                }
            }
        }
    }
}
