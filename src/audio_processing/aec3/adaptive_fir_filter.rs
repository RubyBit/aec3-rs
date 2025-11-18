use crate::audio_processing::aec3::aec3_common::{
    Aec3Optimization, FFT_LENGTH, FFT_LENGTH_BY_2, FFT_LENGTH_BY_2_PLUS_1, get_time_domain_length,
};
use crate::audio_processing::aec3::aec3_fft::Aec3Fft;
use crate::audio_processing::aec3::fft_data::FftData;
use crate::audio_processing::aec3::render_buffer::RenderBuffer;
use crate::audio_processing::logging::apm_data_dumper::ApmDataDumper;

/// Computes and stores the frequency response of the filter.
pub fn compute_frequency_response(
    num_partitions: usize,
    h: &[Vec<FftData>],
    h2: &mut Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
) {
    assert!(num_partitions <= h.len());
    if h2.len() < num_partitions {
        h2.resize(num_partitions, [0.0; FFT_LENGTH_BY_2_PLUS_1]);
    }
    for response in h2.iter_mut().take(num_partitions) {
        response.fill(0.0);
    }

    if num_partitions == 0 {
        return;
    }

    let num_render_channels = h[0].len();
    for p in 0..num_partitions {
        for ch in 0..num_render_channels {
            for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                let re = h[p][ch].re[k];
                let im = h[p][ch].im[k];
                let magnitude = re * re + im * im;
                h2[p][k] = h2[p][k].max(magnitude);
            }
        }
    }
}

/// Adapts the filter partitions as H(t+1)=H(t)+G(t)*conj(X(t)).
pub fn adapt_partitions(
    render_buffer: &RenderBuffer,
    g: &FftData,
    num_partitions: usize,
    h: &mut [Vec<FftData>],
) {
    assert!(num_partitions <= h.len());
    if num_partitions == 0 {
        return;
    }

    let render_fft = render_buffer.fft_buffer();
    let mut index = render_buffer.position();
    let len = render_fft.len();
    let num_render_channels = render_fft[index].len();

    for p in 0..num_partitions {
        for ch in 0..num_render_channels {
            let x = &render_fft[index][ch];
            let h_entry = &mut h[p][ch];
            for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                h_entry.re[k] += x.re[k] * g.re[k] + x.im[k] * g.im[k];
                h_entry.im[k] += x.re[k] * g.im[k] - x.im[k] * g.re[k];
            }
        }
        index = if index + 1 < len { index + 1 } else { 0 };
    }
}

/// Produces the filter output from the current partitions.
pub fn apply_filter(
    render_buffer: &RenderBuffer,
    num_partitions: usize,
    h: &[Vec<FftData>],
    s: &mut FftData,
) {
    assert!(num_partitions <= h.len());
    s.re.fill(0.0);
    s.im.fill(0.0);

    if num_partitions == 0 {
        return;
    }

    let render_fft = render_buffer.fft_buffer();
    let mut index = render_buffer.position();
    let len = render_fft.len();
    let num_render_channels = render_fft[index].len();

    for p in 0..num_partitions {
        for ch in 0..num_render_channels {
            let x = &render_fft[index][ch];
            let h_entry = &h[p][ch];
            for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                s.re[k] += x.re[k] * h_entry.re[k] - x.im[k] * h_entry.im[k];
                s.im[k] += x.re[k] * h_entry.im[k] + x.im[k] * h_entry.re[k];
            }
        }
        index = if index + 1 < len { index + 1 } else { 0 };
    }
}

/// Frequency-domain adaptive FIR filter mirroring the WebRTC reference implementation.
pub struct AdaptiveFirFilter {
    _data_dumper: ApmDataDumper,
    fft: Aec3Fft,
    optimization: Aec3Optimization,
    num_render_channels: usize,
    max_size_partitions: usize,
    size_change_duration_blocks: i32,
    one_by_size_change_duration_blocks: f32,
    current_size_partitions: usize,
    target_size_partitions: usize,
    old_target_size_partitions: usize,
    size_change_counter: i32,
    h: Vec<Vec<FftData>>,
    partition_to_constrain: usize,
}

impl AdaptiveFirFilter {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        max_size_partitions: usize,
        initial_size_partitions: usize,
        size_change_duration_blocks: usize,
        num_render_channels: usize,
        optimization: Aec3Optimization,
        data_dumper: ApmDataDumper,
    ) -> Self {
        assert!(max_size_partitions >= initial_size_partitions);
        assert!(size_change_duration_blocks > 0);
        assert!(num_render_channels > 0);

        let mut h = Vec::with_capacity(max_size_partitions);
        for _ in 0..max_size_partitions {
            h.push(vec![FftData::default(); num_render_channels]);
        }

        let size_change_duration_blocks = size_change_duration_blocks as i32;
        let mut filter = Self {
            _data_dumper: data_dumper,
            fft: Aec3Fft::new(),
            optimization,
            num_render_channels,
            max_size_partitions,
            size_change_duration_blocks,
            one_by_size_change_duration_blocks: 1.0 / size_change_duration_blocks as f32,
            current_size_partitions: initial_size_partitions,
            target_size_partitions: initial_size_partitions,
            old_target_size_partitions: initial_size_partitions,
            size_change_counter: 0,
            h,
            partition_to_constrain: 0,
        };
        zero_filter(0, filter.max_size_partitions, &mut filter.h);
        filter.set_size_partitions(initial_size_partitions, true);
        filter
    }

    pub fn handle_echo_path_change(&mut self) {
        zero_filter(
            self.current_size_partitions,
            self.max_size_partitions,
            &mut self.h,
        );
    }

    pub fn size_partitions(&self) -> usize {
        self.current_size_partitions
    }

    pub fn max_filter_size_partitions(&self) -> usize {
        self.max_size_partitions
    }

    pub fn set_size_partitions(&mut self, size: usize, immediate_effect: bool) {
        let size = size.min(self.max_size_partitions);
        self.target_size_partitions = size;
        if immediate_effect {
            let old_size = self.current_size_partitions;
            self.current_size_partitions = size;
            self.old_target_size_partitions = size;
            zero_filter(old_size, self.current_size_partitions, &mut self.h);
            if self.current_size_partitions > 0 {
                self.partition_to_constrain = self
                    .partition_to_constrain
                    .min(self.current_size_partitions - 1);
            } else {
                self.partition_to_constrain = 0;
            }
            self.size_change_counter = 0;
        } else {
            self.size_change_counter = self.size_change_duration_blocks;
        }
    }

    pub fn filter(&self, render_buffer: &RenderBuffer, s: &mut FftData) {
        match self.optimization {
            Aec3Optimization::Sse2 | Aec3Optimization::Neon | Aec3Optimization::None => {
                apply_filter(render_buffer, self.current_size_partitions, &self.h, s)
            }
        }
    }

    pub fn adapt(&mut self, render_buffer: &RenderBuffer, g: &FftData) {
        self.adapt_and_update_size(render_buffer, g);
        self.constrain();
    }

    pub fn adapt_with_impulse_response(
        &mut self,
        render_buffer: &RenderBuffer,
        g: &FftData,
        impulse_response: &mut Vec<f32>,
    ) {
        self.adapt_and_update_size(render_buffer, g);
        self.constrain_and_update_impulse_response(impulse_response);
    }

    pub fn compute_frequency_response(&self, h2: &mut Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>) {
        compute_frequency_response(self.current_size_partitions, &self.h, h2);
    }

    pub fn scale_filter(&mut self, factor: f32) {
        for partition in &mut self.h {
            for channel in partition {
                for value in channel.re.iter_mut() {
                    *value *= factor;
                }
                for value in channel.im.iter_mut() {
                    *value *= factor;
                }
            }
        }
    }

    pub fn set_filter(&mut self, num_partitions: usize, h: &[Vec<FftData>]) {
        let limit = num_partitions.min(self.current_size_partitions);
        assert!(h.len() >= limit);
        for p in 0..limit {
            assert_eq!(self.h[p].len(), h[p].len());
            for ch in 0..self.num_render_channels {
                self.h[p][ch] = h[p][ch].clone();
            }
        }
    }

    pub fn get_filter(&self) -> &[Vec<FftData>] {
        &self.h
    }

    fn adapt_and_update_size(&mut self, render_buffer: &RenderBuffer, g: &FftData) {
        self.update_size();
        match self.optimization {
            Aec3Optimization::Sse2 | Aec3Optimization::Neon | Aec3Optimization::None => {
                adapt_partitions(render_buffer, g, self.current_size_partitions, &mut self.h)
            }
        }
    }

    fn constrain(&mut self) {
        if self.current_size_partitions == 0 {
            return;
        }
        let mut h_time = [0.0f32; FFT_LENGTH];
        let partition = self
            .partition_to_constrain
            .min(self.current_size_partitions - 1);
        for ch in 0..self.num_render_channels {
            self.fft.ifft(&self.h[partition][ch], &mut h_time);
            for sample in &mut h_time[..FFT_LENGTH_BY_2] {
                *sample *= 1.0 / FFT_LENGTH_BY_2 as f32;
            }
            h_time[FFT_LENGTH_BY_2..].fill(0.0);
            self.fft.fft(&mut h_time, &mut self.h[partition][ch]);
        }
        self.partition_to_constrain = if partition + 1 < self.current_size_partitions {
            partition + 1
        } else {
            0
        };
    }

    fn constrain_and_update_impulse_response(&mut self, impulse_response: &mut Vec<f32>) {
        if self.current_size_partitions == 0 {
            impulse_response.clear();
            return;
        }

        let required_capacity = get_time_domain_length(self.max_size_partitions);
        if impulse_response.capacity() < required_capacity {
            impulse_response.reserve(required_capacity - impulse_response.capacity());
        }
        impulse_response.resize(get_time_domain_length(self.current_size_partitions), 0.0);

        let mut h_time = [0.0f32; FFT_LENGTH];
        let partition = self
            .partition_to_constrain
            .min(self.current_size_partitions - 1);
        let base = partition * FFT_LENGTH_BY_2;
        impulse_response[base..base + FFT_LENGTH_BY_2].fill(0.0);

        for ch in 0..self.num_render_channels {
            self.fft.ifft(&self.h[partition][ch], &mut h_time);
            for sample in &mut h_time[..FFT_LENGTH_BY_2] {
                *sample *= 1.0 / FFT_LENGTH_BY_2 as f32;
            }
            h_time[FFT_LENGTH_BY_2..].fill(0.0);

            if ch == 0 {
                impulse_response[base..base + FFT_LENGTH_BY_2]
                    .copy_from_slice(&h_time[..FFT_LENGTH_BY_2]);
            } else {
                for (dst, &src) in impulse_response[base..base + FFT_LENGTH_BY_2]
                    .iter_mut()
                    .zip(&h_time[..FFT_LENGTH_BY_2])
                {
                    if dst.abs() < src.abs() {
                        *dst = src;
                    }
                }
            }

            self.fft.fft(&mut h_time, &mut self.h[partition][ch]);
        }

        self.partition_to_constrain = if partition + 1 < self.current_size_partitions {
            partition + 1
        } else {
            0
        };
    }

    fn update_size(&mut self) {
        let old_size = self.current_size_partitions;
        if self.size_change_counter > 0 {
            self.size_change_counter -= 1;
            let change_factor =
                self.size_change_counter as f32 * self.one_by_size_change_duration_blocks;
            let averaged = self.old_target_size_partitions as f32 * change_factor
                + self.target_size_partitions as f32 * (1.0 - change_factor);
            self.current_size_partitions = averaged.max(1.0) as usize;
            if self.current_size_partitions > 0 {
                self.partition_to_constrain = self
                    .partition_to_constrain
                    .min(self.current_size_partitions - 1);
            } else {
                self.partition_to_constrain = 0;
            }
        } else {
            self.current_size_partitions = self.target_size_partitions;
            self.old_target_size_partitions = self.target_size_partitions;
        }
        zero_filter(old_size, self.current_size_partitions, &mut self.h);
    }
}

fn zero_filter(start: usize, end: usize, h: &mut [Vec<FftData>]) {
    if start >= end {
        return;
    }
    for partition in &mut h[start..end] {
        for channel in partition {
            channel.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::config::EchoCanceller3Config;
    use crate::audio_processing::aec3::adaptive_fir_filter_erl::compute_erl;
    use crate::audio_processing::aec3::aec_state::AecState;
    use crate::audio_processing::aec3::aec3_common::{
        BLOCK_SIZE, FFT_LENGTH, FFT_LENGTH_BY_2, FFT_LENGTH_BY_2_PLUS_1, detect_optimization,
        get_time_domain_length, num_bands_for_rate,
    };
    use crate::audio_processing::aec3::aec3_fft::{Aec3Fft, Window};
    use crate::audio_processing::aec3::delay_estimate::DelayEstimate;
    use crate::audio_processing::aec3::echo_path_variability::{
        DelayAdjustment, EchoPathVariability,
    };
    use crate::audio_processing::aec3::render_delay_buffer::RenderDelayBuffer;
    use crate::audio_processing::aec3::render_signal_analyzer::RenderSignalAnalyzer;
    use crate::audio_processing::aec3::shadow_filter_update_gain::ShadowFilterUpdateGain;
    use crate::audio_processing::aec3::subtractor_output::SubtractorOutput;
    use crate::audio_processing::logging::apm_data_dumper::ApmDataDumper;
    use crate::audio_processing::utility::cascaded_biquad_filter::{
        BiQuadCoefficients, CascadedBiQuadFilter,
    };
    use crate::test_support::echo_canceller_test_tools::{DelayBuffer, randomize_sample_vector};
    use crate::test_support::random::Random;

    #[test]
    fn filter_statistics_access() {
        let optimization = detect_optimization();
        let data_dumper = ApmDataDumper::new(42);
        let filter = AdaptiveFirFilter::new(9, 9, 250, 1, optimization, data_dumper);
        let mut h2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; filter.max_filter_size_partitions()];
        filter.compute_frequency_response(&mut h2);
        let mut erl = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        compute_erl(optimization, &h2, &mut erl);
        assert!(erl.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn filter_size_matches_configuration() {
        let optimization = detect_optimization();
        for size in 1..5 {
            let data_dumper = ApmDataDumper::new(7);
            let filter = AdaptiveFirFilter::new(size, size, 10, 1, optimization, data_dumper);
            assert_eq!(size, filter.size_partitions());
        }
    }

    #[test]
    fn filter_and_adapt_converges_on_synthetic_echo_paths() {
        const SAMPLE_RATE_HZ: i32 = 48_000;
        const NUM_BANDS: usize = num_bands_for_rate(SAMPLE_RATE_HZ);
        const NUM_BLOCKS_PER_RENDER_CHANNEL: usize = 1000;
        const HIGHPASS: BiQuadCoefficients = BiQuadCoefficients {
            b: [0.97261, -1.94523, 0.97261],
            a: [-1.94448, 0.94598],
        };
        const DELAYS: [usize; 5] = [0, 64, 150, 200, 301];

        for &num_capture_channels in &[1usize, 2, 4] {
            for &num_render_channels in &[1usize, 2, 3, 6, 8] {
                let optimization = detect_optimization();
                let mut config = EchoCanceller3Config::default();
                config.delay.default_delay = 1;
                let data_dumper = ApmDataDumper::new(1234);
                let mut filter = AdaptiveFirFilter::new(
                    config.filter.main.length_blocks,
                    config.filter.main.length_blocks,
                    config.filter.config_change_duration_blocks,
                    num_render_channels,
                    optimization,
                    data_dumper,
                );
                let mut h2 =
                    vec![
                        vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; filter.max_filter_size_partitions()];
                        num_capture_channels
                    ];
                let mut impulse_responses =
                    vec![
                        vec![0.0f32; get_time_domain_length(filter.max_filter_size_partitions())];
                        num_capture_channels
                    ];
                let fft = Aec3Fft::new();
                let mut render_delay_buffer =
                    RenderDelayBuffer::new(config.clone(), SAMPLE_RATE_HZ, num_render_channels);
                let mut gain = ShadowFilterUpdateGain::new(
                    config.filter.shadow.clone(),
                    config.filter.config_change_duration_blocks,
                );
                let mut random_generator = Random::new(42);
                let mut x = vec![vec![vec![0.0f32; BLOCK_SIZE]; num_render_channels]; NUM_BANDS];
                let mut n = vec![0.0f32; BLOCK_SIZE];
                let mut y = vec![0.0f32; BLOCK_SIZE];
                let mut aec_state =
                    AecState::new(EchoCanceller3Config::default(), num_capture_channels);
                let mut render_signal_analyzer = RenderSignalAnalyzer::new(&config);
                let delay_estimate: Option<DelayEstimate> = None;
                let mut e = vec![0.0f32; BLOCK_SIZE];
                let mut s_scratch = [0.0f32; FFT_LENGTH];
                let mut output = vec![SubtractorOutput::new(); num_capture_channels];
                for o in &mut output {
                    o.reset();
                }
                let mut s_freq = FftData::default();
                let mut g_freq = FftData::default();
                let mut e_freq = FftData::default();
                let y2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels];
                let e2_main = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels];
                let mut render_power = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
                const SCALE: f32 = 1.0 / FFT_LENGTH_BY_2 as f32;
                let variability = EchoPathVariability::new(false, DelayAdjustment::None, false);

                for &delay_samples in &DELAYS {
                    let mut delay_buffer =
                        vec![DelayBuffer::<f32>::new(delay_samples); num_render_channels];
                    let mut x_hp_filter: Vec<CascadedBiQuadFilter> = (0..num_render_channels)
                        .map(|_| CascadedBiQuadFilter::with_coefficients(HIGHPASS, 1))
                        .collect();
                    let mut y_hp_filter = CascadedBiQuadFilter::with_coefficients(HIGHPASS, 1);
                    let num_blocks_to_process = NUM_BLOCKS_PER_RENDER_CHANNEL * num_render_channels;
                    let mut delayed_samples = vec![vec![0.0f32; BLOCK_SIZE]; num_render_channels];
                    for j in 0..num_blocks_to_process {
                        y.fill(0.0);
                        for ch in 0..num_render_channels {
                            randomize_sample_vector(&mut random_generator, &mut x[0][ch]);
                            let delayed = &mut delayed_samples[ch];
                            delay_buffer[ch].delay(&x[0][ch], delayed);
                            for k in 0..BLOCK_SIZE {
                                y[k] += delayed[k] / num_render_channels as f32;
                            }
                        }

                        randomize_sample_vector(&mut random_generator, &mut n);
                        let noise_scaling = 1.0 / 100.0 / num_render_channels as f32;
                        for k in 0..BLOCK_SIZE {
                            y[k] += n[k] * noise_scaling;
                        }

                        for ch in 0..num_render_channels {
                            x_hp_filter[ch].process_in_place(&mut x[0][ch]);
                        }
                        y_hp_filter.process_in_place(&mut y);

                        render_delay_buffer.insert(&x);
                        if j == 0 {
                            render_delay_buffer.reset();
                        }
                        render_delay_buffer.prepare_capture_processing();
                        let render_buffer = render_delay_buffer.render_buffer();

                        let min_delay = aec_state.min_direct_path_filter_delay();
                        let delay_partitions = if min_delay >= 0 {
                            Some(min_delay as usize)
                        } else {
                            None
                        };
                        render_signal_analyzer.update(&render_buffer, delay_partitions);

                        filter.filter(&render_buffer, &mut s_freq);
                        fft.ifft(&s_freq, &mut s_scratch);
                        for k in 0..BLOCK_SIZE {
                            let estimate = s_scratch[k + FFT_LENGTH_BY_2] * SCALE;
                            e[k] = (y[k] - estimate).clamp(-32_768.0, 32_767.0);
                        }
                        fft.zero_padded_fft(&e, Window::Rectangular, &mut e_freq);
                        for channel in &mut output {
                            for k in 0..BLOCK_SIZE {
                                channel.s_main[k] = SCALE * s_scratch[k + FFT_LENGTH_BY_2];
                            }
                        }

                        render_buffer.spectral_sum(filter.size_partitions(), &mut render_power);
                        gain.compute(
                            &render_power,
                            &render_signal_analyzer,
                            &e_freq,
                            filter.size_partitions(),
                            false,
                            &mut g_freq,
                        );
                        filter.adapt_with_impulse_response(
                            &render_buffer,
                            &g_freq,
                            &mut impulse_responses[0],
                        );
                        aec_state.handle_echo_path_change(&variability);
                        filter.compute_frequency_response(&mut h2[0]);
                        aec_state.update(
                            delay_estimate.clone(),
                            &h2,
                            &impulse_responses,
                            &render_buffer,
                            &e2_main,
                            &y2,
                            &output,
                        );
                    }

                    let e_energy: f32 = e.iter().map(|v| v * v).sum();
                    let y_energy: f32 = y.iter().map(|v| v * v).sum();
                    assert!(
                        1000.0 * e_energy < y_energy,
                        "filter failed to attenuate echo for channels c={}, r={}, delay={}",
                        num_capture_channels,
                        num_render_channels,
                        delay_samples
                    );
                }
            }
        }
    }
}
