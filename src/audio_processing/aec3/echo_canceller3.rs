use std::collections::VecDeque;

use crate::api::{
    config::EchoCanceller3Config,
    control::{EchoControl, Metrics},
};
use crate::audio_processing::aec3::aec3_common::{
    BLOCK_SIZE, MAX_NUM_BANDS, RENDER_TRANSFER_QUEUE_SIZE_FRAMES, SUB_FRAME_LENGTH,
    num_bands_for_rate, valid_full_band_rate,
};
use crate::audio_processing::aec3::api_call_jitter_metrics::ApiCallJitterMetrics;
use crate::audio_processing::aec3::block_delay_buffer::BlockDelayBuffer;
use crate::audio_processing::aec3::block_framer::BlockFramer;
use crate::audio_processing::aec3::block_processor::BlockProcessor;
use crate::audio_processing::aec3::config_selector::ConfigSelector;
use crate::audio_processing::aec3::frame_blocker::FrameBlocker;
use crate::audio_processing::aec3::multi_channel_content_detector::MultiChannelContentDetector;
use crate::audio_processing::audio_buffer::AudioBuffer;
use crate::audio_processing::high_pass_filter::HighPassFilter;
use crate::audio_processing::logging::apm_data_dumper::{ApmDataDumper, DiagnosticLevel};

const LINEAR_OUTPUT_BANDS: usize = 1;
const SUB_FRAMES_PER_FRAME: usize = AudioBuffer::SPLIT_BAND_SIZE / SUB_FRAME_LENGTH;

type Tensor3 = Vec<Vec<Vec<f32>>>;
type Tensor2 = Vec<Vec<f32>>;

trait BlockProcessorBackend {
    fn process_capture(
        &mut self,
        level_change: bool,
        saturated_microphone_signal: bool,
        linear_output: Option<&mut Tensor3>,
        capture_block: &mut Tensor3,
    );

    fn buffer_render(&mut self, render_block: &[Tensor2]);

    fn update_echo_leakage_status(&mut self, leakage_detected: bool);

    fn set_audio_buffer_delay(&mut self, delay_ms: i32);

    fn metrics(&self) -> Metrics;
}

impl BlockProcessorBackend for BlockProcessor {
    fn process_capture(
        &mut self,
        level_change: bool,
        saturated_microphone_signal: bool,
        linear_output: Option<&mut Tensor3>,
        capture_block: &mut Tensor3,
    ) {
        BlockProcessor::process_capture(
            self,
            level_change,
            saturated_microphone_signal,
            linear_output,
            capture_block,
        );
    }

    fn buffer_render(&mut self, render_block: &[Tensor2]) {
        BlockProcessor::buffer_render(self, render_block);
    }

    fn update_echo_leakage_status(&mut self, leakage_detected: bool) {
        BlockProcessor::update_echo_leakage_status(self, leakage_detected);
    }

    fn set_audio_buffer_delay(&mut self, delay_ms: i32) {
        BlockProcessor::set_audio_buffer_delay(self, delay_ms);
    }

    fn metrics(&self) -> Metrics {
        BlockProcessor::get_metrics(self)
    }
}

struct SwapQueue<T> {
    capacity: usize,
    queue: VecDeque<T>,
}

impl<T> SwapQueue<T> {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            queue: VecDeque::with_capacity(capacity),
        }
    }

    fn insert(&mut self, value: T) -> bool {
        if self.queue.len() >= self.capacity {
            return false;
        }
        self.queue.push_back(value);
        true
    }

    fn remove(&mut self) -> Option<T> {
        self.queue.pop_front()
    }
}

struct RenderWriter {
    data_dumper: ApmDataDumper,
    num_bands: usize,
    num_channels: usize,
    high_pass_filter: HighPassFilter,
    render_queue_input_frame: Tensor3,
}

impl RenderWriter {
    fn new(data_dumper: ApmDataDumper, num_bands: usize, num_channels: usize) -> Self {
        let high_pass_filter = HighPassFilter::new(16_000, num_channels);
        let render_queue_input_frame =
            allocate_tensor(num_bands, num_channels, AudioBuffer::SPLIT_BAND_SIZE);
        Self {
            data_dumper,
            num_bands,
            num_channels,
            high_pass_filter,
            render_queue_input_frame,
        }
    }

    fn insert(&mut self, input: &AudioBuffer, queue: &mut SwapQueue<Tensor3>) {
        assert_eq!(self.num_bands, input.num_bands());
        assert_eq!(self.num_channels, input.num_channels());
        assert_eq!(AudioBuffer::SPLIT_BAND_SIZE, input.num_frames_per_band());

        self.data_dumper.dump_wav(
            DiagnosticLevel::Developer,
            "aec3_render_input",
            AudioBuffer::SPLIT_BAND_SIZE,
            input.split_band(0, 0),
            16_000,
            1,
        );

        copy_buffer_into_frame(input, &mut self.render_queue_input_frame);
        self.high_pass_filter
            .process(&mut self.render_queue_input_frame[0]);

        let _ = queue.insert(self.render_queue_input_frame.clone());
    }
}

pub struct EchoCanceller3 {
    data_dumper: ApmDataDumper,
    config_selector: ConfigSelector,
    multichannel_content_detector: MultiChannelContentDetector,
    sample_rate_hz: i32,
    num_bands: usize,
    /// Number of render channels supplied through the API.
    num_render_input_channels: usize,
    /// Number of render channels actually processed: either
    /// `num_render_input_channels` when proper multichannel content is
    /// detected, or 1, with the render signal downmixed.
    num_render_channels_to_aec: usize,
    num_capture_channels: usize,
    export_linear_output: bool,
    render_writer: RenderWriter,
    render_transfer_queue: SwapQueue<Tensor3>,
    render_blocker: FrameBlocker,
    capture_blocker: FrameBlocker,
    output_framer: BlockFramer,
    linear_output_framer: Option<BlockFramer>,
    block_processor: Box<dyn BlockProcessorBackend>,
    render_queue_output_frame: Tensor3,
    render_block: Tensor3,
    capture_block: Tensor3,
    linear_output_block: Option<Tensor3>,
    render_sub_frame: Tensor3,
    capture_sub_frame: Tensor3,
    linear_output_sub_frame: Option<Tensor3>,
    block_delay_buffer: BlockDelayBuffer,
    api_call_metrics: ApiCallJitterMetrics,
    saturated_microphone_signal: bool,
}

impl EchoCanceller3 {
    pub fn new(
        config: EchoCanceller3Config,
        sample_rate_hz: i32,
        num_render_channels: usize,
        num_capture_channels: usize,
    ) -> Self {
        Self::with_multichannel_config(
            config,
            None,
            sample_rate_hz,
            num_render_channels,
            num_capture_channels,
        )
    }

    /// Creates an echo canceller that switches to `multichannel_config` while
    /// multichannel render content is detected.
    ///
    /// The two configs must agree on `delay.fixed_capture_delay_samples`,
    /// `filter.export_linear_aec_output`, `multi_channel.detect_stereo_content`
    /// and `multi_channel.stereo_detection_timeout_threshold_seconds`, since
    /// those are applied outside the switchable part of the pipeline.
    pub fn with_multichannel_config(
        config: EchoCanceller3Config,
        multichannel_config: Option<EchoCanceller3Config>,
        sample_rate_hz: i32,
        num_render_channels: usize,
        num_capture_channels: usize,
    ) -> Self {
        assert!(valid_full_band_rate(sample_rate_hz));
        let num_bands = num_bands_for_rate(sample_rate_hz);
        assert!(num_bands > 0 && num_bands <= MAX_NUM_BANDS);

        let config = adjust_config(config);
        let multichannel_config = multichannel_config.map(adjust_config);

        let config_selector =
            ConfigSelector::new(config, multichannel_config, num_render_channels);
        let multichannel_content_detector = {
            let active = config_selector.active_config();
            MultiChannelContentDetector::new(
                active.multi_channel.detect_stereo_content,
                num_render_channels,
                active.multi_channel.stereo_detection_threshold,
                active.multi_channel.stereo_detection_timeout_threshold_seconds,
                active.multi_channel.stereo_detection_hysteresis_seconds,
            )
        };

        let data_dumper = ApmDataDumper::new_unique();
        let render_writer = RenderWriter::new(data_dumper.clone(), num_bands, num_render_channels);
        let render_transfer_queue = SwapQueue::new(RENDER_TRANSFER_QUEUE_SIZE_FRAMES);
        let capture_blocker = FrameBlocker::new(num_bands, num_capture_channels);
        let output_framer = BlockFramer::new(num_bands, num_capture_channels);

        let export_linear_output = config_selector
            .active_config()
            .filter
            .export_linear_aec_output;
        let linear_output_framer = export_linear_output
            .then(|| BlockFramer::new(LINEAR_OUTPUT_BANDS, num_capture_channels));
        let linear_output_block = export_linear_output
            .then(|| allocate_tensor(LINEAR_OUTPUT_BANDS, num_capture_channels, BLOCK_SIZE));
        let linear_output_sub_frame = export_linear_output
            .then(|| allocate_tensor(LINEAR_OUTPUT_BANDS, num_capture_channels, SUB_FRAME_LENGTH));

        let render_queue_output_frame =
            allocate_tensor(num_bands, num_render_channels, AudioBuffer::SPLIT_BAND_SIZE);
        let capture_block = allocate_tensor(num_bands, num_capture_channels, BLOCK_SIZE);
        let capture_sub_frame = allocate_tensor(num_bands, num_capture_channels, SUB_FRAME_LENGTH);

        let block_delay_buffer = BlockDelayBuffer::new(
            num_bands,
            AudioBuffer::SPLIT_BAND_SIZE,
            config_selector
                .active_config()
                .delay
                .fixed_capture_delay_samples,
        );

        // The render path is sized to the channel count actually processed.
        // `ConfigSelector` was constructed with this same detection state, so
        // this matches what `initialize` computes; it is inlined here only to
        // avoid building a block processor that would be immediately replaced.
        let num_render_channels_to_aec = if multichannel_content_detector
            .is_proper_multi_channel_content_detected()
        {
            num_render_channels
        } else {
            1
        };
        let render_blocker = FrameBlocker::new(num_bands, num_render_channels_to_aec);
        let render_block = allocate_tensor(num_bands, num_render_channels_to_aec, BLOCK_SIZE);
        let render_sub_frame =
            allocate_tensor(num_bands, num_render_channels_to_aec, SUB_FRAME_LENGTH);
        let block_processor: Box<dyn BlockProcessorBackend> = Box::new(BlockProcessor::new(
            config_selector.active_config().clone(),
            sample_rate_hz,
            num_render_channels_to_aec,
            num_capture_channels,
        ));

        Self {
            data_dumper,
            config_selector,
            multichannel_content_detector,
            sample_rate_hz,
            num_bands,
            num_render_input_channels: num_render_channels,
            num_render_channels_to_aec,
            num_capture_channels,
            export_linear_output,
            render_writer,
            render_transfer_queue,
            render_blocker,
            capture_blocker,
            output_framer,
            linear_output_framer,
            block_processor,
            render_queue_output_frame,
            render_block,
            capture_block,
            linear_output_block,
            render_sub_frame,
            capture_sub_frame,
            linear_output_sub_frame,
            block_delay_buffer,
            api_call_metrics: ApiCallJitterMetrics::new(),
            saturated_microphone_signal: false,
        }
    }

    /// Injects a block processor, mirroring the reference's
    /// `SetBlockProcessorForTesting`.
    ///
    /// Note that a later change in multichannel content detection re-runs
    /// [`Self::initialize`] and replaces the injected processor, exactly as in
    /// the reference.
    #[cfg(test)]
    fn with_block_processor(
        config: EchoCanceller3Config,
        sample_rate_hz: i32,
        num_render_channels: usize,
        num_capture_channels: usize,
        block_processor: Box<dyn BlockProcessorBackend>,
    ) -> Self {
        let mut aec3 = Self::new(
            config,
            sample_rate_hz,
            num_render_channels,
            num_capture_channels,
        );
        aec3.block_processor = block_processor;
        aec3
    }

    /// Sizes the render path to the number of channels actually processed and
    /// (re)creates the block processor with the currently active config.
    fn initialize(&mut self) {
        let multichannel_content_detected = self
            .multichannel_content_detector
            .is_proper_multi_channel_content_detected();

        self.num_render_channels_to_aec = if multichannel_content_detected {
            self.num_render_input_channels
        } else {
            1
        };

        self.config_selector.update(multichannel_content_detected);

        self.render_block = allocate_tensor(
            self.num_bands,
            self.num_render_channels_to_aec,
            BLOCK_SIZE,
        );
        self.render_sub_frame = allocate_tensor(
            self.num_bands,
            self.num_render_channels_to_aec,
            SUB_FRAME_LENGTH,
        );
        self.render_blocker = FrameBlocker::new(self.num_bands, self.num_render_channels_to_aec);

        self.block_processor = Box::new(BlockProcessor::new(
            self.config_selector.active_config().clone(),
            self.sample_rate_hz,
            self.num_render_channels_to_aec,
            self.num_capture_channels,
        ));
    }

    pub fn update_echo_leakage_status(&mut self, leakage_detected: bool) {
        self.block_processor
            .update_echo_leakage_status(leakage_detected);
    }

    /// Enables or disables diagnostic dumping globally.
    ///
    /// When the `diagnostics` feature is disabled, this is a no-op.
    pub fn set_diagnostics_enabled(enabled: bool) {
        ApmDataDumper::set_activated(enabled);
    }

    /// Sets the global diagnostics output directory.
    ///
    /// When the `diagnostics` feature is disabled, this is a no-op.
    pub fn set_diagnostics_output_directory<P: AsRef<std::path::Path>>(path: P) {
        ApmDataDumper::set_output_directory(path);
    }

    /// Sets the global diagnostics verbosity level.
    ///
    /// When the `diagnostics` feature is disabled, this is a no-op.
    pub fn set_diagnostics_level(level: DiagnosticLevel) {
        ApmDataDumper::set_diagnostics_level(level);
    }

    /// Starts a new logical recording set.
    ///
    /// This can be useful to separate multiple runs inside the same log.
    pub fn initiate_new_set_of_recordings(&self) {
        self.data_dumper.initiate_new_set_of_recordings();
    }

    /// Picks a config from the render channel count alone.
    ///
    /// This has no equivalent in the reference. It predates multichannel
    /// content detection and applies the multichannel tuning to *every* stream
    /// with more than one render channel, including upmixed mono — which is
    /// what the detector exists to avoid.
    #[deprecated(
        since = "0.3.2",
        note = "pass EchoCanceller3Config::default() and \
                EchoCanceller3Config::create_default_multichannel_config() to \
                EchoCanceller3::with_multichannel_config instead, which applies \
                the multichannel tuning only while stereo content is detected"
    )]
    pub fn create_default_config(
        num_render_channels: usize,
        num_capture_channels: usize,
    ) -> EchoCanceller3Config {
        let _ = num_capture_channels;
        if num_render_channels > 1 {
            return EchoCanceller3Config::create_default_multichannel_config();
        }
        EchoCanceller3Config::default()
    }

    fn analyze_render_internal(&mut self, render: &AudioBuffer) {
        assert_eq!(self.num_render_input_channels, render.num_channels());
        assert_eq!(self.num_bands, render.num_bands());
        assert_eq!(AudioBuffer::SPLIT_BAND_SIZE, render.num_frames_per_band());

        self.data_dumper.dump_raw_usize(
            DiagnosticLevel::DeepDebug,
            "aec3_call_order",
            EchoCanceller3ApiCall::Render as usize,
        );
        self.render_writer
            .insert(render, &mut self.render_transfer_queue);
    }

    fn analyze_capture_internal(&mut self, capture: &AudioBuffer) {
        self.data_dumper.dump_wav(
            DiagnosticLevel::Developer,
            "aec3_capture_analyze_input",
            capture.num_frames(),
            capture.channel(0),
            self.sample_rate_hz as usize,
            1,
        );

        self.saturated_microphone_signal = false;
        for ch in 0..capture.num_channels() {
            if detect_saturation(capture.channel(ch)) {
                self.saturated_microphone_signal = true;
                break;
            }
        }
    }

    fn process_capture_internal(
        &mut self,
        capture: &mut AudioBuffer,
        mut linear_output: Option<&mut AudioBuffer>,
        level_change: bool,
    ) {
        assert_eq!(self.num_bands, capture.num_bands());
        assert_eq!(self.num_capture_channels, capture.num_channels());
        assert_eq!(AudioBuffer::SPLIT_BAND_SIZE, capture.num_frames_per_band());

        if let Some(ref linear_output) = linear_output {
            if !self.export_linear_output {
                panic!("Linear AEC output requested but not configured");
            }
            assert_eq!(LINEAR_OUTPUT_BANDS, linear_output.num_bands());
            assert_eq!(self.num_capture_channels, linear_output.num_channels());
            assert_eq!(
                AudioBuffer::SPLIT_BAND_SIZE,
                linear_output.num_frames_per_band(),
            );
        }

        self.data_dumper.dump_raw_usize(
            DiagnosticLevel::DeepDebug,
            "aec3_call_order",
            EchoCanceller3ApiCall::Capture as usize,
        );

        self.api_call_metrics.report_capture_call();

        if self
            .config_selector
            .active_config()
            .delay
            .fixed_capture_delay_samples
            > 0
        {
            self.block_delay_buffer.delay_signal(capture);
        }

        self.data_dumper.dump_wav(
            DiagnosticLevel::Developer,
            "aec3_capture_input",
            AudioBuffer::SPLIT_BAND_SIZE,
            capture.split_band(0, 0),
            16_000,
            1,
        );

        self.empty_render_queue();

        let saturated = self.saturated_microphone_signal;
        for sub_frame_index in 0..SUB_FRAMES_PER_FRAME {
            let linear_output_ref = linear_output.as_deref_mut();
            self.process_capture_frame_content(
                capture,
                linear_output_ref,
                level_change,
                saturated,
                sub_frame_index,
            );
        }

        self.process_remaining_capture_frame_content(level_change, saturated);

        self.data_dumper.dump_wav(
            DiagnosticLevel::Developer,
            "aec3_capture_output",
            AudioBuffer::SPLIT_BAND_SIZE,
            capture.split_band(0, 0),
            16_000,
            1,
        );
    }

    fn process_capture_frame_content(
        &mut self,
        capture: &mut AudioBuffer,
        linear_output: Option<&mut AudioBuffer>,
        level_change: bool,
        saturated_microphone_signal: bool,
        sub_frame_index: usize,
    ) {
        fill_audio_buffer_sub_frame(capture, sub_frame_index, &mut self.capture_sub_frame);

        self.capture_blocker
            .insert_sub_frame_and_extract_block(&self.capture_sub_frame, &mut self.capture_block);

        let linear_block = self.linear_output_block.as_mut();
        self.block_processor.process_capture(
            level_change,
            saturated_microphone_signal,
            linear_block,
            &mut self.capture_block,
        );

        self.output_framer
            .insert_block_and_extract_sub_frame(&self.capture_block, &mut self.capture_sub_frame);
        write_audio_buffer_sub_frame(capture, sub_frame_index, &self.capture_sub_frame);

        if let (Some(linear_output_buffer), Some(sub_frame), Some(framer)) = (
            linear_output,
            self.linear_output_sub_frame.as_mut(),
            self.linear_output_framer.as_mut(),
        ) {
            if let Some(linear_block) = self.linear_output_block.as_mut() {
                framer.insert_block_and_extract_sub_frame(linear_block, sub_frame);
                write_audio_buffer_sub_frame(linear_output_buffer, sub_frame_index, sub_frame);
            }
        }
    }

    fn process_remaining_capture_frame_content(
        &mut self,
        level_change: bool,
        saturated_microphone_signal: bool,
    ) {
        if !self.capture_blocker.is_block_available() {
            return;
        }

        self.capture_blocker.extract_block(&mut self.capture_block);
        let linear_block = self.linear_output_block.as_mut();
        self.block_processor.process_capture(
            level_change,
            saturated_microphone_signal,
            linear_block,
            &mut self.capture_block,
        );
        self.output_framer.insert_block(&self.capture_block);

        if let (Some(framer), Some(linear_block)) = (
            self.linear_output_framer.as_mut(),
            self.linear_output_block.as_mut(),
        ) {
            framer.insert_block(linear_block);
        }
    }

    fn empty_render_queue(&mut self) {
        while let Some(frame) = self.render_transfer_queue.remove() {
            self.api_call_metrics.report_render_call();
            self.render_queue_output_frame = frame;

            if self
                .multichannel_content_detector
                .update_detection(&self.render_queue_output_frame)
            {
                // Reinitialize the AEC when proper stereo is detected.
                self.initialize();
            }

            // When the echo canceller processes mono but the render signal
            // carries stereo, downmix by averaging rather than by
            // selecting channel 0.
            let proper_downmix_needed = self
                .multichannel_content_detector
                .is_temporary_multi_channel_content_detected();

            for sub_frame_index in 0..SUB_FRAMES_PER_FRAME {
                fill_tensor_sub_frame(
                    &self.render_queue_output_frame,
                    proper_downmix_needed,
                    sub_frame_index,
                    &mut self.render_sub_frame,
                );
                self.render_blocker.insert_sub_frame_and_extract_block(
                    &self.render_sub_frame,
                    &mut self.render_block,
                );
                self.block_processor.buffer_render(&self.render_block);
            }

            while self.render_blocker.is_block_available() {
                self.render_blocker.extract_block(&mut self.render_block);
                self.block_processor.buffer_render(&self.render_block);
            }
        }
    }
}

enum EchoCanceller3ApiCall {
    Capture = 0,
    Render = 1,
}

impl EchoControl for EchoCanceller3 {
    type Buffer = AudioBuffer;

    fn analyze_render(&mut self, render: &mut Self::Buffer) {
        self.analyze_render_internal(render);
    }

    fn analyze_capture(&mut self, capture: &mut Self::Buffer) {
        self.analyze_capture_internal(capture);
    }

    fn process_capture(&mut self, capture: &mut Self::Buffer, level_change: bool) {
        self.process_capture_internal(capture, None, level_change);
    }

    fn process_capture_with_linear_output(
        &mut self,
        capture: &mut Self::Buffer,
        linear_output: &mut Self::Buffer,
        level_change: bool,
    ) {
        self.process_capture_internal(capture, Some(linear_output), level_change);
    }

    fn metrics(&self) -> Metrics {
        let mut metrics = self.block_processor.metrics();

        // ApiCallJitterMetrics::Jitter uses i32::MAX for the uninitialized min.
        // Surface a stable, user-friendly value instead. Prob should do that..
        let render_min = self.api_call_metrics.render_jitter().min();
        let capture_min = self.api_call_metrics.capture_jitter().min();

        metrics.render_jitter_min = (render_min != i32::MAX).then_some(render_min).unwrap_or(0);
        metrics.render_jitter_max = self.api_call_metrics.render_jitter().max();
        metrics.capture_jitter_min = (capture_min != i32::MAX)
            .then_some(capture_min)
            .unwrap_or(0);
        metrics.capture_jitter_max = self.api_call_metrics.capture_jitter().max();

        metrics
    }

    fn set_audio_buffer_delay(&mut self, delay_ms: i32) {
        self.block_processor.set_audio_buffer_delay(delay_ms);
    }

    fn active_processing(&self) -> bool {
        true
    }
}

fn detect_saturation(samples: &[f32]) -> bool {
    samples
        .iter()
        .any(|sample| *sample >= 32700.0 || *sample <= -32700.0)
}

fn adjust_config(config: EchoCanceller3Config) -> EchoCanceller3Config {
    config
}

fn allocate_tensor(bands: usize, channels: usize, length: usize) -> Tensor3 {
    (0..bands)
        .map(|_| (0..channels).map(|_| vec![0.0; length]).collect::<Vec<_>>())
        .collect::<Vec<_>>()
}

fn copy_buffer_into_frame(buffer: &AudioBuffer, frame: &mut Tensor3) {
    debug_assert_eq!(frame.len(), buffer.num_bands());
    for band in 0..buffer.num_bands() {
        debug_assert_eq!(frame[band].len(), buffer.num_channels());
        for channel in 0..buffer.num_channels() {
            let source = buffer.split_band(channel, band);
            frame[band][channel].copy_from_slice(source);
        }
    }
}

fn fill_audio_buffer_sub_frame(
    buffer: &AudioBuffer,
    sub_frame_index: usize,
    storage: &mut Tensor3,
) {
    let offset = sub_frame_index * SUB_FRAME_LENGTH;
    debug_assert_eq!(storage.len(), buffer.num_bands());
    for band in 0..storage.len() {
        debug_assert_eq!(storage[band].len(), buffer.num_channels());
        for channel in 0..storage[band].len() {
            let source = buffer.split_band(channel, band);
            storage[band][channel].copy_from_slice(&source[offset..offset + SUB_FRAME_LENGTH]);
        }
    }
}

fn write_audio_buffer_sub_frame(
    buffer: &mut AudioBuffer,
    sub_frame_index: usize,
    storage: &Tensor3,
) {
    let offset = sub_frame_index * SUB_FRAME_LENGTH;
    debug_assert_eq!(storage.len(), buffer.num_bands());
    for band in 0..storage.len() {
        debug_assert_eq!(storage[band].len(), buffer.num_channels());
        for channel in 0..storage[band].len() {
            let dest = buffer.split_band_mut(channel, band);
            dest[offset..offset + SUB_FRAME_LENGTH].copy_from_slice(&storage[band][channel]);
        }
    }
}

fn fill_tensor_sub_frame(
    frame: &Tensor3,
    proper_downmix_needed: bool,
    sub_frame_index: usize,
    storage: &mut Tensor3,
) {
    let offset = sub_frame_index * SUB_FRAME_LENGTH;
    let frame_num_channels = frame[0].len();
    let sub_frame_num_channels = storage[0].len();

    if frame_num_channels > sub_frame_num_channels {
        debug_assert_eq!(sub_frame_num_channels, 1);
        for band in 0..frame.len() {
            let destination = &mut storage[band][0];
            destination.copy_from_slice(&frame[band][0][offset..offset + SUB_FRAME_LENGTH]);
            if proper_downmix_needed {
                // When a proper downmix is needed (which is the case when
                // proper stereo is present in the echo reference signal but the
                // echo canceller does the processing in mono) downmix the echo
                // reference by averaging the channel content (otherwise
                // downmixing is done by selecting channel 0).
                for channel in 1..frame_num_channels {
                    let source = &frame[band][channel][offset..offset + SUB_FRAME_LENGTH];
                    for k in 0..SUB_FRAME_LENGTH {
                        destination[k] += source[k];
                    }
                }
                let one_by_num_channels = 1.0 / frame_num_channels as f32;
                for value in destination.iter_mut() {
                    *value *= one_by_num_channels;
                }
            }
        }
        return;
    }

    for band in 0..frame.len() {
        debug_assert_eq!(frame[band].len(), storage[band].len());
        for channel in 0..frame[band].len() {
            storage[band][channel]
                .copy_from_slice(&frame[band][channel][offset..offset + SUB_FRAME_LENGTH]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AudioBuffer, BLOCK_SIZE, BlockProcessorBackend, EchoCanceller3, HighPassFilter, Metrics,
        RENDER_TRANSFER_QUEUE_SIZE_FRAMES, Tensor2, Tensor3, num_bands_for_rate,
    };
    use crate::api::config::EchoCanceller3Config;
    use crate::api::control::EchoControl;
    use std::cell::RefCell;
    use std::collections::VecDeque;
    use std::rc::Rc;

    const NUM_FRAMES_TO_PROCESS: usize = 20;
    const RENDER_PIPELINE_QUEUE_TEST_FRAMES: usize = 30;
    const NUM_FULL_BLOCKS_PER_FRAME: usize = AudioBuffer::SPLIT_BAND_SIZE / BLOCK_SIZE;
    const EXPECTED_BLOCKS_PER_RUN: usize =
        NUM_FRAMES_TO_PROCESS * AudioBuffer::SPLIT_BAND_SIZE / BLOCK_SIZE;

    #[derive(Clone, Copy)]
    enum EchoPathChangeTestVariant {
        None,
        OneSticky,
        OneNonSticky,
    }

    #[derive(Clone, Copy)]
    enum EchoLeakageTestVariant {
        None,
        FalseSticky,
        TrueSticky,
        TrueNonSticky,
    }

    #[derive(Clone, Copy)]
    enum SaturationTestVariant {
        None,
        OneNegative,
        OnePositive,
    }

    struct EchoCanceller3Tester {
        sample_rate_hz: i32,
        num_bands: usize,
        fullband_frame_length: usize,
        capture_buffer: AudioBuffer,
        render_buffer: AudioBuffer,
    }

    impl EchoCanceller3Tester {
        fn new(sample_rate_hz: i32) -> Self {
            let num_bands = num_bands_for_rate(sample_rate_hz);
            let fullband_frame_length = (sample_rate_hz as usize) / 100;
            let capture_buffer = AudioBuffer::new(
                fullband_frame_length,
                1,
                fullband_frame_length,
                1,
                fullband_frame_length,
            );
            let render_buffer = AudioBuffer::new(
                fullband_frame_length,
                1,
                fullband_frame_length,
                1,
                fullband_frame_length,
            );
            Self {
                sample_rate_hz,
                num_bands,
                fullband_frame_length,
                capture_buffer,
                render_buffer,
            }
        }

        fn needs_band_split(&self) -> bool {
            self.sample_rate_hz > 16_000
        }

        fn optional_band_split(&mut self) {
            if self.needs_band_split() {
                self.capture_buffer.split_into_frequency_bands();
                self.render_buffer.split_into_frequency_bands();
            }
        }

        fn populate_capture_frame(&mut self, frame_index: usize, offset: i32) {
            for band in 0..self.num_bands {
                let dest = self.capture_buffer.split_band_mut(0, band);
                for (sample_idx, sample) in dest
                    .iter_mut()
                    .enumerate()
                    .take(AudioBuffer::SPLIT_BAND_SIZE)
                {
                    *sample = multiband_sample(frame_index, sample_idx, band, offset);
                }
            }
        }

        fn populate_render_subbands(&mut self, frame_index: usize, offset: i32) {
            for band in 0..self.num_bands {
                let dest = self.render_buffer.split_band_mut(0, band);
                for (sample_idx, sample) in dest
                    .iter_mut()
                    .enumerate()
                    .take(AudioBuffer::SPLIT_BAND_SIZE)
                {
                    *sample = multiband_sample(frame_index, sample_idx, band, offset);
                }
            }
        }

        fn populate_render_fullband(&mut self, frame_index: usize, offset: i32) {
            let channel = self.render_buffer.channel_mut(0);
            for (sample_idx, sample) in channel
                .iter_mut()
                .enumerate()
                .take(AudioBuffer::SPLIT_BAND_SIZE)
            {
                *sample = fullband_sample(frame_index, sample_idx, offset);
            }
        }

        fn append_render_samples(&self, storage: &mut Vec<f32>) {
            let band = self.render_buffer.split_band(0, 0);
            storage.extend_from_slice(&band[..AudioBuffer::SPLIT_BAND_SIZE]);
        }

        fn append_capture_samples(&self, storage: &mut Vec<f32>) {
            let band = self.capture_buffer.split_band(0, 0);
            storage.extend_from_slice(&band[..AudioBuffer::SPLIT_BAND_SIZE]);
        }

        fn run_capture_transport_verification_test(&mut self) {
            let mut aec3 = EchoCanceller3::with_block_processor(
                EchoCanceller3Config::default(),
                self.sample_rate_hz,
                1,
                1,
                Box::new(CaptureTransportVerificationProcessor),
            );

            for frame_index in 0..NUM_FRAMES_TO_PROCESS {
                aec3.analyze_capture(&mut self.capture_buffer);
                self.optional_band_split();
                self.populate_capture_frame(frame_index, 0);
                self.populate_render_fullband(frame_index, 0);
                aec3.analyze_render(&mut self.render_buffer);
                aec3.process_capture(&mut self.capture_buffer, false);
                assert!(verify_multiband_frame(
                    &self.capture_buffer,
                    self.num_bands,
                    frame_index,
                    -64
                ));
            }
        }

        fn run_render_transport_verification_test(&mut self) {
            let mut aec3 = EchoCanceller3::with_block_processor(
                EchoCanceller3Config::default(),
                self.sample_rate_hz,
                1,
                1,
                Box::new(RenderTransportVerificationProcessor::new()),
            );
            let mut render_input = vec![Vec::new()];
            let mut capture_output = Vec::new();

            for frame_index in 0..NUM_FRAMES_TO_PROCESS {
                aec3.analyze_capture(&mut self.capture_buffer);
                self.optional_band_split();
                self.populate_capture_frame(frame_index, 100);
                self.populate_render_subbands(frame_index, 0);
                self.append_render_samples(&mut render_input[0]);
                aec3.analyze_render(&mut self.render_buffer);
                aec3.process_capture(&mut self.capture_buffer, false);
                self.append_capture_samples(&mut capture_output);
            }

            let mut hp_filter = HighPassFilter::new(16_000, 1);
            hp_filter.process(&mut render_input);
            assert!(verify_sequence(&render_input[0], &capture_output, -64));
        }

        fn run_echo_path_change_verification_test(&mut self, variant: EchoPathChangeTestVariant) {
            let (processor, state) = RecordingBlockProcessor::new();
            let mut aec3 = EchoCanceller3::with_block_processor(
                EchoCanceller3Config::default(),
                self.sample_rate_hz,
                1,
                1,
                Box::new(processor),
            );

            for frame_index in 0..NUM_FRAMES_TO_PROCESS {
                let level_change = match variant {
                    EchoPathChangeTestVariant::None => false,
                    EchoPathChangeTestVariant::OneSticky => true,
                    EchoPathChangeTestVariant::OneNonSticky => frame_index == 0,
                };

                aec3.analyze_capture(&mut self.capture_buffer);
                self.optional_band_split();
                self.populate_capture_frame(frame_index, 0);
                self.populate_render_fullband(frame_index, 0);
                aec3.analyze_render(&mut self.render_buffer);
                aec3.process_capture(&mut self.capture_buffer, level_change);
            }

            let recorded = state.borrow();
            assert_eq!(recorded.render_call_count, EXPECTED_BLOCKS_PER_RUN);
            assert_eq!(recorded.capture_calls.len(), EXPECTED_BLOCKS_PER_RUN);
            match variant {
                EchoPathChangeTestVariant::None => {
                    assert!(recorded.capture_calls.iter().all(|(level, _)| !level));
                }
                EchoPathChangeTestVariant::OneSticky => {
                    assert!(recorded.capture_calls.iter().all(|(level, _)| *level));
                }
                EchoPathChangeTestVariant::OneNonSticky => {
                    for (idx, (level, _)) in recorded.capture_calls.iter().enumerate() {
                        if idx < NUM_FULL_BLOCKS_PER_FRAME {
                            assert!(*level);
                        } else {
                            assert!(!*level);
                        }
                    }
                }
            }
        }

        fn run_echo_leakage_verification_test(&mut self, variant: EchoLeakageTestVariant) {
            let (processor, state) = RecordingBlockProcessor::new();
            let mut aec3 = EchoCanceller3::with_block_processor(
                EchoCanceller3Config::default(),
                self.sample_rate_hz,
                1,
                1,
                Box::new(processor),
            );

            for frame_index in 0..NUM_FRAMES_TO_PROCESS {
                match variant {
                    EchoLeakageTestVariant::None => {}
                    EchoLeakageTestVariant::FalseSticky => {
                        if frame_index == 0 {
                            aec3.update_echo_leakage_status(false);
                        }
                    }
                    EchoLeakageTestVariant::TrueSticky => {
                        if frame_index == 0 {
                            aec3.update_echo_leakage_status(true);
                        }
                    }
                    EchoLeakageTestVariant::TrueNonSticky => {
                        aec3.update_echo_leakage_status(frame_index == 0);
                    }
                }

                aec3.analyze_capture(&mut self.capture_buffer);
                self.optional_band_split();
                self.populate_capture_frame(frame_index, 0);
                self.populate_render_fullband(frame_index, 0);
                aec3.analyze_render(&mut self.render_buffer);
                aec3.process_capture(&mut self.capture_buffer, false);
            }

            let recorded = state.borrow();
            assert_eq!(recorded.render_call_count, EXPECTED_BLOCKS_PER_RUN);
            match variant {
                EchoLeakageTestVariant::None => assert!(recorded.leakage_updates.is_empty()),
                EchoLeakageTestVariant::FalseSticky => {
                    assert_eq!(recorded.leakage_updates, vec![false]);
                }
                EchoLeakageTestVariant::TrueSticky => {
                    assert_eq!(recorded.leakage_updates, vec![true]);
                }
                EchoLeakageTestVariant::TrueNonSticky => {
                    let mut expected = vec![true];
                    expected.extend(std::iter::repeat(false).take(NUM_FRAMES_TO_PROCESS - 1));
                    assert_eq!(recorded.leakage_updates, expected);
                }
            }
        }

        fn run_capture_saturation_verification_test(&mut self, variant: SaturationTestVariant) {
            let (processor, state) = RecordingBlockProcessor::new();
            let mut aec3 = EchoCanceller3::with_block_processor(
                EchoCanceller3Config::default(),
                self.sample_rate_hz,
                1,
                1,
                Box::new(processor),
            );

            for frame_index in 0..NUM_FRAMES_TO_PROCESS {
                {
                    let channel = self.capture_buffer.channel_mut(0);
                    for sample in channel.iter_mut().take(self.fullband_frame_length) {
                        *sample = 0.0;
                    }
                    match variant {
                        SaturationTestVariant::None => {}
                        SaturationTestVariant::OneNegative => {
                            if frame_index == 0 {
                                channel[10] = -32768.0;
                            }
                        }
                        SaturationTestVariant::OnePositive => {
                            if frame_index == 0 {
                                channel[10] = 32767.0;
                            }
                        }
                    }
                }

                aec3.analyze_capture(&mut self.capture_buffer);
                self.optional_band_split();
                self.populate_capture_frame(frame_index, 0);
                self.populate_render_fullband(frame_index, 0);
                aec3.analyze_render(&mut self.render_buffer);
                aec3.process_capture(&mut self.capture_buffer, false);
            }

            let recorded = state.borrow();
            assert_eq!(recorded.render_call_count, EXPECTED_BLOCKS_PER_RUN);
            match variant {
                SaturationTestVariant::None => {
                    assert!(
                        recorded
                            .capture_calls
                            .iter()
                            .all(|(_, saturated)| !saturated)
                    );
                }
                SaturationTestVariant::OneNegative | SaturationTestVariant::OnePositive => {
                    for (idx, (_, saturated)) in recorded.capture_calls.iter().enumerate() {
                        if idx < NUM_FULL_BLOCKS_PER_FRAME {
                            assert!(*saturated);
                        } else {
                            assert!(!*saturated);
                        }
                    }
                }
            }
        }

        fn run_render_swap_queue_verification_test(&mut self) {
            let mut aec3 = EchoCanceller3::with_block_processor(
                EchoCanceller3Config::default(),
                self.sample_rate_hz,
                1,
                1,
                Box::new(RenderTransportVerificationProcessor::new()),
            );
            let mut render_input = vec![Vec::new()];
            let mut capture_output = Vec::new();

            for frame_index in 0..RENDER_TRANSFER_QUEUE_SIZE_FRAMES {
                if self.needs_band_split() {
                    self.render_buffer.split_into_frequency_bands();
                }
                self.populate_render_subbands(frame_index, 0);
                self.append_render_samples(&mut render_input[0]);
                aec3.analyze_render(&mut self.render_buffer);
            }

            for frame_index in 0..RENDER_TRANSFER_QUEUE_SIZE_FRAMES {
                aec3.analyze_capture(&mut self.capture_buffer);
                if self.needs_band_split() {
                    self.capture_buffer.split_into_frequency_bands();
                }
                self.populate_capture_frame(frame_index, 0);
                aec3.process_capture(&mut self.capture_buffer, false);
                self.append_capture_samples(&mut capture_output);
            }

            let mut hp_filter = HighPassFilter::new(16_000, 1);
            hp_filter.process(&mut render_input);
            assert!(verify_sequence(&render_input[0], &capture_output, -64));
        }

        fn run_render_pipeline_swap_queue_overrun_return_value_test(&mut self) {
            let mut aec3 =
                EchoCanceller3::new(EchoCanceller3Config::default(), self.sample_rate_hz, 1, 1);

            for _ in 0..2 {
                for frame_index in 0..RENDER_PIPELINE_QUEUE_TEST_FRAMES {
                    if self.needs_band_split() {
                        self.render_buffer.split_into_frequency_bands();
                    }
                    self.populate_render_fullband(frame_index, 0);
                    aec3.analyze_render(&mut self.render_buffer);
                }
            }
        }
    }

    struct CaptureTransportVerificationProcessor;

    impl BlockProcessorBackend for CaptureTransportVerificationProcessor {
        fn process_capture(
            &mut self,
            _level_change: bool,
            _saturated_microphone_signal: bool,
            _linear_output: Option<&mut Tensor3>,
            _capture_block: &mut Tensor3,
        ) {
        }

        fn buffer_render(&mut self, _render_block: &[Tensor2]) {}

        fn update_echo_leakage_status(&mut self, _leakage_detected: bool) {}

        fn set_audio_buffer_delay(&mut self, _delay_ms: i32) {}

        fn metrics(&self) -> Metrics {
            Metrics::default()
        }
    }

    struct RenderTransportVerificationProcessor {
        render_blocks: VecDeque<Tensor3>,
    }

    impl RenderTransportVerificationProcessor {
        fn new() -> Self {
            Self {
                render_blocks: VecDeque::new(),
            }
        }
    }

    impl BlockProcessorBackend for RenderTransportVerificationProcessor {
        fn process_capture(
            &mut self,
            _level_change: bool,
            _saturated_microphone_signal: bool,
            _linear_output: Option<&mut Tensor3>,
            capture_block: &mut Tensor3,
        ) {
            if let Some(block) = self.render_blocks.pop_front() {
                block_copy(&block, capture_block);
            }
        }

        fn buffer_render(&mut self, render_block: &[Tensor2]) {
            self.render_blocks.push_back(block_clone(render_block));
        }

        fn update_echo_leakage_status(&mut self, _leakage_detected: bool) {}

        fn set_audio_buffer_delay(&mut self, _delay_ms: i32) {}

        fn metrics(&self) -> Metrics {
            Metrics::default()
        }
    }

    #[derive(Default)]
    struct RecordingState {
        capture_calls: Vec<(bool, bool)>,
        render_call_count: usize,
        leakage_updates: Vec<bool>,
    }

    struct RecordingBlockProcessor {
        shared: Rc<RefCell<RecordingState>>,
    }

    impl RecordingBlockProcessor {
        fn new() -> (Self, Rc<RefCell<RecordingState>>) {
            let shared = Rc::new(RefCell::new(RecordingState::default()));
            (
                Self {
                    shared: shared.clone(),
                },
                shared,
            )
        }
    }

    impl BlockProcessorBackend for RecordingBlockProcessor {
        fn process_capture(
            &mut self,
            level_change: bool,
            saturated_microphone_signal: bool,
            _linear_output: Option<&mut Tensor3>,
            _capture_block: &mut Tensor3,
        ) {
            self.shared
                .borrow_mut()
                .capture_calls
                .push((level_change, saturated_microphone_signal));
        }

        fn buffer_render(&mut self, _render_block: &[Tensor2]) {
            self.shared.borrow_mut().render_call_count += 1;
        }

        fn update_echo_leakage_status(&mut self, leakage_detected: bool) {
            self.shared
                .borrow_mut()
                .leakage_updates
                .push(leakage_detected);
        }

        fn set_audio_buffer_delay(&mut self, _delay_ms: i32) {}

        fn metrics(&self) -> Metrics {
            Metrics::default()
        }
    }

    fn block_clone(block: &[Tensor2]) -> Tensor3 {
        block
            .iter()
            .map(|band| {
                band.iter()
                    .map(|channel| channel.clone())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    }

    fn block_copy(src: &[Tensor2], dst: &mut Tensor3) {
        for (dst_band, src_band) in dst.iter_mut().zip(src.iter()) {
            for (dst_channel, src_channel) in dst_band.iter_mut().zip(src_band.iter()) {
                dst_channel.copy_from_slice(src_channel);
            }
        }
    }

    fn multiband_sample(frame_index: usize, sample_idx: usize, band: usize, offset: i32) -> f32 {
        let value = frame_index * AudioBuffer::SPLIT_BAND_SIZE + sample_idx;
        let value = value as i32 + offset;
        if value > 0 {
            5000.0 * band as f32 + value as f32
        } else {
            0.0
        }
    }

    fn fullband_sample(frame_index: usize, sample_idx: usize, offset: i32) -> f32 {
        let value = frame_index * AudioBuffer::SPLIT_BAND_SIZE + sample_idx;
        let value = value as i32 + offset;
        value.max(0) as f32
    }

    fn verify_multiband_frame(
        buffer: &AudioBuffer,
        num_bands: usize,
        frame_index: usize,
        offset: i32,
    ) -> bool {
        for band in 0..num_bands {
            let data = buffer.split_band(0, band);
            for (sample_idx, &sample) in data.iter().enumerate().take(AudioBuffer::SPLIT_BAND_SIZE)
            {
                let reference = multiband_sample(frame_index, sample_idx, band, offset);
                if (reference - sample).abs() > f32::EPSILON {
                    return false;
                }
            }
        }
        true
    }

    fn verify_sequence(reference: &[f32], candidate: &[f32], offset: i32) -> bool {
        for (idx, &sample) in candidate.iter().enumerate() {
            let reference_index = idx as i32 + offset;
            if reference_index >= 0 {
                let ref_idx = reference_index as usize;
                if ref_idx >= reference.len() {
                    return false;
                }
                if (reference[ref_idx] - sample).abs() > f32::EPSILON {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn capture_bitexactness() {
        for &rate in &[16_000, 32_000, 48_000] {
            EchoCanceller3Tester::new(rate).run_capture_transport_verification_test();
        }
    }

    #[test]
    fn render_bitexactness() {
        for &rate in &[16_000, 32_000, 48_000] {
            EchoCanceller3Tester::new(rate).run_render_transport_verification_test();
        }
    }

    #[test]
    fn render_swap_queue() {
        EchoCanceller3Tester::new(16_000).run_render_swap_queue_verification_test();
    }

    #[test]
    fn render_swap_queue_overrun_return_value() {
        for &rate in &[16_000, 32_000, 48_000] {
            EchoCanceller3Tester::new(rate)
                .run_render_pipeline_swap_queue_overrun_return_value_test();
        }
    }

    #[test]
    fn capture_saturation() {
        let variants = [
            SaturationTestVariant::None,
            SaturationTestVariant::OneNegative,
            SaturationTestVariant::OnePositive,
        ];
        for &rate in &[16_000, 32_000, 48_000] {
            for &variant in &variants {
                EchoCanceller3Tester::new(rate).run_capture_saturation_verification_test(variant);
            }
        }
    }

    #[test]
    fn echo_path_change() {
        let variants = [
            EchoPathChangeTestVariant::None,
            EchoPathChangeTestVariant::OneSticky,
            EchoPathChangeTestVariant::OneNonSticky,
        ];
        for &rate in &[16_000, 32_000, 48_000] {
            for &variant in &variants {
                EchoCanceller3Tester::new(rate).run_echo_path_change_verification_test(variant);
            }
        }
    }

    #[test]
    fn echo_leakage() {
        let variants = [
            EchoLeakageTestVariant::None,
            EchoLeakageTestVariant::FalseSticky,
            EchoLeakageTestVariant::TrueSticky,
            EchoLeakageTestVariant::TrueNonSticky,
        ];
        for &rate in &[16_000, 32_000, 48_000] {
            for &variant in &variants {
                EchoCanceller3Tester::new(rate).run_echo_leakage_verification_test(variant);
            }
        }
    }

    /// Drives `num_frames` render frames through the canceller, writing
    /// `values[ch]` into every sample of render channel `ch`.
    fn feed_render_frames(aec3: &mut EchoCanceller3, values: &[f32], num_frames: usize) {
        let num_channels = values.len();
        let fullband_frame_length = 16_000 / 100;
        let mut render = AudioBuffer::new(
            fullband_frame_length,
            num_channels,
            fullband_frame_length,
            num_channels,
            fullband_frame_length,
        );
        let mut capture = AudioBuffer::new(
            fullband_frame_length,
            1,
            fullband_frame_length,
            1,
            fullband_frame_length,
        );

        for _ in 0..num_frames {
            for (ch, &value) in values.iter().enumerate() {
                render.channel_mut(ch).fill(value);
            }
            aec3.analyze_render(&mut render);
            aec3.analyze_capture(&mut capture);
            aec3.process_capture(&mut capture, false);
        }
    }

    /// Upmixed mono render content must keep the canceller in the mono
    /// configuration, and stereo must switch it to the multichannel one.
    #[test]
    fn multichannel_config_is_selected_only_for_stereo() {
        let mut multichannel_config = EchoCanceller3Config::create_default_multichannel_config();
        // Distinguish the two configs by a field the selector does not require
        // to match, so the active one can be identified.
        multichannel_config.delay.default_delay += 1;
        let mono_default_delay = EchoCanceller3Config::default().delay.default_delay;

        // Detection is hysteresis-limited to 2 seconds by default, so drive
        // enough frames to cross it.
        const NUM_FRAMES: usize = 300;

        let mut aec3 = EchoCanceller3::with_multichannel_config(
            EchoCanceller3Config::default(),
            Some(multichannel_config.clone()),
            16_000,
            2,
            1,
        );
        feed_render_frames(&mut aec3, &[0.5, 0.5], NUM_FRAMES);
        assert!(
            !aec3
                .multichannel_content_detector
                .is_proper_multi_channel_content_detected(),
            "upmixed mono was treated as stereo"
        );
        assert_eq!(aec3.num_render_channels_to_aec, 1);
        assert_eq!(
            aec3.config_selector.active_config().delay.default_delay,
            mono_default_delay
        );

        let mut aec3 = EchoCanceller3::with_multichannel_config(
            EchoCanceller3Config::default(),
            Some(multichannel_config.clone()),
            16_000,
            2,
            1,
        );
        feed_render_frames(&mut aec3, &[0.5, -0.5], NUM_FRAMES);
        assert!(
            aec3.multichannel_content_detector
                .is_proper_multi_channel_content_detected(),
            "stereo was not detected"
        );
        assert_eq!(aec3.num_render_channels_to_aec, 2);
        assert_eq!(
            aec3.config_selector.active_config().delay.default_delay,
            multichannel_config.delay.default_delay
        );
    }

    /// Without a multichannel config the render path still collapses to mono
    /// for upmixed content and expands once stereo appears.
    #[test]
    fn render_channel_count_tracks_detection_without_multichannel_config() {
        const NUM_FRAMES: usize = 300;

        let mut aec3 = EchoCanceller3::new(EchoCanceller3Config::default(), 16_000, 2, 1);
        feed_render_frames(&mut aec3, &[0.5, 0.5], NUM_FRAMES);
        assert_eq!(aec3.num_render_channels_to_aec, 1);

        feed_render_frames(&mut aec3, &[0.5, -0.5], NUM_FRAMES);
        assert_eq!(aec3.num_render_channels_to_aec, 2);
    }

    /// Downmixing to mono averages the channels when a proper downmix is
    /// requested, and selects channel 0 otherwise.
    #[test]
    fn sub_frame_downmix_averages_only_when_requested() {
        let frame: Tensor3 = vec![vec![
            vec![1.0f32; 2 * super::SUB_FRAME_LENGTH],
            vec![3.0f32; 2 * super::SUB_FRAME_LENGTH],
        ]];

        let mut storage: Tensor3 = vec![vec![vec![0.0f32; super::SUB_FRAME_LENGTH]]];
        super::fill_tensor_sub_frame(&frame, false, 0, &mut storage);
        assert!(
            storage[0][0].iter().all(|&v| v == 1.0),
            "channel 0 should be selected verbatim without a proper downmix"
        );

        super::fill_tensor_sub_frame(&frame, true, 0, &mut storage);
        assert!(
            storage[0][0].iter().all(|&v| v == 2.0),
            "a proper downmix should average the channels"
        );

        // The second sub-frame must be downmixed from its own sample range.
        super::fill_tensor_sub_frame(&frame, true, 1, &mut storage);
        assert!(storage[0][0].iter().all(|&v| v == 2.0));
    }

    /// With detection disabled the channel count is fixed by the API, matching
    /// the reference's behaviour when `detect_stereo_content` is false.
    #[test]
    fn detection_disabled_keeps_all_render_channels() {
        let mut config = EchoCanceller3Config::default();
        config.multi_channel.detect_stereo_content = false;

        let mut aec3 = EchoCanceller3::new(config, 16_000, 2, 1);
        assert_eq!(aec3.num_render_channels_to_aec, 2);
        feed_render_frames(&mut aec3, &[0.5, 0.5], 50);
        assert_eq!(aec3.num_render_channels_to_aec, 2);
    }
}
