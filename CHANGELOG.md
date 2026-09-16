# Changelog (WIP)

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## UNRELEASED

## [0.4.0] - 2026-09-16

### Added
- `audio_processing::post_filter::PostFilter`, a 4-section Chebyshev type 2
  low-pass that removes content above 19.5 kHz. `create_if_needed` returns `None`
  below 48 kHz, where no filtering is required.
- `nodes::post_filter`, the corresponding graph node. Pass-through below 48 kHz.
- `pipelines::linear::LinearPipelineBuilder::enable_post_filter` (`true`),
  `LinearPipeline::reset_post_filter`, and `LinearPipelineHandles::post_filter`.
- `CascadedBiQuadFilter::from_coefficients`, for cascades whose stages have
  different coefficients.
- `audio_processing::aec3::block::Block`, a contiguous multiband, multichannel
  block.
- `EchoCanceller3::set_capture_output_usage`, also on the `EchoControl` trait
  (default no-op) and on `BlockProcessor`/`EchoRemover`. When the capture output
  is unused, the residual echo estimate, suppression gain and suppression filter
  are skipped and the capture block passes through untouched; the linear filter
  keeps adapting.
- `nodes::aec3` gains a `capture_output_used_in` control port, matching the port
  `nodes::agc2` already has.
- `AecState::erle_unbounded` and `ErleEstimator::erle_unbounded`, the ERLE
  without the `erle.max_l`/`erle.max_h` cap.
- SSE2/AVX2/NEON cores for the pre-echo accumulated error, so enabling
  `delay.detect_pre_echo` costs essentially nothing. On an Apple M-series core
  the matched filter went from 22.6 to 13.0 us per 4 ms block with detection on,
  against 12.8 us with it off.
- Pre-echo detection, behind `delay.detect_pre_echo` (`true`). The matched
  filter tracks the error of the filter truncated at every fourth tap and, once
  it has 50 updates, reports the earliest reflection that already explains the
  capture signal. `MatchedFilterLagAggregator` histograms that lag and reports
  it as the delay, so an early weak reflection followed by a stronger one aligns
  to the early one and stays inside the filter window.
- Config fields that were previously hardcoded or absent, all validated and
  clamped: `delay.detect_pre_echo` (`true`),
  `delay.delay_estimate_smoothing_delay_found` (`0.7`),
  `filter.high_pass_filter_echo_reference` (`false`), a `comfort_noise` section
  with `noise_floor_dbfs` (`-96.03406`), `ep_strength.nearend_len` (`0.83`),
  `ep_strength.erle_onset_compensation_in_dominant_nearend` (`false`), and on
  `suppressor`: `lf_smoothing_during_initial_phase` (`true`),
  `last_permanent_lf_smoothing_band` (`0`), `last_lf_smoothing_band` (`5`),
  `last_lf_band` (`5`), `first_hf_band` (`8`), and
  `dominant_nearend_detection.use_unbounded_echo_spectrum` (`true`).

### Fixed
- Behaviour: the dominant nearend decision runs against the uncapped residual
  echo spectrum. It previously used the capped one, so a filter performing better
  than `erle.max_l`/`erle.max_h` was not credited for it and nearend was
  under-detected, over-suppressing doubletalk.
- Behaviour: the residual echo estimate uses the uncompensated ERLE while the
  dominant nearend state is active. It previously always used the
  onset-compensated ERLE, which is lower, so the residual echo estimate was too
  high during doubletalk. `erle_onset_compensation_in_dominant_nearend` restores
  the old behaviour.
- Behaviour: the reverb tail uses the milder `ep_strength.nearend_len` decay
  during the dominant nearend state.
- The render reference was always high-pass filtered on its way to the block
  processor. It is now filtered only when `filter.high_pass_filter_echo_reference`
  is set, which is off by default; low frequencies were being removed from the
  echo reference but not from capture. Two unit tests had been written against
  the old behaviour.
- Comfort noise is estimated from the linear filter output spectrum when the
  linear estimate is usable. It was always estimated from the microphone
  spectrum, which still contains the echo and biases the noise floor high. The
  nearend spectrum selection also now happens before the AEC state update.
- Behaviour: low-frequency gain smoothing in the suppressor also covers bands at
  or below `suppressor.last_permanent_lf_smoothing_band` when echo exceeds
  nearend. With the default of `0`, band 0 is now smoothed in that case.

### Changed
- Behaviour: `pipelines::linear` applies the fullband post filter by default. At
  48 kHz this attenuates content above 19.5 kHz; lower rates are unaffected. Opt
  out with `enable_post_filter(false)`.
- `MatchedFilter::new` takes a second smoothing value and `update` takes a
  `use_slow_smoothing` flag, so the delay estimator can switch to
  `delay.delay_estimate_smoothing_delay_found` once
  `MatchedFilterLagAggregator::reliable_delay_found` (now public) reports a
  reliable delay. Both values default to `0.7`.
- `ComfortNoiseGenerator::new` takes the noise floor in dBFS. The previously
  hardcoded floor equals the default.
- AEC3 passes 64-sample blocks as `Block` instead of `Vec<Vec<Vec<f32>>>`, so a
  block is one allocation rather than one per band plus one per channel per band.
  This changes the signatures of `EchoCanceller3`'s block path: `BlockProcessor`,
  `EchoRemover`, `RenderDelayBuffer`, `RenderDelayController`,
  `EchoPathDelayEstimator`, `Subtractor`, `SuppressionGain`, `SuppressionFilter`,
  `AlignmentMixer`, `FrameBlocker`, `BlockFramer`, `BlockBuffer` and
  `RenderBuffer::block`. Frames and subframes are unchanged.
- `BlockBuffer::new` no longer takes a frame length; a block is always
  `BLOCK_SIZE` samples.
- `SubbandErleEstimator` and `SignalDependentErleEstimator` keep the ERLE without
  onset handling separately from the onset-compensated one, and the subband
  estimator additionally keeps an uncapped stream. The `erle` accessors on those
  types, on `ErleEstimator` and on `AecState` take an `onset_compensated` flag.
- `ResidualEchoEstimator::estimate` takes `dominant_nearend` and fills a second,
  uncapped residual echo spectrum; `SuppressionGain::get_gain` takes it too.
- `ReverbDecayEstimator::decay`, `ReverbModelEstimator::reverb_decay` and
  `AecState::reverb_decay` take a `mild` flag.
- `MatchedFilter` selects the winning filter itself and reports a single
  `LagEstimate { lag, pre_echo_lag }` through `best_lag_estimate`, replacing the
  per-filter `lag_estimates` slice and its `accuracy`/`reliable`/`updated`
  fields. `MatchedFilter::reset` takes a `full_reset` flag.
- `MatchedFilterLagAggregator::new` takes the whole `Delay` config, and
  `aggregate` takes `Option<LagEstimate>`. `get_delay_at_highest_peak` is
  exposed as `delay_at_highest_peak`.
- The delay headroom is subtracted in `MatchedFilterLagAggregator`, in
  downsampled units, instead of in `compute_buffer_delay` at full rate. The two
  agree whenever `delay.delay_headroom_samples` divides by
  `delay.down_sampling_factor`, which holds for the defaults.

### Removed
- `AecState::erle_uncertainty` and the residual echo estimator branch using it.
  It returned a value only when the echo was saturated, which the caller already
  handled first, so the branch was unreachable.
- Four `#[should_panic]` tests that asserted a wrongly sized block is rejected at
  runtime. `Block` makes the size structural, so those inputs cannot be
  constructed.
- The matched filter test asserting one lag estimate per configured filter, and
  the lag aggregator test selecting the most accurate of several estimates. The
  matched filter now picks the winner itself, so neither has anything to check.

## [0.3.2] - 2026-08-12

### Added
- Multichannel content detection: `MultiChannelContentDetector` classifies
  render content as stereo or upmixed mono, and `ConfigSelector` picks
  the config to match. Ported from WebRTC with their unit tests.
- `multi_channel` config section: `detect_stereo_content` (`true`),
  `stereo_detection_threshold` (`0.0`),
  `stereo_detection_timeout_threshold_seconds` (`300`),
  `stereo_detection_hysteresis_seconds` (`2.0`).
- `EchoCanceller3::with_multichannel_config` and
  `EchoCanceller3Config::create_default_multichannel_config`.
- `suppressor.conservative_hf_suppression` (`false`) and
  `suppressor.high_frequency_suppression.{limiting_gain_band,
  bands_in_limiting_gain}` (`16`, `1`), which parameterise the previously
  hardcoded gain limiting window.
- `echo_model.model_reverb_in_nonlinear_mode` (`true`) and
  `ep_strength.use_conservative_tail_frequency_response` (`true`).

### Changed
- Behaviour: the render signal is processed in mono unless multichannel content
  is detected, instead of always using the constructor's channel count. A
  temporary-stereo downmix averages the channels rather than taking channel 0.
- Behaviour: high-frequency gain limiting is applied only outside the
  dominant-nearend state, on clock drift, or under
  `conservative_hf_suppression`, it previously ran on every block and
  over-suppressed during doubletalk.
- Behaviour: the band-29 upper gain bound is now part of
  `conservative_hf_suppression` and off by default; it was always applied.
- Behaviour: `use_conservative_tail_frequency_response` raises the reverb tail
  estimate to the measured filter tail. The step was missing, so the tail could
  be underestimated.
- The `Aec3` node defaults to the reference multichannel config when the caller
  sets none; a caller-provided config is used for both modes.

### Deprecated
- `EchoCanceller3::create_default_config`, which picks tuning from the render
  channel count alone and so applies the multichannel tuning to upmixed mono.
  Use `EchoCanceller3::with_multichannel_config`.

## [0.3.1] - 2026-06-24

### Fixed
- Neon optimizations are now compiled only for `aarch64` targets a nd not generic `arm` targets, which do not support the required instructions. This fixes build failures on `armv7`.

## [0.3.0] - 2026-06-08

### Changed 
- Changed in block processor to properly handle skipped capture blocks.
- Graph side-input alignment is now sequence-based and strict (breaking):
  - Added `MatchPolicy::Fifo` (explicit queue-order matching) and
    `MatchPolicy::BySequence` (match on equal `PacketMeta::sequence`).
  - `PacketMeta::sequence` is now `Option<u64>` so "not stamped" is
    representable; default packets no longer share sequence `0`.
  - A required `BySequence` dependency with unstamped packets on either side
    now fails with `GraphError::MissingSequenceForAlignment` instead of
    silently falling back to FIFO matching (the error names the port whose
    packets are missing the stamp).
  - `BySequence` dependency packets older than the pending trigger are pruned
    during matching (sequences are monotonic, so they can never match a
    current or future trigger); persistent mismatches no longer accumulate
    until the dependency queue rejects pushes.
  - Reads on an `AlignOn` dependency port with no match in the current
    execution now return `None` instead of falling back to the queue head,
    so a node can never consume a wrong-frame side packet.
  - The noise suppression analysis side input aligns with `BySequence`.
- `GraphBuilder::build` now rejects zero-capacity queues with
  `GraphError::InvalidQueueCapacity`; capacity zero previously behaved
  inconsistently across overflow policies.
- `GraphBuilder::source` no longer takes a `QueueConfig` (breaking); the
  argument was ignored because queues live on inputs and sinks.
- `GraphError::QueueFull` now reports the actual port name.
- Split `aec3::graph` into `port`, `packet`, `node`, `builder`, and `runtime`
  submodules. All public items are re-exported from `aec3::graph`, so paths
  are unchanged.

### Removed
- Timestamp-based matching (breaking): `MatchPolicy::AnyAvailable`,
  `ExactTimestamp`, `WithinSkew`, and `LatestBefore` are gone along with the
  never-constructed `MissingTimestampForAlignment` and `CrossClockAlignment`
  errors. `Timestamp` and `ClockId` remain on `PacketMeta` as opaque
  pass-through metadata; alignment is sequence-based.
- `ReusablePortData` (breaking): defined but never used by the runtime.
- `NodeFactory::describe`, `NodeIoBuilder`, and `GraphBuilder::add_factory`
  (breaking): a parallel node-registration path that nothing used; register
  ports via `NodeSpec` and `GraphBuilder::register_input`/`register_output`.
- `RuntimeOptions` (breaking): `Runtime::new(spec)` no longer takes the empty
  options struct.

### Added
- Doc comments for `GraphBuilder` and `Runtime` APIs.
- Doc comments for the `LinearPipeline` builder API.
- File to File example utilizing the `LinearPipeline` builder API.
- Module documentation for `aec3::graph` covering the single-threaded runtime
  model and the interim caveat about non-reject overflow on trigger inputs.

## [0.2.0] - 2026-04-28

### Fixed

- Matched the full-band ERLE estimator to the WebRTC AEC3 reference by removing
  the incorrect low-band max cap and using the reference smoothing behavior.
  This fixes `echo_return_loss_enhancement` being artificially stuck near 6 dB
  and lets downstream AEC state use the uncapped full-band ERLE estimate.

### Added

- Introduced a new graph-based pipeline construction system for audio processing.
  - Replaces the previous rigid `VoipAec3` API with a general-purpose DAG execution model.
  - Supports arbitrary nodes, ports, and edges with typed audio and control streams.
  - Enables fan-in/fan-out pipelines (e.g. AEC + NS + AGC2 with shared intermediate signals).
  - Zero copy where possible with shared packet handles and copy-on-write buffers.

- Added experimental pipeline builder API for ergonomic graph construction (similar to previous `VoipAec3` API).
  - Provides a linear “audio pipeline” abstraction over the underlying DAG.
  - Supports common DSP chains (capture → HPF -> AEC → NS → AGC2) without explicit graph wiring.


### Removed
- The old `VoipAec3` API and related types have been removed in favor of the new graph-based system. See examples for how to build equivalent pipelines with the new API.

## [0.1.8] - 2026-04-23

### Changed 

- Added SIMD optimizations to aec3 core modules such as adaptive_fir_filter/erl and generic SIMD optimized vector math modules (utilized in the RNN vad in agc2 and in aec3 cng, supression filter and supression gain modules).

###### TODO: Add past versions as well and also go back and tag releases to reference here

[unreleased]: https://github.com/RubyBit/aec3-rs/compare/v0.4.0..HEAD
[0.4.0]: https://github.com/RubyBit/aec3-rs/compare/v0.3.2..v0.4.0
[0.3.2]: https://github.com/RubyBit/aec3-rs/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/RubyBit/aec3-rs/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/RubyBit/aec3-rs/compare/v0.2.1...v0.3.0
[0.2.0]: https://github.com/RubyBit/aec3-rs/compare/v0.1.8...v0.2.0
[0.1.8]: https://github.com/RubyBit/aec3-rs/compare/v0.1.7...v0.1.8
