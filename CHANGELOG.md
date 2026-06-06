# Changelog (WIP)

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

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

[unreleased]: https://github.com/RubyBit/aec3-rs/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/RubyBit/aec3-rs/compare/v0.1.8...v0.2.0
[0.1.8]: https://github.com/RubyBit/aec3-rs/compare/v0.1.7...v0.1.8
