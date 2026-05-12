# Changelog (WIP)

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Changed 
- Changed in block processor to properly handle skipped capture blocks.

### Added
- Doc comments for the `LinearPipeline` builder API.
- File to File example utilizing the `LinearPipeline` builder API.

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
