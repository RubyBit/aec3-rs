use std::sync::atomic::{AtomicUsize, Ordering};

/// Minimal stand-in for the WebRTC `ApmDataDumper` helper. In the reference
/// implementation the class dumps intermediate data for offline diagnostics.
/// For now we simply keep track of unique instance identifiers and ignore the
/// data payloads.
#[derive(Clone, Debug)]
pub struct ApmDataDumper {
    instance_index: usize,
}

impl ApmDataDumper {
    pub fn new(instance_index: usize) -> Self {
        Self { instance_index }
    }

    pub fn new_unique() -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let index = COUNTER.fetch_add(1, Ordering::Relaxed);
        Self::new(index)
    }

    pub fn instance_index(&self) -> usize {
        self.instance_index
    }

    pub fn dump_raw_f32(&self, _name: &str, _value: f32) {}

    pub fn dump_raw_f32_slice(&self, _name: &str, _values: &[f32]) {}

    pub fn dump_raw_i32(&self, _name: &str, _value: i32) {}

    pub fn dump_raw_i32_slice(&self, _name: &str, _values: &[i32]) {}

    pub fn dump_raw_usize(&self, _name: &str, _value: usize) {}

    pub fn dump_raw_usize_slice(&self, _name: &str, _values: &[usize]) {}

    pub fn dump_wav(
        &self,
        _name: &str,
        _num_samples: usize,
        _samples: &[f32],
        _sample_rate_hz: usize,
        _num_channels: usize,
    ) {
    }
}
