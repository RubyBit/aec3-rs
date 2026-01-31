use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(feature = "diagnostics")]
use serde::{Deserialize, Serialize};

/// Controls the verbosity of diagnostic dumping.
#[cfg_attr(feature = "diagnostics", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum DiagnosticLevel {
    Production = 0,
    Developer = 1,
    DeepDebug = 2,
}

impl Default for DiagnosticLevel {
    fn default() -> Self {
        DiagnosticLevel::Production
    }
}

#[cfg(feature = "diagnostics")]
mod diagnostics {
    use std::collections::HashMap;
    use std::fs::{create_dir_all, File, OpenOptions};
    use std::io::{BufWriter, Write};
    use std::path::{Path, PathBuf};
    use std::sync::{Mutex, OnceLock};

    use super::DiagnosticLevel;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
    pub enum DiagnosticRecord {
        Header { version: u32 },
        NewSet { set_index: u64 },
        RawF32 {
            level: DiagnosticLevel,
            set_index: u64,
            instance: usize,
            name: String,
            value: f32,
        },
        RawF32Slice {
            level: DiagnosticLevel,
            set_index: u64,
            instance: usize,
            name: String,
            values: Vec<f32>,
        },
        RawI32 {
            level: DiagnosticLevel,
            set_index: u64,
            instance: usize,
            name: String,
            value: i32,
        },
        RawI32Slice {
            level: DiagnosticLevel,
            set_index: u64,
            instance: usize,
            name: String,
            values: Vec<i32>,
        },
        RawUsize {
            level: DiagnosticLevel,
            set_index: u64,
            instance: usize,
            name: String,
            value: usize,
        },
        RawUsizeSlice {
            level: DiagnosticLevel,
            set_index: u64,
            instance: usize,
            name: String,
            values: Vec<usize>,
        },
        WavRef {
            level: DiagnosticLevel,
            set_index: u64,
            instance: usize,
            name: String,
            path: String,
            sample_rate_hz: usize,
            num_channels: usize,
        },
    }

    struct WavSink {
        //path: PathBuf,
        writer: hound::WavWriter<BufWriter<File>>,
        spec: hound::WavSpec,
    }

    struct LoggerState {
        activated: bool,
        level: DiagnosticLevel,
        output_dir: PathBuf,
        log_path: PathBuf,
        writer: Option<BufWriter<File>>,
        set_index: u64,
        wav_writers: HashMap<String, WavSink>,
    }

    impl LoggerState {
        fn new() -> Self {
            let output_dir = PathBuf::from("aec3_diagnostics");
            let log_path = output_dir.join("aec3_diagnostics.log");
            Self {
                activated: false,
                level: DiagnosticLevel::default(),
                output_dir,
                log_path,
                writer: None,
                set_index: 0,
                wav_writers: HashMap::new(),
            }
        }

        fn ensure_writer(&mut self) -> std::io::Result<()> {
            if self.writer.is_some() {
                return Ok(());
            }
            create_dir_all(&self.output_dir)?;
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.log_path)?;
            let mut writer = BufWriter::new(file);
            if self.log_path.metadata()?.len() == 0 {
                let header = DiagnosticRecord::Header { version: 2 };
                write_record(&mut writer, &header)?;
            }
            self.writer = Some(writer);
            Ok(())
        }

        fn finalize_wavs(&mut self) {
            for (_, sink) in self.wav_writers.drain() {
                let _ = sink.writer.finalize();
            }
        }
    }

    impl Drop for LoggerState {
        fn drop(&mut self) {
            self.finalize_wavs();
        }
    }

    fn global_state() -> &'static Mutex<LoggerState> {
        static STATE: OnceLock<Mutex<LoggerState>> = OnceLock::new();
        STATE.get_or_init(|| Mutex::new(LoggerState::new()))
    }

    fn write_record(writer: &mut BufWriter<File>, record: &DiagnosticRecord) -> std::io::Result<()> {
        let bytes = postcard::to_allocvec(record).map_err(|err| {
            std::io::Error::new(std::io::ErrorKind::Other, format!("postcard encode: {err}"))
        })?;
        let len = bytes.len() as u32;
        writer.write_all(&len.to_le_bytes())?;
        writer.write_all(&bytes)?;
        writer.flush()?;
        Ok(())
    }

    fn log_record(record: DiagnosticRecord) {
        let mut state = match global_state().lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        if !state.activated {
            return;
        }
        write_record_locked(&mut state, &record);
    }

    fn write_record_locked(state: &mut LoggerState, record: &DiagnosticRecord) {
        if state.ensure_writer().is_err() {
            return;
        }
        if let Some(writer) = state.writer.as_mut() {
            let _ = write_record(writer, record);
        }
    }

    fn should_log(state: &LoggerState, level: DiagnosticLevel) -> bool {
        state.activated && level <= state.level
    }

    fn sanitize_name(name: &str) -> String {
        name.chars()
            .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
            .collect()
    }

    pub(super) fn set_activated(activated: bool) {
        let mut state = match global_state().lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        if !activated {
            state.finalize_wavs();
            state.writer = None;
        }
        state.activated = activated;
        if activated {
            let _ = state.ensure_writer();
        }
    }

    pub(super) fn set_level(level: DiagnosticLevel) {
        let mut state = match global_state().lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        state.level = level;
    }

    pub(super) fn set_output_directory(path: &Path) {
        let mut state = match global_state().lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        state.finalize_wavs();
        state.output_dir = path.to_path_buf();
        state.log_path = state.output_dir.join("aec3_diagnostics.log");
        state.writer = None;
    }

    pub(super) fn initiate_new_set_of_recordings() {
        let mut state = match global_state().lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        state.finalize_wavs();
        state.set_index = state.set_index.saturating_add(1);
        let set_index = state.set_index;
        drop(state);
        log_record(DiagnosticRecord::NewSet { set_index });
    }

    pub(super) fn dump_raw_f32(level: DiagnosticLevel, instance: usize, name: &str, value: f32) {
        let mut state = match global_state().lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        if !should_log(&state, level) {
            return;
        }
        let set_index = state.set_index;
        write_record_locked(&mut state, &DiagnosticRecord::RawF32 {
            level,
            set_index,
            instance,
            name: name.to_string(),
            value,
        });
    }

    pub(super) fn dump_raw_f32_slice(
        level: DiagnosticLevel,
        instance: usize,
        name: &str,
        values: &[f32],
    ) {
        let mut state = match global_state().lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        if !should_log(&state, level) {
            return;
        }
        let set_index = state.set_index;
        write_record_locked(&mut state, &DiagnosticRecord::RawF32Slice {
            level,
            set_index,
            instance,
            name: name.to_string(),
            values: values.to_vec(),
        });
    }

    pub(super) fn dump_raw_i32(level: DiagnosticLevel, instance: usize, name: &str, value: i32) {
        let mut state = match global_state().lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        if !should_log(&state, level) {
            return;
        }
        let set_index = state.set_index;
        write_record_locked(&mut state, &DiagnosticRecord::RawI32 {
            level,
            set_index,
            instance,
            name: name.to_string(),
            value,
        });
    }

    pub(super) fn dump_raw_i32_slice(
        level: DiagnosticLevel,
        instance: usize,
        name: &str,
        values: &[i32],
    ) {
        let mut state = match global_state().lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        if !should_log(&state, level) {
            return;
        }
        let set_index = state.set_index;
        write_record_locked(&mut state, &DiagnosticRecord::RawI32Slice {
            level,
            set_index,
            instance,
            name: name.to_string(),
            values: values.to_vec(),
        });
    }

    pub(super) fn dump_raw_usize(
        level: DiagnosticLevel,
        instance: usize,
        name: &str,
        value: usize,
    ) {
        let mut state = match global_state().lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        if !should_log(&state, level) {
            return;
        }
        let set_index = state.set_index;
        write_record_locked(&mut state, &DiagnosticRecord::RawUsize {
            level,
            set_index,
            instance,
            name: name.to_string(),
            value,
        });
    }

    pub(super) fn dump_raw_usize_slice(
        level: DiagnosticLevel,
        instance: usize,
        name: &str,
        values: &[usize],
    ) {
        let mut state = match global_state().lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        if !should_log(&state, level) {
            return;
        }
        let set_index = state.set_index;
        write_record_locked(&mut state, &DiagnosticRecord::RawUsizeSlice {
            level,
            set_index,
            instance,
            name: name.to_string(),
            values: values.to_vec(),
        });
    }

    pub(super) fn dump_wav(
        level: DiagnosticLevel,
        instance: usize,
        name: &str,
        num_samples: usize,
        samples: &[f32],
        sample_rate_hz: usize,
        num_channels: usize,
    ) {
        let _ = num_samples;
        let mut state = match global_state().lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        if !should_log(&state, level) {
            return;
        }
        if state.ensure_writer().is_err() {
            return;
        }

        let set_index = state.set_index;
        let safe_name = sanitize_name(name);
        let key = format!(
            "{set}:{inst}:{name}:{rate}:{ch}",
            set = set_index,
            inst = instance,
            name = safe_name,
            rate = sample_rate_hz,
            ch = num_channels
        );

        if !state.wav_writers.contains_key(&key) {
            let file_name = format!(
                "aec3_{safe_name}_set{set}_inst{inst}.wav",
                safe_name = safe_name,
                set = set_index,
                inst = instance
            );
            let path = state.output_dir.join(file_name);
            let Some(parent) = path.parent() else {
                return;
            };
            if create_dir_all(parent).is_err() {
                return;
            }
            let spec = hound::WavSpec {
                channels: num_channels as u16,
                sample_rate: sample_rate_hz as u32,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            };
            let writer = match hound::WavWriter::create(&path, spec) {
                Ok(writer) => writer,
                Err(_) => return,
            };
            state.wav_writers.insert(
                key.clone(),
                WavSink {
                    writer,
                    spec,
                },
            );
            write_record_locked(&mut state, &DiagnosticRecord::WavRef {
                level,
                set_index,
                instance,
                name: name.to_string(),
                path: path.to_string_lossy().to_string(),
                sample_rate_hz,
                num_channels,
            });
        }

        let mut write_failed = false;
        if let Some(sink) = state.wav_writers.get_mut(&key) {
            if sink.spec.sample_rate != sample_rate_hz as u32
                || sink.spec.channels != num_channels as u16
            {
                write_failed = true;
            } else {
                for sample in samples {
                    if sink.writer.write_sample(*sample).is_err() {
                        write_failed = true;
                        break;
                    }
                }
            }
        }
        if write_failed {
            if let Some(sink) = state.wav_writers.remove(&key) {
                let _ = sink.writer.finalize();
            }
        }
    }

    // fn current_set_index() -> u64 {
    //     let state = match global_state().lock() {
    //         Ok(state) => state,
    //         Err(poisoned) => poisoned.into_inner(),
    //     };
    //     state.set_index
    // }

}

/// Optional diagnostic logger for dumping AEC3 internal data.
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

    pub fn set_activated(activated: bool) {
        #[cfg(feature = "diagnostics")]
        diagnostics::set_activated(activated);
        #[cfg(not(feature = "diagnostics"))]
        let _ = activated;
    }

    pub fn set_diagnostics_level(level: DiagnosticLevel) {
        #[cfg(feature = "diagnostics")]
        diagnostics::set_level(level);
        #[cfg(not(feature = "diagnostics"))]
        let _ = level;
    }

    pub fn set_output_directory<P: AsRef<std::path::Path>>(path: P) {
        #[cfg(feature = "diagnostics")]
        diagnostics::set_output_directory(path.as_ref());
        #[cfg(not(feature = "diagnostics"))]
        let _ = path;
    }

    pub fn initiate_new_set_of_recordings(&self) {
        #[cfg(feature = "diagnostics")]
        diagnostics::initiate_new_set_of_recordings();
    }

    pub fn dump_raw_f32(&self, level: DiagnosticLevel, name: &str, value: f32) {
        #[cfg(feature = "diagnostics")]
        diagnostics::dump_raw_f32(level, self.instance_index, name, value);
        #[cfg(not(feature = "diagnostics"))]
        let _ = (level, name, value);
    }

    pub fn dump_raw_f32_slice(&self, level: DiagnosticLevel, name: &str, values: &[f32]) {
        #[cfg(feature = "diagnostics")]
        diagnostics::dump_raw_f32_slice(level, self.instance_index, name, values);
        #[cfg(not(feature = "diagnostics"))]
        let _ = (level, name, values);
    }

    pub fn dump_raw_i32(&self, level: DiagnosticLevel, name: &str, value: i32) {
        #[cfg(feature = "diagnostics")]
        diagnostics::dump_raw_i32(level, self.instance_index, name, value);
        #[cfg(not(feature = "diagnostics"))]
        let _ = (level, name, value);
    }

    pub fn dump_raw_i32_slice(&self, level: DiagnosticLevel, name: &str, values: &[i32]) {
        #[cfg(feature = "diagnostics")]
        diagnostics::dump_raw_i32_slice(level, self.instance_index, name, values);
        #[cfg(not(feature = "diagnostics"))]
        let _ = (level, name, values);
    }

    pub fn dump_raw_usize(&self, level: DiagnosticLevel, name: &str, value: usize) {
        #[cfg(feature = "diagnostics")]
        diagnostics::dump_raw_usize(level, self.instance_index, name, value);
        #[cfg(not(feature = "diagnostics"))]
        let _ = (level, name, value);
    }

    pub fn dump_raw_usize_slice(&self, level: DiagnosticLevel, name: &str, values: &[usize]) {
        #[cfg(feature = "diagnostics")]
        diagnostics::dump_raw_usize_slice(level, self.instance_index, name, values);
        #[cfg(not(feature = "diagnostics"))]
        let _ = (level, name, values);
    }

    pub fn dump_wav(
        &self,
        level: DiagnosticLevel,
        name: &str,
        num_samples: usize,
        samples: &[f32],
        sample_rate_hz: usize,
        num_channels: usize,
    ) {
        #[cfg(feature = "diagnostics")]
        diagnostics::dump_wav(
            level,
            self.instance_index,
            name,
            num_samples,
            samples,
            sample_rate_hz,
            num_channels,
        );
        #[cfg(not(feature = "diagnostics"))]
        let _ = (level, name, num_samples, samples, sample_rate_hz, num_channels);
    }
}
