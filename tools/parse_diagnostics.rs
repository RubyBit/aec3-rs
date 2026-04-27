use std::env;
use std::fs::File;
use std::io::{self, Read};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
enum DiagnosticLevel {
    Production = 0,
    Developer = 1,
    DeepDebug = 2,
}

#[derive(Debug, Serialize, Deserialize)]
enum DiagnosticRecord {
    Header {
        version: u32,
    },
    NewSet {
        set_index: u64,
    },
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

fn main() -> io::Result<()> {
    let mut args = env::args().skip(1);
    let log_path = args
        .next()
        .unwrap_or_else(|| "aec3_diagnostics/aec3_diagnostics.log".to_string());

    let mut file = File::open(&log_path)?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;

    let mut offset = 0usize;
    let mut record_index = 0usize;
    while offset + 4 <= data.len() {
        let len = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;
        if offset + len > data.len() {
            eprintln!("truncated record at index {record_index}");
            break;
        }
        let payload = &data[offset..offset + len];
        match postcard::from_bytes::<DiagnosticRecord>(payload) {
            Ok(record) => match record {
                DiagnosticRecord::RawF32Slice {
                    level,
                    set_index,
                    instance,
                    name,
                    values,
                } => {
                    println!(
                        "[{record_index}] RawF32Slice {{ level: {level:?}, set_index: {set_index}, instance: {instance}, name: \"{name}\", values: {} }}",
                        format_slice(&values, |&v| v == 0.0)
                    );
                }
                DiagnosticRecord::RawI32Slice {
                    level,
                    set_index,
                    instance,
                    name,
                    values,
                } => {
                    println!(
                        "[{record_index}] RawI32Slice {{ level: {level:?}, set_index: {set_index}, instance: {instance}, name: \"{name}\", values: {} }}",
                        format_slice(&values, |&v| v == 0)
                    );
                }
                DiagnosticRecord::RawUsizeSlice {
                    level,
                    set_index,
                    instance,
                    name,
                    values,
                } => {
                    println!(
                        "[{record_index}] RawUsizeSlice {{ level: {level:?}, set_index: {set_index}, instance: {instance}, name: \"{name}\", values: {} }}",
                        format_slice(&values, |&v| v == 0)
                    );
                }
                _ => println!("[{record_index}] {record:#?}"),
            },
            Err(err) => println!("[{record_index}] decode error: {err}"),
        }
        offset += len;
        record_index += 1;
    }

    Ok(())
}

fn format_slice<T: std::fmt::Display>(slice: &[T], is_zero: impl Fn(&T) -> bool) -> String {
    if slice.is_empty() {
        return "[]".to_string();
    }

    if slice.iter().all(is_zero) {
        return format!("[all zeros, len={}]", slice.len());
    }

    if slice.len() <= 8 {
        let items: Vec<_> = slice.iter().map(|v| v.to_string()).collect();
        return format!("[{}]", items.join(", "));
    }

    let first_four: Vec<_> = slice[..4].iter().map(|v| v.to_string()).collect();
    let last_four: Vec<_> = slice[slice.len() - 4..]
        .iter()
        .map(|v| v.to_string())
        .collect();

    format!(
        "[{}, ..., {}, len={}]",
        first_four.join(", "),
        last_four.join(", "),
        slice.len()
    )
}
