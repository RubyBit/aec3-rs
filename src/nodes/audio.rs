use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AudioFormat {
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub frames_per_channel: u16,
}

impl AudioFormat {
    pub const fn new(sample_rate_hz: u32, channels: u16, frames_per_channel: u16) -> Self {
        Self {
            sample_rate_hz,
            channels,
            frames_per_channel,
        }
    }

    pub const fn ten_ms(sample_rate_hz: u32, channels: u16) -> Self {
        Self {
            sample_rate_hz,
            channels,
            frames_per_channel: (sample_rate_hz / 100) as u16,
        }
    }

    pub fn sample_count(self) -> usize {
        self.frames_per_channel as usize * self.channels as usize
    }

    pub fn schema_key(self) -> String {
        format!(
            "audio:{}:{}:{}",
            self.sample_rate_hz, self.channels, self.frames_per_channel
        )
    }
}

type PoolMap = HashMap<AudioFormat, Vec<Vec<f32>>>;

thread_local! {
    static AUDIO_POOL: RefCell<PoolMap> = RefCell::new(HashMap::new());
}

fn take_buffer(format: AudioFormat) -> Vec<f32> {
    AUDIO_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        pool.entry(format)
            .or_default()
            .pop()
            .unwrap_or_else(|| vec![0.0; format.sample_count()])
    })
}

fn recycle_buffer(format: AudioFormat, mut data: Vec<f32>) {
    data.clear();
    data.resize(format.sample_count(), 0.0);
    AUDIO_POOL.with(|pool| {
        pool.borrow_mut().entry(format).or_default().push(data);
    });
}

#[derive(Debug)]
struct PooledBuffer {
    format: AudioFormat,
    data: Vec<f32>,
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        recycle_buffer(self.format, std::mem::take(&mut self.data));
    }
}

#[derive(Debug, Clone)]
pub struct AudioBufferLease {
    format: AudioFormat,
    inner: Arc<PooledBuffer>,
}

impl AudioBufferLease {
    pub fn silence(format: AudioFormat) -> Self {
        let mut data = take_buffer(format);
        data.fill(0.0);
        Self {
            format,
            inner: Arc::new(PooledBuffer { format, data }),
        }
    }

    pub fn from_interleaved(format: AudioFormat, interleaved: &[f32]) -> Self {
        let mut lease = Self::silence(format);
        lease.make_mut().copy_from_slice(interleaved);
        lease
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.inner.data
    }

    pub fn make_mut(&mut self) -> &mut [f32] {
        if Arc::get_mut(&mut self.inner).is_none() {
            let mut data = take_buffer(self.format);
            data.copy_from_slice(&self.inner.data);
            self.inner = Arc::new(PooledBuffer {
                format: self.format,
                data,
            });
        }

        Arc::get_mut(&mut self.inner)
            .expect("buffer must be unique after copy-on-write")
            .data
            .as_mut_slice()
    }

    pub fn format(&self) -> AudioFormat {
        self.format
    }
}

#[derive(Debug, Clone)]
pub struct AudioChunk {
    pub format: AudioFormat,
    pub data: AudioBufferLease,
}

impl AudioChunk {
    pub fn silence(format: AudioFormat) -> Self {
        Self {
            format,
            data: AudioBufferLease::silence(format),
        }
    }

    pub fn from_interleaved(format: AudioFormat, interleaved: &[f32]) -> Self {
        Self {
            format,
            data: AudioBufferLease::from_interleaved(format, interleaved),
        }
    }

    pub fn samples(&self) -> &[f32] {
        self.data.as_slice()
    }

    pub fn samples_mut(&mut self) -> &mut [f32] {
        self.data.make_mut()
    }
}
