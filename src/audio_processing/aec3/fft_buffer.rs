use super::fft_data::FftData;

pub struct FftBuffer {
    size: usize,
    pub buffer: Vec<Vec<FftData>>,
    pub write: usize,
    pub read: usize,
}

impl FftBuffer {
    pub fn new(size: usize, num_channels: usize) -> Self {
        assert!(size > 0);
        assert!(num_channels > 0);
        let mut buffer = Vec::with_capacity(size);
        for _ in 0..size {
            let mut block = Vec::with_capacity(num_channels);
            for _ in 0..num_channels {
                block.push(FftData::default());
            }
            buffer.push(block);
        }
        Self {
            size,
            buffer,
            write: 0,
            read: 0,
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn inc_index(&self, index: usize) -> usize {
        if index + 1 < self.size { index + 1 } else { 0 }
    }

    pub fn dec_index(&self, index: usize) -> usize {
        if index > 0 { index - 1 } else { self.size - 1 }
    }

    pub fn offset_index(&self, index: usize, offset: isize) -> usize {
        let size = self.size as isize;
        let mut value = index as isize + offset;
        value %= size;
        if value < 0 {
            value += size;
        }
        value as usize
    }

    pub fn update_write_index(&mut self, offset: isize) {
        self.write = self.offset_index(self.write, offset);
    }

    pub fn inc_write_index(&mut self) {
        self.write = self.inc_index(self.write);
    }

    pub fn dec_write_index(&mut self) {
        self.write = self.dec_index(self.write);
    }

    pub fn update_read_index(&mut self, offset: isize) {
        self.read = self.offset_index(self.read, offset);
    }

    pub fn inc_read_index(&mut self) {
        self.read = self.inc_index(self.read);
    }

    pub fn dec_read_index(&mut self) {
        self.read = self.dec_index(self.read);
    }
}
