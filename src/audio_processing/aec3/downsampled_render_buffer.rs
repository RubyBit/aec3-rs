pub struct DownsampledRenderBuffer {
    size: usize,
    pub buffer: Vec<f32>,
    pub write: usize,
    pub read: usize,
}

impl DownsampledRenderBuffer {
    pub fn new(size: usize) -> Self {
        assert!(size > 0);
        Self {
            size,
            buffer: vec![0.0; size],
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
        assert!(self.size > 0);
        let offset_abs = offset.abs() as usize;
        assert!(offset_abs <= self.size);
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

#[cfg(test)]
mod tests {
    use super::DownsampledRenderBuffer;

    #[test]
    fn wraps_indices_correctly() {
        let mut buffer = DownsampledRenderBuffer::new(5);
        buffer.write = 4;
        buffer.inc_write_index();
        assert_eq!(buffer.write, 0);
        buffer.dec_write_index();
        assert_eq!(buffer.write, 4);
        buffer.read = 0;
        buffer.dec_read_index();
        assert_eq!(buffer.read, 4);
    }

    #[test]
    fn offset_index_handles_negative_offsets() {
        let buffer = DownsampledRenderBuffer::new(8);
        assert_eq!(buffer.offset_index(2, -3), 7);
        assert_eq!(buffer.offset_index(0, -1), 7);
        assert_eq!(buffer.offset_index(7, 2), 1);
    }
}
