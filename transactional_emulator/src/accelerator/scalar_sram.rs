//! Scalar SRAM storage plus byte-level preload/dump encoding.

use half::{bf16, f16};

pub(super) struct ScalarSram {
    intsram: Vec<u32>,
    fpsram: Vec<bf16>,
}

impl ScalarSram {
    pub(super) fn new() -> Self {
        Self {
            intsram: vec![0; 1024],
            fpsram: vec![bf16::ZERO; 1024],
        }
    }

    pub(super) fn load_fpsram_from_f16_bytes(&mut self, bytes: &[u8]) {
        let fp_vals = decode_fpsram_f16_bytes(bytes);
        self.fpsram[..fp_vals.len()].copy_from_slice(&fp_vals);
    }

    pub(super) fn load_intsram_from_u32_bytes(&mut self, bytes: &[u8]) {
        let int_vals = decode_intsram_u32_bytes(bytes);
        self.intsram[..int_vals.len()].copy_from_slice(&int_vals);
    }

    pub(super) fn read_fp(&self, addr: usize) -> bf16 {
        self.fpsram[addr]
    }

    pub(super) fn write_fp(&mut self, addr: usize, value: bf16) {
        self.fpsram[addr] = value;
    }

    pub(super) fn read_int(&self, addr: usize) -> u32 {
        self.intsram[addr]
    }

    pub(super) fn write_int(&mut self, addr: usize, value: u32) {
        self.intsram[addr] = value;
    }

    pub(super) fn read_fp_window(&self, start: usize, len: usize) -> &[bf16] {
        &self.fpsram[start..start + len]
    }

    pub(super) fn log_debug_contents(&self) {
        tracing::debug!("INT SRAM Contents: \n {:?}", self.intsram);
        tracing::debug!("FP SRAM Contents: \n {:?}", self.fpsram);
    }

    pub(super) fn fpsram_to_le_bytes(&self) -> Vec<u8> {
        self.fpsram.iter().flat_map(|f| f.to_le_bytes()).collect()
    }
}

fn decode_fpsram_f16_bytes(bytes: &[u8]) -> Vec<bf16> {
    bytes
        .chunks_exact(std::mem::size_of::<f16>())
        .map(|chunk| {
            let bits = u16::from_ne_bytes([chunk[0], chunk[1]]);
            bf16::from_f32(f32::from(f16::from_bits(bits)))
        })
        .collect()
}

fn decode_intsram_u32_bytes(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(std::mem::size_of::<u32>())
        .map(|chunk| u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use half::{bf16, f16};

    use super::ScalarSram;

    #[test]
    fn scalar_sram_decodes_preloads_and_ignores_trailing_bytes() {
        let mut sram = ScalarSram::new();

        let mut fp_bytes = Vec::new();
        fp_bytes.extend_from_slice(&f16::from_f32(1.5).to_bits().to_ne_bytes());
        fp_bytes.extend_from_slice(&f16::from_f32(-2.0).to_bits().to_ne_bytes());
        fp_bytes.push(0xff);

        let mut int_bytes = Vec::new();
        int_bytes.extend_from_slice(&0x1122_3344u32.to_ne_bytes());
        int_bytes.extend_from_slice(&7u32.to_ne_bytes());
        int_bytes.extend_from_slice(&[0xaa, 0xbb]);

        sram.load_fpsram_from_f16_bytes(&fp_bytes);
        sram.load_intsram_from_u32_bytes(&int_bytes);

        assert_eq!(sram.read_fp(0), bf16::from_f32(1.5));
        assert_eq!(sram.read_fp(1), bf16::from_f32(-2.0));
        assert_eq!(sram.read_int(0), 0x1122_3344);
        assert_eq!(sram.read_int(1), 7);
    }

    #[test]
    fn scalar_sram_supports_scalar_reads_writes_windows_and_dump_bytes() {
        let mut sram = ScalarSram::new();

        sram.write_fp(4, bf16::from_f32(3.5));
        sram.write_fp(5, bf16::from_f32(-0.5));
        sram.write_int(6, 42);

        assert_eq!(
            sram.read_fp_window(4, 2),
            &[bf16::from_f32(3.5), bf16::from_f32(-0.5)]
        );
        assert_eq!(sram.read_int(6), 42);

        let dump = sram.fpsram_to_le_bytes();
        let mut expected = Vec::new();
        expected.extend_from_slice(&bf16::from_f32(3.5).to_le_bytes());
        expected.extend_from_slice(&bf16::from_f32(-0.5).to_le_bytes());
        let start = 4 * std::mem::size_of::<bf16>();
        let end = start + 2 * std::mem::size_of::<bf16>();
        assert_eq!(&dump[start..end], expected.as_slice());
    }
}
