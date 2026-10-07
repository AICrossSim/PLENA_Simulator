//! Scalar SRAM storage plus byte-level preload/dump encoding.

use half::f16;
use quantize::{DataType, FpType};

pub(super) struct ScalarSram {
    intsram: Vec<u32>,
    fpsram: Vec<f32>,
    fp_type: DataType,
}

impl ScalarSram {
    pub(super) fn new(fp_type: DataType) -> Self {
        assert!(
            matches!(fp_type, DataType::Fp(_)),
            "SCALAR_FP must be a floating-point type"
        );
        assert!(
            fp_type.size_in_bits() <= 16,
            "scalar SRAM preload/dump words are 16-bit containers"
        );
        Self {
            intsram: vec![0; 1024],
            fpsram: vec![0.0; 1024],
            fp_type,
        }
    }

    pub(super) fn load_fpsram_from_bytes(&mut self, bytes: &[u8]) {
        let values = if self.fp_type == DataType::Fp(FpType::BF16) {
            // Preserve the established emulator artifact ABI: BF16 scalar
            // configurations receive constants encoded as IEEE f16 words.
            decode_fpsram_f16_bytes(bytes)
        } else {
            bytes
                .chunks_exact(2)
                .map(|chunk| {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]) as u32;
                    self.fp_type.convert_bits_to_f32(bits)
                })
                .collect()
        };
        for (dst, value) in self.fpsram.iter_mut().zip(values) {
            *dst = quantize_scalar(self.fp_type, value);
        }
    }

    pub(super) fn load_fpsram_from_bf16_bytes(&mut self, bytes: &[u8]) {
        assert!(bytes.len().is_multiple_of(2), "truncated BF16 preload");
        assert!(
            bytes.len() / 2 <= self.fpsram.len(),
            "BF16 preload exceeds FP SRAM"
        );
        for (slot, bytes) in self.fpsram.iter_mut().zip(bytes.chunks_exact(2)) {
            *slot = bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]]));
        }
    }

    pub(super) fn load_intsram_from_u32_bytes(&mut self, bytes: &[u8]) {
        let int_vals = decode_intsram_u32_bytes(bytes);
        self.intsram[..int_vals.len()].copy_from_slice(&int_vals);
    }

    pub(super) fn read_fp(&self, addr: usize) -> f32 {
        self.fpsram[addr]
    }

    pub(super) fn write_fp(&mut self, addr: usize, value: f32) {
        self.fpsram[addr] = quantize_scalar(self.fp_type, value);
    }

    pub(super) fn read_int(&self, addr: usize) -> u32 {
        self.intsram[addr]
    }

    pub(super) fn write_int(&mut self, addr: usize, value: u32) {
        self.intsram[addr] = value;
    }

    pub(super) fn read_fp_window(&self, start: usize, len: usize) -> &[f32] {
        &self.fpsram[start..start + len]
    }

    /// Bulk counterpart of `read_fp_window`, used by `S_MAP_FP_V`.
    ///
    /// Panics on overrun rather than truncating: an out-of-range FP_MEM base is a
    /// compiler bug, and silently dropping the tail would corrupt a Mamba chunk's
    /// per-row decay scalars with no diagnostic.
    pub(super) fn write_fp_window(&mut self, start: usize, values: &[bf16]) {
        let end = start + values.len();
        assert!(
            end <= self.fpsram.len(),
            "S_MAP_FP_V would write FP_MEM[{start}..{end}) past the {}-entry file",
            self.fpsram.len()
        );
        self.fpsram[start..end].copy_from_slice(values);
    }

    pub(super) fn log_debug_contents(&self) {
        tracing::debug!("INT SRAM Contents: \n {:?}", self.intsram);
        tracing::debug!("FP SRAM Contents: \n {:?}", self.fpsram);
    }

    pub(super) fn fpsram_to_le_bytes(&self) -> Vec<u8> {
        self.fpsram
            .iter()
            .flat_map(|value| {
                let bits = self.fp_type.bits_from_f32(*value) as u16;
                bits.to_le_bytes()
            })
            .collect()
    }

    pub(super) fn intsram_to_le_bytes(&self) -> Vec<u8> {
        self.intsram.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    pub(super) fn intsram_to_le_bytes(&self) -> Vec<u8> {
        self.intsram.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
}

fn quantize_scalar(fp_type: DataType, value: f32) -> f32 {
    fp_type.convert_bits_to_f32(fp_type.bits_from_f32(value))
}

fn decode_fpsram_f16_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(std::mem::size_of::<f16>())
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            f32::from(f16::from_bits(bits))
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
    use half::f16;
    use quantize::{DataType, FpType};

    use super::ScalarSram;

    #[test]
    fn explicit_bf16_preload_preserves_bits_including_signed_zero_and_subnormals() {
        let mut sram = ScalarSram::new();
        let bits = [0x3f80_u16, 0x8000, 0x0001, 0xc020];
        let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        sram.load_fpsram_from_bf16_bytes(&bytes);
        for (i, expected) in bits.into_iter().enumerate() {
            assert_eq!(sram.read_fp(i).to_bits(), expected);
        }
        assert_eq!(&sram.fpsram_to_le_bytes()[..bytes.len()], bytes.as_slice());
    }

    #[test]
    #[should_panic(expected = "truncated BF16 preload")]
    fn explicit_bf16_preload_rejects_partial_values() {
        ScalarSram::new().load_fpsram_from_bf16_bytes(&[0x80]);
    }

    #[test]
    #[should_panic(expected = "BF16 preload exceeds FP SRAM")]
    fn explicit_bf16_preload_rejects_capacity_overflow() {
        ScalarSram::new().load_fpsram_from_bf16_bytes(&[0; 2050]);
    }

    #[test]
    fn scalar_sram_decodes_preloads_and_ignores_trailing_bytes() {
        let mut sram = ScalarSram::new(DataType::Fp(FpType::BF16));

        let mut fp_bytes = Vec::new();
        fp_bytes.extend_from_slice(&f16::from_f32(1.5).to_bits().to_ne_bytes());
        fp_bytes.extend_from_slice(&f16::from_f32(-2.0).to_bits().to_ne_bytes());
        fp_bytes.push(0xff);

        let mut int_bytes = Vec::new();
        int_bytes.extend_from_slice(&0x1122_3344u32.to_ne_bytes());
        int_bytes.extend_from_slice(&7u32.to_ne_bytes());
        int_bytes.extend_from_slice(&[0xaa, 0xbb]);

        sram.load_fpsram_from_bytes(&fp_bytes);
        sram.load_intsram_from_u32_bytes(&int_bytes);

        assert_eq!(sram.read_fp(0), 1.5);
        assert_eq!(sram.read_fp(1), -2.0);
        assert_eq!(sram.read_int(0), 0x1122_3344);
        assert_eq!(sram.read_int(1), 7);
    }

    #[test]
    fn scalar_sram_supports_scalar_reads_writes_windows_and_dump_bytes() {
        let mut sram = ScalarSram::new(DataType::Fp(FpType::BF16));

        sram.write_fp(4, 3.5);
        sram.write_fp(5, -0.5);
        sram.write_int(6, 42);

        assert_eq!(sram.read_fp_window(4, 2), &[3.5, -0.5]);
        assert_eq!(sram.read_int(6), 42);

        let dump = sram.fpsram_to_le_bytes();
        let mut expected = Vec::new();
        expected.extend_from_slice(&0x4060u16.to_le_bytes());
        expected.extend_from_slice(&0xbf00u16.to_le_bytes());
        let start = 4 * std::mem::size_of::<u16>();
        let end = start + 2 * std::mem::size_of::<u16>();
        assert_eq!(&dump[start..end], expected.as_slice());
    }

    #[test]
    fn scalar_sram_honors_fp12_raw_words_and_rounding() {
        let fp12 = DataType::Fp(FpType {
            sign: true,
            exponent: 6,
            mantissa: 5,
        });
        let mut sram = ScalarSram::new(fp12);
        let mut preload = Vec::new();
        preload.extend_from_slice(&0x03e0u16.to_le_bytes());
        preload.extend_from_slice(&0x0be0u16.to_le_bytes());
        sram.load_fpsram_from_bytes(&preload);

        assert_eq!(sram.read_fp(0), 1.0);
        assert_eq!(sram.read_fp(1), -1.0);
        sram.write_fp(2, 31.0);

        let dump = sram.fpsram_to_le_bytes();
        assert_eq!(u16::from_le_bytes([dump[0], dump[1]]), 0x03e0);
        assert_eq!(u16::from_le_bytes([dump[2], dump[3]]), 0x0be0);
        assert_eq!(u16::from_le_bytes([dump[4], dump[5]]), 0x047e);
    }
}
