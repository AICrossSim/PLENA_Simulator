//! ISA-visible register banks and scalar config registers.

use half::bf16;

pub(super) struct AcceleratorRegFile {
    // === ISA-indexed register banks ===
    gp_reg: [u32; 16],
    fp_reg: [bf16; 8],
    hbm_addr_reg: [u64; 16],

    // === Global config registers ===
    scale: u32,
    stride: u32,
    bmm_scale: f32,
    v_mask: u32,
    /// Routed-MoE `(num_experts, top_k)` policy written by `C_SET_TOPK_REG` and
    /// consumed by `V_TOPK rmask=15`.
    ///
    /// `None` until written so `V_TOPK` can trap on an unset policy. A plain `u32`
    /// defaulting to 0 would unpack to `top_k=0`, and `topk_softmax` asserts
    /// `topk > 0` — the program would abort with "topk must be positive", which
    /// says nothing about the missing `C_SET_TOPK_REG`.
    topk_policy: Option<u32>,
}

impl AcceleratorRegFile {
    pub(super) fn new() -> Self {
        Self {
            gp_reg: [0; 16],
            fp_reg: [bf16::ZERO; 8],
            hbm_addr_reg: [0; 16],
            scale: 0,
            stride: 1,
            // Fixed emulator default for head_dim=16: 1/sqrt(16).
            // The current opcode dispatch does not expose a writer for this.
            bmm_scale: 0.25,
            v_mask: 0,
            topk_policy: None,
        }
    }

    /// Read a general-purpose register by its 4-bit ISA encoding.
    pub(super) fn read_gp(&self, r: u8) -> u32 {
        self.gp_reg[r as usize]
    }

    /// Read a floating-point register by its 3-bit ISA encoding.
    pub(super) fn read_fp(&self, r: u8) -> bf16 {
        self.fp_reg[r as usize]
    }

    /// Read an HBM address register by its 4-bit ISA encoding.
    pub(super) fn read_hbm(&self, r: u8) -> u64 {
        self.hbm_addr_reg[r as usize]
    }

    /// Write a general-purpose register by its 4-bit ISA encoding.
    pub(super) fn write_gp(&mut self, r: u8, v: u32) {
        self.gp_reg[r as usize] = v;
    }

    /// Write a floating-point register by its 3-bit ISA encoding.
    pub(super) fn write_fp(&mut self, r: u8, v: bf16) {
        self.fp_reg[r as usize] = v;
    }

    /// Write an HBM address register by its 4-bit ISA encoding.
    pub(super) fn write_hbm(&mut self, r: u8, v: u64) {
        self.hbm_addr_reg[r as usize] = v;
    }

    pub(super) fn scale(&self) -> u32 {
        self.scale
    }

    pub(super) fn set_scale(&mut self, v: u32) {
        self.scale = v;
    }

    pub(super) fn stride(&self) -> u32 {
        self.stride
    }

    pub(super) fn set_stride(&mut self, v: u32) {
        self.stride = v;
    }

    pub(super) fn bmm_scale(&self) -> f32 {
        self.bmm_scale
    }

    pub(super) fn v_mask(&self) -> u32 {
        self.v_mask
    }

    pub(super) fn set_v_mask(&mut self, v: u32) {
        self.v_mask = v;
    }

    pub(super) fn set_topk_policy(&mut self, v: u32) {
        self.topk_policy = Some(v);
    }

    /// Unpack the `C_SET_TOPK_REG` value into `(num_experts, top_k)`.
    ///
    /// The packing is `(num_experts << 8) | top_k`, chosen so every shape up to
    /// 16383 experts fits a single 22-bit `S_ADDI_INT` immediate on the compiler
    /// side. `None` means no `C_SET_TOPK_REG` has executed.
    ///
    /// Field validity is deliberately not checked here: `topk_softmax` already
    /// asserts `0 < top_k <= num_experts`, and duplicating that as a second,
    /// differently-worded assertion would let the two drift apart. This only owns
    /// the bit layout, which is the half the compiler has to agree with.
    pub(super) fn topk_policy(&self) -> Option<(usize, usize)> {
        self.topk_policy
            .map(|packed| ((packed >> 8) as usize, (packed & 0xFF) as usize))
    }

    /// `dst_gp = op(read_gp(src1), read_gp(src2))`. Helper for binary GP-to-GP
    /// instructions (S_ADD_INT / S_SUB_INT / S_MUL_INT).
    pub(super) fn binop_gp<F: FnOnce(u32, u32) -> u32>(
        &mut self,
        dst: u8,
        src1: u8,
        src2: u8,
        op: F,
    ) {
        let v = op(self.read_gp(src1), self.read_gp(src2));
        self.write_gp(dst, v);
    }

    /// `dst_fp = op(read_fp(src1), read_fp(src2))`. Helper for binary FP-to-FP
    /// instructions (S_ADD_FP / S_SUB_FP / S_MAX_FP / S_MUL_FP).
    pub(super) fn binop_fp<F: FnOnce(bf16, bf16) -> bf16>(
        &mut self,
        dst: u8,
        src1: u8,
        src2: u8,
        op: F,
    ) {
        let v = op(self.read_fp(src1), self.read_fp(src2));
        self.write_fp(dst, v);
    }
}

#[cfg(test)]
mod tests {
    use half::bf16;

    use super::AcceleratorRegFile;

    #[test]
    fn new_register_file_uses_isa_defaults() {
        let regs = AcceleratorRegFile::new();

        assert_eq!(regs.read_gp(3), 0);
        assert_eq!(regs.read_fp(2), bf16::ZERO);
        assert_eq!(regs.read_hbm(4), 0);
        assert_eq!(regs.scale(), 0);
        assert_eq!(regs.stride(), 1);
        assert_eq!(regs.bmm_scale(), 0.25);
        assert_eq!(regs.v_mask(), 0);
    }

    #[test]
    fn register_file_reads_writes_and_binary_ops_use_isa_indices() {
        let mut regs = AcceleratorRegFile::new();

        regs.write_gp(1, 10);
        regs.write_gp(2, 3);
        regs.binop_gp(3, 1, 2, u32::wrapping_sub);
        regs.write_fp(1, bf16::from_f32(1.5));
        regs.write_fp(2, bf16::from_f32(2.0));
        regs.binop_fp(3, 1, 2, std::ops::Mul::mul);
        regs.write_hbm(7, 0x1234_5678);
        regs.set_scale(64);
        regs.set_stride(4);
        regs.set_v_mask(0b1010);

        assert_eq!(regs.read_gp(3), 7);
        assert_eq!(regs.read_fp(3), bf16::from_f32(3.0));
        assert_eq!(regs.read_hbm(7), 0x1234_5678);
        assert_eq!(regs.scale(), 64);
        assert_eq!(regs.stride(), 4);
        assert_eq!(regs.v_mask(), 0b1010);
    }

    /// Pins the `(num_experts << 8) | top_k` layout against the real MoE shapes the
    /// compiler packs. This bit layout is a cross-repo contract with
    /// `moe_router_select_v0`; nothing else checks that the two agree.
    #[test]
    fn topk_policy_unpacks_every_supported_moe_shape() {
        let cases = [
            (32u32, 4u32), // GPT-OSS
            (128, 8),      // Qwen3-30B-A3B
            (60, 4),       // Qwen2-MoE
            (64, 6),       // DeepSeek-V2-Lite
            (256, 8),      // DeepSeek-V3 / Kimi K2
            (16, 1),       // Llama-4 Scout
            (16383, 255),  // widest shape the 22-bit S_ADDI_INT immediate admits
        ];
        for (experts, top_k) in cases {
            let packed = (experts << 8) | top_k;
            assert!(
                packed < (1 << 22),
                "packed {packed} for {experts}/{top_k} does not fit a 22-bit immediate"
            );
            let mut regs = AcceleratorRegFile::new();
            regs.set_topk_policy(packed);
            assert_eq!(
                regs.topk_policy(),
                Some((experts as usize, top_k as usize)),
                "round-trip failed for {experts} experts / top-{top_k}"
            );
        }
    }

    #[test]
    fn topk_policy_is_none_until_c_set_topk_reg_runs() {
        // V_TOPK rmask=15 relies on this to trap with a message naming the missing
        // C_SET_TOPK_REG, instead of unpacking 0 into top_k=0 and tripping
        // topk_softmax's unrelated "topk must be positive" assertion.
        assert_eq!(AcceleratorRegFile::new().topk_policy(), None);
    }
}
