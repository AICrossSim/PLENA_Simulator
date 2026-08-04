from __future__ import annotations

import unittest

from runtime_paths import simulator_root

EMULATOR_ROOT = simulator_root() / "transactional_emulator"


class VectorMaskContractTests(unittest.TestCase):
    def test_mask_opcode_reaches_register_state(self) -> None:
        op_source = (EMULATOR_ROOT / "src" / "op.rs").read_text(
            encoding="utf-8"
        )
        dispatch_source = (
            EMULATOR_ROOT / "src" / "accelerator" / "dispatch.rs"
        ).read_text(encoding="utf-8")

        self.assertIn("0x2E => Self::C_SET_V_MASK_REG", op_source)
        self.assertIn(
            "self.reg_file.set_v_mask(self.reg_file.read_gp(*rd))",
            dispatch_source,
        )
        self.assertIn("let segments = *VLEN / *HLEN", dispatch_source)

    def test_masked_element_ops_preserve_destination_lanes(self) -> None:
        source = (EMULATOR_ROOT / "src" / "vector_machine.rs").read_text(
            encoding="utf-8"
        )

        self.assertGreaterEqual(
            source.count("let destination = self.vram.read(vd).await;"),
            8,
        )
        self.assertGreaterEqual(
            source.count("if head_is_selected(mask, head)"),
            10,
        )

    def test_masked_reductions_visit_only_selected_segments(self) -> None:
        source = (EMULATOR_ROOT / "src" / "vector_machine.rs").read_text(
            encoding="utf-8"
        )
        reduce_sum = source[source.index("pub(crate) async fn reduce_sum") :]
        reduce_sum = reduce_sum[: reduce_sum.index("pub(crate) async fn reduce_max")]
        reduce_max = source[source.index("pub(crate) async fn reduce_max") :]

        self.assertIn("if head_is_selected(mask, head)", reduce_sum)
        self.assertIn("result += value", reduce_sum)
        self.assertNotIn("shallow_clone", reduce_sum)
        self.assertIn("if head_is_selected(mask, head)", reduce_max)
        self.assertIn("result = f32::max(result, value)", reduce_max)
        self.assertNotIn("shallow_clone", reduce_max)


if __name__ == "__main__":
    unittest.main()
