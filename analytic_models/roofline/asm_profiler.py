"""
PLENA Decoder Layer ASM Profiler
Parses generated_asm_code.asm and reports instruction counts + estimated cycles
per pipeline section.

Usage:
    python3 analytic_models/roofline/asm_profiler.py
    python3 analytic_models/roofline/asm_profiler.py path/to/generated_asm_code.asm
"""

import sys
import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VLEN = 64  # vector lane width
MLEN = 64  # matrix tile size

# Cycle costs per instruction prefix
CYCLE_COSTS = {
    "S_": 1,
    "C_": 1,
    "H_PREFETCH_V": VLEN,
    "H_PREFETCH_M": MLEN,
    "V_": VLEN,
    "M_MM": MLEN,
    "M_BMM": MLEN,
    "M_BMV": MLEN,
}

SECTION_ORDER = [
    "conv2d",
    "data_loading",
    "embedding_add",
    "rms_norm_1",
    "rope",
    "flash_attention",
    "ffn",
    "lm_head",
    "rms_norm_2",
    "other",
]

DEFAULT_ASM_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "transactional_emulator",
    "testbench",
    "build",
    "generated_asm_code.asm",
)

# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------


def detect_section(comment, current_section, seen_rms_norm):
    """
    Given a comment line (stripped, without leading ';'), return the new
    current_section and updated seen_rms_norm count.

    Returns (new_section, new_seen_rms_norm).
    """
    text = comment.strip()

    # conv2d (patch embedding): im2col setup + systolic matmul on im2col output
    CONV2D_TRIGGERS = [
        "im2col",
        "linear_out",
        "W_2d",
        "im2col_out",
    ]
    for trigger in CONV2D_TRIGGERS:
        if trigger in text:
            return "conv2d", seen_rms_norm

    # data_loading: any Load_Batch header
    if "Load_Batch" in text:
        return "data_loading", seen_rms_norm

    # embedding_add: VRAM Matrix Add or embedding_add
    if "VRAM Matrix Add" in text or "embedding_add" in text:
        return "embedding_add", seen_rms_norm

    # RMS Norm detection - must distinguish rms_norm_1 vs rms_norm_2
    # Only count a new RMS block when we are NOT already inside an rms_norm
    # section, to avoid double-counting consecutive RMS comment lines.
    is_rms = ("Normalize (rms)" in text) or ("RMS Norm" in text)
    if is_rms:
        if current_section in ("rms_norm_1", "rms_norm_2"):
            # Still inside the same RMS block — do not increment
            return current_section, seen_rms_norm
        new_count = seen_rms_norm + 1
        # Use modulo so multi-layer tiled ASM is handled correctly:
        # odd occurrences (1,3,5,...) = rms_norm_1 (pre-attention)
        # even occurrences (2,4,6,...) = rms_norm_2 (post-ffn)
        if new_count % 2 == 1:
            return "rms_norm_1", new_count
        else:
            return "rms_norm_2", new_count

    # rope
    if "RoPE" in text:
        return "rope", seen_rms_norm

    # flash_attention triggers
    FA_TRIGGERS = [
        "Online Softmax",
        "Init Online Softmax",
        "PV Multiply",
        "Final Scale O",
        "Load SubMatrix Row K",
        "Reset VRAM rows",
        "Reset MRAM",
    ]
    for trigger in FA_TRIGGERS:
        if trigger in text:
            return "flash_attention", seen_rms_norm

    # lm_head (full-sequence vocabulary projection, used by LLaDA)
    LM_HEAD_TRIGGERS = [
        "LM head",
        "lm_head",
        "vocab projection",
        "Vocab Projection",
    ]
    for trigger in LM_HEAD_TRIGGERS:
        if trigger in text:
            return "lm_head", seen_rms_norm

    # ffn
    FFN_TRIGGERS = [
        "FFN Generation",
        "FFN Gate",
        "FFN Upsize",
        "FFN Downsize",
    ]
    for trigger in FFN_TRIGGERS:
        if trigger in text:
            return "ffn", seen_rms_norm

    # No match — keep current section
    return current_section, seen_rms_norm


# ---------------------------------------------------------------------------
# Cycle cost for a single instruction line
# ---------------------------------------------------------------------------


def instruction_cycles(line):
    """Return the estimated cycle cost for one instruction line."""
    tok = line.split()[0] if line.split() else ""

    # Specific H_ prefetch variants first (more specific before generic prefix)
    if tok == "H_PREFETCH_V":
        return CYCLE_COSTS["H_PREFETCH_V"]
    if tok == "H_PREFETCH_M":
        return CYCLE_COSTS["H_PREFETCH_M"]

    # Matrix multiply ops
    if tok in ("M_MM", "M_BMM", "M_BMV"):
        return CYCLE_COSTS[tok]

    # Generic prefixes
    if tok.startswith("S_"):
        return CYCLE_COSTS["S_"]
    if tok.startswith("C_"):
        return CYCLE_COSTS["C_"]
    if tok.startswith("V_"):
        return CYCLE_COSTS["V_"]
    if tok.startswith("M_"):
        # Any other M_ instruction — treat as MLEN
        return MLEN
    if tok.startswith("H_"):
        # Any other H_ instruction — treat as VLEN
        return VLEN

    # Unknown — default 1 cycle
    return 1


# ---------------------------------------------------------------------------
# Instruction type classification
# ---------------------------------------------------------------------------


def classify_instr_type(line):
    """Return one of: S, C, H, V, M, other."""
    tok = line.split()[0] if line.split() else ""
    if tok.startswith("S_"):
        return "S"
    if tok.startswith("C_"):
        return "C"
    if tok.startswith("H_"):
        return "H"
    if tok.startswith("V_"):
        return "V"
    if tok.startswith("M_"):
        return "M"
    return "other"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def parse_asm(path):
    """
    Parse the ASM file and return:
        section_instrs  : dict[section_name -> int]   instruction count
        section_cycles  : dict[section_name -> int]   estimated cycles
        type_counts     : dict[type_char -> int]       instruction-type counts
        total_instrs    : int
        total_cycles    : int
    """
    section_instrs = {s: 0 for s in SECTION_ORDER}
    section_cycles = {s: 0 for s in SECTION_ORDER}
    type_counts = {"S": 0, "C": 0, "H": 0, "V": 0, "M": 0, "other": 0}

    current_section = "other"
    seen_rms_norm = 0
    loop_stack = []

    with open(path) as fh:
        for raw_line in fh:
            line = raw_line.strip()

            # Empty line — skip
            if not line:
                continue

            # Comment line — check for section header
            if line.startswith(";"):
                comment = line[1:]
                current_section, seen_rms_norm = detect_section(comment, current_section, seen_rms_norm)
                continue

            # Instruction line — compute dynamic multiplier from loop stack
            dynamic_multiplier = 1
            for count in loop_stack:
                dynamic_multiplier *= count

            tokens = line.split()
            tok = tokens[0] if tokens else ""

            if tok == "C_LOOP_START":
                # Count the loop start instruction itself (at current multiplier)
                cycles = instruction_cycles(line) * dynamic_multiplier
                itype = classify_instr_type(line)
                section_instrs[current_section] += dynamic_multiplier
                section_cycles[current_section] += cycles
                if itype in type_counts:
                    type_counts[itype] += dynamic_multiplier
                else:
                    type_counts["other"] += dynamic_multiplier
                # Then push the loop count
                try:
                    loop_count = int(tokens[-1])
                except (ValueError, IndexError):
                    loop_count = 1
                loop_stack.append(loop_count)
                continue  # already counted above

            elif tok == "C_LOOP_END":
                # Count the loop end instruction at current multiplier
                cycles = instruction_cycles(line) * dynamic_multiplier
                itype = classify_instr_type(line)
                section_instrs[current_section] += dynamic_multiplier
                section_cycles[current_section] += cycles
                if itype in type_counts:
                    type_counts[itype] += dynamic_multiplier
                else:
                    type_counts["other"] += dynamic_multiplier
                # Pop the loop stack
                if loop_stack:
                    loop_stack.pop()
                continue  # already counted above

            # Regular instruction: count with dynamic multiplier
            cycles = instruction_cycles(line) * dynamic_multiplier
            itype = classify_instr_type(line)
            section_instrs[current_section] += dynamic_multiplier
            section_cycles[current_section] += cycles
            if itype in type_counts:
                type_counts[itype] += dynamic_multiplier
            else:
                type_counts["other"] += dynamic_multiplier

    total_instrs = sum(section_instrs.values())
    total_cycles = sum(section_cycles.values())

    return section_instrs, section_cycles, type_counts, total_instrs, total_cycles


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(asm_path, section_instrs, section_cycles, type_counts, total_instrs, total_cycles):
    SEP = "=" * 60
    THIN = chr(0x2500) * 57  # ─────

    rel_path = os.path.relpath(asm_path)

    print(SEP)
    print("  PLENA ASM Profiler (dynamic, loops expanded)")
    print(f"  Source: {rel_path}")
    print(SEP)
    print()
    print("Section breakdown (instruction count):")

    for sec in SECTION_ORDER:
        n_instr = section_instrs[sec]
        n_cyc = section_cycles[sec]
        pct_i = 100.0 * n_instr / total_instrs if total_instrs else 0.0
        pct_c = 100.0 * n_cyc / total_cycles if total_cycles else 0.0
        print(f"  {sec:<18} : {n_instr:>6} instr  ({pct_i:>5.1f}%)  est. {n_cyc:>7} cycles  ({pct_c:>5.1f}%)")

    print("  " + THIN)
    print("  {:<18} : {:>6} instr           est. {:>7} cycles".format("TOTAL", total_instrs, total_cycles))
    print()
    print("Instruction type breakdown:")

    type_labels = [
        ("S", "S_*  (scalar) "),
        ("C", "C_*  (control)"),
        ("H", "H_*  (HBM)   "),
        ("V", "V_*  (vector) "),
        ("M", "M_*  (matmul) "),
    ]
    for key, label in type_labels:
        n = type_counts.get(key, 0)
        pct = 100.0 * n / total_instrs if total_instrs else 0.0
        print(f"  {label} : {n:>6}  ({pct:>5.1f}%)")

    print(SEP)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    if len(sys.argv) >= 2:
        asm_path = sys.argv[1]
    else:
        asm_path = os.path.normpath(DEFAULT_ASM_PATH)

    if not os.path.isfile(asm_path):
        sys.stderr.write(f"ERROR: ASM file not found: {asm_path}\n")
        sys.exit(1)

    section_instrs, section_cycles, type_counts, total_instrs, total_cycles = parse_asm(asm_path)
    print_report(asm_path, section_instrs, section_cycles, type_counts, total_instrs, total_cycles)


if __name__ == "__main__":
    main()
