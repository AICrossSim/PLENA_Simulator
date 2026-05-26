#!/usr/bin/env python3
"""GP/slot consistency tracer for v2-emitted PLENA ISA.

Reads a generated_asm_code.asm produced by the v2 backend with inline
MIR-trace tags (``; %res=op(%ops)`` on each instr line, plus
``; SPILL %x ...`` / ``; RELOAD %x ...`` comments). Simulates the GP file
and IntRAM-slot file symbolically — tracking *which MIR %value* each GP
and slot currently holds — and reports the first place where an
instruction READS a GP whose held %value does NOT match the %value the
trace tag says that operand should be. That mismatch is exactly the
register-allocation corruption class (double-occupancy / read-after-evict
/ stale reload) we keep hitting.

Usage:
    python3 trace_gp.py <generated_asm_code.asm>

Output (to stdout, terminal-friendly — only mismatches + a summary):
    line N: READ gpK expected %A but holds %B   <instr>
No diff dumps; full per-line state is NOT printed.

This is a STATIC checker: it does not execute hardware, only follows the
emitted dataflow. A clean report means GP allocation is self-consistent
(the bug, if any, is elsewhere — numeric/semantic). A mismatch pinpoints
the corrupting line.
"""
import re
import sys


GP_RE = re.compile(r"\bgp(\d+)\b")
# instr line:  MNEMONIC args...   ; %res=op(%a,%b)   (trace tag optional)
TAG_RE = re.compile(r";\s*(%\w+)=\w+\(([^)]*)\)")
SPILL_RE = re.compile(r";\s*SPILL\s+(%\w+|id\d+)\s*\(value@\d+\)\s*->\s*intram\[(\d+)\]")
RELOAD_RE = re.compile(r";\s*RELOAD\s+(%\w+)\s*\(value@\d+\)\s*from\s*intram\[(\d+)\]\s*->\s*gp(\d+)")
LDINT_RE = re.compile(r"^\s*S_LD_INT\s+gp(\d+),\s*gp0,\s*(\d+)")
STINT_RE = re.compile(r"^\s*S_ST_INT\s+gp(\d+),\s*gp0,\s*(\d+)")
LOOP_START_RE = re.compile(r"^\s*C_LOOP_START\s+gp(\d+),\s*(\d+)")
LOOP_END_RE = re.compile(r"^\s*C_LOOP_END\s+gp(\d+)")
# ``; for <var> in [a, b) -- hw counter gpC, idx ram[N]`` — declares that
# IntRAM slot N is loop <var>'s software index. A later
# ``S_LD_INT gpK, gp0, N`` reloads the loop_var (NOT a spilled value).
LOOP_HDR_RE = re.compile(
    r";\s*for\s+(\w+)\s+in\s+\[[^)]*\)\s*--\s*hw counter gp\d+,\s*idx ram\[(\d+)\]"
)


def main(path):
    lines = open(path).read().splitlines()

    # gp_holds[k] = the %name currently in gpK (or None / a raw marker).
    # We seed nothing; values appear as instructions define them.
    gp_holds = {}        # gp -> %name (the SSA value that wrote it)
    slot_holds = {}      # intram slot -> %name (what was last stored)
    idx_slot_var = {}    # intram slot -> loop_var %name (idx backing store)
    # loop counter gps: written by C_LOOP_START, must not be data-read.
    counter_gps = set()

    mismatches = []
    n_instr = 0

    for i, raw in enumerate(lines, start=1):
        line = raw.rstrip()
        if not line.strip():
            continue

        # --- loop header comment: slot N backs loop_var <var> ---
        m = LOOP_HDR_RE.search(line)
        if m:
            var, slot = m.group(1), int(m.group(2))
            idx_slot_var[slot] = f"%{var}"
            slot_holds[slot] = f"%{var}"
            continue

        # --- SPILL: a %value's GP is stored to a slot, GP freed ---
        m = SPILL_RE.search(line)
        if m:
            val, slot = m.group(1), int(m.group(2))
            slot_holds[slot] = val
            continue

        # --- RELOAD: slot -> gp, gp now holds that %value ---
        m = RELOAD_RE.search(line)
        if m:
            val, slot, gp = m.group(1), int(m.group(2)), int(m.group(3))
            held = slot_holds.get(slot)
            if held is not None and held != val:
                mismatches.append(
                    (i, f"RELOAD intram[{slot}] -> gp{gp} as {val} "
                        f"but slot holds {held}", line.strip()))
            gp_holds[gp] = val
            continue

        # --- C_LOOP_START gpK, N : gpK becomes a counter ---
        m = LOOP_START_RE.match(line)
        if m:
            counter_gps.add(int(m.group(1)))
            gp_holds[int(m.group(1))] = "<counter>"
            continue
        m = LOOP_END_RE.match(line)
        if m:
            continue

        # --- raw S_LD_INT gpK, gp0, slot (loop idx reload etc.) ---
        m = LDINT_RE.match(line)
        if m:
            gp, slot = int(m.group(1)), int(m.group(2))
            gp_holds[gp] = slot_holds.get(slot, f"<slot{slot}>")
            continue
        m = STINT_RE.match(line)
        if m:
            gp, slot = int(m.group(1)), int(m.group(2))
            # idx slots permanently belong to their loop_var (init store
            # of gp0=0 and the +1 write-back are both "still the var").
            if slot in idx_slot_var:
                continue
            slot_holds[slot] = gp_holds.get(gp, f"<gp{gp}>")
            continue

        # --- a tagged instruction line: check operand GPs ---
        tag = TAG_RE.search(line)
        if tag is None:
            continue
        n_instr += 1
        res = tag.group(1)
        op_names = [o.strip() for o in tag.group(2).split(",") if o.strip()]
        # the ISA operands (gp tokens) on the code part (before ';')
        code = line.split(";", 1)[0]
        gp_toks = GP_RE.findall(code)
        # gp_toks[0] is usually the dst (result). The rest are sources,
        # in the same order as op_names (PLENA prepends dst).
        # Map source operands -> their gp, compare held %value.
        # dst:
        if res and gp_toks:
            dst_gp = int(gp_toks[0])
            src_gps = [int(x) for x in gp_toks[1:]]
        else:
            dst_gp = None
            src_gps = [int(x) for x in gp_toks]

        # Compare each source operand's gp against the %name it should be.
        # op_names may include non-%value operands (immediates, gp0, fN);
        # only check the ones that are %values and map positionally to a gp.
        val_ops = [o for o in op_names if o.startswith("%") and o != "%gp0"
                   and not o.endswith("_gp0")]
        # Heuristic positional match: align val_ops to src_gps tail.
        for o, g in zip(val_ops, src_gps):
            held = gp_holds.get(g)
            if held is None:
                continue  # unknown (e.g. operand was an immediate-in-gp)
            if held != o:
                mismatches.append(
                    (i, f"READ gp{g} expected {o} but holds {held}",
                     line.strip()))

        # Record the result definition.
        if res and dst_gp is not None:
            gp_holds[dst_gp] = res

    print(f"[trace_gp] scanned {n_instr} tagged instrs; "
          f"{len(mismatches)} mismatch(es)")
    for ln, msg, src in mismatches[:60]:
        print(f"  line {ln}: {msg}")
        print(f"           {src[:90]}")
    if len(mismatches) > 60:
        print(f"  ... +{len(mismatches)-60} more")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: trace_gp.py <generated_asm_code.asm>")
        sys.exit(2)
    main(sys.argv[1])
