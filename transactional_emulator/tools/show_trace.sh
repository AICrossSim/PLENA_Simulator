#!/usr/bin/env bash
# Render the emulator latency trace into a visual timeline + bubble
# report. Reads transactional_emulator/testbench/build/trace.json
# (produced by the emulator's dump_trace) and writes:
#
#   build/trace.svg   -- Gantt timeline, one row per engine; open in
#                        any browser. Gaps between blocks == bubbles.
#   build/trace.html  -- same timeline embedded in a page (open in
#                        VS Code's Simple Browser / Live Preview).
#   stdout            -- per-engine busy / idle (bubble) breakdown.
#
# Zero dependencies: pure-Python stdlib, no matplotlib / browser.
#
# Usage (from repo root or anywhere):
#     bash transactional_emulator/tools/show_trace.sh
#     bash transactional_emulator/tools/show_trace.sh path/to/trace.json
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD="$REPO_ROOT/transactional_emulator/testbench/build"
TRACE="${1:-$BUILD/trace.json}"
SVG="$BUILD/trace.svg"
HTML="$BUILD/trace.html"

if [ ! -f "$TRACE" ]; then
    echo "!! trace not found: $TRACE"
    echo "   run the emulator first (just build-emulator-debug <kernel>)"
    exit 1
fi

TRACE="$TRACE" SVG="$SVG" HTML="$HTML" python3 - <<'PYEOF'
import json, os, sys

trace_path = os.environ["TRACE"]
svg_path   = os.environ["SVG"]
html_path  = os.environ["HTML"]

with open(trace_path) as f:
    events = json.load(f)

if not events:
    print("trace is empty -- nothing to render")
    sys.exit(0)

# Chrome-trace events: ts/dur are in microseconds. Group by tid (engine).
# Each event: {name, cat, tid, ts, dur}
raw = {}             # tid -> list of (ts, end)
cat_of = {}          # tid -> category label
for e in events:
    tid = e["tid"]
    raw.setdefault(tid, []).append((e["ts"], e["ts"] + e["dur"]))
    cat_of[tid] = e.get("cat", str(tid))

t_min = min(e["ts"] for e in events)
t_max = max(e["ts"] + e["dur"] for e in events)
span  = t_max - t_min or 1.0

# Coalesce: merge adjacent same-engine spans that touch (no gap) into
# one block. Hundreds of thousands of back-to-back 1-cycle ops collapse
# into a few "busy" bars — invisible individually anyway — while every
# GAP is kept exactly (a gap == a pipeline bubble, the whole point).
# Keeps the SVG to a few hundred rects instead of ~750k.
#
# `eps` tolerates floating-point ts noise: spans closer than this are
# treated as touching. 1e-4 us = 0.1 ns, well below any real op.
eps = 1e-4
rows = {}            # tid -> list of (ts, dur, name)
n_raw = len(events)
for tid, spans in raw.items():
    spans.sort()
    cur_s, cur_e, cur_n = None, None, 0
    for s, e in spans:
        if cur_s is None:
            cur_s, cur_e, cur_n = s, e, 1
        elif s <= cur_e + eps:
            cur_e = max(cur_e, e)
            cur_n += 1
        else:
            rows.setdefault(tid, []).append(
                (cur_s, cur_e - cur_s, f"{cat_of[tid]} busy x{cur_n}"))
            cur_s, cur_e, cur_n = s, e, 1
    if cur_s is not None:
        rows.setdefault(tid, []).append(
            (cur_s, cur_e - cur_s, f"{cat_of[tid]} busy x{cur_n}"))
n_merged = sum(len(v) for v in rows.values())
print(f" coalesced {n_raw} raw events -> {n_merged} blocks")

# ---- text bubble report -------------------------------------------------
print("=" * 64)
print(" Latency trace -- per-engine bubble report")
print("=" * 64)
print(f" total wall span : {span:.3f} us  ({t_min:.3f} -> {t_max:.3f})")
print(f" events          : {len(events)}")
print("-" * 64)
print(f" {'engine':<10} {'busy us':>12} {'idle us':>12} {'busy%':>8} {'ops':>6}")
print("-" * 64)
for tid in sorted(rows):
    spans = sorted(rows[tid])
    busy = sum(d for _, d, _ in spans)
    # idle within the engine's own active window (first start -> last end)
    eng_start = spans[0][0]
    eng_end   = max(ts + d for ts, d, _ in spans)
    eng_span  = eng_end - eng_start or 1.0
    idle = eng_span - busy
    pct  = 100.0 * busy / eng_span
    print(f" {cat_of[tid]:<10} {busy:>12.3f} {idle:>12.3f} "
          f"{pct:>7.1f}% {len(spans):>6}")
print("=" * 64)

# ---- SVG Gantt ----------------------------------------------------------
W, ROW_H, PAD, LBL = 1400, 38, 14, 90
plot_w = W - 2 * PAD - LBL
tids = sorted(rows)
H = 2 * PAD + 30 + len(tids) * (ROW_H + 8)

palette = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759",
           "#76b7b2", "#edc948", "#b07aa1", "#ff9da7"]

def x(ts):  return PAD + LBL + (ts - t_min) / span * plot_w

parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" '
         f'height="{H}" font-family="monospace" font-size="11">']
parts.append(f'<rect width="{W}" height="{H}" fill="#fafafa"/>')
parts.append(f'<text x="{PAD}" y="{PAD+10}" font-size="13" '
             f'font-weight="bold">PLENA latency trace '
             f'({span:.1f} us, {len(events)} ops) -- gaps = bubbles</text>')

for i, tid in enumerate(tids):
    y = PAD + 30 + i * (ROW_H + 8)
    color = palette[i % len(palette)]
    # row background = engine active window, so bubbles show as gaps
    parts.append(f'<rect x="{PAD+LBL}" y="{y}" width="{plot_w}" '
                 f'height="{ROW_H}" fill="#ececec"/>')
    parts.append(f'<text x="{PAD}" y="{y+ROW_H/2+4}" '
                 f'font-weight="bold">{cat_of[tid]}</text>')
    for ts, dur, name in rows[tid]:
        bx = x(ts)
        bw = max(0.6, dur / span * plot_w)
        parts.append(f'<rect x="{bx:.2f}" y="{y+3}" width="{bw:.2f}" '
                     f'height="{ROW_H-6}" fill="{color}" '
                     f'stroke="#222" stroke-width="0.3">'
                     f'<title>{name}: {dur:.3f}us @ {ts:.3f}us</title></rect>')

# time axis ticks
for k in range(11):
    tx = PAD + LBL + k / 10 * plot_w
    tval = t_min + k / 10 * span
    parts.append(f'<line x1="{tx:.1f}" y1="{PAD+26}" x2="{tx:.1f}" '
                 f'y2="{H-PAD}" stroke="#ccc" stroke-width="0.5"/>')
    parts.append(f'<text x="{tx:.1f}" y="{H-PAD+12}" font-size="9" '
                 f'text-anchor="middle">{tval:.1f}us</text>')

parts.append('</svg>')
svg_text = "\n".join(parts)
with open(svg_path, "w") as f:
    f.write(svg_text)

# HTML wrapper — the SVG inlined into a page so it opens directly in
# VS Code's Simple Browser / Live Preview (no SVG extension needed).
html = (
    "<!doctype html><html><head><meta charset='utf-8'>"
    "<title>PLENA latency trace</title>"
    "<style>body{margin:12px;background:#fff;"
    "font-family:monospace}</style></head><body>\n"
    + svg_text +
    "\n</body></html>\n"
)
with open(html_path, "w") as f:
    f.write(html)

print(f" SVG timeline written to:  {svg_path}")
print(f" HTML timeline written to: {html_path}")
print(" open the .html in VS Code's Simple Browser / Live Preview;")
print(" hover a block for op name + timing")
PYEOF
