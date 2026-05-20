#!/usr/bin/env python3
"""Render a PLENA execution trace into a self-contained interactive HTML
report.

Reads two files produced by the emulator's ``dump_trace`` protocol
command:

* ``<path>.bin``      — packed ``TraceEntry`` records (24 bytes each)
* ``<path>.meta.json`` — engine / op_tag mapping + hardware constants

Outputs a single ``.html`` file with an embedded Canvas-based Gantt
chart, op-type summary table, and engine utilization breakdown. No
server needed — double-click to open.

Usage::

    python3 tools/render_trace.py /tmp/smoke_trace.bin -o /tmp/smoke_trace.html

The HTML inlines the binary as base64 (small kernels) so the file is
self-contained and trivially shareable.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


TRACE_DTYPE = np.dtype([
    ("start_picos", "<u8"),
    ("duration_picos", "<u4"),
    ("engine", "u1"),
    ("op_tag", "u1"),
    ("pad", "<u2"),
    ("hbm_read", "<u4"),
    ("hbm_written", "<u4"),
])


# Six engine lane colors. Picked to read well on a dark background and
# stay distinguishable when binned alongside each other. PrefetchM and
# PrefetchV use related warm tones so the two HBM-side lanes obviously
# belong to the same family, but stay separable side-by-side.
ENGINE_COLORS = {
    0: "#58a6ff",  # Matrix    – blue
    1: "#3fb950",  # Vector    – green
    2: "#d29922",  # Scalar    – amber
    3: "#f85149",  # PrefetchM – red       (HBM → MRAM, bulk weight loads)
    4: "#ff8a65",  # PrefetchV – coral     (HBM ↔ VRAM, activation pre/store)
    5: "#8b949e",  # Control   – grey
}


def fmt_picos(p: int) -> str:
    if p < 1_000:
        return f"{p} ps"
    if p < 1_000_000:
        return f"{p/1e3:.2f} ns"
    if p < 1_000_000_000:
        return f"{p/1e6:.2f} µs"
    if p < 1_000_000_000_000:
        return f"{p/1e9:.2f} ms"
    return f"{p/1e12:.3f} s"


def build_op_breakdown(arr: np.ndarray, op_tag_names: dict[int, str]) -> list[dict]:
    rows = []
    total = int(arr["duration_picos"].sum()) or 1
    for tag in sorted(set(arr["op_tag"].tolist())):
        mask = arr["op_tag"] == tag
        count = int(mask.sum())
        cycles = int(arr["duration_picos"][mask].sum())
        rows.append({
            "tag": int(tag),
            "name": op_tag_names.get(int(tag), f"tag_{tag}"),
            "count": count,
            "cycles": cycles,
            "pct": 100 * cycles / total,
            "avg_per_op": cycles / count if count else 0,
        })
    rows.sort(key=lambda r: -r["cycles"])
    return rows


def build_engine_summary(arr: np.ndarray, engine_names: dict[int, str]) -> list[dict]:
    rows = []
    total = int(arr["duration_picos"].sum()) or 1
    for engine_id in sorted(set(arr["engine"].tolist())):
        mask = arr["engine"] == engine_id
        count = int(mask.sum())
        cycles = int(arr["duration_picos"][mask].sum())
        rows.append({
            "id": int(engine_id),
            "name": engine_names.get(int(engine_id), f"engine_{engine_id}"),
            "count": count,
            "cycles": cycles,
            "pct": 100 * cycles / total,
            "color": ENGINE_COLORS.get(int(engine_id), "#888"),
        })
    return rows


def render_html(
    bin_bytes: bytes,
    meta: dict,
    arr: np.ndarray,
    engine_summary: list[dict],
    op_breakdown: list[dict],
    trace_name: str,
) -> str:
    bin_b64 = base64.b64encode(bin_bytes).decode("ascii")
    bin_size_mb = len(bin_bytes) / 1_048_576

    total_picos = int(meta.get("total_sim_picos") or arr["duration_picos"].sum())
    entry_count = int(meta.get("entry_count", len(arr)))
    hardware = meta.get("hardware", {})

    payload = {
        "engines": meta.get("engines", []),
        "op_tags": meta.get("op_tags", []),
        "trace_start_picos": int(meta.get("trace_start_picos", 0)),
        "total_sim_picos": total_picos,
        "entry_count": entry_count,
        "overflowed": meta.get("overflowed", False),
        "engine_summary": engine_summary,
        "op_breakdown": op_breakdown,
        "hardware": hardware,
    }
    payload_json = json.dumps(payload)

    return _HTML_TEMPLATE.format(
        trace_name=trace_name,
        bin_b64=bin_b64,
        bin_size_mb=f"{bin_size_mb:.2f}",
        entry_count=entry_count,
        total_sim=fmt_picos(total_picos),
        payload_json=payload_json,
        engine_colors_json=json.dumps(ENGINE_COLORS),
    )


# ----------------------------------------------------------------------
# HTML template
# ----------------------------------------------------------------------
# Single self-contained file. Inline JS draws onto a Canvas, so we can
# show hundreds of thousands of ops without DOM thrashing.
# ----------------------------------------------------------------------
_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PLENA trace — {trace_name}</title>
<style>
  :root {{
    --bg: #0d1117; --panel: #161b22; --panel-2: #1f2630;
    --border: #30363d; --text: #c9d1d9; --muted: #8b949e;
    --accent: #58a6ff;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 13px; line-height: 1.5;
  }}
  header {{
    padding: 12px 20px; background: var(--panel);
    border-bottom: 1px solid var(--border);
    display: flex; gap: 20px; align-items: baseline; flex-wrap: wrap;
  }}
  header h1 {{ font-size: 16px; font-weight: 600; margin: 0; }}
  header .muted {{ color: var(--muted); font-size: 12px; }}
  main {{ padding: 14px 20px; display: grid; gap: 14px; }}
  section {{
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 6px; padding: 12px 16px;
  }}
  section h2 {{
    margin: 0 0 10px; font-size: 12px; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.05em; font-weight: 500;
  }}
  .grid-4 {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }}
  .stat {{
    background: var(--panel-2); border: 1px solid var(--border);
    border-radius: 6px; padding: 8px 10px;
  }}
  .stat .label {{ font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }}
  .stat .value {{ font-size: 16px; font-family: ui-monospace, monospace; }}

  .gantt-shell {{
    background: var(--panel-2); border: 1px solid var(--border);
    border-radius: 6px; padding: 8px; overflow: hidden;
  }}
  .gantt-controls {{
    display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
    margin-bottom: 8px; font-size: 12px; color: var(--muted);
  }}
  .gantt-controls button {{
    background: var(--panel-2); color: var(--text); border: 1px solid var(--border);
    border-radius: 4px; padding: 3px 10px; font-size: 12px; cursor: pointer;
  }}
  .gantt-controls button:hover {{ background: var(--panel); border-color: var(--accent); }}
  #gantt-canvas {{
    display: block; width: 100%; height: 260px;
    background: #0a0e13; border-radius: 4px;
    cursor: crosshair;
  }}
  #gantt-tooltip {{
    position: fixed; pointer-events: none;
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 4px; padding: 6px 10px; font-size: 11px;
    font-family: ui-monospace, monospace; line-height: 1.5;
    display: none; z-index: 100; box-shadow: 0 4px 12px rgba(0,0,0,0.5);
  }}

  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  th, td {{ padding: 5px 8px; border-bottom: 1px solid var(--border); text-align: left; }}
  th {{ color: var(--muted); font-weight: 500; font-size: 11px; }}
  td.num {{ text-align: right; font-family: ui-monospace, monospace; }}
  td.name {{ font-family: ui-monospace, monospace; }}

  .legend {{
    display: flex; gap: 12px; flex-wrap: wrap; font-size: 11px;
    margin-bottom: 6px;
  }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; }}
  .legend-swatch {{ width: 12px; height: 12px; border-radius: 2px; }}

  .twocol {{ display: grid; grid-template-columns: 1.4fr 1fr; gap: 14px; }}
  @media (max-width: 1000px) {{ .twocol {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<header>
  <h1>PLENA execution trace</h1>
  <span class="muted">{trace_name}</span>
  <span class="muted">·  {entry_count} ops  ·  {total_sim} sim time  ·  {bin_size_mb} MB binary</span>
</header>

<main>
  <section>
    <h2>Summary</h2>
    <div class="grid-4" id="summary-stats"></div>
  </section>

  <section class="twocol">
    <div>
      <h2>Gantt (binned heatmap → zoom to see raw ops)</h2>
      <div class="gantt-shell">
        <div class="legend" id="gantt-legend"></div>
        <div class="gantt-controls">
          <span>Zoom: <span id="zoom-level">1.0×</span></span>
          <button id="zoom-in">＋</button>
          <button id="zoom-out">－</button>
          <button id="zoom-reset">reset</button>
          <span style="margin-left: auto;">Drag to pan · scroll to zoom · hover for op details</span>
        </div>
        <canvas id="gantt-canvas"></canvas>
        <div id="gantt-tooltip"></div>
      </div>
    </div>
    <div>
      <h2>Engine breakdown</h2>
      <table>
        <thead><tr><th>engine</th><th class="num">ops</th><th class="num">cycles</th><th class="num">%</th></tr></thead>
        <tbody id="engine-tbody"></tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>Op-type breakdown (sorted by total cycles)</h2>
    <table>
      <thead><tr>
        <th>op</th><th>engine</th>
        <th class="num">count</th><th class="num">total cycles</th>
        <th class="num">avg / op</th><th class="num">% of run</th>
      </tr></thead>
      <tbody id="op-tbody"></tbody>
    </table>
  </section>
</main>

<script>
const META = {payload_json};
const ENGINE_COLORS = {engine_colors_json};
const BIN_B64 = "{bin_b64}";

// ---------- decode binary ----------
function decodeTrace(b64) {{
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const u8 = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  const recordSize = 24;
  const n = bin.length / recordSize;
  // Use a struct-of-arrays layout for cache-friendly Canvas drawing.
  const start  = new BigUint64Array(n);
  const dur    = new Uint32Array(n);
  const engine = new Uint8Array(n);
  const opTag  = new Uint8Array(n);
  const dv = new DataView(buf);
  for (let i = 0; i < n; i++) {{
    const o = i * recordSize;
    start[i]  = dv.getBigUint64(o, true);
    dur[i]    = dv.getUint32(o + 8, true);
    engine[i] = dv.getUint8(o + 12);
    opTag[i]  = dv.getUint8(o + 13);
  }}
  return {{ start, dur, engine, opTag, n }};
}}

const trace = decodeTrace(BIN_B64);
const startBase = Number(trace.start[0] || 0n);
const totalPicos = Number(META.total_sim_picos);

const opTagName = {{}};
const opTagEngine = {{}};
for (const t of META.op_tags) {{ opTagName[t.tag] = t.name; opTagEngine[t.tag] = t.engine_id; }}
const engineName = {{}};
for (const e of META.engines) engineName[e.id] = e.name;

// ---------- summary tiles ----------
const summary = document.getElementById("summary-stats");
function stat(label, value) {{
  const d = document.createElement("div"); d.className = "stat";
  d.innerHTML = `<div class="label">${{label}}</div><div class="value">${{value}}</div>`;
  return d;
}}
function fmtPs(p) {{
  if (p < 1e3) return `${{p|0}} ps`;
  if (p < 1e6) return `${{(p/1e3).toFixed(2)}} ns`;
  if (p < 1e9) return `${{(p/1e6).toFixed(2)}} µs`;
  if (p < 1e12) return `${{(p/1e9).toFixed(2)}} ms`;
  return `${{(p/1e12).toFixed(3)}} s`;
}}
summary.appendChild(stat("Total sim time", fmtPs(totalPicos)));
summary.appendChild(stat("Op count", trace.n.toLocaleString()));
summary.appendChild(stat("Unique op types", Object.keys(opTagName).filter(t => META.op_breakdown.find(b => b.tag == t)).length));
summary.appendChild(stat("Hardware", `M${{META.hardware.mlen}}·V${{META.hardware.vlen}}·B${{META.hardware.blen}}·H${{META.hardware.hlen}}`));

// ---------- engine summary table ----------
const engineTbody = document.getElementById("engine-tbody");
for (const row of META.engine_summary) {{
  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td><span class="legend-swatch" style="display:inline-block;background:${{row.color}};margin-right:6px;vertical-align:middle;"></span>${{row.name}}</td>
    <td class="num">${{row.count.toLocaleString()}}</td>
    <td class="num">${{fmtPs(row.cycles)}}</td>
    <td class="num">${{row.pct.toFixed(1)}}%</td>`;
  engineTbody.appendChild(tr);
}}

// ---------- op-type table ----------
const opTbody = document.getElementById("op-tbody");
for (const row of META.op_breakdown) {{
  const color = ENGINE_COLORS[opTagEngine[row.tag]] || "#888";
  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td class="name"><span class="legend-swatch" style="display:inline-block;background:${{color}};margin-right:6px;vertical-align:middle;"></span>${{row.name}}</td>
    <td>${{engineName[opTagEngine[row.tag]] || "?"}}</td>
    <td class="num">${{row.count.toLocaleString()}}</td>
    <td class="num">${{fmtPs(row.cycles)}}</td>
    <td class="num">${{fmtPs(row.avg_per_op)}}</td>
    <td class="num">${{row.pct.toFixed(2)}}%</td>`;
  opTbody.appendChild(tr);
}}

// ---------- Gantt legend ----------
const legendEl = document.getElementById("gantt-legend");
for (const e of META.engine_summary) {{
  const item = document.createElement("div"); item.className = "legend-item";
  item.innerHTML = `<span class="legend-swatch" style="background:${{e.color}}"></span>${{e.name}}`;
  legendEl.appendChild(item);
}}

// ---------- Gantt canvas ----------
const canvas = document.getElementById("gantt-canvas");
const ctx = canvas.getContext("2d");
const tooltip = document.getElementById("gantt-tooltip");

// View state
const NUM_LANES = 6;             // M / V / S / PrefM / PrefV / C
const LANE_PAD = 4;
let zoom = 1.0;
let pan = 0;                     // in picos
let dpr = window.devicePixelRatio || 1;

function resizeCanvas() {{
  dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = 260;
  canvas.width = Math.floor(w * dpr);
  canvas.height = Math.floor(h * dpr);
  draw();
}}
window.addEventListener("resize", resizeCanvas);

function laneY(engineId) {{
  const h = canvas.height / dpr;
  const laneH = (h - 10) / NUM_LANES;
  return 5 + engineId * laneH;
}}
function laneH() {{
  return (canvas.height / dpr - 10) / NUM_LANES - LANE_PAD;
}}

function timeToX(picos) {{
  const w = canvas.width / dpr;
  const visible = totalPicos / zoom;
  return ((picos - pan) / visible) * w;
}}
function xToTime(x) {{
  const w = canvas.width / dpr;
  const visible = totalPicos / zoom;
  return pan + (x / w) * visible;
}}

function draw() {{
  const w = canvas.width / dpr;
  const h = canvas.height / dpr;
  ctx.save();
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  // Lane backgrounds + labels
  const lhFull = (h - 10) / NUM_LANES;
  ctx.font = "10px ui-monospace, monospace";
  for (let i = 0; i < NUM_LANES; i++) {{
    ctx.fillStyle = i % 2 ? "rgba(255,255,255,0.02)" : "rgba(255,255,255,0.04)";
    ctx.fillRect(0, 5 + i * lhFull, w, lhFull);
    ctx.fillStyle = "#8b949e";
    ctx.fillText(engineName[i] || `lane_${{i}}`, 4, 5 + i * lhFull + 12);
  }}

  // Pixel-binned drawing (one column per pixel, count busy per lane)
  // This handles 100K+ ops smoothly because we never draw more rects
  // than pixels wide.
  const lh = laneH();
  const visible = totalPicos / zoom;
  const panLeftPs = pan;
  const panRightPs = pan + visible;
  const psPerPx = visible / w;

  // Allocate per-lane "busy fraction" array (one value per pixel column)
  const busy = [];
  for (let i = 0; i < NUM_LANES; i++) busy.push(new Float32Array(Math.ceil(w)));

  // Walk ops in the visible window. Trace start times are monotonic,
  // so we could binary-search the lower bound; for simplicity we linear-
  // scan with early exit.
  const N = trace.n;
  for (let i = 0; i < N; i++) {{
    const s = Number(trace.start[i]) - startBase;
    const d = trace.dur[i];
    if (s + d < panLeftPs) continue;
    if (s > panRightPs) break;
    const eng = trace.engine[i];
    const xStart = Math.max(0, (s - panLeftPs) / psPerPx);
    const xEnd = Math.min(w, (s + d - panLeftPs) / psPerPx);
    if (xEnd <= xStart) continue;
    const px0 = Math.floor(xStart);
    const px1 = Math.ceil(xEnd);
    for (let px = px0; px < px1 && px < w; px++) {{
      // Accumulate ratio of this pixel covered by this op.
      const overlap = Math.min(px + 1, xEnd) - Math.max(px, xStart);
      busy[eng][px] += overlap;
    }}
  }}

  // Render each lane: alpha-blend the busy fraction.
  for (let eng = 0; eng < NUM_LANES; eng++) {{
    const color = ENGINE_COLORS[eng] || "#888";
    const y = 5 + eng * lhFull + LANE_PAD / 2;
    const arr = busy[eng];
    for (let px = 0; px < w; px++) {{
      const f = Math.min(1.0, arr[px]);
      if (f <= 0.01) continue;
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.25 + 0.75 * f;
      ctx.fillRect(px, y, 1, lh);
    }}
  }}
  ctx.globalAlpha = 1.0;

  // If zoomed enough that ops are wider than 2 pixels, draw raw rects
  // on top so individual op boundaries are visible.
  if (psPerPx < 2000) {{   // <2000 ps/px ≈ 2 cycles/px
    for (let i = 0; i < N; i++) {{
      const s = Number(trace.start[i]) - startBase;
      const d = trace.dur[i];
      if (s + d < panLeftPs) continue;
      if (s > panRightPs) break;
      const eng = trace.engine[i];
      const xStart = Math.max(0, (s - panLeftPs) / psPerPx);
      const xEnd = Math.min(w, (s + d - panLeftPs) / psPerPx);
      if (xEnd - xStart < 0.5) continue;
      ctx.fillStyle = ENGINE_COLORS[eng] || "#888";
      ctx.fillRect(xStart, 5 + eng * lhFull + LANE_PAD/2, Math.max(1, xEnd - xStart), lh);
    }}
  }}

  // Time axis ticks
  ctx.fillStyle = "#8b949e";
  ctx.font = "10px ui-monospace, monospace";
  const tickPicos = niceTick(visible / 8);
  const tickStart = Math.ceil(panLeftPs / tickPicos) * tickPicos;
  for (let t = tickStart; t < panRightPs; t += tickPicos) {{
    const x = (t - panLeftPs) / psPerPx;
    ctx.fillRect(x, h - 14, 1, 4);
    ctx.fillText(fmtPs(t), x + 2, h - 2);
  }}

  ctx.restore();
}}

function niceTick(rough) {{
  const mag = Math.pow(10, Math.floor(Math.log10(rough)));
  const norm = rough / mag;
  let mult;
  if (norm < 1.5) mult = 1;
  else if (norm < 3.5) mult = 2;
  else if (norm < 7.5) mult = 5;
  else mult = 10;
  return mult * mag;
}}

// Zoom + pan controls
function clamp(v, lo, hi) {{ return Math.max(lo, Math.min(hi, v)); }}
function applyZoom(newZoom, focusPx) {{
  newZoom = clamp(newZoom, 1, 1e7);
  if (focusPx != null) {{
    const tBefore = xToTime(focusPx);
    zoom = newZoom;
    const visible = totalPicos / zoom;
    pan = clamp(tBefore - (focusPx / (canvas.width / dpr)) * visible, 0, Math.max(0, totalPicos - visible));
  }} else {{
    zoom = newZoom;
    const visible = totalPicos / zoom;
    pan = clamp(pan, 0, Math.max(0, totalPicos - visible));
  }}
  document.getElementById("zoom-level").textContent = `${{zoom.toFixed(1)}}×`;
  draw();
}}

canvas.addEventListener("wheel", e => {{
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const px = e.clientX - rect.left;
  const factor = e.deltaY < 0 ? 1.25 : 0.8;
  applyZoom(zoom * factor, px);
}}, {{ passive: false }});

let dragStartX = null;
let dragStartPan = 0;
canvas.addEventListener("mousedown", e => {{
  dragStartX = e.clientX;
  dragStartPan = pan;
}});
window.addEventListener("mouseup", () => {{ dragStartX = null; }});
canvas.addEventListener("mousemove", e => {{
  const rect = canvas.getBoundingClientRect();
  if (dragStartX != null) {{
    const dx = e.clientX - dragStartX;
    const visible = totalPicos / zoom;
    pan = clamp(dragStartPan - (dx / rect.width) * visible, 0, Math.max(0, totalPicos - visible));
    draw();
    tooltip.style.display = "none";
    return;
  }}
  // Hover tooltip: find op under cursor
  const px = e.clientX - rect.left;
  const py = e.clientY - rect.top;
  const h = canvas.height / dpr;
  const lhFull = (h - 10) / NUM_LANES;
  const eng = Math.floor((py - 5) / lhFull);
  if (eng < 0 || eng >= NUM_LANES) {{ tooltip.style.display = "none"; return; }}
  const t = xToTime(px);
  // Binary search would be neater; linear is fine.
  let found = null;
  for (let i = 0; i < trace.n; i++) {{
    const s = Number(trace.start[i]) - startBase;
    const d = trace.dur[i];
    if (s > t) break;
    if (s + d < t) continue;
    if (trace.engine[i] !== eng) continue;
    found = i; break;
  }}
  if (found == null) {{ tooltip.style.display = "none"; return; }}
  const s = Number(trace.start[found]) - startBase;
  const d = trace.dur[found];
  tooltip.style.display = "block";
  tooltip.style.left = (e.clientX + 14) + "px";
  tooltip.style.top  = (e.clientY + 14) + "px";
  tooltip.innerHTML = `
    <b>${{opTagName[trace.opTag[found]]}}</b><br>
    op #${{found}}<br>
    engine: ${{engineName[trace.engine[found]]}}<br>
    start: ${{fmtPs(s)}}<br>
    duration: ${{fmtPs(d)}}
  `;
}});
canvas.addEventListener("mouseleave", () => {{ tooltip.style.display = "none"; }});

document.getElementById("zoom-in").addEventListener("click", () => applyZoom(zoom * 2));
document.getElementById("zoom-out").addEventListener("click", () => applyZoom(zoom / 2));
document.getElementById("zoom-reset").addEventListener("click", () => {{ pan = 0; applyZoom(1); }});

resizeCanvas();
</script>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace_bin", help="path to the binary trace file (the sidecar is auto-located)")
    ap.add_argument("-o", "--out", help="output HTML path (default: <trace_bin>.html)")
    args = ap.parse_args()

    bin_path = Path(args.trace_bin).expanduser().resolve()
    if not bin_path.is_file():
        print(f"trace file not found: {bin_path}", file=sys.stderr)
        return 1
    meta_path = bin_path.with_name(bin_path.name + ".meta.json")
    if not meta_path.is_file():
        print(f"sidecar metadata not found: {meta_path}", file=sys.stderr)
        return 1

    bin_bytes = bin_path.read_bytes()
    meta = json.loads(meta_path.read_text())
    arr = np.frombuffer(bin_bytes, dtype=TRACE_DTYPE)

    engine_names = {e["id"]: e["name"] for e in meta.get("engines", [])}
    op_tag_names = {t["tag"]: t["name"] for t in meta.get("op_tags", [])}

    engine_summary = build_engine_summary(arr, engine_names)
    op_breakdown = build_op_breakdown(arr, op_tag_names)

    out_path = Path(args.out).expanduser().resolve() if args.out else bin_path.with_suffix(".html")
    html = render_html(
        bin_bytes=bin_bytes,
        meta=meta,
        arr=arr,
        engine_summary=engine_summary,
        op_breakdown=op_breakdown,
        trace_name=bin_path.name,
    )
    out_path.write_text(html, encoding="utf-8")
    print(f"wrote {out_path}  ({len(html)/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
