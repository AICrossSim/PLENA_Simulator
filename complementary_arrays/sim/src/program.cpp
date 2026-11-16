// Compilation of one task for one context: weight records, matrix and vector phases, SRAM reservation and the estimates
// the dispatcher ranks tasks by.
#include <algorithm>
#include <cmath>

#include "model.hpp"

namespace {

struct Packed {
  int bytes, elements;
};

// FP16 scales of one record: 16 rows by group / scale_group scale groups (none for BF16 or out-of-record scales).
int scale_bytes(const Config& cfg, int bits) {
  if (bits == 16 || cfg.scale_group < 0) return 0;
  return 32 * (cfg.scale_group > 0 ? cfg.group / cfg.scale_group : 1);
}

// Stored weights: records of 16 rows by one quantization group along K, holding the values and their scales, padded to
// whole beats.
Packed pack_panel(const Config& cfg, int n0, int nr, int k0, int kr, int bits) {
  int nn = cdiv(n0 + nr, 16) - n0 / 16, gg = cdiv(k0 + kr, cfg.group) - k0 / cfg.group;
  int values = 16 * cfg.group * bits / 8, record = 64 * beats(values + scale_bytes(cfg, bits));
  return {nn * gg * record, nn * gg * 16 * cfg.group};
}

// M x N x K product in passes of m token rows over N panels of np columns. A pass streams the panel's K/k chunks
// through the array; the first pass brings each chunk as stored records from HBM over the weight port and the
// converter, later passes re-deliver the retained BF16 panel. A panel waits until the inputs of the panel `buffers`
// back are consumed. With packed re-reads the retained panel stays in stored form: every pass reads its records over
// the weight port and decodes them into one of `sbanks` streaming banks, which a chunk reuses once the chunk `sbanks`
// back is consumed.
Phase matrix(const Config& cfg, const Shape& c, int M, int N, int K, int weights, int np, bool partial, int buffers,
             int n_offset, int k_offset, int sbanks) {
  Phase p;
  p.kind = MATRIX;
  p.partial = partial;
  const bool bf16 = weights == ROUTER_WEIGHTS, packed = cfg.packed && !bf16 && cfg.bits < 16;
  int start = 0, L = cdiv(K, c.k) * c.m;
  std::vector<int> panel_ends, chunk_ends;
  for (int j = 0; j < N; j += np) {
    int release = int(panel_ends.size()) >= buffers ? panel_ends[panel_ends.size() - buffers] : 0;
    for (int i = 0; i < M; i += c.m) {
      int mr = std::min(c.m, M - i), nr = std::min(np, N - j);
      if (partial)  // FP32 partial sums, one row per cycle
        for (int r = 0; r < mr; r++) {
          p.inputs.push_back({start++, 0, 4 * nr, release});
          p.bytes[1] += 4 * nr;
        }
      for (int t = 0; t < L; t++) {
        int base = (t / c.m) * c.k + t % c.m;
        int q = std::min(c.k / c.m, std::max(0, cdiv(K - base, c.m)));
        Input in{start + t, 0, 2 * mr * q, release, 2 * nr * q};
        if (t % c.m == 0 && (i == 0 || packed)) {
          int ks = (t / c.m) * c.k, kr = std::min(c.k, K - ks), bits = bf16 ? 16 : cfg.bits;
          Packed pk = pack_panel(cfg, n_offset + j, nr, k_offset + ks, kr, bits);
          if (i == 0) p.regions.push_back({int(p.inputs.size()), weights, n_offset + j, nr, k_offset + ks, kr});
          in.w = pk.bytes;
          in.hbm = i == 0 ? pk.bytes : 0;
          in.decode = bits < 16 ? pk.elements : 0;
          p.expanded += 2LL * pk.elements;
          if (packed && int(chunk_ends.size()) >= sbanks)
            in.release = std::max(in.release, chunk_ends[chunk_ends.size() - sbanks]);
          chunk_ends.push_back(int(p.inputs.size()) + std::min(c.m, L - t));
        }
        in.reuse = i > 0;
        in.packed = packed && in.reuse;
        p.inputs.push_back(in);
        p.bytes[0] += in.w;
        p.bytes[1] += in.x;
        p.hbm += in.hbm;
        p.localw += in.localw;
        p.decoded += in.decode;
        if (in.reuse) p.reread += in.localw;
        if (in.reuse && !in.packed) p.reread_share += in.localw;
        if (in.packed) p.packed_reread += in.w;
        p.mac_inputs++;
        p.useful_macs += int64_t(mr) * nr * q;
        p.empty_row_slots += int64_t(c.m - mr) * c.n * (c.k / c.m);
      }
      int origin = start + L + 2 * c.m - 2 + c.depth() + 2 + (partial ? 1 : 0);
      if (int(p.output.size()) < origin + c.m) p.output.resize(origin + c.m);
      for (int r = 0; r < mr; r++) {
        p.output[origin + r] += 4 * nr;
        p.bytes[2] += 4 * nr;
      }
      start += L + (cfg.tile_bubble ? 1 : 0);
    }
    panel_ends.push_back(int(p.inputs.size()));
  }
  p.cycles = int(p.output.size());
  return p;
}

// Vector phase on the array's lanes. With a coefficient, each token's FP32 scale is read before its first chunk.
Phase vector(const Shape& c, int elements, int x, int y, bool coefficient, int row = -1) {
  Phase p;
  p.kind = row < 0 ? VECTOR : ROW;
  p.row = row;
  int work = std::max(1, cdiv(elements, c.lanes));
  p.cycles = work + VECTOR_LATENCY;
  p.output.resize(p.cycles);
  for (int j = 0; j < work; j++) {
    int a = std::min(elements, j * c.lanes), b = std::min(elements, (j + 1) * c.lanes);
    int xx = 2 * int((int64_t(x / 2) * b) / elements - (int64_t(x / 2) * a) / elements);
    if (coefficient) {
      int rows = (x - y) / 4, width = elements / rows;
      xx = 4 * (b - a) + 4 * (cdiv(b, width) - cdiv(a, width));
    }
    p.inputs.push_back({j, 0, xx});
    p.output[j + VECTOR_LATENCY] = 2 * int((int64_t(y / 2) * b) / elements - (int64_t(y / 2) * a) / elements);
  }
  p.bytes = {0, x, y};
  return p;
}

}  // namespace

int64_t stored_bytes(const Config& cfg, int n, int k, int bits) { return pack_panel(cfg, 0, n, 0, k, bits).bytes; }

void quota_rate(const Config& cfg, int64_t& knum, int64_t& kden) {
  int g = cfg.group, values = 16 * g * cfg.bits / 8, record = 64 * beats(values + scale_bytes(cfg, cfg.bits));
  knum = int64_t(cfg.kappa_milli) * cfg.H * 16 * g;
  kden = 1000LL * record;
}

Program compile(const Config& cfg, const Shape& c, int m, TaskKind kind, int n, bool gate, int cap) {
  const bool router = kind == ROUTER, shared = kind == SHARED;
  const int H = cfg.hidden, np = c.n, bits = router ? 16 : cfg.bits;
  gate = shared && gate;
  const bool packed = cfg.packed && !router && cfg.bits < 16;
  const int sbanks = std::max(2, cdiv(cfg.slots, c.m) + 1);  // streaming banks of packed re-reads
  const int conv = cfg.converter && cfg.bits < 16 ? converter_lanes(cfg, c) : cfg.converter_rate;
  Program p;
  p.tm = m;
  // SRAM with tm token rows and retention rm: 1 / 2 one or two full-K panels, 4 / 5 one or two K panels of k
  auto used = [&](int tm, int rm) -> int64_t {
    int kp = rm >= 4 ? std::min(std::max(H, n), c.k) : std::max(H, n), banks = rm == 2 || rm == 5 ? 2 : 1;
    int nn = 16 * cdiv(np + (np % 16 ? 15 : 0), 16),
        kk = cfg.group * cdiv(c.k + (c.k % cfg.group ? cfg.group - 1 : 0), cfg.group),
        kg = cfg.group * cdiv(kp + (c.k % cfg.group ? cfg.group - 1 : 0), cfg.group);
    int64_t retained =
        packed ? banks * int64_t(pack_panel(cfg, 0, nn, 0, kg, cfg.bits).bytes) + int64_t(sbanks) * nn * kk * 2
               : banks * 2LL * nn * kg;
    return 4LL * tm * H + 10LL * tm * n + (gate ? 4LL * tm : 0) +
           (cfg.slots + cfg.stage_slots) * int64_t(pack_panel(cfg, 0, nn, 0, kk, bits).bytes) +
           cfg.slots * std::max(2LL * c.k, 4LL * np) + retained + cfg.ybanks * 4LL * c.m * np +
           16LL * cfg.vslots * c.lanes + (rm >= 4 ? 4LL * tm * np : 0) + int64_t(cfg.converter_latency) * conv * 6;
  };
  auto choose = [&](int tm) {
    for (int rm : {2, 1, 5, 4})
      if (used(tm, rm) <= cap) return rm;
    return -1;
  };
  int rm = choose(p.tm);
  while (p.tm && rm < 0) {
    p.tm = p.tm == 1 ? 0 : std::max(1, p.tm / 2);
    rm = choose(p.tm);
  }
  if (!p.tm) return p;
  p.reserved = used(p.tm, rm);
  p.unique = router ? int64_t(H) * cfg.experts : int64_t(2 * n + int(gate)) * H + int64_t(H) * n;
  if (cfg.control_cycles) {
    Phase ctl;
    ctl.kind = CONTROL;
    ctl.cycles = cfg.control_cycles;
    ctl.output.resize(ctl.cycles);
    p.phases.push_back(ctl);
  }
  p.useful = router ? double(m) * H * cfg.experts : 3.0 * m * H * n + (gate ? double(m) * H : 0);
  auto emit = [&](int M, int N, int K, int weights) {
    if (rm < 4) {
      p.phases.push_back(matrix(cfg, c, M, N, K, weights, np, false, rm, 0, 0, sbanks));
      return;
    }
    // K panels: each drains before the next, which reloads the partial sums
    for (int j = 0; j < N; j += np)
      for (int kk = 0; kk < K; kk += c.k)
        p.phases.push_back(
            matrix(cfg, c, M, std::min(np, N - j), std::min(c.k, K - kk), weights, np, kk > 0, rm - 3, j, kk, sbanks));
  };
  for (int row = 0; row < m; row += p.tm) {
    int M = std::min(p.tm, m - row);
    p.passes += cdiv(M, c.m);
    if (router) {
      emit(M, cfg.experts, H, ROUTER_WEIGHTS);
      p.phases.push_back(
          vector(c, M * cfg.experts * (4 + cfg.topk), 4 * M * cfg.experts * (4 + cfg.topk), 4 * M * cfg.topk, false));
      continue;
    }
    emit(M, 2 * n + int(gate), H, GATE_UP);                                                 // gate and up projections
    p.phases.push_back(vector(c, 2 * M * n, 8 * M * n, 2 * M * n, false));                  // SiLU(gate) x up
    emit(M, H, n, DOWN);                                                                    // down projection
    if (!shared) p.phases.push_back(vector(c, M * H, 4 * M * H + 4 * M, 4 * M * H, true));  // routing weight
    if (gate) {
      p.phases.push_back(vector(c, M, 4 * M, 4 * M, false));  // sigmoid of the shared-expert gate
      p.phases.push_back(vector(c, M * H, 4 * M * H + 4 * M, 4 * M * H, true));
    }
    for (int i = 0; i < M; i++)
      p.phases.push_back(vector(c, H, 8 * H, 4 * H, false, row + i));  // combine into each token
  }
  for (size_t i = 0; i < p.phases.size(); i++)
    if (p.phases[i].kind == MATRIX) p.lastmat = int(i);
  if (cfg.drain_inner || cfg.drain_last)
    for (size_t i = 0; i < p.phases.size(); i++) {
      Phase& ph = p.phases[i];
      if (ph.kind != MATRIX) continue;
      ph.hold = !(i + 1 < p.phases.size() && p.phases[i + 1].kind == MATRIX && !p.phases[i + 1].partial);
      bool last = int(i) == p.lastmat;
      if (last && cfg.tail_slot) {
        p.tail_drain = cfg.drain_last ? c.drain : 0;
        continue;
      }
      if (last ? cfg.drain_last : cfg.drain_inner) ph.lat = c.drain;
    }
  // Estimates: the slowest of the phase's own resources; re-deliveries at the quota, or at the port share a first pass
  // at the HBM rate leaves (packed re-reads are weight-port bytes, not re-deliveries).
  int64_t knum = 0, kden = 1;
  if (cfg.kappa_milli > 0) quota_rate(cfg, knum, kden);
  if (cfg.port_mode) {
    knum = std::max(1, cfg.W - cfg.H);
    kden = 2;
  }
  const size_t np1 = p.phases.size() + 1;
  p.suffix.assign(np1, 0.0);
  p.wsuffix.assign(np1, 0);
  p.hsuffix.assign(np1, 0);
  p.rsuffix.assign(np1, 0);
  for (int i = int(p.phases.size()) - 1; i >= 0; i--) {
    Phase& ph = p.phases[i];
    double decode =
        cfg.converter == -1 ? 0.0 : double(ph.decoded) / std::max(1, cfg.converter ? conv : cfg.converter_rate);
    ph.estimate = 8 + std::max({double(ph.cycles), double(ph.bytes[0]) / cfg.W, double(ph.bytes[1]) / cfg.X,
                                double(ph.bytes[2]) / cfg.Y, double(ph.hbm) / cfg.H, decode});
    if (cfg.kappa_milli > 0 || cfg.port_mode)
      ph.estimate = std::max(ph.estimate, 8 + double(ph.reread_share / 2) * double(kden) / double(knum));
    if (ph.lat > 0 && ph.hold) ph.estimate += ph.lat;
    if (cfg.tail_slot && i > p.lastmat) ph.estimate = 0;
    p.suffix[i] = p.suffix[i + 1] + ph.estimate;
    p.wsuffix[i] = p.wsuffix[i + 1] + ph.bytes[0];
    p.hsuffix[i] = p.hsuffix[i + 1] + ph.hbm;
    p.rsuffix[i] = p.rsuffix[i + 1] + ph.reread;
  }
  p.icost = llround(p.suffix[0]);
  p.next_hbm.assign(np1, -1);
  for (int i = int(p.phases.size()) - 1; i >= 0; i--) {
    bool hbm = false;
    for (const Input& in : p.phases[i].inputs)
      if (in.hbm > 0) {
        hbm = true;
        break;
      }
    p.next_hbm[i] = hbm ? i : p.next_hbm[i + 1];
  }
  return p;
}
