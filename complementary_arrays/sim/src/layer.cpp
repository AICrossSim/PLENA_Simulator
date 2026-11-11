// One layer window, cycle by cycle: weight delivery, array progress, HBM and port service, conversion and phase
// completion, including the drain latency and tail slot of systolic arrays.
#include "layer.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>

Layer::Layer(const Config& cfg, const Topology& topo, const Shape* shape, const std::vector<std::vector<Program>>& prog,
             const int32_t* ids, int64_t* out, int64_t* trace, int64_t trace_rows, std::vector<Decision>* log)
    : cfg(cfg),
      topo(topo),
      shape(shape),
      prog(prog),
      ids(ids),
      out(out),
      trace(trace),
      trace_rows(trace_rows),
      log(log),
      B(cfg.batch),
      nc(topo.contexts) {}

int Layer::rotated(int i) const {  // which context of a shared array is served first rotates every cycle
  if (!topo.shared) return i;
  int k = array_of(i), g = topo.nctx[k];
  return topo.ctx[k][(topo.pos[i] + int(t % g)) % g];
}

void Layer::record(int64_t when, int c, int j, int kind, int64_t predicted) {
  if (log) log->push_back({when, c, j, tasks[j].expert, int64_t(tasks[j].tokens.size()), kind, cfg.policy, predicted});
}

void Layer::start(int c, int j, int64_t when) {
  Context& s = st[c];
  s.task = j;
  s.phase = 0;
  s.start = when;
  reset_phase(s, when);
  s.count[C_reserved_bytes] = std::max(s.count[C_reserved_bytes], current(c).reserved);
}

void Layer::promote(int c) {
  int j = st[c].next_task;
  st[c].next_task = -1;
  start(c, j, t + 1);
  record(t + 1, c, j, 2, 0);
}

void Layer::finish_task(int c, int j, const Program& p) {
  double* x = cls_count[array_of(c)][task_class(j)];
  x[K_tasks]++;
  x[K_unique_weights] += p.unique;
  x[K_useful_macs] += p.useful;
  if (p.tm < int(tasks[j].tokens.size())) out[S_multi_group_tasks]++;
  record(t + 1, c, j, 4, 0);
}

void Layer::reset_phase(Context& s, int64_t when) {
  assert(s.fifo[0].empty() && s.fifo[1].empty() && s.y == 0 && s.landing.empty());
  if (s.task >= 0 && s.task == s.pre_task && s.phase == s.pre_phase) {  // the prefetched requests become this phase's
    s.hbm.swap(s.pre);
    s.pre.clear();
    s.landing.swap(s.pre_landing);
    s.pre_landing.clear();
    s.hbm_issued = s.pre_issued;
    s.pre_issued = 0;
    s.pre_task = s.pre_phase = -1;
  } else {
    s.hbm_issued = 0;
    s.hbm.clear();
  }
  s.v = 0;
  s.consumed = 0;
  s.requested = {0, 0};
  s.commit = when;
  s.delivered = 0;
}

// Re-deliveries of retained weights to the arrays, granted before the arrays consume: a fixed budget (port_mode 0), or
// the re-delivery share of the shared port (port_mode 1: both classes get their demand if it fits, else half each with
// the unused part going to the other). The grant is split equally over the arrays with demand, the remainder
// round-robin.
void Layer::deliver() {
  if (!(cfg.kappa_milli > 0 || cfg.port_mode)) return;
  int64_t D = cfg.port_mode ? 0 : int64_t((__int128(t + 1) * knum) / kden - (__int128(t) * knum) / kden);
  int64_t needc[MAXC]{}, needk[MAXA]{}, givek[MAXA]{}, tot = 0;
  for (int c = 0; c < nc; c++) {
    const Context& s = st[c];
    if (s.task < 0) continue;
    const Phase& ph = current(c).phases[s.phase];
    if (s.consumed >= int(ph.inputs.size()) || !limited(ph.inputs[s.consumed])) continue;
    int64_t want = ph.inputs[s.consumed].localw / 2 - s.delivered;
    if (want <= 0 || s.fifo[0].empty()) continue;
    const Request& f = s.fifo[0].front();
    if (f.idx != s.consumed || f.remaining || f.ready > t || f.decode) continue;
    needc[c] = want;
    needk[array_of(c)] += want;
    tot += want;
  }
  if (cfg.port_mode) {
    int64_t F = 0;
    for (int c = 0; c < nc; c++)
      if (st[c].task >= 0) F += std::min(pending(c, WPORT), intake(c));
    int64_t R = 2 * tot, Fb = 64 * F, gR = R + Fb <= cfg.W ? R : std::min(R, std::max<int64_t>(cfg.W / 2, cfg.W - Fb));
    D = gR / 2;
    if (row) {
      row[T_port_need_reread] = tot;
      row[T_port_need_first] = F;
    }
  }
  if (tot > D) out[S_delivery_limited_cycles]++;
  int64_t left = D;
  while (left > 0) {
    int act = 0;
    for (int k = 0; k < topo.arrays; k++) act += givek[k] < needk[k];
    if (!act) break;
    int64_t share = left / act;
    if (!share) {
      for (int z = 0; z < topo.arrays && left > 0; z++) {
        int k = (array_rr + z) % topo.arrays;
        if (givek[k] < needk[k]) {
          givek[k]++;
          left--;
        }
      }
      break;
    }
    for (int k = 0; k < topo.arrays; k++)
      if (givek[k] < needk[k]) {
        int64_t g = std::min(needk[k] - givek[k], share);
        givek[k] += g;
        left -= g;
      }
  }
  array_rr = (array_rr + 1) % topo.arrays;
  for (int k = 0; k < topo.arrays; k++) {
    int g = topo.nctx[k];
    int64_t gk = givek[k];
    for (int i = 0; i < g && gk > 0; i++) {
      int c = topo.ctx[k][(i + int(t % g)) % g];
      int64_t x = std::min(gk, needc[c]);
      st[c].delivered += x;
      st[c].count[C_delivered_weights] += x;
      gk -= x;
      if (row) trace_ctx(TC_port_reread_weights, c) += x;
    }
  }
  if (cfg.port_mode) {
    port_reread = 2 * (D - left);
    first_beats = (cfg.W - port_reread) / 64;
    if (row) row[T_port_first_budget] = first_beats;
  }
}

// Tail slots run their trailing vector phases first, with priority on their array's vector lanes.
void Layer::advance_tails() {
  for (int ci = 0; ci < nc; ci++) {
    int c = rotated(ci), a = array_of(c);
    Context& s = st[c];
    Tail& u = s.tail;
    if (u.task < 0) continue;
    allocated += u.buffer;
    if (u.drain_until >= 0 && t < u.drain_until) {  // the drain of the task's last matrix phase is not complete yet
      s.count[C_tail_wait]++;
      continue;
    }
    const Phase& ph = prog[c][tasks[u.task].type].phases[u.phase];
    const int id = MAXC + c;
    if (ph.kind == ROW && u.lock < 0) {
      int tok = tasks[u.task].tokens[ph.row];
      if (cfg.token_locks && locks[tok] >= 0 && locks[tok] != id) continue;
      u.lock = tok;
      if (cfg.token_locks) locks[tok] = id;
    }
    bool input = u.consumed < int(ph.inputs.size()) && ph.inputs[u.consumed].at == u.v;
    bool ready = !input || (!u.fifo.empty() && u.fifo.front().ready <= t);
    int ynew = u.v < ph.cycles ? ph.output[u.v] : 0, ycap = std::max(64, cfg.ybanks * 8 * shape[c].lanes);
    if (ready && u.y + ynew <= ycap && u.v < ph.cycles && !used_res[a][1]) {
      used_res[a][1] = true;
      if (input) {
        assert(u.fifo.front().idx == u.consumed && u.fifo.front().remaining == 0);
        u.fifo.pop_front();
        u.consumed++;
      }
      u.y += ynew;
      u.v++;
      s.count[C_tail_busy]++;
      cls_count[a][task_class(u.task)][K_busy]++;
    }
    while (int(u.fifo.size()) < cfg.vslots && u.requested < int(ph.inputs.size())) {
      int j = u.requested++, sz = ph.inputs[j].x;
      u.fifo.push_back({j, beats(sz), sz ? NEVER : t});
    }
  }
}

void Layer::advance_contexts() {
  for (int ci = 0; ci < nc; ci++) {
    int c = rotated(ci), a = array_of(c);
    Context& s = st[c];
    int64_t* n = s.count;
    if (row) {
      trace_ctx(TC_task, c) = s.task;
      trace_ctx(TC_phase, c) = s.phase;
    }
    if (s.task < 0) {
      n[C_idle]++;
      set_state(c, IDLE);
      continue;
    }
    const Program& p = current(c);
    const Phase& ph = p.phases[s.phase];
    allocated += p.reserved;
    if (ph.kind == ROW && s.lock < 0) {
      int tok = tasks[s.task].tokens[ph.row];
      if (cfg.token_locks && locks[tok] >= 0 && locks[tok] != c) {
        n[C_merge_stall]++;
        set_state(c, STALL);
        continue;
      }
      s.lock = tok;
      if (cfg.token_locks) locks[tok] = c;
    }
    bool input = s.consumed < int(ph.inputs.size()) && ph.inputs[s.consumed].at == s.v;
    bool ready = true;
    if (input)
      for (int r = 0; r < 2; r++)
        ready = ready && !s.fifo[r].empty() && s.fifo[r].front().ready <= t && s.fifo[r].front().decode == 0;
    bool dwait = input && ready && (cfg.kappa_milli > 0 || cfg.port_mode) && limited(ph.inputs[s.consumed]) &&
                 s.delivered < ph.inputs[s.consumed].localw / 2;
    if (dwait) ready = false;
    int ynew = s.v < ph.cycles ? ph.output[s.v] : 0;
    int ycap = ph.kind != MATRIX ? std::max(64, cfg.ybanks * 8 * shape[c].lanes)
                                 : std::max(64, cfg.ybanks * 4 * shape[c].m * shape[c].n);
    if (!ready) {
      n[C_input_stall]++;
      if (dwait) {
        n[C_delivery_wait]++;
        set_state(c, WAIT_REDELIVERY);
      } else if (!s.fifo[0].empty() && !landed(s, s.fifo[0].front())) {
        n[C_wait_hbm]++;
        set_state(c, WAIT_HBM);
      } else if (!s.fifo[0].empty() && s.fifo[0].front().remaining) {
        n[C_wait_wport]++;
        set_state(c, WAIT_PORT);
      } else if (!s.fifo[1].empty() && s.fifo[1].front().remaining) {
        n[C_wait_x]++;
        set_state(c, WAIT_OTHER);
      } else if (!s.fifo[0].empty() && s.fifo[0].front().decode) {
        n[C_wait_decode]++;
        set_state(c, WAIT_OTHER);
      } else {
        n[C_wait_latency]++;
        set_state(c, WAIT_OTHER);
      }
    } else if (s.y + ynew > ycap) {
      n[C_output_stall]++;
      set_state(c, STALL);
    } else if (s.v < ph.cycles && (topo.shared || (cfg.tail_slot && ph.kind != MATRIX)) && ph.kind <= ROW &&
               used_res[a][ph.kind != MATRIX]) {
      n[C_merge_stall]++;  // another context of the array took its MAC array or vector lanes this cycle
      n[C_share_stall]++;
      set_state(c, STALL);
    } else if (s.v < ph.cycles) {
      if (topo.shared && ph.kind <= ROW) used_res[a][ph.kind != MATRIX] = true;
      if (ph.kind == MATRIX) mat_adv[a] = true;
      if (input) {
        for (int r = 0; r < 2; r++) {
          assert(s.fifo[r].front().idx == s.consumed && s.fifo[r].front().remaining == 0);
          s.fifo[r].pop_front();
        }
        s.consumed++;
        s.delivered = 0;
      }
      s.y += ynew;
      s.v++;
      n[C_enabled]++;
      double* x = cls_count[a][task_class(s.task)];
      x[K_busy]++;
      if (ph.kind == MATRIX) x[K_matrix_busy]++;
      set_state(c, COMPUTE);
    } else {
      n[C_commit_wait]++;
      set_state(c, STALL);
    }
    issue_requests(c, p, ph);
  }
}

// HBM runs ahead of the weight FIFO in stage_slots landing slots per context (empty requests need none); once the
// phase's requests are issued, the same slots prefetch the next phase that reads HBM, of this task or of the reserved
// next one.
void Layer::issue_requests(int c, const Program& p, const Phase& ph) {
  Context& s = st[c];
  const int ninputs = int(ph.inputs.size());
  if (s.hbm.empty()) {
    s.hbm.resize(ninputs);
    for (Request& q : s.hbm) {
      q.remaining = 1;
      q.ready = NEVER;
    }
  }
  while (s.hbm_issued < ninputs) {
    int sz = ph.inputs[s.hbm_issued].w;
    if (int(s.landing.size() + s.pre_landing.size()) >= cfg.stage_slots && sz) break;
    int j = s.hbm_issued++;
    s.hbm[j] = {j, beats(sz), sz ? NEVER : t};
    if (sz) s.landing.push_back(j);
  }
  if (s.hbm_issued == ninputs && s.pre_task < 0) {
    int f = p.next_hbm[s.phase + 1];
    if (f >= 0) {
      s.pre_task = s.task;
      s.pre_phase = f;
    } else if (s.next_task >= 0 && (f = prog[c][tasks[s.next_task].type].next_hbm[0]) >= 0) {
      s.pre_task = s.next_task;
      s.pre_phase = f;
    }
  }
  if (s.hbm_issued == ninputs && s.pre_task >= 0) {
    const Phase& nx = prog[c][tasks[s.pre_task].type].phases[s.pre_phase];
    if (s.pre.empty()) {
      s.pre.resize(nx.inputs.size());
      for (Request& q : s.pre) {
        q.remaining = 1;
        q.ready = NEVER;
      }
      s.pre_issued = 0;
    }
    while (s.pre_issued < int(nx.inputs.size()) && int(s.landing.size() + s.pre_landing.size()) < cfg.stage_slots) {
      int j = s.pre_issued++, sz = nx.inputs[j].w;
      out[s.pre_task == s.task ? S_prefetch_same_task_bytes : S_prefetch_next_task_bytes] += sz;
      s.pre[j] = {j, beats(sz), sz ? NEVER : t};
      if (sz) s.pre_landing.push_back(j);
    }
  }
  for (int r = 0; r < 2; r++)
    while (int(s.fifo[r].size()) < (ph.kind != MATRIX ? cfg.vslots : cfg.slots) && s.requested[r] < ninputs) {
      int j = s.requested[r];
      if (ph.inputs[j].release > s.consumed) break;
      s.requested[r]++;
      int sz = r == 0 ? ph.inputs[j].w : ph.inputs[j].x;
      s.fifo[r].push_back({j, beats(sz), sz ? NEVER : t, r == 0 ? ph.inputs[j].decode : 0});
    }
}

int Layer::pending(int c, int r) const {  // beats a context wants from server r
  const Context& s = st[c];
  if (r == HBM) {
    int n = 0;
    for (int j : s.landing) n += s.hbm[j].remaining;
    for (int j : s.pre_landing) n += s.pre[j].remaining;
    return n;
  }
  if (r == YPORT) return s.y / 64 + (s.v >= current(c).phases[s.phase].cycles && s.y % 64 ? 1 : 0);
  int sum = 0;
  for (const Request& q : s.fifo[r])
    if (r != WPORT || landed(s, q)) sum += q.remaining;
  return sum;
}

// Several contexts per array or more than two contexts: equal shares, the remainder round-robin; with core_caps a
// shared array's weight intake is shared by its contexts.
void Layer::split_equal(int r, const std::array<int, MAXC>& need, std::array<int, MAXC>& give, int budget) {
  std::array<int, MAXA> left;
  left.fill(1 << 30);
  auto capped = [&](int c) { return r == WPORT && topo.caps[array_of(c)]; };
  for (int c = 0; c < nc; c++)
    if (capped(c)) left[array_of(c)] = intake(c);
  int act = 0;
  for (int c = 0; c < nc; c++) act += need[c] > 0;
  if (!act) return;
  int share = budget / act;
  for (int c = 0; c < nc; c++) {
    int g = std::min(need[c], share);
    if (capped(c)) g = std::min(g, left[array_of(c)]);
    give[c] = g;
    left[array_of(c)] -= capped(c) ? g : 0;
    budget -= g;
  }
  for (bool more = true; budget && more;) {
    more = false;
    for (int z = 0; z < nc && budget; z++) {
      int c = (rr[r] + z) % nc;
      if (give[c] < need[c] && (!capped(c) || left[array_of(c)] > 0)) {
        give[c]++;
        if (capped(c)) left[array_of(c)]--;
        budget--;
        more = true;
      }
    }
  }
  rr[r] = (rr[r] + 1) % nc;
}

// Two contexts on their own arrays: equal pairs, then alternating single beats while both want more.
void Layer::split_paired(int r, const std::array<int, MAXC>& need, std::array<int, MAXC>& give, int budget) {
  int a = rr[r], b = 1 - a;
  int paired = std::min({need[a], need[b], budget / 2});
  give[a] = give[b] = paired;
  budget -= 2 * paired;
  while (budget) {
    if (give[a] < need[a]) {
      int n = need[b] > give[b] ? 1 : std::min(budget, need[a] - give[a]);
      give[a] += n;
      budget -= n;
      rr[r] = b;
    } else if (give[b] == need[b])
      break;
    if (budget && give[b] < need[b]) {
      int n = need[a] > give[a] ? 1 : std::min(budget, need[b] - give[b]);
      give[b] += n;
      budget -= n;
      rr[r] = a;
    }
  }
}

void Layer::serve(int c, int r, int b) {
  if (!b) return;
  Context& s = st[c];
  if (r == HBM) {  // the current phase first, spare beats to the prefetched phase
    for (int j : s.landing) {
      Request& q = s.hbm[j];
      if (!q.remaining) continue;
      int n = std::min(b, q.remaining);
      q.remaining -= n;
      b -= n;
      s.count[C_hbm_service] += 64LL * n;
      if (row) trace_ctx(TC_hbm_current, c) += 64LL * n;
      if (!q.remaining) q.ready = t + 1 + cfg.hlatency;
      if (!b) break;
    }
    if (b)
      for (int j : s.pre_landing) {
        Request& q = s.pre[j];
        if (!q.remaining) continue;
        int n = std::min(b, q.remaining);
        q.remaining -= n;
        b -= n;
        s.count[C_hbm_service] += 64LL * n;
        if (row) {
          trace_ctx(TC_hbm_prefetch, c) += 64LL * n;
          trace_ctx(TC_prefetch_task, c) = s.pre_task;
          trace_ctx(TC_prefetch_phase, c) = s.pre_phase;
        }
        if (!q.remaining) q.ready = t + 1 + cfg.hlatency;
        if (!b) break;
      }
    return;
  }
  if (r == YPORT) {
    s.y -= std::min(s.y, b * 64);
    s.commit = t + 1 + cfg.latency;
    return;
  }
  for (Request& q : s.fifo[r]) {
    if (!q.remaining || (r == WPORT && !landed(s, q))) continue;
    int n = std::min(b, q.remaining);
    q.remaining -= n;
    b -= n;
    if (!q.remaining) {
      q.ready = t + 1 + cfg.latency;
      if (r == WPORT && !s.landing.empty() && s.landing.front() == q.idx)
        s.landing.pop_front();  // frees its landing slot
    }
    if (!b) break;
  }
}

void Layer::serve_tail(int c, int r, int b) {
  Tail& u = st[c].tail;
  if (r == YPORT) {
    u.y -= std::min(u.y, b * 64);
    u.commit = t + 1 + cfg.latency;
    return;
  }
  for (Request& q : u.fifo) {
    if (!q.remaining) continue;
    int n = std::min(b, q.remaining);
    q.remaining -= n;
    b -= n;
    if (!q.remaining) q.ready = t + 1 + cfg.latency;
    if (!b) break;
  }
}

void Layer::serve_ports() {
  for (int r = 0; r < 4; r++) {
    std::array<int, MAXC> need{}, give{}, tneed{};
    int budget = r == HBM ? int(((t + 1) * cfg.H) / 64 - (t * cfg.H) / 64)
                          : (r == WPORT   ? cfg.W
                             : r == XPORT ? cfg.X
                                          : cfg.Y) /
                                64;
    if (cfg.port_mode && r == WPORT && first_beats >= 0) budget = int(first_beats);
    for (int c = 0; c < nc; c++)
      if (st[c].task >= 0) need[c] = r == WPORT ? std::min(pending(c, r), intake(c)) : pending(c, r);
    if (cfg.tail_slot && (r == XPORT || r == YPORT))  // a tail slot's reads and writes join its context's demand
      for (int c = 0; c < nc; c++) {
        const Tail& u = st[c].tail;
        if (u.task < 0) continue;
        if (r == XPORT)
          for (const Request& q : u.fifo) tneed[c] += q.remaining;
        else
          tneed[c] = u.y / 64 + (u.v >= prog[c][tasks[u.task].type].phases[u.phase].cycles && u.y % 64 ? 1 : 0);
        need[c] += tneed[c];
      }
    if (nc > 2 || topo.anycaps)
      split_equal(r, need, give, budget);
    else
      split_paired(r, need, give, budget);
    int gsum = 0;
    for (int c = 0; c < nc; c++) gsum += give[c];
    if (cfg.port_mode && r == WPORT) {
      int64_t used = 64LL * gsum + port_reread;
      assert(used <= cfg.W);
      out[S_port_bytes] += used;
      if (used + 64 > cfg.W) out[S_port_saturated_cycles]++;
      if (row) row[T_port_used] = used;
    }
    if (row) {
      const int field[4] = {TC_port_first_bytes, TC_x_bytes, TC_y_bytes, TC_hbm_bytes};
      for (int c = 0; c < nc; c++) trace_ctx(field[r], c) += 64LL * give[c];
    }
    if (r == HBM) {
      bool active = false;
      for (int c = 0; c < nc; c++) active |= st[c].task >= 0;
      if (active && !gsum) out[S_hbm_idle_cycles]++;
    }
    for (int c = 0; c < nc; c++) {
      if (tneed[c] && give[c]) {  // the tail slot is served first from its context's grant
        int g = std::min(give[c], tneed[c]);
        give[c] -= g;
        serve_tail(c, r, g);
      }
    }
    for (int c = 0; c < nc; c++) serve(c, r, give[c]);
  }
}

// One converter decodes landed low-bit records into BF16, at most 64 elements per request and cycle, alternating over
// the contexts; an array consumes at most n k / m weights per cycle (with core_caps shared by its contexts).
void Layer::convert() {
  if (cfg.bits >= 16) return;
  int budget = cfg.converter_rate;
  std::array<int, MAXC> cap{};
  std::array<int, MAXA> capk{};
  for (int c = 0; c < nc; c++) cap[c] = shape[c].n * shape[c].k / shape[c].m;
  for (int c = 0; c < nc; c++)
    if (topo.caps[array_of(c)]) capk[array_of(c)] = shape[c].n * shape[c].k / shape[c].m;
  int first = int(t % nc);
  for (bool progress = true; budget && progress;) {
    progress = false;
    for (int z = 0; z < nc && budget; z++) {
      int c = (first + z) % nc;
      Context& s = st[c];
      int& lim = topo.caps[array_of(c)] ? capk[array_of(c)] : cap[c];
      if (s.task < 0 || !lim) continue;
      for (Request& q : s.fifo[0]) {
        if (q.remaining || q.ready > t || !q.decode) continue;
        int n = std::min({budget, lim, q.decode, 64});
        q.decode -= n;
        lim -= n;
        budget -= n;
        progress = true;
        if (row) trace_ctx(TC_decoded, c) += n;
        if (!q.decode) q.ready = t + 1 + cfg.converter_latency;
        break;
      }
    }
  }
}

void Layer::complete_tails() {
  for (int c = 0; c < nc; c++) {
    Context& s = st[c];
    Tail& u = s.tail;
    if (u.task < 0) continue;
    const Program& p = prog[c][tasks[u.task].type];
    const Phase& ph = p.phases[u.phase];
    if (!(u.v >= ph.cycles && u.y == 0 && t + 1 >= u.commit)) continue;
    assert(u.consumed == int(ph.inputs.size()) && u.fifo.empty() && ph.kind != MATRIX && ph.bytes[0] == 0 &&
           ph.localw == 0);
    if (ph.kind == ROW) {
      if (cfg.token_locks) {
        assert(locks[u.lock] == MAXC + c);
        locks[u.lock] = -1;
      }
      u.lock = -1;
      out[S_rows]++;
    }
    if (++u.phase == int(p.phases.size())) {
      if (u.task == 0) router_done = true;
      finished[u.task] = 1;
      finish_task(c, u.task, p);
      s.count[C_finish] = std::max(s.count[C_finish], t + 1);
      done++;
      u.task = -1;
      u.buffer = 0;
    } else {
      u.v = u.consumed = u.requested = 0;
      u.commit = t + 1;
    }
  }
}

// A phase completes when its virtual cycles are done and its outputs are written and committed. With a drain latency a
// dependent next phase starts only when the outputs are complete (meanwhile the context does not claim its array); an
// independent tile stream starts at once and the pending drain carries over. In tail mode the trailing phases after the
// last matrix phase go to the tail slot, if its output buffer fits next to the context's largest program.
void Layer::complete_phases() {
  for (int c = 0; c < nc; c++) {
    Context& s = st[c];
    int64_t* n = s.count;
    if (s.task < 0) continue;
    const Program& p = current(c);
    const Phase& ph = p.phases[s.phase];
    if (!(s.v >= ph.cycles && s.y == 0 && t + 1 >= s.commit)) continue;
    const bool atlast = cfg.tail_slot && s.phase == p.lastmat;
    const bool fits = atlast && topo.maxres[c] + 4LL * p.tm * cfg.hidden <= topo.sram[c];
    if (atlast && !fits && !s.fallback) {
      s.fallback = true;
      int64_t ready = std::max(s.drain_ready, p.tail_drain > 0 ? t + 1 + p.tail_drain : int64_t(0));
      if (ready > t + 1 && s.drain_until < 0) {
        s.drain_until = ready;
        n[C_drain_waits]++;
      }
    }
    if (!atlast && (ph.lat > 0 || s.drain_ready > t + 1) && s.drain_until < 0) {
      int64_t ready = std::max(s.drain_ready, t + 1 + ph.lat);
      if (ph.hold) {
        s.drain_until = ready;
        n[C_drain_waits]++;
      } else {
        s.drain_ready = ready;
        n[C_independent_phases]++;
      }
    }
    if (s.drain_until >= 0 && t + 1 < s.drain_until) {
      n[C_drain_wait]++;
      if (s.phase == p.lastmat) n[C_drain_wait_last]++;
      continue;
    }
    if (s.drain_until >= 0) {
      s.drain_until = -1;
      s.drain_ready = 0;
    } else if (ph.hold)
      s.drain_ready = 0;
    if (atlast && fits && s.tail.task >= 0) {  // the tail slot still holds the previous task's trailing phases
      n[C_tail_blocked]++;
      continue;
    }
    if (atlast && !fits) {
      out[S_tail_fallbacks]++;
      s.fallback = false;
    }
    assert(s.consumed == int(ph.inputs.size()));
    double* x = cls_count[array_of(c)][task_class(s.task)];
    x[K_decoded] += ph.decoded;
    x[K_hbm_bytes] += ph.bytes[0];
    x[K_wport_bytes] += ph.bytes[0];
    x[K_expanded_write_bytes] += ph.expanded;
    x[K_local_read_bytes] += ph.localw;
    x[K_reread_bytes] += ph.reread;
    if (ph.kind == ROW) {
      if (cfg.token_locks) {
        assert(locks[s.lock] == c);
        locks[s.lock] = -1;
      }
      s.lock = -1;
      out[S_rows]++;
    }
    s.phase++;
    if (atlast && fits) {
      Tail& u = s.tail;
      u.task = s.task;
      u.phase = s.phase;
      u.v = u.consumed = u.y = u.requested = 0;
      u.lock = -1;
      u.commit = t + 1;
      u.buffer = 4LL * p.tm * cfg.hidden;
      int64_t ready = std::max(s.drain_ready, p.tail_drain > 0 ? t + 1 + p.tail_drain : int64_t(0));
      u.drain_until = ready > t + 1 ? ready : -1;
      s.drain_ready = 0;
      n[C_tail_handoffs]++;
      assert(u.phase < int(p.phases.size()) && u.fifo.empty());
      record(t + 1, c, s.task, 6, 0);
      s.task = -1;
      if (s.next_task >= 0) promote(c);
    } else if (s.phase == int(p.phases.size())) {
      if (s.task == 0) router_done = true;
      finished[s.task] = 1;
      assert(s.drain_ready == 0 && s.drain_until < 0);
      finish_task(c, s.task, p);
      s.task = -1;
      n[C_finish] = t + 1;
      done++;
      if (s.next_task >= 0) promote(c);
    } else
      reset_phase(s, t + 1);
  }
}

void Layer::write_outputs() {
  out[S_status] = 0;
  out[S_cycles] = t;
  out[S_startup] = startup;
  int64_t hbm = 0;
  for (int c = 0; c < MAXC; c++) {
    if (c >= nc) {
      out[OUT_CONTEXT + C_core * MAXC + c] = -1;
      continue;
    }
    int64_t* n = st[c].count;
    assert(n[C_enabled] + n[C_input_stall] + n[C_output_stall] + n[C_merge_stall] + n[C_commit_wait] + n[C_idle] ==
           t - startup);
    n[C_core] = array_of(c);
    n[C_sram_bytes] = topo.sram[c];
    for (int f = 0; f < NCONTEXT_OUTPUT; f++) out[OUT_CONTEXT + f * MAXC + c] = n[f];
    hbm += n[C_hbm_service];
  }
  assert(hbm <= 64 * ((t * cfg.H) / 64));
  (void)hbm;
  for (int k = 0; k < topo.arrays; k++) out[OUT_ARRAY + A_lanes * MAXA + k] = shape[topo.first[k]].lanes;
  for (int k = 0; k < MAXA; k++)
    for (int q = 0; q < NCLASS; q++)
      for (int f = 0; f < NCLASS_OUTPUT; f++)
        out[OUT_CLASS + (f * MAXA + k) * NCLASS + q] = llround(cls_count[k][q][f]);
  int routed = 0;
  for (const Task& task : tasks) routed += task.expert >= 0 && task.expert < cfg.experts;
  const int H = cfg.hidden, F = cfg.routed_width, S = cfg.shared_width;
  out[S_weight_bytes] = stored_bytes(cfg, cfg.experts, H, 16) +
                        routed * (stored_bytes(cfg, 2 * F, H, cfg.bits) + stored_bytes(cfg, H, F, cfg.bits)) +
                        stored_bytes(cfg, 2 * S + cfg.shared_gate, H, cfg.bits) + stored_bytes(cfg, H, S, cfg.bits);
  assert(out[S_rows] == int64_t(B) * (cfg.topk + 1));
  for (int l : locks) assert(l == -1);
}

void Layer::run() {
  std::fill(out, out + NOUT, 0);
  if (!build_tasks()) {
    out[S_status] = 1;
    return;
  }
  locks.assign(B, -1);
  finished.assign(tasks.size(), 0);
  coldp = int(tasks.size()) - 1;
  const int H = cfg.hidden;
  const int64_t token_buffers =
      6LL * B * H;  // FP32 outputs and BF16 activations of every token, shared by the contexts
  startup =
      std::max(cdiv(B * H, VECTOR_LANES), cdiv(beats(4 * B * H), cfg.Y / 64)) + cfg.latency;  // zeroes the FP32 outputs
  t = startup;
  out[S_peak_sram] = token_buffers;
  if (cfg.kappa_milli > 0) quota_rate(cfg, knum, kden);
  int rc = -1;
  double best = 1e300;
  for (int c = 0; c < nc; c++)
    if (prog[c][B + 1].tm && prog[c][B + 1].suffix[0] < best) {
      best = prog[c][B + 1].suffix[0];
      rc = c;
    }
  if (rc < 0) {
    out[S_status] = 1;
    return;
  }
  start(rc, 0, t);
  record(t, rc, 0, 3, icost(rc, B + 1));
  int64_t last_progress = t, previous = -1;
  while (done < int(tasks.size())) {
    int64_t progress = done;
    for (int c = 0; c < nc; c++) {
      const int64_t* n = st[c].count;
      progress += n[C_enabled] + n[C_hbm_service] + n[C_delivered_weights] + n[C_tail_busy];
    }
    if (progress != previous) {
      last_progress = t;
      previous = progress;
    }
    if (t - last_progress > std::max(100000, cfg.hlatency + 10000)) {
      std::fprintf(stderr, "deadlock: no progress since cycle %lld (batch %d, %d of %zu tasks done)\n",
                   (long long)last_progress, B, done, tasks.size());
      out[S_status] = 2;
      return;
    }
    row = trace && t - startup < trace_rows ? trace + (t - startup) * NTRACE : nullptr;
    if (row) {
      row[T_cycle] = t;
      for (int c = 0; c < MAXC; c++) trace_ctx(TC_prefetch_task, c) = trace_ctx(TC_prefetch_phase, c) = -1;
    }
    if (cfg.policy == REGRET) fill_window();
    dispatch();
    reserve_next();
    deliver();
    allocated = token_buffers;
    std::memset(used_res, 0, sizeof used_res);
    std::memset(mat_adv, 0, sizeof mat_adv);
    if (cfg.tail_slot) advance_tails();
    advance_contexts();
    out[S_peak_sram] = std::max(out[S_peak_sram], allocated);
    if (cfg.drain_inner || cfg.drain_last)  // array idle while one of its contexts waits for a drain
      for (int k = 0; k < topo.arrays; k++) {
        if (mat_adv[k]) continue;
        for (int i = 0; i < topo.nctx[k]; i++) {
          const Context& q = st[topo.ctx[k][i]];
          if (q.drain_until >= 0 && t < q.drain_until) {
            out[OUT_ARRAY + A_drain_idle * MAXA + k]++;
            break;
          }
        }
      }
    serve_ports();
    convert();
    if (cfg.tail_slot) complete_tails();
    complete_phases();
    t++;
  }
  write_outputs();
}
