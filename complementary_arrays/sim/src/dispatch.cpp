// Dispatch: the front-end task order and the policies that hand tasks to free contexts and reserve next tasks.
#include <cmath>
#include <numeric>

#include "layer.hpp"

// Tasks: the router, every routed expert with tokens and the shared expert (or its two halves), in front-end order.
bool Layer::build_tasks() {
  const int E = cfg.experts, K = cfg.topk;
  for (int e = 0; e < E; e++) {
    Task a{e, 0, {}};
    for (int i = 0; i < B; i++)
      for (int k = 0; k < K; k++)
        if (ids[i * K + k] == e) a.tokens.push_back(i);
    if (!a.tokens.empty()) {
      a.type = int(a.tokens.size()) - 1;
      tasks.push_back(a);
    }
  }
  if (cfg.shared_expert) {
    Task shared{E, B, std::vector<int>(B)};
    std::iota(shared.tokens.begin(), shared.tokens.end(), 0);
    tasks.push_back(shared);
    if (cfg.shared_place == 3) {
      shared.type = B + 2;
      tasks.push_back(shared);
    }
  }
  for (const Task& task : tasks) {
    bool fits = false;
    for (int c = 0; c < nc; c++) fits |= prog[c][task.type].tm > 0;
    if (!fits) return false;
  }
  if (cfg.policy == REGRET)  // longest first
    std::stable_sort(tasks.begin(), tasks.end(), [&](const Task& a, const Task& b) {
      double x = cost(0, a.type), y = cost(0, b.type);
      for (int c = 1; c < nc; c++) {
        x = std::min(x, cost(c, a.type));
        y = std::min(y, cost(c, b.type));
      }
      return x > y;
    });
  else  // shared expert, then routed experts by token count (hot to cold)
    std::stable_sort(tasks.begin(), tasks.end(), [&](const Task& a, const Task& b) {
      bool sa = a.expert == E, sb = b.expert == E;
      if (sa != sb) return sa;
      return a.tokens.size() > b.tokens.size();
    });
  tasks.insert(tasks.begin(), Task{-1, B + 1, {}});
  first_routed = !cfg.shared_expert ? 1 : cfg.shared_place == 3 ? 3 : 2;
  return true;
}

double Layer::remaining(int c) const {
  const Context& s = st[c];
  if (s.task < 0) return 0.0;
  const Program& p = current(c);
  const Phase& ph = p.phases[s.phase];
  double r = p.suffix[s.phase + 1] + ph.estimate * std::max(0.0, 1.0 - double(s.v) / ph.cycles) +
             std::max<int64_t>(0, s.commit - t);
  if (s.drain_until >= 0) r += std::max<int64_t>(0, s.drain_until - t);
  return r;
}

int64_t Layer::remaining_int(int c) const {  // a down-counter loaded with the task's rounded estimate
  const Context& s = st[c];
  return s.task < 0 ? 0 : std::max<int64_t>(0, s.start + icost(c, tasks[s.task].type) - t);
}

int64_t Layer::committed(int c) const {
  int64_t r = remaining_int(c);
  if (st[c].next_task >= 0) r += icost(c, tasks[st[c].next_task].type);
  return r;
}

bool Layer::running(int j) const {
  for (int c = 0; c < nc; c++)
    if (st[c].task == j || st[c].next_task == j || st[c].tail.task == j) return true;
  return false;
}

// The context predicted to finish the shared expert first (fb its finish; fc that of context c), -1 if none fits.
int Layer::preferred_shared(int c, int64_t& fb, int64_t& fc) const {
  int pref = -1;
  fb = fc = IINF;
  for (int cc = 0; cc < nc; cc++)
    if (prog[cc][B].tm) {
      int64_t f = remaining_int(cc) + icost(cc, B);
      if (f < fb) {
        fb = f;
        pref = cc;
      }
      if (cc == c) fc = f;
    }
  return pref;
}

// The shared expert (it does not wait for the router) goes to the context predicted to finish it first; a free context
// also takes it while that context is busy, if it finishes it at most 1.5 times later. The halves of a split shared
// expert go to the first context that fits each. Returns the task taken by context c, or -1.
int Layer::take_shared(int c, bool free) {
  if (shared_done) return -1;
  if (cfg.shared_place == 3) {
    for (int h = 0; h < 2; h++)
      if (!half_taken[h] && prog[c][tasks[1 + h].type].tm) {
        half_taken[h] = true;
        shared_done = half_taken[0] && half_taken[1];
        return 1 + h;
      }
    return -1;
  }
  int64_t fb, fc;
  int pref = preferred_shared(c, fb, fc);
  if (pref == c || (free && pref >= 0 && st[pref].task >= 0 && prog[c][B].tm && fc <= fb + (fb >> 1))) {
    shared_done = true;
    return 1;
  }
  return -1;
}

// A band plan charges the shared expert not yet dispatched to its preferred array (scaled by that array's progress
// factor when re-planning); each undispatched half of a split shared expert to its array.
void Layer::charge_shared(int64_t* R, bool scaled) const {
  if (cfg.shared_place == 3) {
    for (int h = 0; h < 2; h++)
      if (!half_taken[h])
        for (int cc = 0; cc < nc; cc++)
          if (prog[cc][tasks[1 + h].type].tm) {
            R[topo.rank[array_of(cc)]] += icost(cc, tasks[1 + h].type);
            break;
          }
    return;
  }
  if (shared_done) return;
  int64_t fb, fc;
  int pref = preferred_shared(-1, fb, fc);
  if (pref >= 0)
    R[topo.rank[array_of(pref)]] += scaled ? icost(pref, B) * factor(array_of(pref)) / 1024 : icost(pref, B);
}

// Pincer: routed tasks are taken from the hot end by the array with more multipliers (on one array: even contexts) and
// from the cold end by the other.
int Layer::pick_pincer(int c, bool free) {
  int js = take_shared(c, free);
  if (js >= 0) return js;
  if (!router_done || hotp > coldp) return -1;
  int k = array_of(c);
  bool hot;
  if (topo.arrays == 1)
    hot = topo.pos[c] % 2 == 0;
  else {
    int64_t mk = shape[topo.first[k]].macs(), mo = shape[topo.first[1 - k]].macs();
    hot = mk > mo || (mk == mo && k == 0);
  }
  int j = hot ? hotp : coldp;
  if (!prog[c][tasks[j].type].tm) return -1;
  if (hot)
    hotp++;
  else
    coldp--;
  return j;
}

// HBM and weight-port bytes per cycle (x1024) of a program on context c from phase `from` on; the port rate counts
// first passes, packed re-reads and re-deliveries.
void Layer::rates(int c, int type, int from, int64_t& hr, int64_t& pr) const {
  const Program& p = prog[c][type];
  const int64_t T = std::max<int64_t>(1, llround(p.suffix[from]));
  hr = p.hsuffix[from] * 1024 / T;
  pr = (p.wsuffix[from] + p.rsuffix[from]) * 1024 / T;
}

// Supply-aware choice between the hot-end task (h) and the cold-end task (c) given the rates of the other contexts
// (Ho, Po): the penalty of a candidate is port overload + HBM idling + half the HBM oversubscription, each normalised;
// within the tie tolerance the context keeps its own end.
static bool supply_choice(bool own_hot, int64_t hh, int64_t ph, int64_t hc, int64_t pc, int64_t Ho, int64_t Po,
                          int64_t W, int64_t H, int tie) {
  auto penalty = [&](int64_t h, int64_t p) {
    const int64_t P = Po + p, Hs = Ho + h;
    return std::max<int64_t>(0, P - 1024 * W) / W + std::max<int64_t>(0, 1024 * H - Hs) / H +
           std::max<int64_t>(0, Hs - 1024 * H) / (2 * H);
  };
  const int64_t a = penalty(hh, ph), b = penalty(hc, pc), tol = int64_t(tie) * 1024 / 1000;
  if (a + tol < b) return true;
  if (b + tol < a) return false;
  return own_hot;
}

// Supply-aware pincer: as the pincer, but a context weighs both end tasks by how well each keeps HBM busy without
// overloading the weight port next to what the other contexts are running. Own end: the tallest array (on one array
// even contexts) the hot end.
int Layer::pick_supply(int c, bool free) {
  int js = take_shared(c, free);
  if (js >= 0) return js;
  if (!router_done || hotp > coldp) return -1;
  const bool own = topo.arrays == 1 ? topo.pos[c] % 2 == 0 : topo.rank[array_of(c)] == 0;
  const int jh = hotp, jc = coldp;
  const bool fh = prog[c][tasks[jh].type].tm > 0, fc = prog[c][tasks[jc].type].tm > 0;
  if (!fh && !fc) return -1;
  bool hot;
  if (jh == jc)
    hot = true;
  else if (fh != fc)
    hot = fh;
  else {
    int64_t Ho = 0, Po = 0, hh, ph, hc, pc;
    for (int cc = 0; cc < nc; cc++)
      if (cc != c && st[cc].task >= 0) {
        int64_t h, q;
        rates(cc, tasks[st[cc].task].type, st[cc].phase, h, q);
        Ho += h;
        Po += q;
      }
    rates(c, tasks[jh].type, 0, hh, ph);
    rates(c, tasks[jc].type, 0, hc, pc);
    hot = supply_choice(own, hh, ph, hc, pc, Ho, Po, cfg.W, cfg.H, cfg.p15_tie);
  }
  if (jh != jc && hot != own) out[S_end_switches]++;
  if (hot) return hotp++;
  return coldp--;
}

int64_t Layer::array_task_cost(int r, int j) const {  // cheapest context of the array of rank r
  int k = topo.byrank[r];
  int64_t b = IINF;
  for (int i = 0; i < topo.nctx[k]; i++) b = std::min(b, icost(topo.ctx[k][i], tasks[j].type));
  return b;
}

int64_t Layer::array_committed(int r) const {
  int k = topo.byrank[r];
  int64_t z = 0;
  for (int i = 0; i < topo.nctx[k]; i++) z += committed(topo.ctx[k][i]);
  return z;
}

// Cut of n tasks with costs bc (rank-major, IINF where a task does not fit) into one contiguous band per rank after the
// committed work R, with the smallest feasible makespan (binary search, greedy fill in rank order); without a feasible
// cut the last band takes the rest and plan_fallback is set.
static void cut_bands(int K, int n, const int64_t* R, const std::vector<int64_t>& bc, int* lo, int* hi, int64_t* out) {
  auto cut = [&](int64_t T) {  // greedy fill in rank order; returns the first task left over
    int i = 0;
    for (int r = 0; r < K; r++) {
      lo[r] = i;
      int64_t acc = R[r];
      while (i < n) {
        int64_t x = bc[size_t(r) * n + i];
        if (x >= IINF || acc + x > T) break;
        acc += x;
        i++;
      }
      hi[r] = i - 1;
    }
    return i;
  };
  int64_t T = 0;
  for (int r = 0; r < K; r++) T = std::max(T, R[r]);
  for (int i = 0; i < n; i++) {
    int64_t w = 0;
    for (int r = 0; r < K; r++) {
      int64_t x = bc[size_t(r) * n + i];
      if (x < IINF) w = std::max(w, x);
    }
    T += w;
  }
  if (cut(T) < n) {
    out[S_band_plan_fallback] = 1;
    hi[K - 1] = n - 1;
  } else {
    int64_t a = 0, b = T;
    while (a < b) {
      int64_t m = a + (b - a) / 2;
      if (cut(m) == n)
        b = m;
      else
        a = m + 1;
    }
    cut(a);
  }
}

// Once the router finishes, the routed tasks are cut into one contiguous band per array (tallest array first) with the
// smallest predicted makespan.
void Layer::plan_bands() {
  planned = true;
  const int K = topo.arrays, N = int(tasks.size()), j0 = first_routed, n = N - j0;
  int64_t R[MAXA];
  for (int r = 0; r < K; r++) R[r] = array_committed(r);
  charge_shared(R, false);
  band_task_cost.assign(size_t(K) * N, IINF);
  std::vector<int64_t> bc(size_t(K) * std::max(n, 1), IINF);
  for (int r = 0; r < K; r++)
    for (int j = j0; j < N; j++) bc[size_t(r) * n + j - j0] = band_task_cost[size_t(r) * N + j] = array_task_cost(r, j);
  int lo[MAXA], hi[MAXA];
  cut_bands(K, n, R, bc, lo, hi, out);
  int64_t makespan = 0;
  for (int r = 0; r < K; r++) {
    band_lo[r] = lo[r] + j0;
    band_hi[r] = hi[r] + j0;
    band_left[r] = 0;
    for (int j = band_lo[r]; j <= band_hi[r]; j++) {
      int64_t x = band_task_cost[size_t(r) * N + j];
      if (x < IINF) band_left[r] += x;
    }
    makespan = std::max(makespan, R[r] + band_left[r]);
    if (log) log->push_back({t, topo.byrank[r], band_lo[r], band_hi[r], r, 5, cfg.policy, R[r] + band_left[r]});
  }
  out[S_band_plan_cycle] = t;
  out[S_band_planned_makespan] = makespan;
}

// Band: a context takes from its own band (the taller array from the hot end, the shorter one from the cold end; on one
// array even contexts from the hot end). With its band empty it steals from the other band at the nearer end;
// p14_steal 2 steals only a task the thief is predicted to finish before the band's owner finishes its band.
int Layer::pick_band(int c, bool free) {
  int js = take_shared(c, free);
  if (js >= 0) return js;
  if (!router_done) return -1;
  if (!planned) plan_bands();
  const int K = topo.arrays, r = topo.rank[array_of(c)], N = int(tasks.size());
  auto take = [&](int b, bool lo) {
    int j = lo ? band_lo[b] : band_hi[b];
    if (!prog[c][tasks[j].type].tm) return -1;
    if (lo)
      band_lo[b]++;
    else
      band_hi[b]--;
    int64_t x = band_task_cost[size_t(b) * N + j];
    if (x < IINF) band_left[b] -= x;
    return j;
  };
  if (band_lo[r] <= band_hi[r]) return take(r, K == 1 ? topo.pos[c] % 2 == 0 : r < K - 1);
  for (int b = 0; b < K; b++) {
    if (b == r || band_lo[b] > band_hi[b]) continue;
    bool lo = b > r;
    int jn = lo ? band_lo[b] : band_hi[b];
    if (cfg.p14_steal == 2 && band_task_cost[size_t(b) * N + jn] < IINF &&
        !(array_committed(r) + array_task_cost(r, jn) < array_committed(b) + band_left[b]))
      continue;
    int j = take(b, lo);
    if (j >= 0) {
      out[S_band_steals]++;
      return j;
    }
  }
  return -1;
}

// Band dispatch with re-planning: costs on an array are scaled by its progress factor, the measured occupancy of the
// tasks it finished (start until they left the context, router excluded) over their estimates (x1024, 1024 before
// the first finish).
int64_t Layer::factor(int k) const { return estimated[k] > 0 ? occupancy[k] * 1024 / estimated[k] : 1024; }

int64_t Layer::replan_cost(int r, int j) const {
  int64_t x = array_task_cost(r, j);
  return x >= IINF ? IINF : x * factor(topo.byrank[r]) / 1024;
}

int64_t Layer::replan_committed(int r) const {
  const int k = topo.byrank[r];
  const int64_t f = factor(k);
  int64_t z = 0;
  for (int i = 0; i < topo.nctx[k]; i++) {
    const Context& s = st[topo.ctx[k][i]];
    if (s.task < 0) continue;
    const int c = topo.ctx[k][i];
    z += std::max<int64_t>(0, s.start + icost(c, tasks[s.task].type) * f / 1024 - t);
    if (s.next_task >= 0) z += icost(c, tasks[s.next_task].type) * f / 1024;
  }
  return z;
}

// The first plan cuts every routed task; later plans re-cut the unassigned tasks (the bands concatenated in rank
// order).
void Layer::replan_bands() {
  const int K = topo.arrays, N = int(tasks.size());
  std::vector<int> L;
  if (planned_done < 0)
    for (int j = first_routed; j < N; j++) L.push_back(j);
  else
    for (int r = 0; r < K; r++) {
      L.insert(L.end(), band_queue[r].begin(), band_queue[r].end());
      band_queue[r].clear();
    }
  const int n = int(L.size());
  int64_t R[MAXA];
  for (int r = 0; r < K; r++) R[r] = replan_committed(r);
  charge_shared(R, true);
  std::vector<int64_t> bc(size_t(K) * std::max(n, 1), IINF);
  for (int r = 0; r < K; r++)
    for (int i = 0; i < n; i++) bc[size_t(r) * n + i] = replan_cost(r, L[i]);
  int lo[MAXA], hi[MAXA];
  cut_bands(K, n, R, bc, lo, hi, out);
  int64_t makespan = 0;
  for (int r = 0; r < K; r++) {
    int64_t z = 0;
    for (int i = lo[r]; i <= hi[r]; i++) {
      band_queue[r].push_back(L[i]);
      if (bc[size_t(r) * n + i] < IINF) z += bc[size_t(r) * n + i];
    }
    makespan = std::max(makespan, R[r] + z);
    if (log)
      log->push_back({t, topo.byrank[r], lo[r] <= hi[r] ? L[lo[r]] : -1, lo[r] <= hi[r] ? L[hi[r]] : -1, r, 5,
                      cfg.policy, R[r] + z});
  }
  out[S_band_plan_cycle] = t;
  out[S_band_planned_makespan] = makespan;
  out[S_band_plans]++;
  planned_done = done;
}

// Takes and steals as the band policy, on bands re-cut whenever a task has finished since the last plan.
int Layer::pick_replan(int c, bool free) {
  int js = take_shared(c, free);
  if (js >= 0) return js;
  if (!router_done) return -1;
  const int K = topo.arrays, r = topo.rank[array_of(c)];
  if (planned_done < 0 || (done != planned_done && K > 1)) replan_bands();
  auto take = [&](int b, bool lo) {
    std::deque<int>& q = band_queue[b];
    if (q.empty()) return -1;
    int j = lo ? q.front() : q.back();
    if (!prog[c][tasks[j].type].tm) return -1;
    if (lo)
      q.pop_front();
    else
      q.pop_back();
    return j;
  };
  if (!band_queue[r].empty()) return take(r, K == 1 ? topo.pos[c] % 2 == 0 : r < K - 1);
  for (int b = 0; b < K; b++) {  // with at most two arrays there is one other band
    if (b == r || band_queue[b].empty()) continue;
    bool lo = b > r;
    int jn = lo ? band_queue[b].front() : band_queue[b].back();
    if (cfg.p14_steal == 2) {
      int64_t finish = replan_committed(b);  // the band owner's predicted finish
      for (int j : band_queue[b])
        if (replan_cost(b, j) < IINF) finish += replan_cost(b, j);
      if (replan_cost(b, jn) < IINF && !(replan_committed(r) + replan_cost(r, jn) < finish)) continue;
    }
    int j = take(b, lo);
    if (j >= 0) {
      out[S_band_steals]++;
      return j;
    }
  }
  return -1;
}

int Layer::pick(int c, bool free) {
  switch (cfg.policy) {
    case BAND:
      return pick_band(c, free);
    case SUPPLY_PINCER:
      return pick_supply(c, free);
    case REPLAN_BAND:
      return pick_replan(c, free);
    default:
      return pick_pincer(c, free);
  }
}

// Regret: a queued task prefers the context predicted to finish it first; take the task whose finish differs most
// between the two contexts, then the longer one. only >= 0: tasks preferring that context; else: preferring a free one.
int Layer::pick_regret(int only, int& core) const {
  int best = -1;
  double regret = -1e300, size = -1, r0 = remaining(0), r1 = remaining(1);
  for (int j = 0; j < int(queue.size()); j++) {
    int tp = tasks[queue[j]].type;
    double f0 = r0 + cost(0, tp), f1 = r1 + cost(1, tp);
    int c = f0 <= f1 ? 0 : 1;
    if (!prog[c][tp].tm) c = 1 - c;
    if (only >= 0 ? c != only : st[c].task >= 0) continue;
    double r = std::abs(f0 - f1), z = prog[c][tp].suffix[0];
    if (r > regret || (r == regret && z > size)) {
      best = j;
      core = c;
      regret = r;
      size = z;
    }
  }
  return best;
}

void Layer::fill_window() {  // regret: a window of the next lookahead tasks in front-end order
  while (router_done && cursor < int(tasks.size()) && int(queue.size()) < cfg.lookahead) {
    if (!finished[cursor] && !running(cursor) && std::find(queue.begin(), queue.end(), cursor) == queue.end())
      queue.push_back(cursor);
    cursor++;
  }
  if (router_done) return;
  for (int j = 1; j < int(tasks.size()); j++)
    if (tasks[j].expert == cfg.experts && std::find(queue.begin(), queue.end(), j) == queue.end()) {
      if (!running(j) && !finished[j]) queue.push_back(j);
      break;
    }
}

void Layer::dispatch() {
  if (cfg.policy == REGRET) {
    while (!queue.empty() && (st[0].task < 0 || st[1].task < 0)) {
      int core = -1, best = pick_regret(-1, core);
      if (best < 0) break;
      int j = queue[best];
      queue.erase(queue.begin() + best);
      start(core, j, t);
      record(t, core, j, 0, 0);
    }
    return;
  }
  for (bool any = true; any;) {
    any = false;
    for (int c = 0; c < nc; c++)
      if (st[c].task < 0) {
        int j = pick(c, true);
        if (j < 0) continue;
        start(c, j, t);
        record(t, c, j, 0, 0);
        any = true;
      }
  }
}

// One-deep task queue: once every HBM request of its task is issued, a busy context reserves its next task, whose first
// phase then prefetches into the same landing slots.
void Layer::reserve_next() {
  if (!cfg.prefetch) return;
  for (int c = 0; c < nc; c++) {
    Context& s = st[c];
    if (s.task < 0 || s.next_task >= 0 || (cfg.policy == REGRET && queue.empty())) continue;
    const Program& p = current(c);
    if (p.next_hbm[s.phase + 1] >= 0 || s.hbm_issued < int(p.phases[s.phase].inputs.size())) continue;
    int j;
    if (cfg.policy == REGRET) {
      int core = -1, best = pick_regret(c, core);
      if (best < 0) continue;
      j = queue[best];
      queue.erase(queue.begin() + best);
    } else {
      j = pick(c, false);
      if (j < 0) continue;
    }
    s.next_task = j;
    out[S_next_reservations]++;
    record(t, c, j, 1, 0);
  }
}
