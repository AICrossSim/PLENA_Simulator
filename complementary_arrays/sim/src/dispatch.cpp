// Dispatch: the front-end task order and the policies that hand tasks to free contexts and reserve next tasks.
#include <cmath>
#include <numeric>

#include "layer.hpp"

// Tasks: the router, every routed expert with tokens and the shared expert, in front-end order.
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
  Task shared{E, B, std::vector<int>(B)};
  std::iota(shared.tokens.begin(), shared.tokens.end(), 0);
  tasks.push_back(shared);
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

// The shared expert (it does not wait for the router) goes to the context predicted to finish it first; a free context
// also takes it while that context is busy, if it finishes it at most 1.5 times later.
bool Layer::take_shared(int c, bool free) {
  if (shared_done) return false;
  int pref = -1;
  int64_t fb = IINF, fc = IINF;
  for (int cc = 0; cc < nc; cc++)
    if (prog[cc][B].tm) {
      int64_t f = remaining_int(cc) + icost(cc, B);
      if (f < fb) {
        fb = f;
        pref = cc;
      }
      if (cc == c) fc = f;
    }
  if (pref == c || (free && pref >= 0 && st[pref].task >= 0 && prog[c][B].tm && fc <= fb + (fb >> 1))) {
    shared_done = true;
    return true;
  }
  return false;
}

// Pincer: routed tasks are taken from the hot end by the array with more multipliers (on one array: even contexts) and
// from the cold end by the other.
int Layer::pick_pincer(int c, bool free) {
  if (take_shared(c, free)) return 1;
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

// Once the router finishes, the routed tasks are cut into one contiguous band per array (tallest array first) with the
// smallest predicted makespan.
void Layer::plan_bands() {
  planned = true;
  const int K = topo.arrays, N = int(tasks.size());
  int64_t R[MAXA];
  for (int r = 0; r < K; r++) R[r] = array_committed(r);
  if (!shared_done) {  // charged to the context the shared expert will go to
    int pref = -1;
    int64_t fb = IINF;
    for (int cc = 0; cc < nc; cc++)
      if (prog[cc][B].tm) {
        int64_t f = remaining_int(cc) + icost(cc, B);
        if (f < fb) {
          fb = f;
          pref = cc;
        }
      }
    if (pref >= 0) R[topo.rank[array_of(pref)]] += icost(pref, B);
  }
  std::vector<int64_t>& task_cost = band_task_cost;
  task_cost.assign(size_t(K) * N, IINF);
  for (int r = 0; r < K; r++)
    for (int j = 2; j < N; j++) task_cost[size_t(r) * N + j] = array_task_cost(r, j);
  int lo[MAXA], hi[MAXA];
  auto cut = [&](int64_t T) {  // greedy fill in rank order; returns the first task left over
    int j = 2;
    for (int r = 0; r < K; r++) {
      lo[r] = j;
      int64_t acc = R[r];
      while (j < N) {
        int64_t x = task_cost[size_t(r) * N + j];
        if (x >= IINF || acc + x > T) break;
        acc += x;
        j++;
      }
      hi[r] = j - 1;
    }
    return j;
  };
  int64_t T = 0;
  for (int r = 0; r < K; r++) T = std::max(T, R[r]);
  for (int j = 2; j < N; j++) {
    int64_t w = 0;
    for (int r = 0; r < K; r++) {
      int64_t x = task_cost[size_t(r) * N + j];
      if (x < IINF) w = std::max(w, x);
    }
    T += w;
  }
  if (cut(T) < N) {  // no contiguous cut respects the SRAM: the last band takes the rest
    out[S_band_plan_fallback] = 1;
    hi[K - 1] = N - 1;
  } else {
    int64_t a = 0, b = T;
    while (a < b) {
      int64_t m = a + (b - a) / 2;
      if (cut(m) == N)
        b = m;
      else
        a = m + 1;
    }
    cut(a);
  }
  int64_t makespan = 0;
  for (int r = 0; r < K; r++) {
    band_lo[r] = lo[r];
    band_hi[r] = hi[r];
    band_left[r] = 0;
    for (int j = lo[r]; j <= hi[r]; j++) {
      int64_t x = task_cost[size_t(r) * N + j];
      if (x < IINF) band_left[r] += x;
    }
    makespan = std::max(makespan, R[r] + band_left[r]);
    if (log) log->push_back({t, topo.byrank[r], lo[r], hi[r], r, 5, cfg.policy, R[r] + band_left[r]});
  }
  out[S_band_plan_cycle] = t;
  out[S_band_planned_makespan] = makespan;
}

// Band: a context takes from its own band (the taller array from the hot end, the shorter one from the cold end; on one
// array even contexts from the hot end). With its band empty it steals from the other band at the nearer end;
// p14_steal 2 steals only a task the thief is predicted to finish before the band's owner finishes its band.
int Layer::pick_band(int c, bool free) {
  if (take_shared(c, free)) return 1;
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
        int j = cfg.policy == BAND ? pick_band(c, true) : pick_pincer(c, true);
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
      j = cfg.policy == BAND ? pick_band(c, false) : pick_pincer(c, false);
      if (j < 0) continue;
    }
    s.next_task = j;
    out[S_next_reservations]++;
    record(t, c, j, 1, 0);
  }
}
