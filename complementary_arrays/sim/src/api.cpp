// C interface for engine.py: field layouts, topology resolution, program compilation and the parallel window loop.
#include <omp.h>

#include <string>

#include "layer.hpp"

#ifdef SIM_RAMULATOR
#include "hbm_ramulator.hpp"
#endif

namespace {

std::string names(std::initializer_list<const char*> list) {
  std::string s;
  for (const char* n : list) s += std::string(s.empty() ? "" : ",") + n;
  return s;
}

#define NAME(name) #name,
const std::string& layout() {
  static const std::string s = "config:" + names({CONFIG_FIELDS(NAME)}) + ";scalar:" + names({SCALAR_OUTPUTS(NAME)}) +
                               ";context:" + names({CONTEXT_OUTPUTS(NAME)}) + ";array:" + names({ARRAY_OUTPUTS(NAME)}) +
                               ";class:" + names({CLASS_OUTPUTS(NAME)}) + ";state:" + names({ARRAY_STATES(NAME)}) +
                               ";trace:" + names({TRACE_FIELDS(NAME)}) +
                               ";trace_context:" + names({TRACE_CONTEXT_FIELDS(NAME)}) +
                               ";trace_array:" + names({TRACE_ARRAY_FIELDS(NAME)}) + ";dims:" + std::to_string(MAXC) +
                               "," + std::to_string(MAXA) + "," + std::to_string(NCLASS);
  return s;
}
#undef NAME

Config read_config(const int64_t* v) {
  Config cfg;
  int i = 0;
#define READ(name) cfg.name = int(v[i++]);
  CONFIG_FIELDS(READ)
#undef READ
  return cfg;
}

struct Chip {
  Topology topo;
  std::vector<Shape> shape;
  // per context: routed expert with 1..B tokens, shared expert (or its first half), router, second shared half
  std::vector<std::vector<Program>> prog;
};

// Contexts c of a shared array run on array c / ctx_per_core; otherwise each context has its own array. Vector lanes
// are split over the arrays in proportion to their multipliers; SRAM after the token buffers is split by share (and
// share2 inside pairs for four contexts). Programs are compiled for the token counts in the windows and those flagged
// in provision (one flag per count 1..B); the largest reservation among them decides whether a tail buffer fits. The
// shared expert is compiled for every context, for the contexts of one array, or as two halves (the first, with the
// gate, on array 0, the second on array 1). Returns 0 or a negative error code.
int prepare(const Config& cfg, const int32_t* shapes, int windows, const int32_t* routes, const int8_t* provision,
            Chip& chip) {
  const int nc = cfg.contexts, B = cfg.batch, E = cfg.experts;
  if ((nc != 2 && nc != 4) || 6LL * B * cfg.hidden >= cfg.capacity) return -1;
  if (B > 1024 || cfg.topk > 64 || cfg.topk > E) return -8;
  Topology& T = chip.topo;
  T.contexts = nc;
  T.shared = cfg.shared_array;
  if (T.shared) {
    if (cfg.ctx_per_core < 1 || nc % cfg.ctx_per_core) return -2;
    T.arrays = nc / cfg.ctx_per_core;
  } else
    T.arrays = nc;
  if (T.arrays > MAXA || (!T.shared && nc != 2)) return -2;
  chip.shape.resize(nc);
  for (int c = 0; c < nc; c++) {
    chip.shape[c] = {shapes[4 * c], shapes[4 * c + 1], shapes[4 * c + 2], 1024, shapes[4 * c + 3]};
    T.array_of[c] = T.shared ? c / cfg.ctx_per_core : c;
  }
  for (int k = 0; k < MAXA; k++) T.first[k] = -1;
  for (int c = 0; c < nc; c++) {
    int k = T.array_of[c];
    T.pos[c] = T.nctx[k];
    T.ctx[k][T.nctx[k]++] = c;
    if (T.first[k] < 0) T.first[k] = c;
  }
  for (int c = 0; c < nc; c++) {
    const Shape &s = chip.shape[c], &f = chip.shape[T.first[T.array_of[c]]];
    if (s.m != f.m || s.n != f.n || s.k != f.k) return -3;
  }
  int64_t macs = 0;
  for (int k = 0; k < T.arrays; k++) {
    T.caps[k] = T.shared && cfg.core_caps;
    T.anycaps |= T.caps[k];
    macs += chip.shape[T.first[k]].macs();
  }
  if (macs > (1 << 20)) return -5;
  int given = 0;
  for (int k = 0; k < T.arrays; k++) {
    int lanes = k == T.arrays - 1 ? VECTOR_LANES - given
                                  : std::max(1, int(int64_t(VECTOR_LANES) * chip.shape[T.first[k]].macs() / macs));
    if (lanes < 1) return -5;
    given += lanes;
    for (int i = 0; i < T.nctx[k]; i++) chip.shape[T.ctx[k][i]].lanes = lanes;
  }
  for (int k = 0; k < T.arrays; k++) T.byrank[k] = k;
  std::stable_sort(T.byrank, T.byrank + T.arrays, [&](int a, int b) {
    const Shape &x = chip.shape[T.first[a]], &y = chip.shape[T.first[b]];
    return x.m != y.m ? x.m > y.m : x.macs() > y.macs();
  });
  for (int r = 0; r < T.arrays; r++) T.rank[T.byrank[r]] = r;
  for (int c = 0; c < nc; c++) {
    int64_t sram = 1LL * (cfg.capacity - 6 * B * cfg.hidden);
    T.sram[c] = nc == 4 ? sram * (c < 2 ? cfg.share : 1000 - cfg.share) / 1000 *
                              (c % 2 == 0 ? cfg.share2 : 1000 - cfg.share2) / 1000
                        : sram * (c == 0 ? cfg.share : 1000 - cfg.share) / 1000;
  }
  if (cfg.shared_place < 0 || cfg.shared_place > 3 || (cfg.shared_place >= 2 && T.arrays != 2)) return -9;
  if (cfg.policy != REGRET && cfg.policy != PINCER && cfg.policy != BAND && cfg.policy != SUPPLY_PINCER &&
      cfg.policy != REPLAN_BAND)
    return -4;
  if (cfg.policy == REGRET && nc != 2) return -4;
  std::vector<char> needed(provision, provision + B);
  for (int s = 0; s < windows; s++) {
    std::vector<int> counts(E, 0);
    for (int j = 0; j < B * cfg.topk; j++) {
      int e = routes[int64_t(s) * B * cfg.topk + j];
      if (e < 0 || e >= E) return -6;
      counts[e]++;
    }
    for (int n : counts)
      if (n > 0) {
        if (n > B) return -6;
        needed[n - 1] = 1;
      }
  }
  chip.prog.assign(nc, std::vector<Program>(B + 3));
  const int F = cfg.routed_width, S = cfg.shared_width;
  for (int c = 0; c < nc; c++) {
    const int cap = int(T.sram[c]), k = T.array_of[c];
    const Shape& sh = chip.shape[c];
    for (int m = 1; m <= B; m++)
      if (needed[m - 1]) chip.prog[c][m - 1] = compile(cfg, sh, m, ROUTED, F, false, cap);
    if (cfg.shared_expert && cfg.shared_place == 3) {
      if (k == 0)
        chip.prog[c][B] = compile(cfg, sh, B, SHARED, S / 2, cfg.shared_gate, cap);
      else
        chip.prog[c][B + 2] = compile(cfg, sh, B, SHARED, S - S / 2, false, cap);
    } else if (cfg.shared_expert && (cfg.shared_place == 0 || k == cfg.shared_place - 1))
      chip.prog[c][B] = compile(cfg, sh, B, SHARED, S, cfg.shared_gate, cap);
    chip.prog[c][B + 1] = compile(cfg, sh, B, ROUTER, F, false, cap);
    for (const Program& p : chip.prog[c])
      if (p.tm) T.maxres[c] = std::max(T.maxres[c], p.reserved);
  }
  return 0;
}

}  // namespace

extern "C" const char* sim_layout() { return layout().c_str(); }

namespace {

int run_windows(const int64_t* config, const int32_t* shapes, int windows, const int32_t* routes,
                const int8_t* provision, int threads, int64_t* out, const xcheck::Options* ram) {
  const Config cfg = read_config(config);
  Chip chip;
  if (int err = prepare(cfg, shapes, windows, routes, provision, chip)) return err;
  if (ram && cfg.shared_place == 3) return -10;
  const int64_t stride = int64_t(cfg.batch) * cfg.topk;
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1)
  for (int s = 0; s < windows; s++)
    Layer(cfg, chip.topo, chip.shape.data(), chip.prog, routes + s * stride, out + int64_t(s) * NOUT, nullptr, 0,
          nullptr, ram, s)
        .run();
  return 0;
}

}  // namespace

// windows x (batch x topk) routes -> windows x NOUT outputs. shapes: per context m, n, k, drain latency.
extern "C" int sim_run(const int64_t* config, const int32_t* shapes, int windows, const int32_t* routes,
                       const int8_t* provision, int threads, int64_t* out) {
  return run_windows(config, shapes, windows, routes, provision, threads, out, nullptr);
}

#ifdef SIM_RAMULATOR
// As sim_run with the HBM requests served in closed loop through Ramulator (ram_config: its configuration text; extra
// cycles from the last returned transfer to a usable request; cap on lines issued per cycle; address layout; lines
// tried per context and cycle; log: file prefix of the per-window logs, empty for none).
extern "C" int sim_run_ramulator(const int64_t* config, const int32_t* shapes, int windows, const int32_t* routes,
                                 const int8_t* provision, int threads, const char* ram_config, int extra, int cap,
                                 int layout, int look, const char* log, int64_t* out) {
  xcheck::Options o;
  o.config = ram_config;
  o.extra = extra;
  o.cap = cap;
  o.layout = layout;
  o.look = look;
  o.log = log ? log : "";
  return run_windows(config, shapes, windows, routes, provision, threads, out, &o);
}
#endif

// Compiled program of one expert task on one array (shape: m, n, k, lanes, drain): estimate, compute cycles, HBM bytes.
extern "C" int sim_program(const int64_t* config, const int32_t* shape, int tokens, int shared, int cap, double* out) {
  const Config cfg = read_config(config);
  const Program p =
      compile(cfg, Shape{shape[0], shape[1], shape[2], shape[3], shape[4]}, tokens, shared ? SHARED : ROUTED,
              shared ? cfg.shared_width : cfg.routed_width, shared && cfg.shared_gate, cap);
  if (!p.tm) return -1;
  out[0] = p.suffix[0];
  out[1] = out[2] = 0;
  for (const Phase& ph : p.phases) {
    out[1] += ph.cycles;
    out[2] += double(ph.hbm);
  }
  return 0;
}

// One window with the per-cycle trace (rows x NTRACE) and the dispatch decision log (up to max_decisions x 8).
extern "C" int sim_trace(const int64_t* config, const int32_t* shapes, const int32_t* routes, const int8_t* provision,
                         int64_t* out, int64_t* rows, int64_t max_rows, int64_t* decisions, int max_decisions,
                         int* ndecisions) {
  const Config cfg = read_config(config);
  Chip chip;
  if (int err = prepare(cfg, shapes, 1, routes, provision, chip)) return err;
  std::vector<Decision> log;
  Layer(cfg, chip.topo, chip.shape.data(), chip.prog, routes, out, rows, max_rows, &log).run();
  *ndecisions = std::min(max_decisions, int(log.size()));
  for (int i = 0; i < *ndecisions; i++)
    for (int j = 0; j < 8; j++) decisions[8 * i + j] = log[i][j];
  return int(log.size()) > max_decisions ? -7 : 0;
}
