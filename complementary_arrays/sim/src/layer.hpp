// Cycle-level simulation of one layer window: task contexts on the arrays, dispatch, HBM, ports and converter.
#pragma once
#include <algorithm>
#include <deque>
#include <limits>

#include "model.hpp"

constexpr int64_t NEVER = std::numeric_limits<int64_t>::max();

struct Topology {
  int contexts = 0, arrays = 0;
  bool shared = false;                // contexts of one array share its MAC array and vector lanes
  int array_of[MAXC]{}, pos[MAXC]{};  // array of each context, position among the array's contexts
  int nctx[MAXA]{}, ctx[MAXA][MAXC]{}, first[MAXA]{};
  bool caps[MAXA]{}, anycaps = false;    // the array's contexts share its weight-port and converter intake
  int rank[MAXA]{}, byrank[MAXA]{};      // band dispatch: rank 0 is the tallest array
  int64_t sram[MAXC]{}, maxres[MAXC]{};  // SRAM bytes of each context; its largest program reservation
};

// A transfer in beats; usable from cycle `ready` once no beats remain (and, for weights, nothing is left to decode).
struct Request {
  int idx, remaining;
  int64_t ready;
  int decode = 0;
};

struct Tail {  // trailing phases of the task a context last handed off (tail mode)
  int task = -1, phase = 0, v = 0, consumed = 0, y = 0, lock = -1, requested = 0;
  int64_t commit = 0, drain_until = -1, buffer = 0;
  std::deque<Request> fifo;
};

struct Context {
  int task = -1, phase = 0, next_task = -1;
  // virtual cycle of the phase, inputs consumed, output bytes still to write, token lock held
  int v = 0, consumed = 0, y = 0, lock = -1;
  int64_t start = 0, commit = 0, delivered = 0;  // delivered: weights of the head input re-delivered so far
  // HBM requests of the current phase, and those already issued for the prefetched phase (pre_task, pre_phase)
  std::vector<Request> hbm, pre;
  std::deque<int> landing, pre_landing;  // requests that hold a landing slot
  int hbm_issued = 0, pre_issued = 0, pre_task = -1, pre_phase = -1;
  std::array<int, 2> requested{};
  std::array<std::deque<Request>, 2> fifo;    // weight and activation FIFOs
  int64_t drain_until = -1, drain_ready = 0;  // the dependent next phase starts at drain_until; pending drain end
  bool fallback = false;                      // this task's trailing phases run in place
  Tail tail;
  int64_t count[NCONTEXT_OUTPUT]{};  // output counters, C_*
};

// expert: -1 router, `experts` the shared expert; type: program index (tokens - 1, B shared expert, B + 1 router)
struct Task {
  int expert, type;
  std::vector<int> tokens;
};

using Decision = std::array<int64_t, 8>;

class Layer {
 public:
  Layer(const Config& cfg, const Topology& topo, const Shape* shape, const std::vector<std::vector<Program>>& prog,
        const int32_t* ids, int64_t* out, int64_t* trace = nullptr, int64_t trace_rows = 0,
        std::vector<Decision>* log = nullptr);
  void run();

 private:
  const Config& cfg;
  const Topology& topo;
  const Shape* shape;
  const std::vector<std::vector<Program>>& prog;
  const int32_t* ids;
  int64_t *out, *trace, trace_rows;
  std::vector<Decision>* log;
  const int B, nc;

  std::vector<Task> tasks;
  std::vector<int> queue, locks;
  std::vector<char> finished;
  std::array<Context, MAXC> st;
  int64_t t = 0, startup = 0, allocated = 0;
  int done = 0, cursor = 1, hotp = 2, coldp = 0;
  bool router_done = false, shared_done = false;
  std::array<int, 4> rr{};  // round-robin pointers of the servers
  bool planned = false;     // band dispatch: unassigned range of each band and its estimated cost
  int band_lo[MAXA]{}, band_hi[MAXA]{};
  int64_t band_left[MAXA]{};
  std::vector<int64_t> band_task_cost;
  int64_t knum = 0, kden = 1;                 // re-delivery budget, knum / kden weights per cycle
  int64_t port_reread = 0, first_beats = -1;  // shared port this cycle: re-delivery bytes, beats left for first passes
  int array_rr = 0;
  // per array this cycle: MAC array / vector lanes taken; a matrix phase advanced
  bool used_res[MAXA][2]{}, mat_adv[MAXA]{};
  double cls_count[MAXA][NCLASS][NCLASS_OUTPUT]{};
  int64_t* row = nullptr;  // trace row of the current cycle

  enum Server { WPORT, XPORT, YPORT, HBM };

  const Program& current(int c) const { return prog[c][tasks[st[c].task].type]; }
  int array_of(int c) const { return topo.array_of[c]; }
  // weight beats per cycle a context's array takes in
  int intake(int c) const { return std::max(1, 2 * shape[c].n * shape[c].k / shape[c].m / 64); }
  int task_class(int j) const {
    int e = tasks[j].expert;
    return e < 0 ? 2 : e == cfg.experts ? 1 : 0;
  }
  double cost(int c, int type) const { return prog[c][type].tm ? prog[c][type].suffix[0] : 1e300; }
  int64_t icost(int c, int type) const { return prog[c][type].tm ? prog[c][type].icost : IINF; }
  bool landed(const Context& s, const Request& q) const {
    return s.hbm[q.idx].remaining == 0 && s.hbm[q.idx].ready <= t;
  }
  bool limited(const Input& in) const { return in.localw > 0 && in.reuse; }
  int64_t& trace_ctx(int field, int c) { return row[NTRACE_FIELD + field * MAXC + c]; }
  void set_state(int c, int state) {
    if (row) trace_ctx(TC_state, c) = state;
  }
  int rotated(int i) const;
  void record(int64_t when, int c, int j, int kind, int64_t predicted);

  bool build_tasks();
  void start(int c, int j, int64_t when);
  void promote(int c);
  void finish_task(int c, int j, const Program& p);
  void reset_phase(Context& s, int64_t when);
  double remaining(int c) const;
  int64_t remaining_int(int c) const;
  int64_t committed(int c) const;
  bool running(int j) const;
  bool take_shared(int c, bool free);
  int pick_pincer(int c, bool free);
  int64_t array_task_cost(int r, int j) const;
  int64_t array_committed(int r) const;
  void plan_bands();
  int pick_band(int c, bool free);
  int pick_regret(int only, int& core) const;
  void fill_window();
  void dispatch();
  void reserve_next();
  void deliver();
  void advance_tails();
  void advance_contexts();
  void issue_requests(int c, const Program& p, const Phase& ph);
  int pending(int c, int r) const;
  void split_equal(int r, const std::array<int, MAXC>& need, std::array<int, MAXC>& give, int budget);
  void split_paired(int r, const std::array<int, MAXC>& need, std::array<int, MAXC>& give, int budget);
  void serve(int c, int r, int b);
  void serve_tail(int c, int r, int b);
  void serve_ports();
  void convert();
  void complete_tails();
  void complete_phases();
  void write_outputs();
};
