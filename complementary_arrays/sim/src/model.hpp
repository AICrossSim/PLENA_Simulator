// Shared definitions of the layer model: configuration, array shapes, compiled programs and the output layouts.
#pragma once
#include <array>
#include <cstdint>
#include <vector>

constexpr int MAXC = 4;             // task contexts
constexpr int MAXA = 2;             // arrays
constexpr int NCLASS = 3;           // task classes: routed expert, shared expert, router
constexpr int VECTOR_LANES = 2048;  // split over the arrays in proportion to their multipliers
constexpr int VECTOR_LATENCY = 4;
constexpr int64_t IINF = int64_t(1e15);

inline int cdiv(int a, int b) { return (a + b - 1) / b; }
inline int beats(int bytes) { return cdiv(bytes, 64); }  // HBM and SRAM ports move 64-byte beats

// Configuration of one simulation, passed from Python as int64 values in this order. Bandwidths are bytes per cycle:
// W weight port, X activation reads, Y output writes, H HBM. latency / hlatency: SRAM / HBM read latency.
// kappa_milli: re-delivery budget of port_mode 0 in thousandths of the HBM weight rate (0: unlimited).
// drain_inner / drain_last: a phase that reads a matrix phase's outputs waits for the array's drain (inside a task /
// after its last matrix phase); tile_bubble: one idle cycle between tiles; tail_slot: a task's trailing phases leave
// the context.
// clang-format off
#define CONFIG_FIELDS(F)                                                                                               \
  F(batch) F(topk) F(experts) F(hidden) F(routed_width) F(shared_width) F(shared_gate) F(contexts) F(shared_array)     \
  F(ctx_per_core) F(core_caps) F(W) F(X) F(Y) F(H) F(capacity) F(latency) F(hlatency) F(control_cycles) F(bits)        \
  F(group) F(converter_rate) F(converter_latency) F(port_mode) F(kappa_milli) F(policy) F(p14_steal) F(lookahead)      \
  F(slots) F(stage_slots) F(vslots) F(ybanks) F(share) F(share2) F(token_locks) F(drain_inner) F(drain_last)           \
  F(tile_bubble) F(tail_slot)
// clang-format on

struct Config {
#define CONFIG_MEMBER(name) int name;
  CONFIG_FIELDS(CONFIG_MEMBER)
#undef CONFIG_MEMBER
};

enum Policy { REGRET = 2, PINCER = 13, BAND = 14 };

struct Shape {
  int m, n, k;       // rows (tokens per pass), columns, reduction width of the array
  int lanes = 1024;  // vector lanes
  int drain = 0;     // cycles from a phase's last output row until the last column has it
  int macs() const { return n * k; }
  int depth() const {  // levels of a cell's adder tree
    int d = 0;
    for (int x = k / m - 1; x; x >>= 1) d++;
    return d;
  }
};

struct Input {      // operands of one virtual cycle of a phase
  int at;           // virtual cycle that consumes them
  int w;            // stored weight bytes: landed from HBM, then moved over the weight port
  int x;            // activation or partial-sum bytes
  int release = 0;  // requested only after this many inputs of the phase were consumed (buffer reuse)
  int localw = 0;   // BF16 weight bytes delivered to the array
  int decode = 0;   // elements the converter decodes before the array may use them
  int reuse = 0;    // delivered again from the retained panel (token passes after the first)
};

enum Kind { MATRIX, VECTOR, ROW, CONTROL };  // ROW: combine into one token, serialised by token locks

struct Phase {
  int kind = MATRIX, row = -1, cycles = 0;
  std::vector<Input> inputs;
  std::vector<int> output;         // bytes written per virtual cycle
  std::array<int64_t, 3> bytes{};  // weight-port, activation and output bytes
  int64_t localw = 0, reread = 0, decoded = 0, expanded = 0;
  double estimate = 0;   // dispatch estimate of its cycles
  int lat = 0;           // drain after completion before its outputs are complete
  bool hold = false;     // the next phase reads its outputs
  bool partial = false;  // reloads partial sums
};

struct Program {  // one task (an expert or the router for m tokens) compiled for one context
  std::vector<Phase> phases;
  std::vector<double> suffix;                   // estimate from each phase to the end
  std::vector<int> next_hbm;                    // first phase at or after each index that reads HBM, -1 if none
  int tm = 0;                                   // token rows per pass group; 0: does not fit the context's SRAM
  int64_t reserved = 0, icost = 0, unique = 0;  // SRAM bytes, rounded total estimate, unique weight elements
  double useful = 0;                            // useful MACs
  int lastmat = -1, tail_drain = 0;             // last matrix phase; drain the tail slot waits for
};

Program compile(const Config& cfg, const Shape& c, int m, bool shared, bool router, int cap);
int64_t stored_bytes(const Config& cfg, int n, int k, int bits);
void quota_rate(const Config& cfg, int64_t& knum, int64_t& kden);

// Output row of one window: scalars, then per-context, per-array and per-(array, task class) blocks, each field-major.
// A per-cycle trace row holds TRACE_FIELDS, then the per-context TRACE_CONTEXT_FIELDS, field-major.
// clang-format off
#define SCALAR_OUTPUTS(F)                                                                                              \
  F(status) F(cycles) F(startup) F(rows) F(peak_sram) F(hbm_idle_cycles) F(next_reservations)                          \
  F(prefetch_same_task_bytes) F(prefetch_next_task_bytes) F(band_plan_cycle) F(band_planned_makespan) F(band_steals)   \
  F(band_plan_fallback) F(delivery_limited_cycles) F(port_bytes) F(port_saturated_cycles) F(tail_fallbacks)            \
  F(multi_group_tasks) F(weight_bytes)
#define CONTEXT_OUTPUTS(F)                                                                                             \
  F(core) F(sram_bytes) F(enabled) F(input_stall) F(output_stall) F(merge_stall) F(commit_wait) F(idle) F(share_stall) \
  F(hbm_service) F(delivery_wait) F(delivered_weights) F(wait_hbm) F(wait_wport) F(wait_x) F(wait_latency)             \
  F(wait_decode) F(finish) F(reserved_bytes) F(drain_wait) F(drain_waits) F(independent_phases) F(drain_wait_last)     \
  F(tail_handoffs) F(tail_wait) F(tail_busy) F(tail_blocked)
#define ARRAY_OUTPUTS(F) F(lanes) F(drain_idle)
#define CLASS_OUTPUTS(F)                                                                                               \
  F(tasks) F(unique_weights) F(decoded) F(hbm_bytes) F(wport_bytes) F(expanded_write_bytes) F(local_read_bytes)        \
  F(busy) F(matrix_busy) F(useful_macs) F(reread_bytes)
#define TRACE_FIELDS(F) F(cycle) F(port_need_reread) F(port_need_first) F(port_used) F(port_first_budget)
#define TRACE_CONTEXT_FIELDS(F)                                                                                        \
  F(state) F(task) F(phase) F(prefetch_task) F(prefetch_phase) F(hbm_bytes) F(hbm_current) F(hbm_prefetch)             \
  F(port_first_bytes) F(port_reread_weights) F(decoded) F(x_bytes) F(y_bytes)
// clang-format on

#define ENUM_S(name) S_##name,
#define ENUM_C(name) C_##name,
#define ENUM_A(name) A_##name,
#define ENUM_K(name) K_##name,
#define ENUM_T(name) T_##name,
#define ENUM_TC(name) TC_##name,
enum { SCALAR_OUTPUTS(ENUM_S) NSCALAR };
enum { CONTEXT_OUTPUTS(ENUM_C) NCONTEXT_OUTPUT };
enum { ARRAY_OUTPUTS(ENUM_A) NARRAY_OUTPUT };
enum { CLASS_OUTPUTS(ENUM_K) NCLASS_OUTPUT };
enum { TRACE_FIELDS(ENUM_T) NTRACE_FIELD };
enum { TRACE_CONTEXT_FIELDS(ENUM_TC) NTRACE_CONTEXT_FIELD };
#undef ENUM_S
#undef ENUM_C
#undef ENUM_A
#undef ENUM_K
#undef ENUM_T
#undef ENUM_TC

constexpr int OUT_CONTEXT = NSCALAR, OUT_ARRAY = OUT_CONTEXT + NCONTEXT_OUTPUT * MAXC;
constexpr int OUT_CLASS = OUT_ARRAY + NARRAY_OUTPUT * MAXA, NOUT = OUT_CLASS + NCLASS_OUTPUT * MAXA * NCLASS;
constexpr int NTRACE = NTRACE_FIELD + NTRACE_CONTEXT_FIELD * MAXC;

// Context states of the trace: what the context did in the cycle.
enum State { IDLE, COMPUTE, WAIT_HBM, WAIT_REDELIVERY, WAIT_PORT, WAIT_OTHER, STALL };
