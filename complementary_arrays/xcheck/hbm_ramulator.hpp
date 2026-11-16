// Closed-loop HBM service through Ramulator 2 for the layer simulator (cross-check of the ideal HBM stream).
//
// The simulator's fixed-rate HBM server (H bytes per cycle shared by the contexts' landing requests) is replaced by
// Ramulator: every HBM request a context places in its landing queue is registered in program order and read as
// 16-byte transfers at addresses of one weight image; a request is complete when all its transfers have returned and
// every older request of its context is complete, and usable `extra` cycles later. The dispatcher keeps the nominal H
// in its estimates; only the service changes. Requires libramulator (Ramulator 2 with its C API) at link time.
//
// Weight image (Geometry): per layer, the router (BF16), the shared expert (gate/up, then down) and every routed
// expert (gate/up, then down) are contiguous matrices in that order. A matrix is stored in records of 16 columns x
// `group` reduction rows, the packed format the weight port reads; layout bit 0 = 0 keeps every 16-column strip with all
// its K records contiguous (one image for every chip), bit 0 = 1 makes every request contiguous (weights tiled for the
// reading array: an upper bound). Bit 1 XORs the nine channel bits [5, 14) of Ramulator's interleave with bits [14, 23)
// and [23, 32) (a bijection); bit 2 (with bit 0 = 0) pads every strip by 32 bytes so that consecutive strips do not
// start on the same channels.
//
// Issue discipline per cycle, contexts in turn starting at t mod contexts: look = 0 issues the transfers of a context's
// oldest unissued request in order, one 64-byte line per turn, and a rejected transfer stops that context for the cycle;
// look = W > 0 tries the unissued lines among each context's oldest W, and a rejected transfer blocks only its channel
// for the cycle. cap > 0 limits the lines issued per cycle.
//
// Integration into the layer simulator (sim/src), one Hbm object per window when enabled:
//   1. Each HBM input of a packed weight panel carries its Region (matrix tag 0 gate/up, 1 down, 10 router; expert, or
//      `experts` for the shared expert; columns n0, nr; rows k0, kr), set where the program builder packs the panel;
//      every Request gets an id field (-1 until registered).
//   2. Per cycle, in place of the HBM budget of serve_ports: for every active context, register (add) the requests of
//      its landing queue, then those of its prefetch queue, that have beats left and no id; call issue(t) and tick(t);
//      then walk each context's landing and prefetch queues in order with `older = false` and set every registered
//      request's remaining beats to pending(id, older) when that is smaller, crediting the difference to the context's
//      HBM service; a request that reaches zero becomes ready at t + 1 + extra() and is passed to complete(id, t). An
//      unregistered request with beats left sets `older`. Finally account(bytes given this cycle).
//   3. At the end of the window, write_log(window, cycles, init) if logging is on.
#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <deque>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

extern "C" {
struct ramulator;
ramulator* ramulator_new(const char* config);
void ramulator_finalize(ramulator*);
bool ramulator_request(ramulator*, uint64_t addr, bool write, void (*callback)(void*), void* data, int size);
float ramulator_period(ramulator*);  // ns per Ramulator clock
void ramulator_tick(ramulator*);
}

namespace xcheck {

struct Options {
  std::string config;    // Ramulator configuration (JSON text)
  int extra = 65;        // cycles from the last returned transfer to a usable request (100 for an unloaded line)
  int cap = 0;           // lines issued per cycle, 0 = no limit
  int layout = 0;        // address map bits, see above
  int look = 0;          // lines tried per context and cycle, 0 = in order
  int cycle_ps = 1000;   // engine cycle
  std::string log;       // file prefix of the per-window log, empty = none
};

struct Geometry {
  int experts = 0, hidden = 0, width = 0, shared_width = 0, shared_gate = 0;  // E, H, F, S, gate row of the shared expert
  int group = 32, bits = 16;               // record rows and weight bits
  int record_bytes = 0, router_record_bytes = 0;  // one record in the weight format and in BF16, multiples of 64
};

struct Region {
  int expert, tag, n0, nr, k0, kr;
};

class Hbm {
 public:
  static constexpr int MAXCTX = 64, BINS = 8192, CHANNELS = 512;

  Hbm(const Options& o, const Geometry& g, int contexts) : opt(o), geo(g), nc(contexts) {
    assert(nc <= MAXCTX && geo.record_bytes % 64 == 0 && geo.router_record_bytes % 64 == 0);
    r = ramulator_new(opt.config.c_str());
    assert(r);
    tck = double(std::lround(double(ramulator_period(r)) * 1000.0));
    current() = this;
  }
  ~Hbm() {
    if (r) ramulator_finalize(r);
  }
  Hbm(const Hbm&) = delete;
  Hbm& operator=(const Hbm&) = delete;

  int extra() const { return opt.extra; }

  // Register a request of `lines` 64-byte beats of context ctx in program order; returns its id.
  int64_t add(int ctx, const Region& g, int64_t lines, int64_t t) {
    Req q;
    q.ctx = ctx;
    q.expert = g.expert;
    q.tag = g.tag;
    q.bits = g.tag == 10 ? 16 : geo.bits;
    q.n0 = g.n0;
    q.nr = g.nr;
    q.k0 = g.k0;
    q.kr = g.kr;
    q.lines = lines;
    q.enq = t;
    int N, K;
    dims(q.expert, q.tag, N, K);
    const int rb = rec(q.bits), cg0 = q.n0 / 16, nn = cdiv(q.n0 + q.nr, 16) - cg0, kg0 = q.k0 / geo.group,
              gg = cdiv(q.k0 + q.kr, geo.group) - kg0;
    assert(64 * lines == int64_t(nn) * gg * rb);                     // request = whole records of its panel
    assert(cg0 + nn <= cdiv(N, 16) && kg0 + gg <= cdiv(K, geo.group));  // inside its matrix
    if (opt.look > 0) q.part.assign(lines, 0);
    q_.push_back(std::move(q));
    const int64_t id = int64_t(q_.size()) - 1;
    fifo[ctx].push_back(id);
    return id;
  }

  // One cycle of issue.
  void issue(int64_t t) {
    if (next == 0) next = double(t) * opt.cycle_ps;
    const int64_t cap = opt.cap > 0 ? 4LL * opt.cap : std::numeric_limits<int64_t>::max();
    int64_t n = 0;
    std::array<bool, MAXCTX> blocked{};
    bool any = opt.look <= 0;
    const int start = int(t % nc);
    if (opt.look > 0) {
      std::array<char, CHANNELS> busy_ch{};
      for (int z = 0; z < nc && n < cap; z++) {
        auto& f = fifo[(start + z) % nc];
        while (!f.empty() && q_[f.front()].issued == 4 * q_[f.front()].lines) f.pop_front();
        int64_t seen = 0;
        for (size_t fi = 0; fi < f.size() && seen < opt.look && n < cap; fi++) {
          const int64_t id = f[fi];
          Req& q = q_[id];
          while (q.nextl < q.lines && q.part[q.nextl] == 4) q.nextl++;
          for (int64_t l = q.nextl; l < q.lines && seen < opt.look && n < cap; l++) {
            if (q.part[l] == 4) continue;
            seen++;
            while (q.part[l] < 4 && n < cap) {
              const uint64_t a = addr(q, 4 * l + q.part[l]);
              const int ch = int((a >> 5) & (CHANNELS - 1));
              if (busy_ch[ch]) break;
              if (!send(a, id, t)) {
                busy_ch[ch] = 1;
                rejects++;
                break;
              }
              if (q.first < 0) q.first = t;
              q.last = t;
              q.part[l]++;
              q.issued++;
              n++;
            }
          }
        }
      }
    }
    while (any && n < cap) {
      any = false;
      for (int z = 0; z < nc && n < cap; z++) {
        const int c = (start + z) % nc;
        if (blocked[c]) continue;
        auto& f = fifo[c];
        while (!f.empty() && q_[f.front()].issued == 4 * q_[f.front()].lines) f.pop_front();
        if (f.empty()) {
          blocked[c] = true;
          continue;
        }
        const int64_t id = f.front();
        Req& q = q_[id];
        const int64_t end = (q.issued / 4 + 1) * 4;
        while (q.issued < end && n < cap) {
          if (!send(addr(q, q.issued), id, t)) {
            blocked[c] = true;
            rejects++;
            break;
          }
          if (q.first < 0) q.first = t;
          q.last = t;
          q.issued++;
          n++;
        }
        if (!blocked[c]) any = true;
      }
    }
  }

  // Advance Ramulator to the end of engine cycle t (call after issue).
  void tick(int64_t t) {
    unissued = false;
    for (int c = 0; c < nc && !unissued; c++)
      for (int64_t id : fifo[c])
        if (q_[id].issued < 4 * q_[id].lines) {
          unissued = true;
          break;
        }
    busy_now = outstanding > 0;
    now = t;
    current() = this;
    while (next < double(t + 1) * opt.cycle_ps) {
      ramulator_tick(r);
      next += tck;
    }
  }

  // Beats still pending for request id: one while its data is complete but an older request of its context is not.
  int64_t pending(int64_t id, bool& older) const {
    const Req& q = q_[id];
    int64_t left = (4 * q.lines - q.done + 3) / 4;
    if (older && left == 0) left = 1;
    if (left > 0) older = true;
    return left;
  }

  void complete(int64_t id, int64_t t) { q_[id].fin = t; }

  // End of cycle: bytes credited to the contexts this cycle.
  void account(int64_t given) {
    bytes += given;
    if (busy_now || unissued) busy_cycles++;
    if (unissued) {
      backlog_cycles++;
      backlog_bytes += given;
    }
  }

  // Per-window log: summary JSON (bytes, busy and backlogged cycles, transfer latency histogram) and request table.
  void write_log(int window, int64_t cycles, int64_t init) const {
    assert(outstanding == 0);
    if (opt.log.empty()) return;
    const std::string stem = opt.log + "_w" + std::to_string(window);
    std::ofstream f(stem + ".json");
    f << "{\"window\":" << window << ",\"cycles\":" << cycles << ",\"init\":" << init << ",\"bytes\":" << bytes
      << ",\"busy_cycles\":" << busy_cycles << ",\"backlog_cycles\":" << backlog_cycles
      << ",\"backlog_bytes\":" << backlog_bytes << ",\"transfers\":" << transfers << ",\"rejects\":" << rejects
      << ",\"lat_sum\":" << lat_sum << ",\"lat_n\":" << lat_n << ",\"tck_ps\":" << tck << ",\"extra\":" << opt.extra
      << ",\"cap\":" << opt.cap << ",\"layout\":" << opt.layout << ",\"requests\":" << q_.size() << ",\"hist\":[";
    int last = 0;
    for (int i = 0; i < BINS; i++)
      if (hist[i]) last = i;
    for (int i = 0; i <= last; i++) f << (i ? "," : "") << hist[i];
    f << "]}\n";
    std::ofstream g(stem + "_req.csv");
    g << "ctx,expert,tag,bits,n0,nr,k0,kr,lines,enq,first,last,fin\n";
    for (const Req& q : q_)
      g << q.ctx << "," << q.expert << "," << q.tag << "," << q.bits << "," << q.n0 << "," << q.nr << "," << q.k0 << ","
        << q.kr << "," << q.lines << "," << q.enq << "," << q.first << "," << q.last << "," << q.fin << "\n";
  }

 private:
  struct Req {
    int ctx, expert, tag, bits, n0, nr, k0, kr;
    int64_t lines, issued = 0, done = 0, enq = 0, first = -1, last = -1, fin = -1, nextl = 0;
    std::vector<uint8_t> part;  // transfers issued per line (look > 0)
  };

  Options opt;
  Geometry geo;
  int nc;
  ramulator* r = nullptr;
  double tck = 0, next = 0;
  int64_t now = 0;
  std::vector<Req> q_;
  std::array<std::deque<int64_t>, MAXCTX> fifo;
  std::vector<int64_t> hist = std::vector<int64_t>(BINS, 0);  // transfer latency in engine cycles, issue to return
  int64_t bytes = 0, busy_cycles = 0, backlog_cycles = 0, backlog_bytes = 0, transfers = 0, rejects = 0,
          outstanding = 0, lat_sum = 0, lat_n = 0;
  bool busy_now = false, unissued = false;

  static Hbm*& current() {
    static thread_local Hbm* h = nullptr;
    return h;
  }
  static int cdiv(int a, int b) { return (a + b - 1) / b; }

  static void returned(void* d) {
    Hbm* w = current();
    const uint64_t v = uint64_t(uintptr_t(d));
    const int64_t id = int64_t(v >> 32), issued_at = int64_t(v & 0xffffffffULL);
    w->q_[id].done++;
    w->outstanding--;
    const int64_t l = w->now - issued_at;
    w->hist[std::min<int64_t>(l, BINS - 1)]++;
    w->lat_sum += l;
    w->lat_n++;
  }

  bool send(uint64_t a, int64_t id, int64_t t) {
    if (!ramulator_request(r, a, false, returned, (void*)uintptr_t((uint64_t(id) << 32) | uint64_t(t)), 16)) return false;
    transfers++;
    outstanding++;
    return true;
  }

  int rec(int bits) const { return bits == 16 ? geo.router_record_bytes : geo.record_bytes; }

  void dims(int expert, int tag, int& N, int& K) const {
    if (tag == 10) {
      N = geo.experts;
      K = geo.hidden;
      return;
    }
    const int n = expert == geo.experts ? geo.shared_width : geo.width;
    if (tag == 0) {
      N = 2 * n + (expert == geo.experts && geo.shared_gate ? 1 : 0);
      K = geo.hidden;
    } else {
      N = geo.hidden;
      K = n;
    }
  }

  // Bytes of an N x K matrix, plus `pad` bytes per 16-column strip.
  int64_t msize(int N, int K, int bits, int pad) const {
    return int64_t(cdiv(N, 16)) * (int64_t(cdiv(K, geo.group)) * rec(bits) + pad);
  }

  uint64_t mbase(int expert, int tag) const {
    const int pad = (opt.layout & 4) ? 32 : 0, E = geo.experts, H = geo.hidden, S = geo.shared_width, F = geo.width;
    const int64_t rt = msize(E, H, 16, pad), sgu = msize(2 * S + geo.shared_gate, H, geo.bits, pad),
                  sdn = msize(H, S, geo.bits, pad), gu = msize(2 * F, H, geo.bits, pad), dn = msize(H, F, geo.bits, pad);
    if (tag == 10) return 0;
    if (expert == E) return rt + (tag == 0 ? 0 : sgu);
    return rt + sgu + sdn + int64_t(expert) * (gu + dn) + (tag == 0 ? 0 : gu);
  }

  // Address of transfer x (16 bytes) of request q.
  uint64_t addr(const Req& q, int64_t x) const {
    int N, K;
    dims(q.expert, q.tag, N, K);
    const int rb = rec(q.bits), lr = rb / 64, KG = cdiv(K, geo.group);
    const int cg0 = q.n0 / 16, nn = cdiv(q.n0 + q.nr, 16) - cg0, kg0 = q.k0 / geo.group,
              gg = cdiv(q.k0 + q.kr, geo.group) - kg0;
    const int64_t line = x / 4, sub = x % 4;
    uint64_t a;
    if ((opt.layout & 1) == 0) {
      const int64_t strip = line / (int64_t(gg) * lr), off = line % (int64_t(gg) * lr);
      const int64_t span = int64_t(KG) * rb + ((opt.layout & 4) ? 32 : 0);
      a = mbase(q.expert, q.tag) + int64_t(cg0 + strip) * span + int64_t(kg0) * rb + off * 64;
    } else {
      a = mbase(q.expert, q.tag) + (int64_t(cg0) * KG + int64_t(kg0) * nn) * rb + line * 64;
    }
    a += 16 * sub;
    if (opt.layout & 2) a ^= (((a >> 14) ^ (a >> 23) ^ (a >> 32)) & 511ULL) << 5;
    return a;
  }
};

}  // namespace xcheck
