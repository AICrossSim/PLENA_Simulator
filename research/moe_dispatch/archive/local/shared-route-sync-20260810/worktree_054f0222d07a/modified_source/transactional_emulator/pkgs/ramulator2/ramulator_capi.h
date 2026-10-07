#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

struct ramulator;

typedef struct ramulator_channel_telemetry {
    uint64_t cycles;
    uint64_t read_requests;
    uint64_t read_requests_served;
    uint64_t write_requests;
    uint64_t write_requests_served;
    uint64_t row_hits;
    uint64_t row_misses;
    uint64_t row_conflicts;
} ramulator_channel_telemetry;

ramulator* ramulator_new(const char *config);

void ramulator_finalize(ramulator*);

bool ramulator_request(ramulator *val, uint64_t addr, bool write, void (*callback)(void*), void *data, int size);

float ramulator_period(ramulator *val);

void ramulator_tick(ramulator *val);

/** Snapshot native controller counters for one zero-based DRAM channel. */
bool ramulator_get_channel_telemetry(
    ramulator *val,
    uint32_t channel,
    ramulator_channel_telemetry *out
);

#ifdef __cplusplus
}
#endif
