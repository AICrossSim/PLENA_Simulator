#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

struct ramulator;

ramulator* ramulator_new(const char *config);

void ramulator_finalize(ramulator*);

bool ramulator_request(ramulator *val, uint64_t addr, bool write, void (*callback)(void*), void *data, int size);

float ramulator_period(ramulator *val);

void ramulator_tick(ramulator *val);

// Versioned calibration interface. All getters run under the Rust model lock.
uint32_t ramulator_capi_version();
uint32_t ramulator_tx_bytes(ramulator *val);
// YAML snapshot; returns required bytes including NUL, without partial writes.
uint64_t ramulator_stats(ramulator *val, char *buffer, uint64_t capacity);
const char *ramulator_library_path();

#ifdef __cplusplus
}
#endif
