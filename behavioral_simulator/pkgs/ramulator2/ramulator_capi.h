#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

struct ramulator;

ramulator* ramulator_new(const char *config);

void ramulator_finalize(ramulator*);

bool ramulator_request(ramulator *val, uint64_t addr, bool write, void (*callback)(void*), void *data);

float ramulator_period(ramulator *val);

void ramulator_tick(ramulator *val);

#ifdef __cplusplus
}
#endif
