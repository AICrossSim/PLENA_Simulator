#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

struct dramsys;

dramsys* dramsys_new(const char *config_path);

void dramsys_finalize(dramsys*);

bool dramsys_request(dramsys *val, uint64_t addr, bool write, void (*callback)(void*), void *data);

float dramsys_period(dramsys *val);

void dramsys_tick(dramsys *val);

#ifdef __cplusplus
}
#endif
