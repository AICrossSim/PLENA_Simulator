/* Exact integer DFS acceleration only. Not a hardware/HBM simulator.
 * Visit order, incumbent ties, pruning and counters match the Python DFS.
 * The Python caller proves all sums fit signed int64 before entry.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

typedef struct {
  int ng, nr;
  const int64_t *offset, *add, *alt_dep, *suffix, *dep;
  int64_t *stack, *path, *best_path;
  int64_t best;
  uint64_t nodes, leaves, pruned;
} Context;

static void visit(Context *s, int g, int64_t current_dep) {
  s->nodes++;
  int64_t *loads = s->stack + (size_t)g * s->nr;
  int64_t lower = current_dep > s->dep[g] ? current_dep : s->dep[g];
  for (int j = 0; j < s->nr; j++) {
    int64_t value = loads[j] + s->suffix[(size_t)g*s->nr+j];
    if (value > lower) lower = value;
  }
  if (lower >= s->best) { s->pruned++; return; }
  if (g == s->ng) {
    s->leaves++; s->best = lower;
    memcpy(s->best_path, s->path, (size_t)s->ng*sizeof(int64_t));
    return;
  }
  int64_t *next = s->stack + (size_t)(g+1)*s->nr;
  for (int64_t a=s->offset[g]; a<s->offset[g+1]; a++) {
    for (int j=0; j<s->nr; j++) next[j]=loads[j]+s->add[(size_t)a*s->nr+j];
    s->path[g]=a-s->offset[g];
    int64_t next_dep=current_dep>s->alt_dep[a] ? current_dep : s->alt_dep[a];
    visit(s,g+1,next_dep);
  }
}

int exact_visit(int ng, int nr, const int64_t *offset, const int64_t *add,
                const int64_t *alt_dep, const int64_t *suffix,
                const int64_t *dep, const int64_t *initial,
                int64_t *best_path, int64_t *best,
                uint64_t *nodes, uint64_t *leaves, uint64_t *pruned) {
  if (ng<0 || ng>512 || nr<0 || nr>128) return 1;
  size_t size=(size_t)(ng+1)*(nr ? nr : 1);
  int64_t *stack=calloc(size,sizeof(int64_t));
  int64_t *path=calloc((size_t)(ng ? ng : 1),sizeof(int64_t));
  if (!stack || !path) { free(stack);free(path);return 2; }
  memcpy(stack,initial,(size_t)nr*sizeof(int64_t));
  Context s={ng,nr,offset,add,alt_dep,suffix,dep,stack,path,best_path,INT64_MAX,0,0,0};
  visit(&s,0,0);
  *best=s.best; *nodes=s.nodes; *leaves=s.leaves; *pruned=s.pruned;
  free(stack);free(path);
  return s.leaves ? 0 : 3;
}
