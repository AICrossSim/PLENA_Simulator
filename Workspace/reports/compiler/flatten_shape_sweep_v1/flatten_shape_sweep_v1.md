# Flattened MatrixMachine Shape Sweep v1

All points use the latest combined compiler path (`rtl-v4`, selector
hoisting, reduction overwrite, K-major broadcast, loop-AGU-v1) and
ideal-II1 compute timing. Matrix SRAM capacity is held at 8,388,608
FP entries (`MLEN * depth`) in every shape.

> The large Matrix shapes are structural timing/area extrapolations,
> and broadcast Matrix RTL remains unvalidated.

## Main Findings

- At 1,048,576 PEs, `2048/512` is 35.72% slower than `1024/1024` for `seq=482, batch=16`, while using 18.91% more area.
- At `seq=8192`, the same flattened shape is 27.08% faster. Its area-latency product is 13.30% lower, whereas the short-context product is 61.38% higher.
- The long-context gain is not a direct MatrixMachine throughput gain. Matrix work changes from 14.434M to 19.432M cycles, while Vector falls from 22.888M to 13.599M and Scalar from 24.165M to 12.140M.
- A moderate flattening from `512/512` to `1024/256` at 262,144 PEs is 37.35% faster at `seq=8192`; the more extreme `2048/128` point gives back part of that gain.
- Tail behavior is decisive. `2048/512` is 12.44% faster at `seq=4096`, but is effectively tied (0.00% slower) at `seq=4097`. The single-row tail invokes full-width BMM because active-row BMM is unavailable.
- At `seq=8192`, changing `1024/1024` to `2048/512` increases physical HBM reads by 8.74% and writes by 100.00%. The evaluated points remain compute-bound, but this traffic matters for energy.

The current implementation therefore has a real long-context shape
benefit, but it should be described as a whole-machine tiling benefit
caused largely by wider `MLEN=VLEN`, not as proof that the flattened
systolic datapath alone has higher effective throughput.

## short_b16

### PE budget 262,144

| MLEN/BLEN | Ratio | Area mm2 | Compute Mcy | Roofline ms/layer | Matrix Mcy | Vector Mcy | vs least-flat |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512/512 | 1.0 | 102.749 | 37.371 | 39.621 | 21.619 | 10.924 | +0.00% |
| 1024/256 | 4.0 | 121.344 | 39.594 | 40.992 | 27.723 | 7.896 | -3.46% |
| 2048/128 | 16.0 | 127.501 | 78.593 | 80.024 | 64.909 | 7.019 | -101.98% |

### PE budget 524,288

| MLEN/BLEN | Ratio | Area mm2 | Compute Mcy | Roofline ms/layer | Matrix Mcy | Vector Mcy | vs least-flat |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024/512 | 2.0 | 220.602 | 26.373 | 27.747 | 14.604 | 7.896 | +0.00% |
| 2048/256 | 8.0 | 237.824 | 43.813 | 45.218 | 32.618 | 7.019 | -62.96% |

### PE budget 1,048,576

| MLEN/BLEN | Ratio | Area mm2 | Compute Mcy | Roofline ms/layer | Matrix Mcy | Vector Mcy | vs least-flat |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024/1024 | 1.0 | 379.667 | 19.947 | 21.309 | 8.207 | 7.896 | +0.00% |
| 2048/512 | 4.0 | 451.446 | 27.531 | 28.921 | 16.972 | 7.019 | -35.72% |

## long_4096

### PE budget 262,144

| MLEN/BLEN | Ratio | Area mm2 | Compute Mcy | Roofline ms/layer | Matrix Mcy | Vector Mcy | vs least-flat |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512/512 | 1.0 | 102.749 | 41.424 | 42.705 | 14.288 | 12.993 | +0.00% |
| 1024/256 | 4.0 | 121.344 | 29.394 | 30.192 | 15.105 | 7.250 | +29.30% |
| 2048/128 | 16.0 | 127.501 | 42.160 | 43.012 | 32.868 | 4.702 | -0.72% |

### PE budget 524,288

| MLEN/BLEN | Ratio | Area mm2 | Compute Mcy | Roofline ms/layer | Matrix Mcy | Vector Mcy | vs least-flat |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024/512 | 2.0 | 220.602 | 22.811 | 23.597 | 8.569 | 7.250 | +0.00% |
| 2048/256 | 8.0 | 237.824 | 25.120 | 25.959 | 16.717 | 4.702 | -10.01% |

### PE budget 1,048,576

| MLEN/BLEN | Ratio | Area mm2 | Compute Mcy | Roofline ms/layer | Matrix Mcy | Vector Mcy | vs least-flat |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024/1024 | 1.0 | 379.667 | 19.666 | 20.445 | 5.438 | 7.250 | +0.00% |
| 2048/512 | 4.0 | 451.446 | 17.069 | 17.901 | 8.896 | 4.702 | +12.44% |

## long_4097

### PE budget 262,144

| MLEN/BLEN | Ratio | Area mm2 | Compute Mcy | Roofline ms/layer | Matrix Mcy | Vector Mcy | vs least-flat |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512/512 | 1.0 | 102.749 | 44.153 | 45.547 | 16.633 | 13.240 | +0.00% |
| 1024/256 | 4.0 | 121.344 | 33.281 | 34.336 | 18.570 | 7.522 | +24.61% |
| 2048/128 | 16.0 | 127.501 | 59.219 | 60.455 | 48.383 | 5.144 | -32.73% |

### PE budget 524,288

| MLEN/BLEN | Ratio | Area mm2 | Compute Mcy | Roofline ms/layer | Matrix Mcy | Vector Mcy | vs least-flat |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024/512 | 2.0 | 220.602 | 25.368 | 26.409 | 10.718 | 7.522 | +0.00% |
| 2048/256 | 8.0 | 237.824 | 33.692 | 34.909 | 24.457 | 5.144 | -32.19% |

### PE budget 1,048,576

| MLEN/BLEN | Ratio | Area mm2 | Compute Mcy | Roofline ms/layer | Matrix Mcy | Vector Mcy | vs least-flat |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024/1024 | 1.0 | 379.667 | 21.986 | 23.019 | 7.353 | 7.522 | +0.00% |
| 2048/512 | 4.0 | 451.446 | 21.815 | 23.021 | 12.990 | 5.144 | -0.00% |

## long_8192

### PE budget 262,144

| MLEN/BLEN | Ratio | Area mm2 | Compute Mcy | Roofline ms/layer | Matrix Mcy | Vector Mcy | vs least-flat |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512/512 | 1.0 | 102.749 | 134.177 | 136.411 | 36.528 | 42.762 | +0.00% |
| 1024/256 | 4.0 | 121.344 | 83.774 | 85.467 | 33.526 | 22.888 | +37.35% |
| 2048/128 | 16.0 | 127.501 | 97.743 | 99.404 | 67.392 | 13.599 | +27.13% |

### PE budget 524,288

| MLEN/BLEN | Ratio | Area mm2 | Compute Mcy | Roofline ms/layer | Matrix Mcy | Vector Mcy | vs least-flat |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024/512 | 2.0 | 220.602 | 70.664 | 72.333 | 20.518 | 22.888 | +0.00% |
| 2048/256 | 8.0 | 237.824 | 62.928 | 64.564 | 35.067 | 13.599 | +10.74% |

### PE budget 1,048,576

| MLEN/BLEN | Ratio | Area mm2 | Compute Mcy | Roofline ms/layer | Matrix Mcy | Vector Mcy | vs least-flat |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024/1024 | 1.0 | 379.667 | 64.552 | 66.208 | 14.434 | 22.888 | +0.00% |
| 2048/512 | 4.0 | 451.446 | 46.657 | 48.278 | 19.432 | 13.599 | +27.08% |

## Claim Boundary

- Equal PE count does not imply equal modeled area; the report gives
  both values and fixes Matrix SRAM bit capacity instead of tile count.
- `ideal-II1` makes every Vector/Scalar/Control instruction one cycle.
  The result is a DSE architectural estimate, not cycle-exact RTL.
- Matrix timing is measured only at small shapes and structurally
  extrapolated here. Broadcast BMM remains RTL-unvalidated.
- `MLEN=VLEN` couples Matrix shape to Vector tiling. The reported
  end-to-end gain cannot be attributed to MatrixMachine alone.
