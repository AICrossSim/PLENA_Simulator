"""
TVM tilelang end-to-end tests for the transactional emulator.

These tests use the tilelang_tvm_compiler to compile kernels written in the
tilelang DSL (TVM PrimFunc) to PLENA ISA, then run them through the Rust
transactional emulator and verify numerical correctness against CPU golden.

Tests follow the pattern:
  1. Use TvmTestbenchSpec to define kernel + parameters
  2. Call run(spec) to compile and generate simulation environment
  3. Call run_and_assert() to execute emulator and verify results

Available tests:
  - linear_min_test.py: Multi-tile GEMM (C = A @ B^T)
  - gelu_min_test.py: GELU activation (tanh approximation)

Run with:
  just test-tilelang-linear-min
  just test-tilelang-gelu-min
"""
