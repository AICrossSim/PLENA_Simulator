# Broad unit checks, initial attempt

Actual unedited logs and JUnit results are retained. Research: 281 tests and 282 subtests passed. Analytical: 387 passed, 67 failed, 14 errors, 10 skipped. Rust did not execute tests because default rustup toolchain was unavailable. Failures are not represented as passes. Compiler integration was still in progress. The supplied frozen native fixture is source-hash matched and distinct from fresh Rust compilation.

Commands, package versions, source commit and binary SHA are in UNIT_CHECKS.json. FAILURE_CLASSIFICATION.json records observed source/fixture causes and newly generated scratch temp relocation. No GPU benchmarks or checkpoint inference were run.
