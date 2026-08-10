#!/usr/bin/env bash
# Build libramulator.so with the host toolchain for machines where the Nix
# dev shell cannot materialise its Rust toolchain. Mirrors
# pkgs/ramulator2/default.nix exactly: same upstream revision, same patch,
# same C API wrapper, RelWithDebInfo, unstripped. The library lands in
# target/host-deps/ramulator2/lib; export LIBRARY_PATH and LD_LIBRARY_PATH
# to that directory before running cargo build.
set -euo pipefail

EMULATOR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG_DIR="$EMULATOR_ROOT/pkgs/ramulator2"
DEPS_DIR="$EMULATOR_ROOT/target/host-deps"
SRC_DIR="$DEPS_DIR/ramulator2-src"
OUT_DIR="$DEPS_DIR/ramulator2"

RAMULATOR_REV="be93be78055d922aa1d4d33e15bcc8f2b0c61a9d"
RAMULATOR_REPO="https://github.com/CMU-SAFARI/ramulator2"

mkdir -p "$DEPS_DIR"
if [[ ! -d "$SRC_DIR/.git" ]]; then
    git clone "$RAMULATOR_REPO" "$SRC_DIR"
fi
git -C "$SRC_DIR" fetch --quiet origin
git -C "$SRC_DIR" checkout --quiet "$RAMULATOR_REV"
git -C "$SRC_DIR" clean -fdx --quiet

git -C "$SRC_DIR" apply "$PKG_DIR/hbm_nbl.patch"
cp "$PKG_DIR/ramulator_capi.cc" "$SRC_DIR/src/frontend/impl/external_wrapper/ramulator_capi.cc"
cp "$PKG_DIR/ramulator_capi.h" "$SRC_DIR/src/frontend/impl/external_wrapper/ramulator_capi.h"
sed -i "/gem5_frontend.cpp/aimpl\/external_wrapper\/ramulator_capi.cc" \
    "$SRC_DIR/src/frontend/CMakeLists.txt"

cmake -S "$SRC_DIR" -B "$SRC_DIR/build" -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build "$SRC_DIR/build" -j "${BUILD_JOBS:-8}"

# The library links at the source root, one level above the build tree,
# matching the packaged derivation's installPhase.
mkdir -p "$OUT_DIR/lib"
cp "$SRC_DIR/libramulator.so" "$OUT_DIR/lib/"
echo "libramulator.so -> $OUT_DIR/lib"
echo "export LIBRARY_PATH=$OUT_DIR/lib:\${LIBRARY_PATH:-}"
echo "export LD_LIBRARY_PATH=$OUT_DIR/lib:\${LD_LIBRARY_PATH:-}"
