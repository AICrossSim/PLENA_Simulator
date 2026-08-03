{
  stdenv,
  fetchFromGitHub,
  cmake,
  pkgs,
  lib,
}: let
  yaml-cpp = fetchFromGitHub {
    owner = "jbeder";
    repo = "yaml-cpp";
    rev = "yaml-cpp-0.9.0";
    hash = "sha256-+FOsPQY44h1g9tEw3O281LkiYKXdW2jnFKw+oTRkhGw=";
  };
in
  stdenv.mkDerivation {
    pname = "ramulator";
    version = "2.1-unstable-2026-07-30";

    src = fetchFromGitHub {
      owner = "CMU-SAFARI";
      repo = "ramulator2";
      rev = "b3efdc5019a312874961a8c226097eb0581f2b5f";
      hash = "sha256-cVHRY2TbYl+srtqH+jV0JJ3G/TWIJ1yGO83ngucIve4=";
    };

    postPatch = ''
      cp ${./ramulator_capi.cc} src/ramulator/frontend/impl/ramulator_capi.cc
      cp ${./ramulator_capi.h} src/ramulator/frontend/impl/ramulator_capi.h
      sed -i "/impl\/external.cpp/aimpl\/ramulator_capi.cc" src/ramulator/frontend/CMakeLists.txt
    '';

    buildInputs = [
      pkgs.fmt_10
    ];

    nativeBuildInputs = [
      cmake
    ];
    cmakeFlags = [
	(lib.cmakeBool "RAMULATOR_PYTHON_BINDINGS" false)
      "-DFETCHCONTENT_SOURCE_DIR_YAML-CPP=${yaml-cpp}"
      "-DFETCHCONTENT_SOURCE_DIR_FMT=${pkgs.fmt_10.src}"
    ];

    # Code can be a bit buggy, so..
    cmakeBuildType = "RelWithDebInfo";
    dontStrip = true;

    installPhase = ''
      mkdir -p $out/lib
      cp ../libramulator.so $out/lib
    '';
  }
