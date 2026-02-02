{
  lib,
  stdenv,
  fetchFromGitHub,
  cmake,
  pkgs,
  nlohmann_json,
  sqlite,
  systemc,
}: let
  dramutils = fetchFromGitHub {
    owner = "tukl-msd";
    repo = "DRAMUtils";
    rev = "v1.11.0";
    hash = "sha256-KQ/BIvjhSreLtZi/Qi0OnJdHjrcgCLK7Nnjb2uxsKVE=";
  };

  drampower = fetchFromGitHub {
    owner = "tukl-msd";
    repo = "DRAMPower";
    rev = "v5.4.1";
    hash = "sha256-Xm/SpFazIF8pOefAuj5M1vpzcYo0GYnT/lrsop6SL/E=";
  };
in
  stdenv.mkDerivation rec {
    pname = "DRAMSys";
    version = "5.2.0";

    src = fetchFromGitHub {
      owner = "tukl-msd";
      repo = "DRAMSys";
      rev = "v${version}";
      hash = "sha256-iq0KbfC27XKeNldRwvfog6fY+NIjiIq9zwy+/9Yugi4=";
    };

    postPatch = ''
      cp ${./dramsys_capi.cc} src/libdramsys/DRAMSys/dramsys_capi.cc
      cp ${./dramsys_capi.h} src/libdramsys/DRAMSys/dramsys_capi.h
      sed -i "/SimConfig.cpp/aDRAMSys\/dramsys_capi.cc" src/libdramsys/CMakeLists.txt
    '';

    nativeBuildInputs = [
        cmake
    ];

    propagatedBuildInputs = [
        nlohmann_json
        sqlite
        systemc
    ];

    cmakeFlags = [
      (lib.cmakeBool "DRAMSYS_USE_FETCH_CONTENT_NLOHMANN_JSON" false)
      (lib.cmakeBool "DRAMSYS_USE_FETCH_CONTENT_SQLITE3" false)
      (lib.cmakeBool "DRAMSYS_USE_FETCH_CONTENT_SYSTEMC" false)
      "-DFETCHCONTENT_SOURCE_DIR_DRAMUTILS=${dramutils}"
      "-DFETCHCONTENT_SOURCE_DIR_DRAMPOWER=${drampower}"
    ];

    cmakeBuildType = "RelWithDebInfo";
    dontStrip = true;

    installPhase = ''
      mkdir $out
      cp -r lib $out/
    '';
  }
