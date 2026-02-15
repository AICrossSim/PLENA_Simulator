{
  stdenv,
  fetchFromGitHub,
  cmake,
  pkgs,
}: let
  yaml-cpp = fetchFromGitHub {
    owner = "jbeder";
    repo = "yaml-cpp";
    rev = "yaml-cpp-0.7.0";
    hash = "sha256-2tFWccifn0c2lU/U1WNg2FHrBohjx8CXMllPJCevaNk=";
  };

  spdlog = fetchFromGitHub {
    owner = "gabime";
    repo = "spdlog";
    rev = "v1.11.0";
    hash = "sha256-kA2MAb4/EygjwiLEjF9EA7k8Tk//nwcKB1+HlzELakQ=";
  };

  argparse = fetchFromGitHub {
    owner = "p-ranav";
    repo = "argparse";
    rev = "v2.9";
    hash = "sha256-vbf4kePi5gfg9ub4aP1cCK1jtiA65bUS9+5Ghgvxt/E=";
  };

  # Patched source derivation which fixes the wrong installation path in argparse.
  # We can't take pkgs.argparse directly because ramulator2's cmake expects the source.
  argparseFixed = stdenv.mkDerivation {
    pname = "argparse";
    version = "2.9";

    src = argparse;
    postPatch = pkgs.argparse.postPatch;
    installPhase = ''
      cp -r . $out
    '';
  };
in
  stdenv.mkDerivation {
    pname = "ramulator2";
    version = "0-unstable-2025-05-07";

    src = fetchFromGitHub {
      owner = "CMU-SAFARI";
      repo = "ramulator2";
      rev = "e442c64b2c0db7afd9d23173925d636ea2895a36";
      hash = "sha256-OnodNdG1kFTEbtOkCX7aRIKX8ZyBmYmGgL4XSzhF0rc=";
    };

    patches = [
      ./fix_density.patch
    ];

    postPatch = ''
      cp ${./ramulator_capi.cc} src/frontend/impl/external_wrapper/ramulator_capi.cc
      cp ${./ramulator_capi.h} src/frontend/impl/external_wrapper/ramulator_capi.h
      sed -i "/gem5_frontend.cpp/aimpl\/external_wrapper\/ramulator_capi.cc" src/frontend/CMakeLists.txt
    '';

    nativeBuildInputs = [
      cmake
    ];
    cmakeFlags = [
      "-DFETCHCONTENT_SOURCE_DIR_YAML-CPP=${yaml-cpp}"
      "-DFETCHCONTENT_SOURCE_DIR_SPDLOG=${spdlog}"
      "-DFETCHCONTENT_SOURCE_DIR_ARGPARSE=${argparseFixed}"
    ];

    # Code can be a bit buggy, so..
    cmakeBuildType = "RelWithDebInfo";
    dontStrip = true;

    installPhase = ''
      mkdir -p $out/lib
      cp ../libramulator.so $out/lib
    '';
  }
