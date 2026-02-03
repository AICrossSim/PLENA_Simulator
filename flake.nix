{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    systems.url = "github:nix-systems/default-linux";
    flake-utils = {
      url = "github:numtide/flake-utils";
      inputs.systems.follows = "systems";
    };
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    ...
  } @ inputs: let
    lib = nixpkgs.lib;
  in
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {
        inherit system;
      };
      rustPlatform = pkgs.rustPlatform;
      llvm14 = pkgs.llvmPackages_14;
    in rec {
      # ---------- Formatter ----------
      formatter = pkgs.alejandra;

      # ---------- Packages ----------
      packages =
        let
          customPkgs = if builtins.pathExists ./behavioral_simulator/pkgs
                      then import ./behavioral_simulator/pkgs { inherit pkgs; }
                      else {};
        in customPkgs // rec {
        # Build the behavioral simulator as a Nix package
        behavioral-simulator = rustPlatform.buildRustPackage {
          pname = "behavioral-simulator";
          version = "0.1.0";
          src = pkgs.lib.cleanSource ./behavioral_simulator;

          # Use the Cargo.lock from the behavioral_simulator subdir
          cargoLock = {
            lockFile = ./behavioral_simulator/Cargo.lock;
          };

          # Add any system libraries your Rust crate needs
          buildInputs = with pkgs; [
            openssl
          ] ++ (if customPkgs ? ramulator2 then [ customPkgs.ramulator2 ] else []);

          nativeBuildInputs = with pkgs; [
            pkg-config
          ];

          # Set up library paths for ramulator2
          preBuild = if customPkgs ? ramulator2 then ''
            export LD_LIBRARY_PATH="${customPkgs.ramulator2}/lib:$LD_LIBRARY_PATH"
            export LIBRARY_PATH="${customPkgs.ramulator2}/lib:$LIBRARY_PATH"
          '' else "";

          # Set environment variables if needed
          # RUSTFLAGS = "-C target-cpu=native";
        };

        # Make behavioral-simulator the default package
        default = behavioral-simulator;
      };

      # ---------- Development Shells ----------
      devShells =
        let
          customPkgs = if builtins.pathExists ./behavioral_simulator/pkgs
                      then import ./behavioral_simulator/pkgs { inherit pkgs; }
                      else {};
        in {
        default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Include ramulator2 from your custom packages
            (if customPkgs ? ramulator2 then customPkgs.ramulator2 else null)

            # --- Verilog/SystemVerilog toolchain ---
            verilator
            verible

            # --- Compilers / build tools ---
            gcc
            gnumake
            cmake
            ninja
            pkg-config
            autoconf
            flex
            bison
            ccache
            help2man

            # --- LLVM/Clang (plus specific 14.x, if needed) ---
            clang
            llvmPackages.clang-unwrapped
            llvmPackages.lld
            clang-tools
            llvm14.clang
            llvm14.lld

            # --- General dev / utils ---
            git
            wget
            unzip
            vim
            htop
            xdg-utils
            parallel
            just

            # --- Crypto / SSL / IDN ---
            openssl
            libidn

            # --- Performance / NUMA ---
            gperftools
            numactl

            # --- Python ---
            python312
            python312Packages.pip
            python312Packages.sphinx

            # --- Math / BLAS / LAPACK / Fortran ---
            openblas
            lapack
            gfortran

            # --- Graphics / docs ---
            graphviz

            # --- Multimedia / FFmpeg (libavformat, libswscale) ---
            ffmpeg

            # --- SDL 1.2 + SDL2 stacks ---
            SDL
            SDL_image
            SDL_mixer
            SDL_ttf
            smpeg
            portmidi
            SDL2
            SDL2_image
            SDL2_mixer
            SDL2_ttf
            xorg.libXtst
          ];

          nativeBuildInputs = with pkgs; [
            rustup
            uv
          ];

          # Set up environment for ramulator2 library
          shellHook = let
            ramulatorPath = if customPkgs ? ramulator2 then "${customPkgs.ramulator2}/lib" else "";
          in ''
            export PYTHONPATH="$PWD:$PWD/tools:''${PYTHONPATH:-}"
            ${if customPkgs ? ramulator2 then ''
              export LD_LIBRARY_PATH="${ramulatorPath}:$LD_LIBRARY_PATH"
              export LIBRARY_PATH="${ramulatorPath}:$LIBRARY_PATH"
              export PKG_CONFIG_PATH="${ramulatorPath}/pkgconfig:$PKG_CONFIG_PATH"
            '' else ""}

            echo ">>> Toolchain versions:"
            echo "Verilator:    $(verilator --version 2>/dev/null || echo not found)"
            echo "Verible:      $(verible-verilog-format --version 2>/dev/null || echo not found)"
            echo "Clang:        $(clang --version | head -n1 2>/dev/null || echo not found)"
            echo "GCC:          $(gcc --version | head -n1 2>/dev/null || echo not found)"
            echo "CMake:        $(cmake --version | head -n1 2>/dev/null || echo not found)"
            echo "Python 3.12:  $(python3.12 --version 2>/dev/null || echo not found)"
            echo "FFmpeg:       $(ffmpeg -version | head -n1 2>/dev/null || echo not found)"
            echo "Ramulator2:   ${if customPkgs ? ramulator2 then "library at ${ramulatorPath}" else "not available"}"
          '';
        };
      };
    });
}