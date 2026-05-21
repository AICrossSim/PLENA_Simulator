# Docker Setup for PLENA Simulator

This directory contains Docker configuration that encapsulates the Nix development environment.

## Files

- `Dockerfile` - Multi-stage build with dev, builder, and runtime targets
- `docker-compose.yml` - Service definitions and volume management

## Quick Start

From the repository root:

```bash
# Build and enter development environment
just docker-dev

# Or manually:
docker compose -f docker/docker-compose.yml build dev
docker compose -f docker/docker-compose.yml up -d dev
docker compose -f docker/docker-compose.yml exec dev bash
```

## Available Commands

| Command | Description |
|---------|-------------|
| `just docker-build-dev` | Build development image |
| `just docker-build-all` | Build all images |
| `just docker-dev` | Start and enter dev container |
| `just docker-run <cmd>` | Run command in dev environment |
| `just docker-test <target>` | Run a test target in Docker |
| `just docker-down` | Stop containers |
| `just docker-clean` | Remove containers and cached volumes |
| `just docker-build-runtime` | Build minimal runtime image |

## Build Targets

### dev (default)
Full development environment with:
- Nix flake environment
- Python 3.12 + all dependencies
- Rust toolchain
- All build tools (gcc, clang, cmake, etc.)

### builder
CI/CD build stage that compiles the transactional emulator via `nix build`.

### runtime
Minimal image containing only the built emulator binary and Python analytical models.

## CUDA Support

For GPU-enabled development:

```bash
docker compose -f docker/docker-compose.yml --profile cuda up -d dev-cuda
docker compose -f docker/docker-compose.yml exec dev-cuda bash
```

## Persistent Volumes

The setup uses named volumes to persist:
- `plena-nix-store` - Nix store for faster rebuilds
- `plena-cargo-cache` - Rust/Cargo build cache
- `plena-venv-cache` - Python virtual environment

To clean all caches: `just docker-clean`
