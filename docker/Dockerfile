# PLENA Simulator Docker Image
# Encapsulates the Nix development environment for portable builds

# Use NixOS as base for native Nix support
FROM docker.io/nixos/nix:latest AS base

# Enable flakes and nix-command
RUN mkdir -p /etc/nix && \
    echo "experimental-features = nix-command flakes" >> /etc/nix/nix.conf && \
    echo "sandbox = false" >> /etc/nix/nix.conf

# Set up working directory
WORKDIR /workspace

# Copy Nix configuration files first for better layer caching
COPY flake.nix flake.lock* ./
COPY transactional_emulator/pkgs ./transactional_emulator/pkgs/

# Pre-build Nix dependencies (this layer will be cached)
RUN nix develop --profile /nix-profile -c true

# ============================================================
# Development image - includes all dev tools
# ============================================================
FROM base AS dev

# Copy the entire repository
COPY . .

# Initialize git submodules if .gitmodules exists
RUN if [ -f .gitmodules ]; then \
        nix develop -c git submodule update --init --recursive || true; \
    fi

# Create Python virtual environment and install dependencies
RUN nix develop -c bash -c '\
    if command -v uv &> /dev/null; then \
        uv venv .venv --python python3.12 && \
        . .venv/bin/activate && \
        uv sync && \
        uv pip install -e . ; \
    fi'

# Set up entry point that activates Nix environment
COPY <<'EOF' /entrypoint.sh
#!/usr/bin/env bash
set -e

cd /workspace

# Activate virtual environment if it exists
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

# Execute command in Nix develop shell
exec nix develop -c "$@"
EOF

RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]

# ============================================================
# Build image - for CI/CD builds
# ============================================================
FROM base AS builder

COPY . .

# Build the transactional emulator
RUN nix build .#transactional-emulator -o result

# ============================================================
# Runtime image - minimal image with just the built artifacts
# ============================================================
FROM docker.io/nixos/nix:latest AS runtime

RUN mkdir -p /etc/nix && \
    echo "experimental-features = nix-command flakes" >> /etc/nix/nix.conf

WORKDIR /app

# Copy built artifacts from builder
COPY --from=builder /workspace/result /app/bin

# Copy Python sources and configs needed at runtime
COPY --from=builder /workspace/analytic_models /app/analytic_models
COPY --from=builder /workspace/plena_settings.toml /app/
COPY --from=builder /workspace/pyproject.toml /app/

ENV PATH="/app/bin/bin:${PATH}"

CMD ["transactional-emulator", "--help"]
