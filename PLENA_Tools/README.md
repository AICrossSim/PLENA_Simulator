# PLENA Tools

Runtime utilities for PLENA RTL simulation and testing.

## Installation

### Using direnv (Recommended)

If you're in the PLENA_RTL project root, the `.envrc` will automatically install this package when you run:

```bash
direnv allow
```

### Manual Installation

From the PLENA_RTL root directory:
```bash
pip install -e PLENA_Tools
```

Or from within the PLENA_Tools directory:
```bash
pip install -e .
```

For development (includes pytest and ruff):
```bash
pip install -e ".[dev]"
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0.0
- Cocotb >= 1.8.0

## Modules

- **plena_utils**: Core utilities including cocotb helpers, logging, and quantization
- **memory_mapping**: HBM memory mapping for RTL simulation
- **verification**: RTL verification utilities
