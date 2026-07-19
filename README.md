# CuTe DSL Cookbook

A personal repository for learning, experimenting with, and documenting CUDA CuTe DSL.

## Contents

- `python/cute/`: CuTe DSL examples, including matrix multiplication, transpose, elementwise operations, and data movement.
- `python/notes/`: Experiments and notes on layouts, TMA, swizzles, copies, and related concepts.
- `python/leetgpu/`: LeetGPU exercise solutions.
- `logs/`: Benchmark results, assembly dumps, and experiment logs.
- `tutorials/`: CUDA learning resources included as Git submodules.
- `agents/`: Agent-facing skills and technical reference material.

## Get Learning Materials

Initialize the submodules after cloning the repository:

```bash
git submodule update --init --recursive
```

## Python environment

The Python environment is managed by `uv` and targets the CUDA 13 generation
of CuTe DSL. `pyproject.toml` and `uv.lock` are the source of truth, and the
virtual environment is created inside the checkout as `.venv`.

Create or update the environment:

```bash
uv lock
uv sync --frozen
```

Activate the environment and verify the installation:

```bash
source .venv/bin/activate
python -c 'import cutlass; import torch; print(cutlass.__version__, torch.__version__)'
```

Machine-specific helper scripts belong in the ignored `scripts/` directory and
must not be committed.
