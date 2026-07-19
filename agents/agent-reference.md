# CUDA / CuTe DSL Agent Reference

Reference entry points for agents that write, debug, and optimize CUDA and CuTe DSL code. Inspect the current code and tests first. For API contracts, instruction semantics, or hardware behavior, prefer local documentation and then confirm against official documentation matching the environment version.

## Start by Verifying the Environment

APIs, PTX instructions, and hardware features are version-dependent. Before implementing code or drawing conclusions, record:

```bash
nvcc --version
nvidia-smi
python3 -c "import cutlass; print(cutlass.__file__)"
```

Do not assume that features from a newer CUDA Toolkit, driver, or SM architecture are available in the current environment.

## Local, Searchable Documentation

The `ptx-isa-markdown` submodule is the first source for low-level questions. It covers PTX ISA 9.1, CUDA Runtime API 13.1, and CUDA Driver API 13.1. When the installed toolkit differs, verify the result with the official documentation.

| Question | Local entry point |
| --- | --- |
| PTX instructions, state spaces, memory consistency, WGMMA, and TMA | [PTX index](ptx-isa-markdown/cuda_skill/references/ptx-docs/INDEX.md) |
| Runtime API: memory, streams, events, graphs, and error codes | [Runtime API index](ptx-isa-markdown/cuda_skill/references/cuda-runtime-docs/INDEX.md) |
| Driver API: contexts, modules, VMM, and launches | [Driver API index](ptx-isa-markdown/cuda_skill/references/cuda-driver-docs/INDEX.md) |
| Nsight Systems and Nsight Compute command patterns | [nsys guide](ptx-isa-markdown/cuda_skill/references/nsys-guide.md), [ncu guide](ptx-isa-markdown/cuda_skill/references/ncu-guide.md) |
| Compute Sanitizer, cuda-gdb, and cuobjdump | [debugging tools guide](ptx-isa-markdown/cuda_skill/references/debugging-tools.md) |
| CuTe DSL experiments in this repository | [`../python/cute/`](../python/cute/), [`../python/notes/`](../python/notes/) |

Example searches:

```bash
rg -n "wgmma\.mma_async" agents/ptx-isa-markdown/cuda_skill/references/ptx-docs
rg -n "cudaMallocAsync" agents/ptx-isa-markdown/cuda_skill/references/cuda-runtime-docs
rg -n "cuMemMap" agents/ptx-isa-markdown/cuda_skill/references/cuda-driver-docs
```

## Official Online Documentation

### CUDA Fundamentals, APIs, and Compilation

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/): programming model, memory hierarchy, synchronization, performance principles, and feature availability.
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/): coalescing, bandwidth, concurrency, and optimization techniques.
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/) / [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/): function signatures, parameter constraints, and error semantics.
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/): use when the local snapshot is incomplete or newer instruction behavior must be confirmed.
- [NVCC Compiler Driver](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/contents.html): `-arch`, `-code`, fatbinaries, JIT, and compilation phases.
- [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/): `cuobjdump`, `nvdisasm`, `cu++filt`, and binary inspection.

### CUTLASS and CuTe DSL

- [CUTLASS documentation](https://docs.nvidia.com/cutlass/latest/index.html): the overall architecture of the C++ templates and Python DSLs.
- [CuTe DSL documentation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html): JIT, layouts, types, debugging, autotuning, and framework integration.
- [CuTe DSL Quick Start](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html): installation and version compatibility.
- [CuTe DSL API](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute.html): concrete `cutlass.cute` types and operations.
- [CuTe DSL examples](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL): prefer source and runnable examples from the matching CUTLASS revision.

### Performance Analysis and Correctness

- [Nsight Systems](https://docs.nvidia.com/nsight-systems/): first locate end-to-end bottlenecks, CPU/GPU gaps, and kernel launch sequences.
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html): then inspect throughput, memory access, and occupancy for the hot kernel.
- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html): use memcheck, racecheck, initcheck, and synccheck to investigate correctness problems.
- [CUDA Samples](https://github.com/NVIDIA/cuda-samples): find runnable official examples of APIs and optimization patterns.

## Working Guidelines

1. Establish reproducible correctness and performance baselines; do not change a kernel based on intuition alone.
2. Link conclusions about Runtime/Driver APIs, PTX modifiers, layouts, or architecture capabilities to a specific local file or official section.
3. Use Nsight Systems to find the hotspot before collecting Nsight Compute metrics for that kernel. Change one testable hypothesis at a time.
4. When code fails, minimize the reproduction and run Compute Sanitizer; inspect generated PTX/SASS when needed.
5. Prefer references that match the installed CUDA version, CUTLASS revision, and GPU architecture. Official source and tests take precedence over secondary explanations.
