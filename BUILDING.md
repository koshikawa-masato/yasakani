# Building yskn.dll from source

The release page has a prebuilt `yskn.dll`. You can also build it yourself. You do not need a GPU to build it, only to run it.
日本語：Visual Studio 2022 Build Tools（C++）、CUDA Toolkit、CMake、Ninja を入れ、「x64 Native Tools Command Prompt for VS 2022」で下の 2 つのコマンドを実行します。

## What the release was built with (2026-11-15)

| | |
|---|---|
| OS | Windows 11, x64 |
| C++ compiler | Microsoft Visual C++ 19.44.35226 (Visual Studio 2022 Build Tools, toolset 14.44.35207) |
| CUDA | NVIDIA CUDA Toolkit 13.2 (nvcc 13.2.78) |
| Build tools | CMake (3.24 or newer is required) and Ninja |
| Target | one DLL with machine code for compute capability 8.6 (`sm_86`, RTX 30 series), 8.9 (`sm_89`, RTX 40 series) and 12.0 (`sm_120`, RTX 50 series), plus PTX for `compute_120`. Tested on the RTX 5090 only |

The `yskn.dll` on the release page is this build (SHA-256 `6f8c20644dfa39ed0dea396133a882ecc4451ec19f7926dd737a90540e08e838`). It was built
from this source with the two commands below (the CUDA compiler was named explicitly with `-DCMAKE_CUDA_COMPILER`). Its RTX 50 code is the
code of the build the YASAKANI speed tables in [docs/MEASUREMENTS.md](docs/MEASUREMENTS.md) were measured with (`0f5d973b…0bbb`, 2026-11-07),
plus two fixes (a missing synchronization between GPU threads, and a scratch buffer per engine for the KV5 encoder); the host code also
checks the GPU when it opens a model. In our checks on the RTX 5090 both builds wrote the same tokens. Other compiler or CUDA versions
have not been tried. Compute capability 12.0 needs a recent CUDA Toolkit.

## Steps

Open the **x64 Native Tools Command Prompt for VS 2022** (so that the compiler is on the path), go to the source folder (the one with
`CMakeLists.txt`) and run:

```bat
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

The result is `build\yskn.dll` (about 56 MB; the CUDA runtime is linked statically, so no CUDA Toolkit is needed to run it). The recorded build of
the release took about four minutes and printed many compiler warnings; they are known.

Pass the file to the tools with `--dll`:

```bat
yasakani info --model-dir D:\models\qwen38 --dll build\yskn.dll
```

`yasakani info` prints the SHA-256 of the DLL. A build made with a different compiler or CUDA version will not be byte-identical to the file on the
release page; that is expected. Check the model files with `yasakani verify`, and the whole installation with `yasakani selfcheck`.

## GPUs

By default the build makes machine code for four GPU generations in one DLL: `sm_70` (Tesla V100, this fork), `sm_86` (RTX 30 series),
`sm_89` (RTX 40 series) and `sm_120` (RTX 50 series). Two features need compute capability 9.0 or newer: programmatic dependent launch
and the bulk L2 prefetch. The code for the older generations is built without them, and the engine uses them only on a GPU that runs the
RTX 50 code. When it opens a model, the engine checks the GPU and stops with a message on GPUs older than compute capability 7.0 and on
GPUs the build has no code for. AMD GPUs and Macs are not supported.

Only the RTX 5090 has been tested. The other cards run code compiled from the same source but not tuned for them. They are slower than
the RTX 5090, and we have not checked that the engine works on each of them.

To build for one generation only (a faster build and a smaller file), name it when you configure, in a new build folder:

```bat
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
```

Use `120` for the RTX 50 series, `89` for the RTX 40 series, `86` for the RTX 30 series or `70` for the Tesla V100. Values below 70 are refused.

One process uses one GPU. On a PC with several GPUs, choose the GPU with the `CUDA_VISIBLE_DEVICES` environment variable.

## Tesla V100 (this fork)

This fork adds the Tesla V100 (Volta, compute capability 7.0). Volta has none of the instructions the kernels use from
Turing and Ampere onward, so `src/volta_compat.cuh` provides fallbacks that are compiled only into the sm_70 code:

* `cp.async` becomes an ordinary synchronous copy; commit is a no-op and wait synchronises the warp's active threads.
* The INT8 `mma.sync` (m16n8k32) is rebuilt from `__dp4a` over warp shuffles, bit for bit the same result.
* The BF16 / FP16 `mma.sync` (m16n8k16) is rebuilt from warp shuffles and scalar FMA in fp32; only the summation order
  differs from the tensor cores.
* `ldmatrix` is rebuilt from warp shuffles, including the transposing form.

The fragment layouts are the ones in the PTX ISA manual; `tools/volta_test/` checks every fallback against a plain
host computation on the GPU itself:

```sh
nvcc -O2 -arch=sm_70 -I src tools/volta_test/volta_test.cu -o /tmp/volta_test && /tmp/volta_test
```

Building sm_70 needs a CUDA 12.x toolkit (12.9 or older; CUDA 13 removed Volta). The driver can be newer. On Linux:

```sh
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc -DCUDAToolkit_ROOT=/usr/local/cuda-12.8
cmake --build build -j
```

Two limits on the V100: its shared memory per block is at most 96 KiB (the default paths fit; the opt-in
`YSKN_ATX2=1` and `YSKN_ATX_NS=4` settings do not, so leave them off), and it is much slower than the RTX 5090 —
the memory bandwidth is about half and the emulated matrix instructions run on the ordinary cores. Expect the
engine to work, not to be fast.

## Layout

* `src/`: the engine (C++ and CUDA); `include/`: the C interface that the Python package binds to; `third_party/`: nlohmann/json (MIT).
* The Python package is built and installed separately (`pip install .` in its folder).
* `tools/convert/`: the program that made the released model files from the original weights (Qwen3.8-27B only). See
  [tools/convert/README.md](tools/convert/README.md).
