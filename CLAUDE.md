# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this fork adds on top of upstream

This is a fork of NVIDIA CUTLASS 4.5.2 with the following local additions:

1. **SM80 native planar-complex bf16 GEMM + example + unit test** (commits `21765063` → `061674fd`). This is the working, runnable feature consumed by `../cmm-cutlass`.
2. **Phase 4b SM120 (consumer Blackwell) planar-complex scaffolding** (commits `67f7d201` → Phase 4b.4 commit). Schedule tags, dispatch policy, collective MMA mainloop, register-accumulator epilogue. **NOT runnable yet** — see "Phase 4b status" section below for the integration gap.

## Phase 4b status (SM120 planar-complex scaffolding)

Built between commits `67f7d201` and the Phase 4b.4 commit, in four phased deliverables targeting consumer Blackwell (RTX 50-series, sm_120):

| Phase | Commit | What landed |
|---|---|---|
| 4b.1 | `67f7d201` | `KernelScheduleSm120PlanarComplexGemm` + `KernelTmaWarpSpecialized1SmPlanarComplexSm120` schedule tags + `MainloopSm120TmaWarpSpecializedPlanarComplex` dispatch policy in `dispatch_policy.hpp`; `sm120_planar_complex_mma_builder.inl` skeleton with a partial-spec predicate matching `arch::Sm120` + the new schedule tag |
| 4b.2 | `026eaac8` | `sm120_mma_warpspecialized_planar_complex.hpp` — full collective MMA: 4 SMEM planes, 4 TMA atoms, `PipelineTmaAsync`, register-resident accumulator packed as `(MMA, MMA_M, MMA_N, 2)`, 4 register MMAs per K-block in planar pattern with operand-level negation |
| 4b.3 | `2451bf80` | `sm120_default_epilogue_planar_complex.hpp` — 2-plane sibling of `DefaultEpilogue`, register-accumulator, direct global stores, no SMEM staging or TMA; `PlanarComplexDefaultSm120` schedule tag + `CollectiveEpilogue` partial spec |
| 4b.4 | (this) | Integration assessment — builder placeholder refinement + `default_gemm_cbfloat16_sm120.h` documentation file. **Not runnable** |

### The integration gap (what's blocking a runnable SM120 planar GEMM)

Upstream CUTLASS 4.5.2 has **no kernel-layer `GemmUniversal` specialization for any CUTLASS-3.x planar-complex mainloop**. Verified by:

```bash
grep -rln "MainloopSm100TmaUmmaWarpSpecializedPlanarComplex" include/cutlass/gemm/kernel/
# (returns nothing — even SM100 planar has no kernel-layer driver)
```

This is why none of the existing `sm100_gemm_planar_*` test files (`test/unit/gemm/device/sm100_gemm_planar_*.cu`) are registered into any `cutlass_test_unit_gemm_device_*` CMakeLists target — there's no kernel layer to compose mainloop + epilogue + scheduler through. The 4b.3 missing-`#include` fix in `collective_epilogue.hpp` for `sm100_epilogue_planar_complex_tma_warpspecialized.hpp` (also added in this fork) lets the SM100 *epilogue* be selected at the builder level, but a full SM100 planar `GemmUniversal` still can't be instantiated.

Phase 5 (multi-day standalone work) would author this kernel-layer template. Approximate scope: ~500-1000 lines, sibling of `sm90_gemm_tma_warpspecialized_cooperative.hpp` (892 lines), adapted for the planar accumulator shape `(MMA, MMA_M, MMA_N, 2)` and the dual-output epilogue dispatch. The Phase 4b.1 builder skeleton's `static_assert` placeholder body has the exact `CollectiveOp` typedef sketch that Phase 5 would substitute in.

### Open architectural question (also for Phase 5)

The Phase 4b.2 mainloop uses **Ampere-style register MMA** (`SM80_16x8x16_F32BF16BF16F32_TN`, supported on sm_120 via legacy `mma.sync.aligned`). Consumer Blackwell bf16 in upstream CUTLASS uses **SM90 GMMA atoms** (descriptor-based, A/B read direct from smem). The register-MMA choice is correctness-first; if Phase 5 benches show insufficient perf win over the existing SM80 path, swap the `mma()` body for an SM90-GMMA-on-sm_120 variant (same TMA/smem machinery, different MMA pattern). The mainloop's class-level interfaces stay the same.

### What 4b shipped that IS useful right now

Even without a runnable kernel, Phase 4b unblocks downstream work:
- The Phase 4b.3 `collective_epilogue.hpp` missing-include fix is a strict upstream bugfix that future SM100 planar work needs anyway
- The scaffolding (dispatch policy, mainloop, epilogue) locks in the type contract so Phase 5's kernel-layer template just needs to compose existing pieces
- Type-system instantiation of `cutlass::gemm::device::GemmCbfloat16Sm120` fails with a clear "kernel-layer template missing" message instead of an opaque template error

---

## Original (still-active) SM80 path

The SM80 native planar-complex bf16 GEMM, example, and unit test landed as commits `21765063` → `061674fd`:

| File | What |
|---|---|
| `include/cutlass/cbfloat16.h` | `using cbfloat16_t = complex<bfloat16_t>` scalar handle |
| `include/cutlass/gemm/kernel/default_gemm_cbfloat16.h` | `GemmCbfloat16` kernel typedef + `device::GemmCbfloat16` adapter (RowMajor A / ColumnMajor B / RowMajor C, fp32 accumulator, 128×128×32 / 64×64×32 / 16×8×16 tile, 3 stages) |
| `examples/cbfloat16_gemm/` | Standalone runnable example + CMakeLists |
| `test/unit/gemm/device/gemm_cbfloat16_bf16_f32_tensor_op_sm80.cu` | 4 GTest cases registered into the existing `cutlass_test_unit_gemm_device_tensorop_planar_complex` target |

Both `examples/CMakeLists.txt` and `test/unit/gemm/device/CMakeLists.txt` got one-line manifest edits to register the new files; no other existing files were modified.

The kernel is consumed by the sibling package `../cmm-cutlass` (a PyTorch C++/CUDA extension). Any change to `GemmCbfloat16`'s template parameters (tile shape, layouts, output dtype) needs a matching update on the cmm-cutlass side.

## Critical gotcha: `Gemm::LayoutC` is transposed

`DefaultGemmPlanarComplexUniversal` internally transposes RowMajor C requests, so even though `default_gemm_cbfloat16.h` declares `LayoutC = RowMajor`, the actual `Gemm::LayoutC` evaluates to **`ColumnMajor`**. Always derive Element/Layout types from `Gemm::`:

```cpp
using ElementA = typename Gemm::ElementA;
using LayoutA  = typename Gemm::LayoutA;
using LayoutC  = typename Gemm::LayoutC;   // ColumnMajor, NOT RowMajor!
```

Hard-coding the user-facing layouts on the caller side silently produces transposed output (`C[m,n]` lands at `C[n,m]`). The unit tests pass because `TestbedPlanarComplex` follows this pattern; commit `061674fd` fixes the example after the bug bit me. The same fix is encoded in `cmm-cutlass/src/cmm_cutlass/csrc/complex_mm_bf16.cu`.

## Build (the cbfloat16 additions only)

The full CUTLASS build is slow (thousands of kernels in the library). Disable everything you don't need:

```bash
mkdir -p build && cd build
cmake -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DCUTLASS_NVCC_ARCHS=80 \
  -DCUTLASS_ENABLE_EXAMPLES=ON \
  -DCUTLASS_ENABLE_TESTS=ON \
  -DCUTLASS_ENABLE_LIBRARY=OFF \
  -DCUTLASS_ENABLE_PROFILER=OFF ..

ninja cbfloat16_gemm
ninja cutlass_test_unit_gemm_device_tensorop_planar_complex
```

`-DCUTLASS_NVCC_ARCHS=80` builds SM80 PTX only — it JITs forward to SM90/100/120 at runtime. Fine for development; add specific arches for native SASS.

### Run

```bash
./examples/cbfloat16_gemm/cbfloat16_gemm
./test/unit/gemm/device/cutlass_test_unit_gemm_device_tensorop_planar_complex \
  --gtest_filter='*bf16*'
```

Each bf16 unit test variant exercises `TestAllGemmPlanarComplex<Gemm>` (multiple M/N/K sizes against `cutlass::reference::host::GemmPlanarComplex`). The 4 cases take ~20s combined on SM120.

## Toolchain workaround: CUDA 13 + Ubuntu 26.04 glibc 2.41

System CUDA 13.0's `crt/math_functions.h` declares `rsqrt`/`rsqrtf`/`sinpi`/`cospi` without `noexcept`, conflicting with glibc 2.41's C23 IEC-60559 extension block. nvcc rejects this at parse time of any CUDA header (cmake's compiler-detection probe fails before our code is even compiled). `/usr/local/cuda/` is read-only in the dev container, so direct patching is blocked.

Workaround in `$HOME/cuda-local`:

- A symlink-farm layout mimicking `/usr/local/cuda/`, with `crt/math_functions.h` replaced by a patched copy (uses NVIDIA's own `__NV_IEC_60559_FUNCS_EXCEPTION_SPECIFIER` macro on `rsqrt`/`rsqrtf`, hoisted above their declarations).
- nvcc binaries are **copied** (not symlinked) into `$HOME/cuda-local/bin/` so `/proc/self/exe` resolves locally and cudafe++ reads our patched headers instead of the system ones.
- Wheel-installed cudart 12.x at `$HOME/cuda-venv/...site-packages/nvidia/cuda_*` supplies the patched-source provenance; the runtime libs come from the system 13.0 install via the farm.

To build against it:

```bash
CUDA_HOME=$HOME/cuda-local PATH=$HOME/cuda-local/bin:$PATH cmake -GNinja ...
```

On a stock CUDA 12.x / Ubuntu ≤24.04 environment none of this is needed — point cmake at the system CUDA and it just works.

## Reference points when touching the cbfloat16 path

When extending the bf16 planar complex kernel (more tile shapes, conjugate transforms, fp16 sibling, complex64 variant), read these in order:

- `include/cutlass/gemm/warp/mma_planar_complex.h:117-173` — the 4-MMA pattern at the warp level. Read this to confirm any planar complex change still maps to the four `real_mma` calls in the right order (the fourth MMA writes to `D.real` with `negate(A.imag)`).
- `include/cutlass/gemm/threadblock/default_mma_planar_complex_multistage.h:91-120` — the threadblock multistage wrapper, which constructs itself by composing `DefaultMma<...>` for the underlying real-valued MMA. New element types must have a `DefaultMma` specialization for `<ElementA, LayoutA, ..., ArchTag, OpClassTensorOp, ...>` at the threadblock level.
- `include/cutlass/gemm/kernel/default_gemm_planar_complex_universal.h:159-228` (pipelined, Stages ≤ 2) and `:275+` (multistage, Stages > 2) — kernel-level defaults; we use the multistage path.
- `test/unit/gemm/device/testbed_planar_complex.h` — the test harness used by `TestAllGemmPlanarComplex<Gemm>`. Pulling `Gemm::ElementA / LayoutA / …` instead of hard-coding is what makes the testbed survive the `LayoutC` transpose.

## Working with the cbfloat16 unit tests

The 4 bf16 GTest cases live in one .cu file but are compiled into `cutlass_test_unit_gemm_device_tensorop_planar_complex` (alongside the fp16 sm70/sm75/sm80 cases). Adding a new variant: append the new .cu to the `cutlass_test_unit_gemm_device_add_executable(...)` call in `test/unit/gemm/device/CMakeLists.txt:551`. The build batches sources (`BATCH_SOURCES ON, BATCH_SIZE 4`) which can hide per-file errors — pass `-j 1` to ninja while iterating.
