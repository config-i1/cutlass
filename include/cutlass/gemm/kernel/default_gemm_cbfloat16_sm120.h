/***************************************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*!
    \file
    \brief Blackwell-consumer (SM120, RTX 50-series) planar-complex bf16 GEMM.

    Sibling of default_gemm_cbfloat16.h (Ampere/SM80) and
    default_gemm_cbfloat16_sm100.h (Blackwell DC/SM100).

    Built on the Phase 5 SM120 planar-complex stack:
      - dispatch_policy.hpp:        MainloopSm120TmaWarpSpecializedPlanarComplex
                                    + KernelTmaWarpSpecialized1SmPlanarComplexSm120
      - sm120_mma_warpspecialized_planar_complex.hpp:  collective MMA
      - sm120_default_epilogue_planar_complex.hpp:     register-acc epilogue
      - sm120_gemm_warpspecialized_planar_complex.hpp: kernel-layer GemmUniversal
      - sm120_planar_complex_mma_builder.inl:          CollectiveBuilder hook

    Variant exposed initially (Phase 5.3 — single tile, expanded after first
    bench in Phase 5.4/5.5):
      GemmCbfloat16Sm120  — 1Sm  cluster<1,1,1>, 128x128x32 MMA tile, 3 stages.

    Output: bf16, RowMajor C (which becomes ColumnMajor inside the kernel —
    same transpose trap as the SM80 path; see CLAUDE.md), fp32 accumulator.

    Build note: requires --gpu-architecture=sm_120 (or sm_120a) and a runtime
    sm_120 device. The kernel will JIT-decline to instantiate on other archs
    via the planar mainloop's GmemTiledCopy = SM90_TMA_LOAD requirement.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/cbfloat16.h"
#include "cutlass/numeric_types.h"

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"

#include "cutlass/arch/mma_sm80.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/sm120_default_epilogue_planar_complex.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {
namespace cbfloat16_sm120_detail {

using ElementA = cutlass::bfloat16_t;
using TransformA = cute::identity;
using ElementPairA = cute::tuple<ElementA, TransformA>;
using LayoutA = cutlass::layout::RowMajor;

using ElementB = cutlass::bfloat16_t;
using TransformB = cute::identity;
using ElementPairB = cute::tuple<ElementB, TransformB>;
using LayoutB = cutlass::layout::ColumnMajor;

using ElementAccumulator = float;
using ElementC = cutlass::bfloat16_t;
using LayoutC = cutlass::layout::RowMajor;

constexpr int AlignmentA = 8;
constexpr int AlignmentB = 8;
constexpr int AlignmentC = 8;

// SM120 planar bf16 builder helper: given a static MmaTileShape (the TileShape
// the kernel operates on) + ClusterShape (must be <1,1,1>), produce the
// GemmKernel type via:
//   1. CollectiveMainloop = SM120 planar bf16 collective builder
//   2. CollectiveEpilogue = the Phase 4b.3 register-accumulator default planar
//      epilogue (constructed directly, not via the epilogue CollectiveBuilder —
//      no SM120 planar epilogue builder exists yet, and the direct construction
//      is simpler for the single supported variant)
//   3. GemmUniversal = the Phase 5.2 SM120 planar kernel-layer

template <class MmaTileShape, class ClusterShape>
struct SmBuild {
  using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized1SmPlanarComplexSm120;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120,
      cutlass::arch::OpClassTensorOp,
      ElementPairA, LayoutA, AlignmentA,
      ElementPairB, LayoutB, AlignmentB,
      ElementAccumulator,
      MmaTileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAuto,
      MainloopSchedule
  >::CollectiveOp;

  // Thread epilogue op: identity LinearCombination (alpha=1, beta=0) over bf16
  // accumulator → bf16 output. Used per-plane independently by the planar
  // epilogue (see sm120_default_epilogue_planar_complex.hpp).
  using ThreadEpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementC, AlignmentC, ElementAccumulator, ElementAccumulator>;

  using CollectiveEpilogue = cutlass::epilogue::collective::Sm120PlanarComplexDefaultEpilogue<
      ElementC,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      ThreadEpilogueOp,
      cutlass::epilogue::PlanarComplexDefaultSm120>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;
};

} // namespace cbfloat16_sm120_detail

// Variant typedef (initially: just one variant for Phase 5.3 iteration).
// Phase 5.5 expands to a tile family analogous to the SM100 header once the
// first variant compiles + runs.
using DefaultGemmCbfloat16Sm120Kernel = typename
    cbfloat16_sm120_detail::SmBuild<
        cute::Shape<cute::_128, cute::_128, cute::_32>,
        cute::Shape<cute::_1, cute::_1, cute::_1>
    >::GemmKernel;

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel

namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

using GemmCbfloat16Sm120 = GemmUniversalAdapter<kernel::DefaultGemmCbfloat16Sm120Kernel>;

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace cutlass
