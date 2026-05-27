/***************************************************************************************************
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief Blackwell-DC (SM100) planar-complex bf16 GEMM instantiations.

    Sibling of default_gemm_cbfloat16.h's Ampere-tagged GemmCbfloat16 family,
    but built on CUTLASS 3.x's CollectiveBuilder API targeting the
    SM100 (Blackwell data-center, B200/B300) TMA + UMMA pipeline. The
    SM100 planar-complex builder lives in
    cutlass/gemm/collective/builders/sm100_planar_complex_umma_builder.inl
    and is gated to `arch::Sm100` only — these kernels do NOT run on
    sm_120 (Blackwell consumer / RTX 50).

    Variants exposed:
      GemmCbfloat16Sm100         — 1Sm  cluster<1,1,1>, 128x128x64 MMA tile
      GemmCbfloat16Sm100_2sm     — 2Sm  cluster<2,1,1>, 128x128x64 MMA tile
      GemmCbfloat16Sm100_64x128  — 1Sm  cluster<1,1,1>,  64x128x64 MMA tile
      GemmCbfloat16Sm100_128x64  — 1Sm  cluster<1,1,1>, 128x64x64  MMA tile

    Output: bf16, RowMajor C, fp32 accumulator. Layout follows the
    canonical SM100 planar bf16 test at
    test/unit/gemm/device/sm100_gemm_planar_cbf16_cbf16_cbf16_tensor_op_cf32.cu.

    Build note: requires --gpu-architecture=sm_100a (arch-conditional flag).
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/cbfloat16.h"
#include "cutlass/numeric_types.h"

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_sm100.hpp"

#include "cutlass/arch/mma_sm100.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {
namespace cbfloat16_sm100_detail {

// Shared type aliases for every variant in this header. They differ only in
// MmaTileShape, ClusterShape, and the 1Sm/2Sm schedule tag.

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

// Generic builder helper: picks the schedule tag pair (1Sm or 2Sm) and the
// epilogue tag from the SmCount template parameter to keep the four
// instantiations below short.

template <class MmaTileShape, class ClusterShape, int SmCount>
struct SmBuild;

template <class MmaTileShape, class ClusterShape>
struct SmBuild<MmaTileShape, ClusterShape, 1> {
  using EpilogueSchedule = cutlass::epilogue::PlanarComplexTmaWarpSpecialized1Sm;
  using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized1SmPlanarComplexSm100;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, 8,
      ElementC, LayoutC, 8,
      EpilogueSchedule
  >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      ElementPairA, LayoutA, 8,
      ElementPairB, LayoutB, 8,
      ElementAccumulator,
      MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopSchedule
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;
};

template <class MmaTileShape, class ClusterShape>
struct SmBuild<MmaTileShape, ClusterShape, 2> {
  using EpilogueSchedule = cutlass::epilogue::PlanarComplexTmaWarpSpecialized2Sm;
  using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized2SmPlanarComplexSm100;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, 8,
      ElementC, LayoutC, 8,
      EpilogueSchedule
  >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      ElementPairA, LayoutA, 8,
      ElementPairB, LayoutB, 8,
      ElementAccumulator,
      MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopSchedule
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;
};

} // namespace cbfloat16_sm100_detail

// Variant typedefs. SmBuild instantiations name each
// (MmaTileShape, ClusterShape, 1Sm-or-2Sm) combo we care about.
using DefaultGemmCbfloat16Sm100Kernel = typename
    cbfloat16_sm100_detail::SmBuild<cute::Shape<cute::_128, cute::_128, cute::_64>,
                                    cute::Shape<cute::_1, cute::_1, cute::_1>, 1>::GemmKernel;

using DefaultGemmCbfloat16Sm100Kernel_2sm = typename
    cbfloat16_sm100_detail::SmBuild<cute::Shape<cute::_128, cute::_128, cute::_64>,
                                    cute::Shape<cute::_2, cute::_1, cute::_1>, 2>::GemmKernel;

using DefaultGemmCbfloat16Sm100Kernel_64x128 = typename
    cbfloat16_sm100_detail::SmBuild<cute::Shape<cute::_64, cute::_128, cute::_64>,
                                    cute::Shape<cute::_1, cute::_1, cute::_1>, 1>::GemmKernel;

using DefaultGemmCbfloat16Sm100Kernel_128x64 = typename
    cbfloat16_sm100_detail::SmBuild<cute::Shape<cute::_128, cute::_64, cute::_64>,
                                    cute::Shape<cute::_1, cute::_1, cute::_1>, 1>::GemmKernel;

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel

namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

using GemmCbfloat16Sm100        = GemmUniversalAdapter<kernel::DefaultGemmCbfloat16Sm100Kernel>;
using GemmCbfloat16Sm100_2sm    = GemmUniversalAdapter<kernel::DefaultGemmCbfloat16Sm100Kernel_2sm>;
using GemmCbfloat16Sm100_64x128 = GemmUniversalAdapter<kernel::DefaultGemmCbfloat16Sm100Kernel_64x128>;
using GemmCbfloat16Sm100_128x64 = GemmUniversalAdapter<kernel::DefaultGemmCbfloat16Sm100Kernel_128x64>;

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace cutlass
