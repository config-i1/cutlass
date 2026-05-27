/***************************************************************************************************
 * Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief CollectiveBuilder partial specialization for SM120 (consumer
           Blackwell, RTX 50-series) planar-complex bf16/f16 GEMM.

    Sibling of sm100_planar_complex_umma_builder.inl, gated to arch::Sm120
    and the local-fork KernelScheduleSm120PlanarComplexGemm tag family.

    LOCAL FORK ADDITION — not present in upstream CUTLASS 4.5.2. The
    upstream library ships SM120 builders only for regular dense, sparse,
    block-scaled, and blockwise GEMM (sm120_mma_builder.inl &
    siblings). Planar-complex on consumer Blackwell needs its own
    pipeline policy (no TMA multicast, ClusterShape forced to
    Shape<1,1,1>, 101 KB SMEM/block instead of 232 KB) which this builder
    encapsulates.

    PHASE STATUS: 4b.1 (this file) provides the partial-specialization
    skeleton — predicate compiles, dispatch policy + schedule tags exist,
    but the CollectiveOp body is a placeholder static_assert. The full
    collective mainloop port lands in Phase 4b.2 alongside the
    sm120_mma_warpspecialized_planar_complex.hpp companion file.
*/

#pragma once

// We don't pull in sm100_common.inl (the SM100 builder's helpers don't
// apply on SM120) — Phase 4b.2 will introduce sm120_common helpers /
// reuse the existing sm120_mma_builder.inl's primitives.

#include "cutlass/detail/dependent_false.hpp"
#include "cutlass/arch/arch.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_mma_decl.hpp"
#include "cutlass/gemm/collective/collective_builder_decl.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////
// Planar Complex f16/bf16 SM120 (consumer Blackwell) kernels builder — skeleton
//
// Predicate mirror of the SM100 builder (sm100_planar_complex_umma_builder.inl)
// but gated to:
//   - arch::Sm120 (not Sm100)
//   - KernelScheduleSm120PlanarComplexGemm (not KernelScheduleSm100PlanarComplexGemm)
//
// Phase 4b.2 will replace the CollectiveOp placeholder below with the real
// CollectiveMma instantiation backed by:
//   - MainloopSm120TmaWarpSpecializedPlanarComplex (already defined)
//   - the new sm120_mma_warpspecialized_planar_complex.hpp collective MMA
//   - SM120 single-CTA TMA atoms (no multicast — sm_120 doesn't support it)
//   - SM120 MMA atoms (from sm120_mma_builder.inl's existing primitives)
/////////////////////////////////////////////////////////////////////////////////////////////////
template <
  class ArchTag,
  class ElementA,
  class GmemLayoutATag,
  class TransformA,
  class ElementB,
  class GmemLayoutBTag,
  class TransformB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class BuilderScheduleTag
>
struct CollectiveBuilder<
    ArchTag,
    arch::OpClassTensorOp,
    cute::tuple<ElementA, TransformA>,
    GmemLayoutATag,
    8,
    cute::tuple<ElementB, TransformB>,
    GmemLayoutBTag,
    8,
    ElementAccumulator,
    TileShape_MNK,    // (MmaAtomShapeM, MmaAtomShapeN, TileK)
    ClusterShape_MNK, // sm_120 requires Shape<_1,_1,_1>; enforced below
    StageCountType,
    BuilderScheduleTag,
    cute::enable_if_t<
      cute::is_same_v<ArchTag, arch::Sm120> &&
      (cute::is_same_v<ElementA, cutlass::half_t> || cute::is_same_v<ElementA, cutlass::bfloat16_t>) &&
      (cute::is_same_v<ElementB, cutlass::half_t> || cute::is_same_v<ElementB, cutlass::bfloat16_t>) &&
      cute::is_base_of_v<KernelScheduleSm120PlanarComplexGemm, BuilderScheduleTag>>>
{
  static_assert(cute::is_static_v<TileShape_MNK>, "TileShape has to be static");
  static_assert(cute::is_same_v<ClusterShape_MNK, cute::Shape<cute::_1, cute::_1, cute::_1>>,
                "SM120 planar-complex requires ClusterShape<_1,_1,_1> — consumer "
                "Blackwell does not support TMA multicast or cluster > 1.");

  // Phase 4b.1 placeholder. Replaced in Phase 4b.2 with the real
  // CollectiveMma<MainloopSm120TmaWarpSpecializedPlanarComplex<...>, ...>.
  // Until then, instantiating this specialization fails with a clear message.
  static_assert(cutlass::detail::dependent_false<ArchTag>,
                "SM120 planar-complex CollectiveBuilder skeleton (cmm-cutlass "
                "Phase 4b.1): the dispatch policy and schedule tag are defined, "
                "but the CollectiveOp body lands in Phase 4b.2 (port of "
                "sm100_mma_warpspecialized_planar_complex.hpp to SM120).");

  using CollectiveOp = void; // unreachable — static_assert above traps first
};

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
