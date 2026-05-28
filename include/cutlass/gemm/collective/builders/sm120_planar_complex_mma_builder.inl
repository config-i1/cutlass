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

#include "cutlass/detail/dependent_false.hpp"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma_sm80.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/collective/collective_mma_decl.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/collective/collective_builder_decl.hpp"
#include "cutlass/gemm/collective/sm120_mma_warpspecialized_planar_complex.hpp"

#include "cute/atom/mma_atom.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/arch/copy_sm75.hpp"
#include "cute/arch/mma_sm80.hpp"
#include "cute/atom/mma_traits_sm90_gmma.hpp"  // for GMMA::Layout_K_SW128_Atom (arch-portable swizzle)

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

  // Phase 5.3 wiring. ElementA/ElementB constrained to bf16 by the predicate;
  // ElementAccumulator defaults to float for this builder.
  static_assert(cute::is_same_v<ElementA, cutlass::bfloat16_t>,
                "SM120 planar bf16 builder: ElementA must be bfloat16_t");
  static_assert(cute::is_same_v<ElementB, cutlass::bfloat16_t>,
                "SM120 planar bf16 builder: ElementB must be bfloat16_t");
  static_assert(cute::is_same_v<ElementAccumulator, float>,
                "SM120 planar bf16 builder: ElementAccumulator must be float");

  // MMA atom: SM80 register MMA bf16xbf16 → fp32, m16n8k16 (TN — A row-major / B col-major).
  // Wrapped in a 2x2 AtomLayoutMNK to give 128 threads per CTA (4 atoms × 32 threads).
  using TiledMma = decltype(cute::make_tiled_mma(
      cute::MMA_Atom<cute::SM80_16x8x16_F32BF16BF16F32_TN>{},
      cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>{}));

  // Dual-TiledMma pair (for builder symmetry with the SM100 sibling).
  // SM120 register MMA has no ScaleIn::Neg descriptor flip; negation is done
  // at the operand register at MMA time (see Phase 4b.2 mainloop's gemm_kblock
  // lambda), so both pair slots hold the same TiledMma.
  using TiledMmaPair = cutlass::gemm::collective::detail::Sm120CollectiveMmaPlanarComplexTiledMmaType<TiledMma, TiledMma>;

  // TMA atoms (single-CTA; sm_120 doesn't support multicast).
  using GmemTiledCopyA = cute::SM90_TMA_LOAD;
  using GmemTiledCopyB = cute::SM90_TMA_LOAD;

  // SMEM layout atoms: K-major swizzle. Use SW64 (8x32 bf16) — it divides our
  // 32-element K-tile evenly. SW128 (8x64) would only work for K=64 tiles.
  // The GMMA-prefixed Layout_K_*_Atom names are arch-portable swizzle patterns
  // (not GMMA-specific despite the namespace).
  using SmemLayoutAtomA = decltype(cute::GMMA::Layout_K_SW64_Atom<ElementA>{});
  using SmemLayoutAtomB = decltype(cute::GMMA::Layout_K_SW64_Atom<ElementB>{});

  // SMEM→RMEM copy atom: ldmatrix.x4 (SM75_U32x4_LDSM_N) — 16 threads cooperate
  // to load 4×8-element bf16 segments per thread per call. Matches the SM80
  // 16x8x16 MMA's expected operand layout.
  using SmemCopyAtomA = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, ElementA>;
  using SmemCopyAtomB = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, ElementB>;

  // Pipeline / scheduler stages. PipelineStages=3 is the standard SM80/SM90
  // multistage default; the SchedulerPipelineStageCount slot isn't materially
  // used by this non-persistent kernel layer, so set it to 1.
  static constexpr int PipelineStages = 3;
  static constexpr int SchedulerPipelineStageCount = 1;

  using DispatchPolicy = cutlass::gemm::MainloopSm120TmaWarpSpecializedPlanarComplex<
      PipelineStages,
      SchedulerPipelineStageCount,
      ClusterShape_MNK,
      cutlass::gemm::KernelTmaWarpSpecialized1SmPlanarComplexSm120>;

  using CollectiveOp = cutlass::gemm::collective::CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      ElementA,
      cutlass::gemm::TagToStrideA_t<GmemLayoutATag>,
      ElementB,
      cutlass::gemm::TagToStrideB_t<GmemLayoutBTag>,
      TiledMmaPair,
      GmemTiledCopyA,
      SmemLayoutAtomA,
      SmemCopyAtomA,
      TransformA,
      GmemTiledCopyB,
      SmemLayoutAtomB,
      SmemCopyAtomB,
      TransformB>;
};

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
