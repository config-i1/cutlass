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
    \brief SM120 (consumer Blackwell, RTX 50-series) planar-complex
           Warp-Specialized collective mainloop.

    Local fork addition — not present in upstream CUTLASS 4.5.2.

    Design: hybrid of sm120_mma_tma.hpp (regular GEMM, register-MMA pipeline)
    and sm100_mma_warpspecialized_planar_complex.hpp (planar 4-MMA pattern,
    4-plane SMEM/TMA layout).

    Adapted FROM SM100 planar:
      - 4 SMEM planes (A_real, A_imag, B_real, B_imag)
      - 4 TMA atoms, 4 loads per K-tile
      - 4 MMAs per K-block in the planar pattern

    Adapted FROM SM120 regular:
      - PipelineTmaAsync<Stages> (not PipelineTmaUmmaAsync — sm_120 doesn't
        use the TMEM/UMMA accumulator-pipeline machinery)
      - Single-CTA TMA loads (SM90_TMA_LOAD; no multicast, no SM100_TMA_2SM)
      - Cluster forced to <1,1,1>
      - Register fragments via partition_fragment_A/B + SmemCopyAtom retiling
      - Register-resident accumulators (no TMEM)

    Planar-complex differences from SM120 regular:
      - 4 SMEM planes + 4 TMA atoms (vs 2)
      - load() issues 4 TMA copies per K-tile under a single producer_acquire,
        sharing the mainloop barrier
      - mma() runs 4 register MMAs per K-block:
          accum_real += A_real * B_real            (no transform)
          accum_imag += A_real * B_imag            (TransformB == conjugate flips sign)
          accum_real += (-)A_imag * B_imag         (TransformA == TransformB flips sign)
          accum_imag += (-)A_imag * B_real         (TransformA == conjugate flips sign)
        Sign flips done via cute::negate on the operand register at MMA time
        (matches the SM80 planar bf16 path's pattern in
         include/cutlass/gemm/warp/mma_planar_complex.h).
      - Accumulator is passed as a single FrgTensorC of shape (MMA, MMA_M, MMA_N, 2):
        inner dim _2 picks real (index 0) / imag (index 1). This keeps the
        FrgTensorC API the same as the SM120 regular mainloop, just with one
        extra mode that the kernel-layer + epilogue both have to plumb.

    Phase 4b.2 deliverable: this file. Compiles standalone, parses cleanly when
    included via the build hub. End-to-end kernel still depends on:
      - Phase 4b.3: SM120 planar epilogue (sm120_epilogue_planar_complex_tma_warpspecialized.hpp)
      - Phase 4b.4: kernel-layer + cmm-cutlass dispatcher wiring + arg construction

    Open architectural question for Phases 4b.3/4b.4 (deferred from 4b.2):
    The MMA-atom choice for bf16 on sm_120 is non-obvious because upstream
    CUTLASS has no SM120 bf16 builder (the existing sm120_mma_builder.inl is
    f8/f6/f4-only; sm120_rr_smem_copy_selector_* assert UseF8f6f4). Consumer
    Blackwell runs bf16 via Hopper SM90 GMMA atoms in upstream practice.
    This mainloop assumes Ampere-style register MMA (matches the SM120 regular
    f8f6f4 pipeline's structure — partition_fragment_A/B + register accumulator),
    which IS supported by sm_120 hardware via the legacy mma.sync.aligned
    instructions. It compiles and would run, but it does not leverage Blackwell-
    class GMMA acceleration. If the Phase 4b.4 bench shows insufficient win
    over the existing SM80 bf16 planar path, the natural next step is to swap
    the register-MMA pipeline here for an SM90-GMMA-on-sm_120 pipeline
    (descriptor-based A/B reads from smem, register accumulator) — that's a
    second rewrite of this file's mma() body, but the TMA load + smem layout
    machinery stays the same.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/detail/dependent_false.hpp"
#include "cutlass/trace.h"
#include "cutlass/numeric_types.h"

#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;

namespace detail {

// Pair struct used by the SM120 planar-complex builder. SM120 register MMA
// has no UMMA-style ScaleIn::Neg descriptor flip — both slots hold the same
// TiledMma; negation happens at the operand register at MMA time. Kept as a
// pair (vs a single type) for source-level symmetry with the SM100 builder
// (Sm100CollectiveMmaPlanarComplexTiledMmaType) — makes diffing easier.
template<class TiledMmaAPos_, class TiledMmaANeg_>
struct Sm120CollectiveMmaPlanarComplexTiledMmaType {
  using TiledMmaAPosAtom = TiledMmaAPos_;
  using TiledMmaANegAtom = TiledMmaANeg_;
};

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  int Stages,
  int SchedulerPipelineStageCount,
  class ClusterShape,
  class KernelScheduleType,
  class TileShape_,
  class ElementA_,
  class StrideA_,
  class ElementB_,
  class StrideB_,
  class TiledMmaPair_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_>
struct CollectiveMma<
    MainloopSm120TmaWarpSpecializedPlanarComplex<Stages, SchedulerPipelineStageCount, ClusterShape, KernelScheduleType>,
    TileShape_,
    ElementA_,
    StrideA_,
    ElementB_,
    StrideB_,
    TiledMmaPair_,
    GmemTiledCopyA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_> {
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopSm120TmaWarpSpecializedPlanarComplex<Stages, SchedulerPipelineStageCount, ClusterShape, KernelScheduleType>;
  using TileShape = TileShape_;

  using TiledMmaPair = TiledMmaPair_;
  // SM120 only ever uses TiledMmaAPosAtom; negation is at operand level (see mma() below).
  using TiledMma = typename TiledMmaPair::TiledMmaAPosAtom;

  using ElementA = ElementA_;
  using StrideA = StrideA_;
  using ElementB = ElementB_;
  using StrideB = StrideB_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using CtaShape_MNK = decltype(shape_div(TileShape{}, ClusterShape{}));
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;

  using RuntimeDataTypeA = void*;
  using RuntimeDataTypeB = void*;

  static constexpr int ThreadCount = size(TiledMma{});

  using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;
  using PipelineParams   = typename MainloopPipeline::Params;
  using PipelineState    = typename cutlass::PipelineState<DispatchPolicy::Stages>;

  // Only one producer thread per CTA (the one elected by elect_one_sync) issues
  // all four TMA loads under a single producer_acquire.
  static constexpr int NumProducerThreadEvents = 1;

  // Cluster invariant — sm_120 doesn't support clusters > 1×1×1.
  static_assert(cute::is_same_v<ClusterShape, cute::Shape<cute::_1, cute::_1, cute::_1>>,
                "SM120 planar-complex requires ClusterShape<_1,_1,_1>; consumer "
                "Blackwell does not support TMA multicast or cluster > 1.");

  static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(not cute::is_void_v<SmemCopyAtomA>,
    "SM120 planar mainloop must specify a copy atom for A operand smem->rmem reads.");
  static_assert(not cute::is_void_v<SmemCopyAtomB>,
    "SM120 planar mainloop must specify a copy atom for B operand smem->rmem reads.");

  // Tile along modes in a way that maximizes the TMA box size (same as SM120 regular).
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      conditional_t< ::cutlass::gemm::detail::is_major<0,StrideA>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      conditional_t< ::cutlass::gemm::detail::is_major<0,StrideB>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));

  static_assert(rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
  static_assert(rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");

  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 2 or more.");
  static_assert(not cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
                not cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
                "MMA atom must source both A and B operands from rmem for this mainloop.");
  static_assert(cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD>,
      "SM120 planar mainloop only supports SM90_TMA_LOAD (single-CTA TMA; "
      "no multicast, no SM100 2SM TMA).");
  static_assert(cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD>,
      "SM120 planar mainloop only supports SM90_TMA_LOAD (single-CTA TMA; "
      "no multicast, no SM100 2SM TMA).");

  // TMA recast: keep input element type as-is for bf16/f16 (the only types the
  // SM120 planar builder accepts), but cast to a size-equivalent uint type for
  // generality / to avoid TMA rounding for non-trivial element types.
  using TmaInternalElementA = uint_bit_t<sizeof_bits_v<ElementA>>;
  using TmaInternalElementB = uint_bit_t<sizeof_bits_v<ElementB>>;

  using SmemAllocTypeA = typename TiledMma::ValTypeA;
  using SmemAllocTypeB = typename TiledMma::ValTypeB;

  // TMA transaction byte counts (per K-tile, summing all 4 planes).
  static constexpr uint32_t TmaTransactionBytesMK = static_cast<uint32_t>(
      cutlass::bits_to_bytes(size(take<0,2>(SmemLayoutA{})) * sizeof_bits<ElementA>::value));
  static constexpr uint32_t TmaTransactionBytesNK = static_cast<uint32_t>(
      cutlass::bits_to_bytes(size(take<0,2>(SmemLayoutB{})) * sizeof_bits<ElementB>::value));
  // 2x for A (real+imag) plus 2x for B (real+imag).
  static constexpr uint32_t TmaTransactionBytes = 2 * (TmaTransactionBytesMK + TmaTransactionBytesNK);

  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _0> {
      alignas(1024) cute::array_aligned<SmemAllocTypeA, cute::cosize_v<SmemLayoutA>> smem_A_real;
      alignas(1024) cute::array_aligned<SmemAllocTypeA, cute::cosize_v<SmemLayoutA>> smem_A_imag;
      alignas(1024) cute::array_aligned<SmemAllocTypeB, cute::cosize_v<SmemLayoutB>> smem_B_real;
      alignas(1024) cute::array_aligned<SmemAllocTypeB, cute::cosize_v<SmemLayoutB>> smem_B_imag;
    } tensors;

    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    alignas(16) PipelineStorage pipeline_storage;
  };
  using TensorStorage   = typename SharedStorage::TensorStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  // Host side kernel arguments — 4 pointer+stride pairs.
  struct Arguments {
    ElementA const* ptr_A_real{nullptr};
    StrideA dA_real{};
    ElementA const* ptr_A_imag{nullptr};
    StrideA dA_imag{};
    ElementB const* ptr_B_real{nullptr};
    StrideB dB_real{};
    ElementB const* ptr_B_imag{nullptr};
    StrideB dB_imag{};
  };

  // Device side kernel params — 4 TMA atoms.
  struct Params {
    // 4 TMA atom types — all identical structurally; we cache the same type 4x.
    using TMA_A = decltype(make_tma_copy(
        GmemTiledCopyA{},
        make_tensor(recast_ptr<TmaInternalElementA>(nullptr), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
        SmemLayoutA{}(_,_,0),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        size<1>(ClusterShape{}))); // mcast along N — but cluster is <1,1,1>, so mcast count = 1
    using TMA_B = decltype(make_tma_copy(
        GmemTiledCopyB{},
        make_tensor(recast_ptr<TmaInternalElementB>(nullptr), repeat_like(StrideB{}, int32_t(0)), StrideB{}),
        SmemLayoutB{}(_,_,0),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
        size<0>(ClusterShape{})));
    TMA_A tma_load_a_real;
    TMA_A tma_load_a_imag;
    TMA_B tma_load_b_real;
    TMA_B tma_load_b_imag;
    uint32_t tma_transaction_bytes = TmaTransactionBytes;
    uint32_t tma_transaction_bytes_mk = TmaTransactionBytesMK;
    uint32_t tma_transaction_bytes_nk = TmaTransactionBytesNK;
  };

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    (void) workspace;

    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    auto ptr_A_real = recast_ptr<TmaInternalElementA>(args.ptr_A_real);
    auto ptr_A_imag = recast_ptr<TmaInternalElementA>(args.ptr_A_imag);
    auto ptr_B_real = recast_ptr<TmaInternalElementB>(args.ptr_B_real);
    auto ptr_B_imag = recast_ptr<TmaInternalElementB>(args.ptr_B_imag);

    Tensor tensor_a_real = make_tensor(ptr_A_real, make_layout(make_shape(M,K,L), args.dA_real));
    Tensor tensor_a_imag = make_tensor(ptr_A_imag, make_layout(make_shape(M,K,L), args.dA_imag));
    Tensor tensor_b_real = make_tensor(ptr_B_real, make_layout(make_shape(N,K,L), args.dB_real));
    Tensor tensor_b_imag = make_tensor(ptr_B_imag, make_layout(make_shape(N,K,L), args.dB_imag));

    typename Params::TMA_A tma_load_a_real = make_tma_copy(
        GmemTiledCopyA{}, tensor_a_real, SmemLayoutA{}(_,_,cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        size<1>(ClusterShape{}));
    typename Params::TMA_A tma_load_a_imag = make_tma_copy(
        GmemTiledCopyA{}, tensor_a_imag, SmemLayoutA{}(_,_,cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        size<1>(ClusterShape{}));
    typename Params::TMA_B tma_load_b_real = make_tma_copy(
        GmemTiledCopyB{}, tensor_b_real, SmemLayoutB{}(_,_,cute::Int<0>{}),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
        size<0>(ClusterShape{}));
    typename Params::TMA_B tma_load_b_imag = make_tma_copy(
        GmemTiledCopyB{}, tensor_b_imag, SmemLayoutB{}(_,_,cute::Int<0>{}),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
        size<0>(ClusterShape{}));

    return {
      tma_load_a_real, tma_load_a_imag,
      tma_load_b_real, tma_load_b_imag,
      TmaTransactionBytes,
      TmaTransactionBytesMK,
      TmaTransactionBytesNK
    };
  }

  template<class ProblemShape>
  static bool
  can_implement(
      ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    bool implementable = true;
    constexpr int min_tma_aligned_elements_A = 128 / cute::sizeof_bits<ElementA>::value;
    implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_A>(cute::make_shape(M,K,L), StrideA{});
    constexpr int min_tma_aligned_elements_B = 128 / cute::sizeof_bits<ElementB>::value;
    implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_B>(cute::make_shape(N,K,L), StrideB{});

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA.\n");
    }
    return implementable;
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& mainloop_params) {
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_a_real.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_a_imag.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_b_real.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_b_imag.get_tma_descriptor());
  }

  /// Set up the data needed by this collective for load and mma.
  /// Returns a tuple of (gA_real_mkl, gA_imag_mkl, gB_real_nkl, gB_imag_nkl).
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_init(ProblemShape_MNKL const& problem_shape_MNKL, Params const& mainloop_params) const {
    using X = Underscore;
    auto [M, N, K, L] = problem_shape_MNKL;

    Tensor mA_real_mkl = mainloop_params.tma_load_a_real.get_tma_tensor(make_shape(M,K,L));
    Tensor mA_imag_mkl = mainloop_params.tma_load_a_imag.get_tma_tensor(make_shape(M,K,L));
    Tensor mB_real_nkl = mainloop_params.tma_load_b_real.get_tma_tensor(make_shape(N,K,L));
    Tensor mB_imag_nkl = mainloop_params.tma_load_b_imag.get_tma_tensor(make_shape(N,K,L));

    Tensor gA_real_mkl = local_tile(mA_real_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});
    Tensor gA_imag_mkl = local_tile(mA_imag_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});
    Tensor gB_real_nkl = local_tile(mB_real_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});
    Tensor gB_imag_nkl = local_tile(mB_imag_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});

    return cute::make_tuple(gA_real_mkl, gA_imag_mkl, gB_real_nkl, gB_imag_nkl);
  }

  /// Perform a collective-scoped matrix multiply-accumulate — Producer Perspective.
  /// Issues 4 TMA copies per K-tile under a single producer_acquire (single barrier).
  template <
    class TensorA, class TensorB,
    class KTileIterator, class BlockCoord
  >
  CUTLASS_DEVICE void
  load(
      Params const& mainloop_params,
      MainloopPipeline pipeline,
      PipelineState smem_pipe_write,
      cute::tuple<TensorA, TensorA, TensorB, TensorB> const& load_inputs,
      BlockCoord const& blk_coord,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      uint32_t block_rank_in_cluster,
      TensorStorage& shared_tensors) {
    int lane_predicate = cute::elect_one_sync();

    if (lane_predicate) {
      Tensor sA_real = make_tensor(make_smem_ptr(shared_tensors.smem_A_real.data()), SmemLayoutA{});
      Tensor sA_imag = make_tensor(make_smem_ptr(shared_tensors.smem_A_imag.data()), SmemLayoutA{});
      Tensor sB_real = make_tensor(make_smem_ptr(shared_tensors.smem_B_real.data()), SmemLayoutB{});
      Tensor sB_imag = make_tensor(make_smem_ptr(shared_tensors.smem_B_imag.data()), SmemLayoutB{});

      // Cluster is <1,1,1> so block_rank_in_cluster is always 0 and there is no
      // multicast — but keep the structure for symmetry with the SM120 regular
      // mainloop in case future hardware lifts the constraint.
      (void) block_rank_in_cluster;
      auto block_tma_a_real = mainloop_params.tma_load_a_real.get_slice(0);
      auto block_tma_a_imag = mainloop_params.tma_load_a_imag.get_slice(0);
      auto block_tma_b_real = mainloop_params.tma_load_b_real.get_slice(0);
      auto block_tma_b_imag = mainloop_params.tma_load_b_imag.get_slice(0);

      Tensor gA_real_mkl = get<0>(load_inputs);
      Tensor gA_imag_mkl = get<1>(load_inputs);
      Tensor gB_real_nkl = get<2>(load_inputs);
      Tensor gB_imag_nkl = get<3>(load_inputs);

      auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
      Tensor gA_real = gA_real_mkl(_,_,m_coord,_,l_coord);
      Tensor gA_imag = gA_imag_mkl(_,_,m_coord,_,l_coord);
      Tensor gB_real = gB_real_nkl(_,_,n_coord,_,l_coord);
      Tensor gB_imag = gB_imag_nkl(_,_,n_coord,_,l_coord);

      Tensor tAgA_real = block_tma_a_real.partition_S(gA_real);
      Tensor tAsA_real = block_tma_a_real.partition_D(sA_real);
      Tensor tAgA_imag = block_tma_a_imag.partition_S(gA_imag);
      Tensor tAsA_imag = block_tma_a_imag.partition_D(sA_imag);
      Tensor tBgB_real = block_tma_b_real.partition_S(gB_real);
      Tensor tBsB_real = block_tma_b_real.partition_D(sB_real);
      Tensor tBgB_imag = block_tma_b_imag.partition_S(gB_imag);
      Tensor tBsB_imag = block_tma_b_imag.partition_D(sB_imag);

      // No multicast on sm_120.
      uint16_t mcast_mask = 0;

      CUTLASS_PRAGMA_NO_UNROLL
      for ( ; k_tile_count > 0; --k_tile_count) {
        pipeline.producer_acquire(smem_pipe_write);

        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);

        int write_stage = smem_pipe_write.index();
        copy(mainloop_params.tma_load_a_real.with(*tma_barrier, mcast_mask), tAgA_real(_,_,_,*k_tile_iter), tAsA_real(_,_,_,write_stage));
        copy(mainloop_params.tma_load_a_imag.with(*tma_barrier, mcast_mask), tAgA_imag(_,_,_,*k_tile_iter), tAsA_imag(_,_,_,write_stage));
        copy(mainloop_params.tma_load_b_real.with(*tma_barrier, mcast_mask), tBgB_real(_,_,_,*k_tile_iter), tBsB_real(_,_,_,write_stage));
        copy(mainloop_params.tma_load_b_imag.with(*tma_barrier, mcast_mask), tBgB_imag(_,_,_,*k_tile_iter), tBsB_imag(_,_,_,write_stage));
        ++k_tile_iter;

        ++smem_pipe_write;
      }
    }
  }

  /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void
  load_tail(MainloopPipeline pipeline, PipelineState smem_pipe_write) {
    int lane_predicate = cute::elect_one_sync();

    if (lane_predicate) {
      pipeline.producer_tail(smem_pipe_write);
    }
  }

  /// Perform a collective-scoped matrix multiply-accumulate — Consumer Perspective.
  /// Accepts a packed accumulator of shape (MMA, MMA_M, MMA_N, 2):
  ///   accum(_,_,_,0) = real accumulator
  ///   accum(_,_,_,1) = imag accumulator
  /// Runs the 4-MMA planar pattern per K-block, with sign flips done via
  /// cute::transform + cute::negate on the relevant operand register at MMA
  /// time (no UMMA ScaleIn::Neg descriptor available on register MMA).
  template <
    class FrgTensorC,
    class BlockCoord
  >
  CUTLASS_DEVICE void
  mma(MainloopPipeline pipeline,
      PipelineState smem_pipe_read,
      FrgTensorC& accum,
      int k_tile_count,
      int thread_idx,
      TensorStorage& shared_tensors,
      Params const& mainloop_params,
      [[maybe_unused]] BlockCoord& blk_crd) {
    using namespace cute;
    (void) mainloop_params;

    static_assert(is_rmem<FrgTensorC>::value, "Accumulator tensor must be rmem resident.");
    static_assert(rank(FrgTensorC{}) == 4,
                  "Accumulator must be MMA-partitioned with a trailing _2 mode (MMA, MMA_M, MMA_N, 2)");
    CUTE_STATIC_ASSERT_V(size<3>(FrgTensorC{}) == _2{},
                         "Accumulator trailing mode must be _2 (planar real/imag).");

    auto accum_real = accum(_,_,_,_0{});
    auto accum_imag = accum(_,_,_,_1{});
    clear(accum_real);
    clear(accum_imag);

    Tensor sA_real = make_tensor(make_smem_ptr(shared_tensors.smem_A_real.data()), SmemLayoutA{});
    Tensor sA_imag = make_tensor(make_smem_ptr(shared_tensors.smem_A_imag.data()), SmemLayoutA{});
    Tensor sB_real = make_tensor(make_smem_ptr(shared_tensors.smem_B_real.data()), SmemLayoutB{});
    Tensor sB_imag = make_tensor(make_smem_ptr(shared_tensors.smem_B_imag.data()), SmemLayoutB{});

    //
    // C accumulators and A/B partitioning
    //
    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);

    Tensor tCrA_real = thread_mma.partition_fragment_A(sA_real(_,_,Int<0>{}));
    Tensor tCrA_imag = thread_mma.partition_fragment_A(sA_imag(_,_,Int<0>{}));
    Tensor tCrB_real = thread_mma.partition_fragment_B(sB_real(_,_,Int<0>{}));
    Tensor tCrB_imag = thread_mma.partition_fragment_B(sB_imag(_,_,Int<0>{}));

    //
    // SMEM->RMEM copy plumbing — same SmemCopyAtomA/B used for all four planes.
    //
    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(thread_idx);
    Tensor tCsA_real = smem_thr_copy_A.partition_S(as_position_independent_swizzle_tensor(sA_real));
    Tensor tCsA_imag = smem_thr_copy_A.partition_S(as_position_independent_swizzle_tensor(sA_imag));
    Tensor tCrA_real_copy_view = smem_thr_copy_A.retile_D(tCrA_real);
    Tensor tCrA_imag_copy_view = smem_thr_copy_A.retile_D(tCrA_imag);

    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(thread_idx);
    Tensor tCsB_real = smem_thr_copy_B.partition_S(as_position_independent_swizzle_tensor(sB_real));
    Tensor tCsB_imag = smem_thr_copy_B.partition_S(as_position_independent_swizzle_tensor(sB_imag));
    Tensor tCrB_real_copy_view = smem_thr_copy_B.retile_D(tCrB_real);
    Tensor tCrB_imag_copy_view = smem_thr_copy_B.retile_D(tCrB_imag);

    // A scratch fragment used to hold a negated copy of A_imag when the MMA pattern
    // requires it (saves the original A_imag for the subsequent imagAcc += A_imag*B_real).
    Tensor tCrA_imag_neg = make_tensor_like(tCrA_imag);

    CUTE_STATIC_ASSERT_V(size<1>(tCsA_real) == size<1>(tCrA_real_copy_view));
    CUTE_STATIC_ASSERT_V(size<2>(tCsA_real) == size<2>(tCrA_real_copy_view));
    CUTE_STATIC_ASSERT_V(size<1>(tCrA_real) == size<1>(accum_real));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB_real) == size<2>(accum_real));
    CUTE_STATIC_ASSERT_V(size<2>(tCsA_real) == size<2>(tCsB_real));
    CUTE_STATIC_ASSERT_V(size<3>(tCsA_real) == size<3>(tCsB_real));
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA_real));
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB_real));

    //
    // PIPELINED MAIN LOOP
    //
    auto K_BLOCK_MAX = size<2>(tCrA_real);

    int read_stage = smem_pipe_read.index();
    auto tCsA_real_stage = tCsA_real(_,_,_,read_stage);
    auto tCsA_imag_stage = tCsA_imag(_,_,_,read_stage);
    auto tCsB_real_stage = tCsB_real(_,_,_,read_stage);
    auto tCsB_imag_stage = tCsB_imag(_,_,_,read_stage);

    // copy_kblock: smem->rmem for all four planes at a given k_block.
    auto copy_kblock = [&](auto k_block) {
      copy(smem_tiled_copy_A, tCsA_real_stage(_,_,k_block), tCrA_real_copy_view(_,_,k_block));
      copy(smem_tiled_copy_A, tCsA_imag_stage(_,_,k_block), tCrA_imag_copy_view(_,_,k_block));
      copy(smem_tiled_copy_B, tCsB_real_stage(_,_,k_block), tCrB_real_copy_view(_,_,k_block));
      copy(smem_tiled_copy_B, tCsB_imag_stage(_,_,k_block), tCrB_imag_copy_view(_,_,k_block));
    };

    // gemm_kblock: 4 register MMAs in the planar pattern.
    //
    //   accum_real += A_real * B_real                       [step 1]
    //   accum_imag += A_real * B_imag      (if TransformB==conjugate, negate B_imag)  [step 2]
    //   accum_real += (-)A_imag * B_imag   (if TransformA != TransformB, no negate)  [step 3]
    //   accum_imag += (-)A_imag * B_real   (if TransformA==conjugate, negate A_imag) [step 4]
    //
    // For the bf16 cmm-cutlass use case both transforms are cute::identity, so the
    // step-3 negate fires (real -= A_imag*B_imag) and step-4 does not.
    auto gemm_kblock = [&](auto k_block) {
      // Step 1: accum_real += A_real * B_real (always)
      cute::gemm(tiled_mma, tCrA_real(_,_,k_block), tCrB_real(_,_,k_block), accum_real);

      // Step 2: accum_imag += A_real * (+/-) B_imag
      if constexpr (cute::is_same_v<TransformB, cute::conjugate>) {
        // Negate B_imag in a scratch view and MMA against it
        auto tCrB_imag_neg_local = make_tensor_like(tCrB_imag(_,_,k_block));
        cute::transform(tCrB_imag(_,_,k_block), tCrB_imag_neg_local, cute::negate{});
        cute::gemm(tiled_mma, tCrA_real(_,_,k_block), tCrB_imag_neg_local, accum_imag);
      } else {
        cute::gemm(tiled_mma, tCrA_real(_,_,k_block), tCrB_imag(_,_,k_block), accum_imag);
      }

      // Step 3: accum_real += (-)A_imag * B_imag (negate iff TransformA == TransformB)
      if constexpr (cute::is_same_v<TransformA, TransformB>) {
        // realAcc -= A_imag * B_imag  → negate A_imag
        cute::transform(tCrA_imag(_,_,k_block), tCrA_imag_neg(_,_,k_block), cute::negate{});
        cute::gemm(tiled_mma, tCrA_imag_neg(_,_,k_block), tCrB_imag(_,_,k_block), accum_real);
      } else {
        cute::gemm(tiled_mma, tCrA_imag(_,_,k_block), tCrB_imag(_,_,k_block), accum_real);
      }

      // Step 4: accum_imag += (+/-)A_imag * B_real
      if constexpr (cute::is_same_v<TransformA, cute::conjugate>) {
        // imagAcc += (-A_imag) * B_real → negate A_imag (reuse scratch if it wasn't used by step 3)
        if constexpr (!cute::is_same_v<TransformA, TransformB>) {
          cute::transform(tCrA_imag(_,_,k_block), tCrA_imag_neg(_,_,k_block), cute::negate{});
        }
        cute::gemm(tiled_mma, tCrA_imag_neg(_,_,k_block), tCrB_real(_,_,k_block), accum_imag);
      } else {
        cute::gemm(tiled_mma, tCrA_imag(_,_,k_block), tCrB_real(_,_,k_block), accum_imag);
      }
    };

    pipeline.consumer_wait(smem_pipe_read);

    copy_kblock(_0{});
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 1; --k_tile_count) {
      for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) {

        auto k_block_next = ((k_block + 1) == K_BLOCK_MAX) ? 0 : (k_block + 1);

        if (k_block == K_BLOCK_MAX - 1) {
          cutlass::arch::NamedBarrier::sync(
            thr_size(tiled_mma), cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
          pipeline.consumer_release(smem_pipe_read);
          ++smem_pipe_read;
          read_stage = smem_pipe_read.index();
          tCsA_real_stage = tCsA_real(_,_,_,read_stage);
          tCsA_imag_stage = tCsA_imag(_,_,_,read_stage);
          tCsB_real_stage = tCsB_real(_,_,_,read_stage);
          tCsB_imag_stage = tCsB_imag(_,_,_,read_stage);
          pipeline.consumer_wait(smem_pipe_read);
        }

        copy_kblock(k_block_next);
        gemm_kblock(k_block);

      });
    }

    // Hoist out last k_tile
    for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) {

      auto k_block_next = ((k_block + 1) == K_BLOCK_MAX) ? 0 : (k_block + 1);

      if (k_block == K_BLOCK_MAX - 1) {
        cutlass::arch::NamedBarrier::sync(
          thr_size(tiled_mma), cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
        pipeline.consumer_release(smem_pipe_read);
        ++smem_pipe_read;
      }

      if (k_block_next > 0) {
        copy_kblock(k_block_next);
      }
      gemm_kblock(k_block);

    });
  }

  /// Perform a Consumer Epilogue to release all buffers (no-op for SM120).
  CUTLASS_DEVICE void
  mma_tail(MainloopPipeline, PipelineState, int) {
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
