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

/*! \file
    \brief Planar-complex default epilogue: register-accumulator, direct-store
           sibling of DefaultEpilogue for the SM120 planar-complex mainloop.

    Local fork addition — companion to
    cutlass/gemm/collective/sm120_mma_warpspecialized_planar_complex.hpp.
    No upstream sibling; the SM100 planar epilogue path is TMEM-based and
    doesn't apply here.

    Contract:
      Accumulator passed in is rmem-resident with shape (MMA, MMA_M, MMA_N, 2);
      inner mode _2 carries (real, imag).
      Writes two output planes (D_real, D_imag) and optionally reads two
      source planes (C_real, C_imag) for alpha/beta scaling.

    Design choices:
      - Mirrors DefaultEpilogue's structure 1:1, just doubled to handle two planes
      - No SMEM staging, no TMA store — direct register→gmem global stores via
        cutlass::arch::global_store, predicated for OOB tile residue
      - Same ThreadEpilogueOp_ (e.g. LinearCombination<bf16, ..>) is run on both
        real and imag accumulators independently. Alpha/beta are real-valued
        (NOT complex) in this minimal-viable variant; complex alpha/beta would
        require a custom op that takes both real and imag accumulators at once,
        and would land in a follow-up alongside the full SM120 planar TMA epilogue.

    Phase 4b.3 deliverable: this file. Standalone-compiles via the include-hub
    addition; not yet exercised by any kernel layer (deferred to Phase 4b.4
    where the kernel-layer specialization + builder wiring lands).
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/memory.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"

#include "cute/tensor.hpp"
#include "cute/numeric/numeric_types.hpp"
#include "cutlass/cuda_host_adapter.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {

/////////////////////////////////////////////////////////////////////////////////////////////////
// Schedule tag — local fork addition. Selects this register-accumulator planar
// epilogue in builder dispatch. The "Default" suffix mirrors the
// non-TMA / non-SMEM-staged DefaultEpilogue lineage.
/////////////////////////////////////////////////////////////////////////////////////////////////
struct PlanarComplexDefaultSm120 {};

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class ElementC_,
  class StrideC_,
  class StrideD_,
  class ThreadEpilogueOp_,
  class EpilogueSchedule_ = PlanarComplexDefaultSm120
>
class Sm120PlanarComplexDefaultEpilogue {
public:
  //
  // Type Aliases
  //
  using EpilogueSchedule = EpilogueSchedule_;
  using DispatchPolicy   = EpilogueSchedule_;

  // Thread-level epilogue operator (e.g. LinearCombination<ElementOutput, ...>)
  using ThreadEpilogueOp   = ThreadEpilogueOp_;
  using ElementOutput      = typename ThreadEpilogueOp::ElementOutput;
  using ElementAccumulator = typename ThreadEpilogueOp::ElementAccumulator;
  using ElementCompute     = typename ThreadEpilogueOp::ElementCompute;
  using ElementScalar      = ElementCompute;
  using ElementC           = ElementC_;
  using StrideC            = StrideC_;
  using ElementD           = typename ThreadEpilogueOp::ElementD;
  using StrideD            = StrideD_;

  using GmemElementC = cute::conditional_t<cute::is_void_v<ElementC>, ElementD, ElementC>;

  using GmemTiledCopyC = void;
  using GmemTiledCopyD = void;

  static const int kOutputAlignment = ThreadEpilogueOp::kCount;
  using AlignmentType = typename cute::uint_bit<sizeof_bits<ElementOutput>::value * kOutputAlignment>::type;

  static_assert(cute::rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]");
  static_assert(cute::rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]");

  struct SharedStorage { };
  using TensorStorage = SharedStorage;

  // Host side epilogue arguments — two planes for C, two planes for D.
  struct Arguments {
    typename ThreadEpilogueOp::Params thread{};
    ElementC const* ptr_C_real = nullptr;
    ElementC const* ptr_C_imag = nullptr;
    StrideC dC_real{};
    StrideC dC_imag{};
    ElementD* ptr_D_real = nullptr;
    ElementD* ptr_D_imag = nullptr;
    StrideD dD_real{};
    StrideD dD_imag{};
  };

  using Params = Arguments;

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      [[maybe_unused]] ProblemShape const& _,
      Arguments const& args,
      [[maybe_unused]] void* workspace) {
    return args;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& /*problem_shape*/, Arguments const& /*args*/) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& /*problem_shape*/, Arguments const& /*args*/,
                       void* /*workspace*/, cudaStream_t /*stream*/,
                       CudaHostAdapter* /*cuda_adapter*/ = nullptr) {
    return cutlass::Status::kSuccess;
  }

  template <class ProblemShape>
  static bool
  can_implement(
      [[maybe_unused]] ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    return true;
  }

  CUTLASS_HOST_DEVICE
  Sm120PlanarComplexDefaultEpilogue(Params const& params_, SharedStorage const& shared_storage = SharedStorage())
      : params(params_), epilogue_op(params_.thread) { }

  CUTLASS_DEVICE
  bool
  is_source_needed() {
    return epilogue_op.is_source_needed();
  }

  /// Apply the epilogue to the planar accumulator and write two output planes.
  ///
  /// Accumulator shape: (MMA, MMA_M, MMA_N, 2). Inner _2 picks real (0) /
  /// imag (1). Mirrors DefaultEpilogue::operator() but executes its body
  /// twice — once per plane — with the matching source/destination pointers
  /// and strides.
  template <
    class ProblemShapeMNKL,
    class BlockShapeMNK,
    class BlockCoordMNKL,
    class FrgEngine, class FrgLayout,
    class TiledMma,
    class ResidueMNK
  >
  CUTLASS_DEVICE void
  operator()(
      ProblemShapeMNKL problem_shape_mnkl,
      BlockShapeMNK blk_shape_MNK,
      BlockCoordMNKL blk_coord_mnkl,
      cute::Tensor<FrgEngine, FrgLayout> const& accumulators,
      TiledMma tiled_mma,
      [[maybe_unused]] ResidueMNK,
      int thread_idx,
      [[maybe_unused]] char*)
  {
    using namespace cute;
    using X = Underscore;

    static_assert(cute::rank(ProblemShapeMNKL{}) == 4, "ProblemShapeMNKL must be rank 4");
    static_assert(is_static<BlockShapeMNK>::value, "ThreadBlock tile shape must be static");
    static_assert(cute::rank(BlockShapeMNK{}) == 3, "BlockShapeMNK must be rank 3");
    static_assert(cute::rank(BlockCoordMNKL{}) == 4, "BlockCoordMNKL must be rank 4");
    static_assert(cute::rank(FrgLayout{}) == 4,
                  "Planar accumulator must be rank-4: (MMA, MMA_M, MMA_N, 2)");
    CUTE_STATIC_ASSERT_V(size<3>(FrgLayout{}) == _2{},
                         "Planar accumulator trailing mode must be _2 (real, imag).");

    auto M = get<0>(problem_shape_mnkl);
    auto N = get<1>(problem_shape_mnkl);
    auto L = get<3>(problem_shape_mnkl);

    auto stride_c_real = detail::get_epilogue_stride<EpilogueSchedule>(params.dC_real);
    auto stride_c_imag = detail::get_epilogue_stride<EpilogueSchedule>(params.dC_imag);
    auto stride_d_real = detail::get_epilogue_stride<EpilogueSchedule>(params.dD_real);
    auto stride_d_imag = detail::get_epilogue_stride<EpilogueSchedule>(params.dD_imag);

    // Slice accumulator into real/imag sub-tensors.
    auto accum_real = accumulators(_,_,_,_0{});
    auto accum_imag = accumulators(_,_,_,_1{});

    // Per-plane writeback lambda. Re-deriving gC/gD/cD per-plane keeps the
    // structure verbatim with DefaultEpilogue::operator() — easier to verify
    // by diff. Optimizing this (hoist OOB pred shared across planes) is left
    // for a follow-up; correctness first.
    auto write_plane = [&](auto& accum_plane,
                           ElementC const* ptr_C, auto stride_c,
                           ElementD* ptr_D, auto stride_d) {
      Tensor mC_mnl = make_tensor(make_gmem_ptr<GmemElementC>(ptr_C), make_shape(M,N,L), stride_c);
      Tensor mD_mnl = make_tensor(make_gmem_ptr(ptr_D), make_shape(M,N,L), stride_d);
      Tensor gC_mnl = local_tile(mC_mnl, blk_shape_MNK, make_coord(_,_,_), Step<_1,_1, X>{});
      Tensor gD_mnl = local_tile(mD_mnl, blk_shape_MNK, make_coord(_,_,_), Step<_1,_1, X>{});

      auto [m_coord, n_coord, k_coord, l_coord] = blk_coord_mnkl;
      Tensor gC = gC_mnl(_,_,m_coord,n_coord,l_coord);
      Tensor gD = gD_mnl(_,_,m_coord,n_coord,l_coord);

      auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
      Tensor tCgD = thr_mma.partition_C(gD);
      Tensor tCgC = thr_mma.partition_C(gC);

      CUTE_STATIC_ASSERT_V(size(tCgC) == size(tCgD),
          "Source and destination must have the same number of elements.");
      CUTE_STATIC_ASSERT_V(size(tCgD) == size(accum_plane),
          "Accumulator count must match destination element count.");

      // OOB predication for tile-quantization residue.
      auto shape_MN = make_shape(M, N);
      Tensor mD_crd = make_identity_tensor(shape_MN);
      Tensor cD_mn = local_tile(mD_crd, take<0,2>(blk_shape_MNK), make_coord(m_coord, n_coord));
      Tensor tCcD_mn = thr_mma.partition_C(cD_mn);
      Tensor cD = make_coord_tensor(cD_mn.layout());
      Tensor tCcD = make_coord_tensor(tCcD_mn.layout());
      auto residue_cD = shape_MN - cD_mn(_0{});
      auto residue_tCcD = shape_MN - tCcD_mn(_0{});

      if (not elem_less(repeat_like(residue_cD, _0{}), residue_cD)) {
        return; // fully OOB
      }

      using FragCType = remove_cvref_t<decltype(tCgC(0))>;
      using FragDType = remove_cvref_t<decltype(tCgD(0))>;

      if (epilogue_op.is_source_needed()) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(accum_plane); ++i) {
          FragCType fragC;
          bool pred = elem_less(tCcD(i), residue_tCcD);
          arch::global_load<FragCType, sizeof(FragCType)>(fragC, &tCgC(i), pred);
          FragDType fragD = epilogue_op(accum_plane(i), fragC);
          arch::global_store<FragDType, sizeof(FragDType)>(fragD, &tCgD(i), pred);
        }
      } else {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(accum_plane); ++i) {
          bool pred = elem_less(tCcD(i), residue_tCcD);
          FragDType fragD = epilogue_op(accum_plane(i));
          arch::global_store<FragDType, sizeof(FragDType)>(fragD, &tCgD(i), pred);
        }
      }
    };

    write_plane(accum_real, params.ptr_C_real, stride_c_real, params.ptr_D_real, stride_d_real);
    write_plane(accum_imag, params.ptr_C_imag, stride_c_imag, params.ptr_D_imag, stride_d_imag);
  }

private:
  Params params;
  ThreadEpilogueOp epilogue_op;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
// CollectiveEpilogue partial specialization — selects this epilogue when the
// builder is queried with PlanarComplexDefaultSm120 schedule. Mirrors the
// pattern used by the DefaultEpilogue specialization in
// default_epilogue.hpp's downstream wiring.
/////////////////////////////////////////////////////////////////////////////////////////////////
template <
  class ElementC,
  class StrideC,
  class StrideD,
  class ThreadEpilogueOp
>
class CollectiveEpilogue<
  PlanarComplexDefaultSm120,
  ElementC, StrideC, StrideD, ThreadEpilogueOp
> : public Sm120PlanarComplexDefaultEpilogue<ElementC, StrideC, StrideD, ThreadEpilogueOp, PlanarComplexDefaultSm120> {
public:
  using Sm120PlanarComplexDefaultEpilogue<ElementC, StrideC, StrideD, ThreadEpilogueOp, PlanarComplexDefaultSm120>::
        Sm120PlanarComplexDefaultEpilogue;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace collective
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
