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
    \brief SM120 (consumer Blackwell) planar-complex GEMM kernel-layer template.

    Local fork addition (Phase 5 of cmm-cutlass). Gated to the
    KernelScheduleSm120PlanarComplexGemm schedule tag family.

    Adapted from sm90_gemm_tma_warpspecialized.hpp (the simpler
    non-cooperative SM90 warp-specialized kernel template) with planar-
    specific changes:
      - 1 producer warp group (1 warp does TMA loads via elect_one_sync)
      - 1 consumer warp group (does MMA + epilogue back-to-back)
      - Planar accumulator allocated with shape (MMA, MMA_M, MMA_N, 2)
        (inner _2 picks real / imag)
      - Uses the Phase 4b.3 register-accumulator DefaultEpilogue-style
        epilogue API (operator() with all args, no LoadPipeline/StorePipeline)
      - Non-persistent: blockIdx.x/y/z map directly to tile coords
      - Cluster forced to <1,1,1>; no cluster_wait/cluster_arrive

    See CLAUDE.md "Phase 5 design (5.1 analysis output)" for the chosen
    Path B (simple-epilogue-API kernel layer) vs Path A (SM90-epilogue-API
    rewrite) tradeoff.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/workspace.h"
#include "cutlass/fast_math.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "cute/tensor.hpp"
#include "cutlass/trace.h"
#include "cutlass/gemm/kernel/gemm_universal_decl.h"

///////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::kernel {

template <
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class TileScheduler_
>
class GemmUniversal<
  ProblemShape_,
  CollectiveMainloop_,
  CollectiveEpilogue_,
  TileScheduler_,
  cute::enable_if_t<cute::is_base_of_v<
      cutlass::gemm::KernelScheduleSm120PlanarComplexGemm,
      typename CollectiveMainloop_::DispatchPolicy::Schedule>>>
{
public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;
  static_assert(cute::rank(ProblemShape{}) == 3 or cute::rank(ProblemShape{}) == 4,
    "ProblemShape{} should be <M,N,K> or <M,N,K,L>");

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using ArchTag   = typename CollectiveMainloop::ArchTag;
  using ElementA  = typename CollectiveMainloop::ElementA;
  using StrideA   = typename CollectiveMainloop::StrideA;
  using ElementB  = typename CollectiveMainloop::ElementB;
  using StrideB   = typename CollectiveMainloop::StrideB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC  = typename CollectiveEpilogue::StrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD  = typename CollectiveEpilogue::StrideD;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  // sm_120 supports only ClusterShape<1,1,1> for planar (no multicast).
  static_assert(cute::is_same_v<ClusterShape, cute::Shape<cute::_1, cute::_1, cute::_1>>,
                "SM120 planar-complex kernel requires ClusterShape<_1,_1,_1>.");

  // We don't drive a persistent scheduler; blockIdx maps to tile directly.
  // TileScheduler_ is accepted for API symmetry but only PersistentScheduler / void
  // are allowed (matching sm90_gemm_tma_warpspecialized.hpp's pattern).
  static_assert(cute::is_void_v<TileScheduler_> ||
                cute::is_same_v<TileScheduler_, cutlass::gemm::PersistentScheduler>,
                "SM120 planar-complex kernel does not support specializing the tile scheduler.");

  // Kernel level shared memory storage. Mainloop and epilogue don't use smem
  // concurrently for this non-persistent kernel — use a union.
  struct SharedStorage {
    union TensorStorage {
      using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;
      using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;
      MainloopTensorStorage mainloop;
      EpilogueTensorStorage epilogue;
    } tensors;

    struct PipelineStorage : cute::aligned_struct<16, _1> {
      using MainloopPipelineStorage = typename CollectiveMainloop::PipelineStorage;
      alignas(16) MainloopPipelineStorage mainloop;
    } pipelines;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);
  static constexpr uint32_t NumLoadWarpGroups = 1;
  static constexpr uint32_t NumMmaWarpGroups = 1;
  static constexpr uint32_t MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMma{})) + (NumLoadWarpGroups * NumThreadsPerWarpGroup);
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  // Device side arguments
  struct Arguments {
    cutlass::gemm::GemmUniversalMode mode{cutlass::gemm::GemmUniversalMode::kGemm};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
  };

  // Kernel entry point API
  struct Params {
    ProblemShape problem_shape{};
    MainloopParams mainloop{};
    EpilogueParams epilogue{};
  };

  //
  // Methods
  //

  static Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    (void) workspace;
    return {
      args.problem_shape,
      CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, workspace),
      CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, workspace)
    };
  }

  static bool
  can_implement(Arguments const& args) {
    bool implementable = (args.mode == GemmUniversalMode::kGemm) or
        (args.mode == GemmUniversalMode::kBatched && cute::rank(ProblemShape{}) == 4);
    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Mode/ProblemShape rank mismatch.\n");
      return implementable;
    }
    implementable &= CollectiveMainloop::can_implement(args.problem_shape, args.mainloop);
    implementable &= CollectiveEpilogue::can_implement(args.problem_shape, args.epilogue);
    return implementable;
  }

  static size_t
  get_workspace_size(Arguments const& /*args*/) {
    return 0;
  }

  static cutlass::Status
  initialize_workspace(Arguments const& /*args*/, void* /*workspace*/ = nullptr,
                       cudaStream_t /*stream*/ = nullptr,
                       CudaHostAdapter* /*cuda_adapter*/ = nullptr) {
    return Status::kSuccess;
  }

  static dim3
  get_grid_shape(Params const& params) {
    auto problem_shape_MNKL = cute::append<4>(params.problem_shape, cute::Int<1>{});
    auto M = cute::get<0>(problem_shape_MNKL);
    auto N = cute::get<1>(problem_shape_MNKL);
    auto L = cute::get<3>(problem_shape_MNKL);
    auto blk_M = cute::size<0>(TileShape{});
    auto blk_N = cute::size<1>(TileShape{});
    return dim3(
      static_cast<uint32_t>((M + blk_M - 1) / blk_M),
      static_cast<uint32_t>((N + blk_N - 1) / blk_N),
      static_cast<uint32_t>(L));
  }

  static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  CUTLASS_DEVICE
  void
  operator()(Params const& params, char* smem_buf) {
    using namespace cute;
    using X = Underscore;

#if (defined(__CUDA_ARCH_FEAT_SM90_ALL) || defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL) || \
     CUDA_ARCH_CONDITIONAL_OR_FAMILY(1200) || CUDA_ARCH_CONDITIONAL_OR_FAMILY(1210))
#  define ENABLE_SM120_PLANAR_KERNEL 1
#endif

#if ! defined(ENABLE_SM120_PLANAR_KERNEL)
    CUTE_INVALID_CONTROL_PATH("ERROR: SM120 planar-complex kernel used without targeting "
                              "sm_120a compute capability. Aborting.\n");
#else

    enum class WarpGroupRole {
      Producer = 0,
      Consumer = 1,
    };
    enum class ProducerWarpRole {
      Mainloop = 0,
      Warp1 = 1,
      Warp2 = 2,
      Warp3 = 3,
    };

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    int thread_idx = int(threadIdx.x);
    int lane_idx = canonical_lane_idx();
    int warp_idx = canonical_warp_idx_sync();
    int warp_idx_in_warp_group = warp_idx % NumWarpsPerWarpGroup;
    int warp_group_thread_idx = thread_idx % NumThreadsPerWarpGroup;
    auto warp_group_role = WarpGroupRole(canonical_warp_group_idx());
    auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);
    int lane_predicate = cute::elect_one_sync();
    uint32_t block_rank_in_cluster = 0; // cluster<1,1,1>

    // Issue TMA descriptor prefetch from a single thread
    if ((warp_idx == 0) && lane_predicate) {
      CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
    }

    // Setup mainloop pipeline
    using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
    typename MainloopPipeline::Params mainloop_pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer && producer_warp_role == ProducerWarpRole::Mainloop) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
    }
    mainloop_pipeline_params.is_leader = warp_group_thread_idx == 0;
    mainloop_pipeline_params.num_consumers = NumThreadsPerWarpGroup;
    mainloop_pipeline_params.transaction_bytes = params.mainloop.tma_transaction_bytes;
    MainloopPipeline mainloop_pipeline(shared_storage.pipelines.mainloop, mainloop_pipeline_params, ClusterShape{});

    typename CollectiveMainloop::PipelineState mainloop_pipe_consumer_state;
    PipelineState mainloop_pipe_producer_state = cutlass::make_producer_start_state<MainloopPipeline>();

    // Cluster wait: sm_120 only supports cluster<1,1,1>, so plain __syncthreads.
    __syncthreads();

    static_assert(cute::rank(StrideA{}) == 3, "StrideA must be rank-3: [M, K, L].");
    static_assert(cute::rank(StrideB{}) == 3, "StrideB must be rank-3: [N, K, L].");

    auto blk_shape = TileShape{};
    TiledMma tiled_mma;

    auto problem_shape_MNKL = append<4>(params.problem_shape, cute::Int<1>{});

    CollectiveMainloop collective_mainloop;
    CollectiveEpilogue collective_epilogue(params.epilogue, shared_storage.tensors.epilogue);

    // Prepare and partition the input tensors.
    // For planar mainloop, load_init returns a 4-element tuple:
    //   get<0>(load_inputs) = gA_real_mkl  shape (BLK_M, BLK_K, m, k, l)
    //   get<1>(load_inputs) = gA_imag_mkl
    //   get<2>(load_inputs) = gB_real_nkl  shape (BLK_N, BLK_K, n, k, l)
    //   get<3>(load_inputs) = gB_imag_nkl
    auto load_inputs = collective_mainloop.load_init(problem_shape_MNKL, params.mainloop);
    static_assert(cute::tuple_size_v<decltype(load_inputs)> == 4,
                  "Planar mainloop load_init must return a 4-element tuple "
                  "(gA_real_mkl, gA_imag_mkl, gB_real_nkl, gB_imag_nkl).");

    Tensor gA_real_mkl = get<0>(load_inputs);
    Tensor gB_real_nkl = get<2>(load_inputs);

    auto m_coord = int(blockIdx.x);
    auto n_coord = int(blockIdx.y);
    auto l_coord = int(blockIdx.z);
    auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

    auto k_tile_iter  = cute::make_coord_iterator(shape<3>(gA_real_mkl));
    auto k_tile_count = size<3>(gA_real_mkl);

    if (warp_group_role == WarpGroupRole::Producer) {
      if (producer_warp_role == ProducerWarpRole::Mainloop) {
        collective_mainloop.load(
          params.mainloop,
          mainloop_pipeline,
          mainloop_pipe_producer_state,
          load_inputs,
          blk_coord,
          k_tile_iter, k_tile_count,
          lane_idx,
          block_rank_in_cluster,
          shared_storage.tensors.mainloop
        );
        mainloop_pipe_producer_state.advance(k_tile_count);
        collective_mainloop.load_tail(mainloop_pipeline, mainloop_pipe_producer_state);
      }
    }
    else if (warp_group_role == WarpGroupRole::Consumer) {
      // Allocate the planar accumulator: rank-4 tensor (MMA, MMA_M, MMA_N, 2).
      // Compute the base 3D shape via partition_fragment_C, then extend
      // with a trailing _2 for the planar real/imag axis.
      auto base_accum = partition_fragment_C(tiled_mma, take<0,2>(blk_shape));   // (MMA, MMA_M, MMA_N)
      auto planar_shape = append(shape(base_accum), _2{});
      Tensor accumulators = make_tensor<ElementAccumulator>(planar_shape);

      collective_mainloop.mma(
        mainloop_pipeline,
        mainloop_pipe_consumer_state,
        accumulators,
        k_tile_count,
        warp_group_thread_idx,
        shared_storage.tensors.mainloop,
        params.mainloop,
        blk_coord
      );

      collective_mainloop.mma_tail(
        mainloop_pipeline,
        mainloop_pipe_consumer_state,
        k_tile_count
      );

      // Synchronize: mainloop done in mainloop smem, epilogue can now reuse the
      // smem union slot for its own tensors.
      __syncthreads();

      // Invoke the planar epilogue directly. The Phase 4b.3 epilogue's
      // operator() signature is:
      //   (problem_shape_MNKL, blk_shape, blk_coord, accumulators, tiled_mma,
      //    residue_MNK, thread_idx, char*)
      // residue_MNK is unused by the epilogue's body but kept for signature
      // compatibility with DefaultEpilogue. We pass a placeholder zero tuple.
      auto residue_MNK = cute::make_tuple(0, 0, 0);
      collective_epilogue(
        problem_shape_MNKL,
        blk_shape,
        blk_coord,
        accumulators,
        tiled_mma,
        residue_MNK,
        warp_group_thread_idx,
        nullptr
      );
    }
#endif
  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::kernel
