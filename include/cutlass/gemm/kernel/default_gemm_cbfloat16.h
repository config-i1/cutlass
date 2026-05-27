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
    \brief Canonical SM80 planar-complex bfloat16 GEMM instantiation.

    Pins the element types (bf16 inputs, fp32 accumulator and output), the row-col layout
    pair matching the m16n8k16 tensor-core instruction, and a 128x128x32 / 64x64x32 /
    16x8x16 tile config with 3-stage software pipelining. Provides both the kernel-level
    GemmKernel and a GemmUniversalAdapter-wrapped device entry point.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/cbfloat16.h"
#include "cutlass/complex.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/default_gemm_planar_complex_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

#include "cutlass/epilogue/thread/linear_combination_planar_complex.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

using DefaultGemmCbfloat16Kernel = typename DefaultGemmPlanarComplexUniversal<
    cutlass::bfloat16_t,
    cutlass::layout::RowMajor,
    cutlass::ComplexTransform::kNone,
    8,
    cutlass::bfloat16_t,
    cutlass::layout::ColumnMajor,
    cutlass::ComplexTransform::kNone,
    8,
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombinationPlanarComplex<
        float,
        4,
        float,
        float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,
    cutlass::arch::OpMultiplyAdd
>::GemmKernel;

/////////////////////////////////////////////////////////////////////////////////////////////////
// Bf16-output variant: same TN-layout / fp32 accumulator as
// DefaultGemmCbfloat16Kernel, but the epilogue writes bf16 directly,
// avoiding a downstream fp32 -> bf16 cast at the Python boundary. fp32
// accumulation inside the kernel is unchanged. Validated by
// test/unit/gemm/device/gemm_cbfloat16_bf16_f32_tensor_op_sm80.cu
// (the bf16t_bf16n_bf16n_tensor_op_f32_16816 case).

// Bf16-output variant tile family. All share TN layout, bf16 output,
// fp32 accumulator, 3 software pipeline stages, OpMultiplyAdd; they
// differ only in ThreadblockShape / WarpShape so a Python-side
// autotuner can pick the best one per (M, K, N).
//
// Tile naming: TB_M x TB_N x TB_K _ W_M x W_N. K-tile is 32 (matching
// the m16n8k16 PTX op pipeline) except where called out. Variants
// chosen to cover the M-heavy COT shapes (M = B*T = 4096-8192,
// K = 384-1536, N = 768-1536):
//
//   128x128x32_64x64   - large square, current default
//   64x128x32_32x64    - half-M, M-skinny
//   128x64x32_64x32    - half-N, N-skinny
//   64x64x32_32x32     - small square (matches cmm-triton's fp32
//                        Blackwell winner)
//   128x128x64_64x64   - larger K-tile, fewer K iters for K=384 W_O

#define _CMM_DEFAULT_GEMM_CBF16_BF16OUT(TBM, TBN, TBK, WM, WN)              \
  typename DefaultGemmPlanarComplexUniversal<                               \
    cutlass::bfloat16_t,                                                    \
    cutlass::layout::RowMajor,                                              \
    cutlass::ComplexTransform::kNone,                                       \
    8,                                                                      \
    cutlass::bfloat16_t,                                                    \
    cutlass::layout::ColumnMajor,                                           \
    cutlass::ComplexTransform::kNone,                                       \
    8,                                                                      \
    cutlass::bfloat16_t,                                                    \
    cutlass::layout::RowMajor,                                              \
    float,                                                                  \
    cutlass::arch::OpClassTensorOp,                                         \
    cutlass::arch::Sm80,                                                    \
    cutlass::gemm::GemmShape<TBM, TBN, TBK>,                                \
    cutlass::gemm::GemmShape<WM,  WN,  TBK>,                                \
    cutlass::gemm::GemmShape<16, 8, 16>,                                    \
    cutlass::epilogue::thread::LinearCombinationPlanarComplex<              \
        cutlass::bfloat16_t, 4, float, float                                \
    >,                                                                      \
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,           \
    3,                                                                      \
    cutlass::arch::OpMultiplyAdd                                            \
  >::GemmKernel

using DefaultGemmCbfloat16Bf16OutKernel          = _CMM_DEFAULT_GEMM_CBF16_BF16OUT(128, 128, 32, 64, 64);
using DefaultGemmCbfloat16Bf16Out_64x128_32x64   = _CMM_DEFAULT_GEMM_CBF16_BF16OUT( 64, 128, 32, 32, 64);
using DefaultGemmCbfloat16Bf16Out_128x64_64x32   = _CMM_DEFAULT_GEMM_CBF16_BF16OUT(128,  64, 32, 64, 32);
using DefaultGemmCbfloat16Bf16Out_64x64_32x32    = _CMM_DEFAULT_GEMM_CBF16_BF16OUT( 64,  64, 32, 32, 32);
using DefaultGemmCbfloat16Bf16Out_128x128k64     = _CMM_DEFAULT_GEMM_CBF16_BF16OUT(128, 128, 64, 64, 64);

#undef _CMM_DEFAULT_GEMM_CBF16_BF16OUT

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel

namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

using GemmCbfloat16        = GemmUniversalAdapter<kernel::DefaultGemmCbfloat16Kernel>;
using GemmCbfloat16Bf16Out = GemmUniversalAdapter<kernel::DefaultGemmCbfloat16Bf16OutKernel>;

// Autotune candidates. Naming matches the kernel typedef tile spec.
using GemmCbfloat16Bf16Out_64x128_32x64 = GemmUniversalAdapter<kernel::DefaultGemmCbfloat16Bf16Out_64x128_32x64>;
using GemmCbfloat16Bf16Out_128x64_64x32 = GemmUniversalAdapter<kernel::DefaultGemmCbfloat16Bf16Out_128x64_64x32>;
using GemmCbfloat16Bf16Out_64x64_32x32  = GemmUniversalAdapter<kernel::DefaultGemmCbfloat16Bf16Out_64x64_32x32>;
using GemmCbfloat16Bf16Out_128x128k64   = GemmUniversalAdapter<kernel::DefaultGemmCbfloat16Bf16Out_128x128k64>;

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace cutlass
