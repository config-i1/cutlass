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

} // namespace kernel

namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

using GemmCbfloat16 = GemmUniversalAdapter<kernel::DefaultGemmCbfloat16Kernel>;

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace cutlass
