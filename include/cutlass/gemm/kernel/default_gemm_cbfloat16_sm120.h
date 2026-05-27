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
    \brief PLACEHOLDER — Blackwell-consumer (SM120, RTX 50-series) planar-
           complex bf16 GEMM instantiation target.

    Sibling of default_gemm_cbfloat16.h (Ampere) and
    default_gemm_cbfloat16_sm100.h (Blackwell data-center). This file marks
    the integration target for the SM120 planar-complex bf16 GEMM but does
    NOT yet expose a usable `cutlass::gemm::device::GemmCbfloat16Sm120`.

    Phase 4b status (as of commit 2451bf80):
      - 4b.1: schedule tag + dispatch policy + builder skeleton ✓
      - 4b.2: collective MMA mainloop (register-accumulator, TMA-loaded, 4-plane,
              planar 4-MMA pattern with operand-level negation) ✓
      - 4b.3: register-accumulator default planar epilogue ✓
      - 4b.4: INTEGRATION GAP — see CLAUDE.md "Phase 4b status" section

    The blocker: upstream CUTLASS 4.5.2 has no kernel-layer GemmUniversal
    specialization for any CUTLASS-3.x planar-complex mainloop. Verified by
    `grep -rln "MainloopSm100TmaUmmaWarpSpecializedPlanarComplex"
        include/cutlass/gemm/kernel/` returning no kernel-layer hits even for
    the SM100 planar mainloop. Authoring the kernel-layer template is a
    ~500-1000 line standalone milestone (sibling of
    sm90_gemm_tma_warpspecialized_cooperative.hpp adapted for the planar
    accumulator shape (MMA, MMA_M, MMA_N, 2)).

    Once that lands, this file would expose:

      using DefaultGemmCbfloat16Sm120Kernel = typename
        cbfloat16_sm120_detail::SmBuild<
          cute::Shape<cute::_128, cute::_128, cute::_32>,   // 100KB-fit tile
          cute::Shape<cute::_1,   cute::_1,   cute::_1>,    // sm_120: no cluster
          1                                                   // 1Sm only
        >::GemmKernel;

      using GemmCbfloat16Sm120 = GemmUniversalAdapter<DefaultGemmCbfloat16Sm120Kernel>;

    plus a small tile family analogous to the SM100 header's variants. Output
    would be bf16, fp32 accumulator, matching the GemmCbfloat16Bf16Out
    contract from default_gemm_cbfloat16.h.

    The expected SMEM budget on consumer Blackwell is 101 KB/block (vs the
    232 KB available on SM100), so the tile shapes shipped here will be a
    subset of the SM100 variants — likely 128x128x32, 64x128x32, 128x64x32,
    and 64x64x32, all with 2-3 pipeline stages.
*/

#pragma once

// Intentionally empty body. This is a Phase 4b.4 documentation placeholder
// (see file-level comment above). Including this header is safe; instantiating
// `cutlass::gemm::device::GemmCbfloat16Sm120` (which is not declared here) is
// blocked until the kernel-layer template lands in Phase 5.

#include "cutlass/cutlass.h"
