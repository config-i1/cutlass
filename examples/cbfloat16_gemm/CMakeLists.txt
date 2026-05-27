
# Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Requires the bf16 m16n8k16 tensor-core instruction (Ampere or newer).
cutlass_example_add_executable(
  cbfloat16_gemm
  cbfloat16_gemm.cu
  )

# --- Out-of-tree build (reference) -------------------------------------------
# To build this example standalone against an installed CUTLASS fork, use:
#
#   cmake_minimum_required(VERSION 3.18)
#   project(cbfloat16_gemm CUDA CXX)
#   set(CMAKE_CUDA_ARCHITECTURES 80)
#   set(CMAKE_CXX_STANDARD 17)
#   set(CMAKE_CUDA_STANDARD 17)
#   set(CUTLASS_DIR "/path/to/cutlass-fork" CACHE PATH "Path to CUTLASS fork root")
#   add_executable(cbfloat16_gemm cbfloat16_gemm.cu)
#   target_include_directories(cbfloat16_gemm PRIVATE
#       ${CUTLASS_DIR}/include
#       ${CUTLASS_DIR}/tools/util/include)
#   target_compile_options(cbfloat16_gemm PRIVATE
#       $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo -O3>)
