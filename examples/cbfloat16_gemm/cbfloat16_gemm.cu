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
    \brief Standalone example exercising the SM80 planar-complex bf16 GEMM.

    Runs a correctness check against cutlass::reference::host::GemmPlanarComplex and
    a perf measurement at M=N=K=1024 reporting TFLOP/s using 8*M*N*K for complex GEMM.

    A complex GEMM C = A * B with A in C^(MxK), B in C^(KxN), C in C^(MxN) is implemented
    as four real bf16 tensor-core MMA calls per warp tile, with fp32 accumulators, by
    reusing CUTLASS's planar-complex pipeline. Storage is planar: real and imaginary
    halves live in disjoint memory regions sharing one base pointer and an imag_stride.

    Important: GemmCbfloat16's user-facing Layout/Element types are derived via
    `using ElementA = typename Gemm::ElementA` etc. CUTLASS internally transposes
    layouts depending on output requirements, so hard-coding the layouts on the
    caller side risks swapping A/B in a way that silently produces transposed
    output. The testbed pattern (also used here) avoids the trap.
*/

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/default_gemm_cbfloat16.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor_planar_complex.h"
#include "cutlass/util/reference/host/gemm_planar_complex.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

using Gemm = cutlass::gemm::device::GemmCbfloat16;

using ElementA = typename Gemm::ElementA;
using LayoutA  = typename Gemm::LayoutA;
using ElementB = typename Gemm::ElementB;
using LayoutB  = typename Gemm::LayoutB;
using ElementC = typename Gemm::ElementC;
using LayoutC  = typename Gemm::LayoutC;
using ElementAccumulator = typename Gemm::ElementAccumulator;
using ElementCompute     = typename Gemm::EpilogueOutputOp::ElementCompute;

struct Options {
  int m = 1024;
  int n = 1024;
  int k = 1024;
  int iterations = 20;
  bool help = false;

  void parse(int argc, char const **argv) {
    cutlass::CommandLine cmd(argc, argv);
    if (cmd.check_cmd_line_flag("help")) help = true;
    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("iterations", iterations);
  }

  void print_usage(std::ostream &os) const {
    os << "cbfloat16_gemm - planar complex bf16 GEMM on SM80\n\n"
       << "Options:\n"
       << "  --help              Print this message\n"
       << "  --m=<int>           GEMM M (default 1024)\n"
       << "  --n=<int>           GEMM N (default 1024)\n"
       << "  --k=<int>           GEMM K (default 1024)\n"
       << "  --iterations=<int>  Timed iterations (default 20)\n";
  }
};

struct Tensors {
  cutlass::HostTensorPlanarComplex<ElementA, LayoutA> A;
  cutlass::HostTensorPlanarComplex<ElementB, LayoutB> B;
  cutlass::HostTensorPlanarComplex<ElementC, LayoutC> C;
  cutlass::HostTensorPlanarComplex<ElementC, LayoutC> D;
  cutlass::HostTensorPlanarComplex<ElementC, LayoutC> D_ref;

  Tensors(int M, int N, int K) {
    A.reset({M, K});
    B.reset({K, N});
    C.reset({M, N});
    D.reset({M, N});
    D_ref.reset({M, N}, /*device_backed=*/false);
  }

  void initialize(uint64_t seed) {
    int scope = 4;
    cutlass::reference::host::TensorFillRandomUniform(A.host_view(), seed,        scope, -scope, 0);
    cutlass::reference::host::TensorFillRandomUniform(B.host_view(), seed * 2019, scope, -scope, 0);
    cutlass::reference::host::TensorFill(C.host_view(),     cutlass::complex<ElementC>());
    cutlass::reference::host::TensorFill(D.host_view(),     cutlass::complex<ElementC>());
    cutlass::reference::host::TensorFill(D_ref.host_view(), cutlass::complex<ElementC>());
    A.sync_device();
    B.sync_device();
    C.sync_device();
    D.sync_device();
  }
};

cutlass::Status run_gemm(Tensors &t,
                         cutlass::gemm::GemmCoord problem_size,
                         cutlass::complex<ElementCompute> alpha,
                         cutlass::complex<ElementCompute> beta) {
  auto lda = t.A.layout().stride(0);
  auto ldb = t.B.layout().stride(0);
  auto ldc = t.C.layout().stride(0);
  auto ldd = t.D.layout().stride(0);

  typename Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    problem_size,
    /*batch_count=*/1,
    {alpha, beta},
    t.A.device_data(), t.A.device_data() + t.A.imaginary_stride(),
    t.B.device_data(), t.B.device_data() + t.B.imaginary_stride(),
    t.C.device_data(), t.C.device_data() + t.C.imaginary_stride(),
    t.D.device_data(), t.D.device_data() + t.D.imaginary_stride(),
    lda, lda, ldb, ldb, ldc, ldc, ldd, ldd
  };

  Gemm gemm;
  return gemm(args);
}

void run_host_reference(Tensors &t,
                        cutlass::gemm::GemmCoord problem_size,
                        cutlass::complex<ElementCompute> alpha,
                        cutlass::complex<ElementCompute> beta) {
  cutlass::reference::host::GemmPlanarComplex<
      ElementA, LayoutA,
      ElementB, LayoutB,
      ElementC, LayoutC,
      ElementAccumulator>(
    problem_size,
    alpha,
    t.A.host_ref(), Gemm::kTransformA,
    t.B.host_ref(), Gemm::kTransformB,
    beta,
    t.C.host_ref(),
    t.D_ref.host_ref());
}

} // namespace

/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **argv) {
  Options opt;
  opt.parse(argc, argv);
  if (opt.help) {
    opt.print_usage(std::cout);
    return 0;
  }

  cudaDeviceProp props{};
  cudaGetDeviceProperties(&props, 0);
  if (props.major < 8) {
    std::cerr << "This example requires SM80 or newer (found SM"
              << props.major << "." << props.minor << ").\n";
    return -1;
  }

  cutlass::complex<ElementCompute> alpha{1.0f, 0.0f};
  cutlass::complex<ElementCompute> beta{0.0f, 0.0f};
  cutlass::gemm::GemmCoord problem_size{opt.m, opt.n, opt.k};

  std::cout << "cbfloat16_gemm: planar complex bf16 GEMM on SM"
            << props.major << "." << props.minor
            << "  (M=" << opt.m << " N=" << opt.n << " K=" << opt.k << ")\n";

  Tensors t(opt.m, opt.n, opt.k);
  t.initialize(/*seed=*/1073);

  auto st = run_gemm(t, problem_size, alpha, beta);
  if (st != cutlass::Status::kSuccess) {
    std::cerr << "GemmCbfloat16 run failed.\n";
    return -2;
  }
  cudaDeviceSynchronize();
  t.D.sync_host();

  run_host_reference(t, problem_size, alpha, beta);

  bool passed = cutlass::reference::host::TensorEquals(t.D.host_view(), t.D_ref.host_view());

  double max_err = 0.0;
  for (int i = 0; i < opt.m * opt.n; ++i) {
    max_err = std::max(max_err, std::abs(double(t.D.host_data()[i])      - double(t.D_ref.host_data()[i])));
    max_err = std::max(max_err, std::abs(double(t.D.host_data_imag()[i]) - double(t.D_ref.host_data_imag()[i])));
  }

  std::cout << "  Correctness  : " << (passed ? "PASS" : "FAIL")
            << "   (max abs error vs host reference = "
            << std::scientific << std::setprecision(3) << max_err << ")\n";

  // Warm-up + timed loop.
  run_gemm(t, problem_size, alpha, beta);
  cudaDeviceSynchronize();

  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  cudaEventRecord(e0);
  for (int i = 0; i < opt.iterations; ++i) {
    run_gemm(t, problem_size, alpha, beta);
  }
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  float ms_total = 0.0f;
  cudaEventElapsedTime(&ms_total, e0, e1);
  double ms_per_iter = double(ms_total) / opt.iterations;

  // 4 real multiplies + 2 real adds per complex multiply-add, 2 flops per FMA
  // -> 8 * M * N * K flops per complex GEMM.
  double flops  = 8.0 * double(opt.m) * double(opt.n) * double(opt.k);
  double tflops = (flops / 1.0e12) / (ms_per_iter / 1.0e3);

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "  Perf         : " << ms_per_iter << " ms/iter   " << tflops << " TFLOP/s\n";
  std::cout << (passed ? "PASS" : "FAIL") << std::endl;

  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return passed ? 0 : 1;
}
