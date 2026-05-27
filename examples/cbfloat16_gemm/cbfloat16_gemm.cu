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

    Runs two correctness checks plus a perf measurement:
      (1) Small problem (M = N = K = 128) against a CPU double-precision reference.
      (2) Default problem (M = N = K = 1024) against the CUTLASS device reference.
      (3) Timed run at the same default size, reported in TFLOP/s using the
          8 * M * N * K flop count for complex GEMM.

    A complex GEMM C = A * B with A in C^(MxK), B in C^(KxN), C in C^(MxN) is
    implemented as four real bf16 tensor-core MMA calls per warp tile, with fp32
    accumulators, by reusing CUTLASS's planar-complex pipeline. Storage is planar:
    real and imaginary halves live in disjoint memory regions sharing one base
    pointer and an imag_stride. Outputs are fp32.
*/

#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <cmath>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/default_gemm_cbfloat16.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor_planar_complex.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/device/gemm_planar_complex.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

using Gemm = cutlass::gemm::device::GemmCbfloat16;

using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementC = float;
using ElementAccumulator = float;
using ElementCompute = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

/// CPU reference: complex matmul in double precision over planar storage.
/// Taken verbatim from the implementation prompt (lines 168-184). Used at
/// small sizes only - the budget for the default 1024^3 case is satisfied by
/// the device-side CUTLASS reference instead.
void complex_mm_ref(
    int M, int N, int K,
    const double* Ar, const double* Ai,   // M x K row-major
    const double* Br, const double* Bi,   // K x N column-major (matching LayoutB)
    double* Cr, double* Ci)               // M x N row-major
{
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      double sr = 0, si = 0;
      for (int k = 0; k < K; k++) {
        // A is row-major: A[m, k] = A[m*K + k]
        // B is column-major: B[k, n] = B[k + n*K]
        double ar = Ar[m * K + k];
        double ai = Ai[m * K + k];
        double br = Br[k + n * K];
        double bi = Bi[k + n * K];
        sr += ar * br - ai * bi;
        si += ar * bi + ai * br;
      }
      // C is row-major: C[m, n] = C[m*N + n]
      Cr[m * N + n] = sr;
      Ci[m * N + n] = si;
    }
  }
}

struct Options {
  int m = 1024;
  int n = 1024;
  int k = 1024;
  int iterations = 20;
  int small_size = 128;
  bool help = false;

  void parse(int argc, char const **argv) {
    cutlass::CommandLine cmd(argc, argv);
    if (cmd.check_cmd_line_flag("help")) help = true;
    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("small_size", small_size);
  }

  void print_usage(std::ostream &os) const {
    os << "cbfloat16_gemm - planar complex bf16 GEMM on SM80\n\n"
       << "Options:\n"
       << "  --help              Print this message\n"
       << "  --m=<int>           GEMM M (default 1024)\n"
       << "  --n=<int>           GEMM N (default 1024)\n"
       << "  --k=<int>           GEMM K (default 1024)\n"
       << "  --iterations=<int>  Timed iterations (default 20)\n"
       << "  --small_size=<int>  CPU-reference cross-check size (default 128)\n";
  }
};

/// Allocates HostTensorPlanarComplex tensors, fills A and B with uniform random
/// values in [-2, 2], and pushes host -> device. The narrow range keeps
/// accumulated error inside the 1e-2 bf16 budget at K up to 1024.
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
    D_ref.reset({M, N});
  }

  void fill_random(uint64_t seed) {
    cutlass::reference::device::BlockFillRandomUniform(
        A.device_data(), A.size() * 2, seed, ElementA(2), ElementA(-2), 0);
    cutlass::reference::device::BlockFillRandomUniform(
        B.device_data(), B.size() * 2, seed * 7u + 1u, ElementB(2), ElementB(-2), 0);
    A.sync_host();
    B.sync_host();
  }
};

/// Runs the bf16 planar complex GEMM into tensors.D and returns Status.
cutlass::Status run_gemm(Tensors &t, int M, int N, int K) {
  typename Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    /*batch_count=*/1,
    {cutlass::complex<ElementCompute>(1.0f, 0.0f),
     cutlass::complex<ElementCompute>(0.0f, 0.0f)},
    t.A.device_data(),
    t.A.device_data_imag(),
    t.B.device_data(),
    t.B.device_data_imag(),
    t.C.device_data(),
    t.C.device_data_imag(),
    t.D.device_data(),
    t.D.device_data_imag(),
    t.A.layout().stride(0),
    t.A.layout().stride(0),
    t.B.layout().stride(0),
    t.B.layout().stride(0),
    t.C.layout().stride(0),
    t.C.layout().stride(0),
    t.D.layout().stride(0),
    t.D.layout().stride(0)
  };

  Gemm gemm;

  size_t workspace_size = Gemm::get_workspace_size(args);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  if (auto s = gemm.can_implement(args); s != cutlass::Status::kSuccess) return s;
  if (auto s = gemm.initialize(args, workspace.get()); s != cutlass::Status::kSuccess) return s;
  return gemm(/*stream=*/nullptr);
}

/// Runs the CUTLASS device reference into tensors.D_ref.
void run_device_reference(Tensors &t, int M, int N, int K) {
  cutlass::reference::device::GemmPlanarComplex<
      ElementA, LayoutA,
      ElementB, LayoutB,
      ElementC, LayoutC,
      ElementCompute,
      ElementAccumulator>(
    {M, N, K},
    cutlass::complex<ElementCompute>(1.0f, 0.0f),
    t.A.device_ref(),
    cutlass::ComplexTransform::kNone,
    t.B.device_ref(),
    cutlass::ComplexTransform::kNone,
    cutlass::complex<ElementCompute>(0.0f, 0.0f),
    t.C.device_ref(),
    t.D_ref.device_ref(),
    ElementAccumulator(0));
}

/// Returns max(|D - D_ref|) across both the real and imaginary planes.
double max_abs_error(Tensors &t) {
  t.D.sync_host();
  t.D_ref.sync_host();
  double max_err = 0.0;
  size_t n_elements = t.D.size() * 2;
  ElementC const *d = t.D.host_data();
  ElementC const *r = t.D_ref.host_data();
  for (size_t i = 0; i < n_elements; ++i) {
    double err = std::abs(double(d[i]) - double(r[i]));
    if (err > max_err) max_err = err;
  }
  return max_err;
}

/// Runs the CPU double-precision reference and compares against the GPU result.
/// Used as a sanity check at a small problem size where the O(M*N*K) cost is
/// negligible (<1s at the default small_size=128).
bool cpu_cross_check(int M, int N, int K, uint64_t seed, double tol) {
  Tensors t(M, N, K);
  t.fill_random(seed);

  if (run_gemm(t, M, N, K) != cutlass::Status::kSuccess) return false;
  cudaDeviceSynchronize();
  t.D.sync_host();

  // Convert bf16 inputs to double for the CPU reference.
  std::vector<double> Ar(size_t(M) * K), Ai(size_t(M) * K);
  std::vector<double> Br(size_t(K) * N), Bi(size_t(K) * N);
  std::vector<double> Cr(size_t(M) * N), Ci(size_t(M) * N);
  for (size_t i = 0; i < Ar.size(); ++i) {
    Ar[i] = double(t.A.host_data()[i]);
    Ai[i] = double(t.A.host_data_imag()[i]);
  }
  for (size_t i = 0; i < Br.size(); ++i) {
    Br[i] = double(t.B.host_data()[i]);
    Bi[i] = double(t.B.host_data_imag()[i]);
  }
  complex_mm_ref(M, N, K, Ar.data(), Ai.data(), Br.data(), Bi.data(), Cr.data(), Ci.data());

  double max_err = 0.0;
  for (size_t i = 0; i < Cr.size(); ++i) {
    max_err = std::max(max_err, std::abs(double(t.D.host_data()[i]) - Cr[i]));
    max_err = std::max(max_err, std::abs(double(t.D.host_data_imag()[i]) - Ci[i]));
  }
  std::cout << "  CPU-ref check  M=N=K=" << M
            << " : max abs error = " << std::scientific << std::setprecision(3) << max_err
            << "  tol=" << tol << "  "
            << (max_err < tol ? "PASS" : "FAIL") << "\n";
  return max_err < tol;
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
  int device = 0;
  cudaGetDeviceProperties(&props, device);
  if (props.major < 8) {
    std::cerr << "This example requires SM80 or newer (found SM"
              << props.major << "." << props.minor << ").\n";
    return -1;
  }

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "cbfloat16_gemm: planar complex bf16 GEMM on SM"
            << props.major << "." << props.minor << "\n\n";

  // -- (1) CPU double-precision cross check at small size ---------------------
  bool small_ok = cpu_cross_check(
      opt.small_size, opt.small_size, opt.small_size,
      /*seed=*/2026u, /*tol=*/1e-1);

  // -- (2) Device-reference correctness at full size --------------------------
  Tensors t(opt.m, opt.n, opt.k);
  t.fill_random(/*seed=*/1729u);

  if (run_gemm(t, opt.m, opt.n, opt.k) != cutlass::Status::kSuccess) {
    std::cerr << "GemmCbfloat16 run failed.\n";
    return -2;
  }
  run_device_reference(t, opt.m, opt.n, opt.k);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "cudaDeviceSynchronize after reference: " << cudaGetErrorString(err) << "\n";
    return -3;
  }

  double max_err = max_abs_error(t);
  bool big_ok = max_err < 1e-2;
  std::cout << "  Device-ref check M=" << opt.m << " N=" << opt.n << " K=" << opt.k
            << " : max abs error = " << std::scientific << std::setprecision(3) << max_err
            << "  tol=1e-2  " << (big_ok ? "PASS" : "FAIL") << "\n\n";

  // -- (3) Performance measurement at full size -------------------------------
  cudaEvent_t e_begin, e_end;
  cudaEventCreate(&e_begin);
  cudaEventCreate(&e_end);

  // Warm-up.
  if (run_gemm(t, opt.m, opt.n, opt.k) != cutlass::Status::kSuccess) {
    std::cerr << "Warm-up run failed.\n";
    return -4;
  }
  cudaDeviceSynchronize();

  cudaEventRecord(e_begin);
  for (int i = 0; i < opt.iterations; ++i) {
    run_gemm(t, opt.m, opt.n, opt.k);
  }
  cudaEventRecord(e_end);
  cudaEventSynchronize(e_end);

  float ms_total = 0.0f;
  cudaEventElapsedTime(&ms_total, e_begin, e_end);
  double ms_per_iter = double(ms_total) / opt.iterations;

  // 4 real multiplies + 2 real adds per complex multiply-add, 2 flops per FMA.
  // -> 8 * M * N * K flops per complex GEMM.
  double flops = 8.0 * double(opt.m) * double(opt.n) * double(opt.k);
  double tflops = (flops / 1.0e12) / (ms_per_iter / 1.0e3);

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "  Perf  M=" << opt.m << " N=" << opt.n << " K=" << opt.k
            << " : " << ms_per_iter << " ms/iter"
            << "  " << tflops << " TFLOP/s\n\n";

  bool overall = small_ok && big_ok;
  std::cout << (overall ? "PASS" : "FAIL") << std::endl;

  cudaEventDestroy(e_begin);
  cudaEventDestroy(e_end);
  return overall ? 0 : 1;
}
