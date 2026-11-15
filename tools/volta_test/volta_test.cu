// SPDX-License-Identifier: Apache-2.0
// Standalone checks for the Volta fallbacks in src/volta_compat.cuh.
//
// Build for a Volta GPU (Tesla V100) with a CUDA toolkit that still knows sm_70 (12.8 or older; CUDA 13 removed it):
//   nvcc -O2 -arch=sm_70 -I src tools/volta_test/volta_test.cu -o /tmp/volta_test && /tmp/volta_test
//
// Every test builds the mma / ldmatrix fragments straight from the layouts in the PTX ISA manual (warp-level
// matrix fragments for mma.m16n8k32 / .m16n8k16 and ldmatrix .m8n8.x4), runs the fallback, and compares against a
// plain host computation. The integer tests must match exactly.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include "volta_compat.cuh"

using namespace yskn;

static int failures = 0;
static void report(const char* name, bool ok) {
  std::printf("%-14s %s\n", name, ok ? "PASS" : "FAIL");
  if (!ok) ++failures;
}

// ---------- mma m16n8k32, s8 x s8 and u8 x s8 ----------

template <bool U8>
__global__ void mma32_kernel(const int8_t* A, const int8_t* B, const int* C0, int* D) {
  const int lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  unsigned a[4], b[2];
  for (int rr = 0; rr < 4; ++rr) {
    unsigned w = 0;
    for (int j = 0; j < 4; ++j) {
      const int i = rr * 4 + j;
      const int row = (i < 4 || (i >= 8 && i < 12)) ? g : g + 8;
      const int k = tig * 4 + (i & 3) + (i >= 8 ? 16 : 0);
      w |= (unsigned)(uint8_t)A[row * 32 + k] << (8 * j);
    }
    a[rr] = w;
  }
  for (int rr = 0; rr < 2; ++rr) {
    unsigned w = 0;
    for (int j = 0; j < 4; ++j) {
      const int i = rr * 4 + j;
      const int k = tig * 4 + (i & 3) + (i >= 4 ? 16 : 0);
      w |= (unsigned)(uint8_t)B[k * 8 + g] << (8 * j);
    }
    b[rr] = w;
  }
  int d[4];
  for (int i = 0; i < 4; ++i) d[i] = C0[(i < 2 ? g : g + 8) * 8 + tig * 2 + (i & 1)];
  if (U8) volta::mma_u8s8(d, a, b[0], b[1]); else volta::mma_s8(d, a, b[0], b[1]);
  for (int i = 0; i < 4; ++i) D[(i < 2 ? g : g + 8) * 8 + tig * 2 + (i & 1)] = d[i];
}

static bool test_mma32(bool u8) {
  std::mt19937 rng(u8 ? 12345 : 999);
  std::vector<int8_t> A(16 * 32), B(32 * 8);
  std::vector<int> C0(16 * 8), D(16 * 8), ref(16 * 8);
  for (auto& x : A) x = (int8_t)(rng() % 256);
  for (auto& x : B) x = (int8_t)(rng() % 256);
  for (auto& x : C0) x = (int)(rng() % 2001) - 1000;
  for (int r = 0; r < 16; ++r)
    for (int c = 0; c < 8; ++c) {
      int s = C0[r * 8 + c];
      for (int k = 0; k < 32; ++k) s += (u8 ? (int)(uint8_t)A[r * 32 + k] : (int)A[r * 32 + k]) * (int)B[k * 8 + c];
      ref[r * 8 + c] = s;
    }
  int8_t *dA, *dB; int *dC, *dD;
  cudaMalloc(&dA, A.size()); cudaMalloc(&dB, B.size()); cudaMalloc(&dC, C0.size() * 4); cudaMalloc(&dD, D.size() * 4);
  cudaMemcpy(dA, A.data(), A.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B.data(), B.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dC, C0.data(), C0.size() * 4, cudaMemcpyHostToDevice);
  if (u8) mma32_kernel<true><<<1, 32>>>(dA, dB, dC, dD); else mma32_kernel<false><<<1, 32>>>(dA, dB, dC, dD);
  const cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { std::printf("  cuda error: %s\n", cudaGetErrorString(e)); return false; }
  cudaMemcpy(D.data(), dD, D.size() * 4, cudaMemcpyDeviceToHost);
  cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dD);
  return D == ref;
}

// ---------- mma m16n8k16, bf16 and f16 ----------

template <bool BF16>
__global__ void mma16_kernel(const uint16_t* A, const uint16_t* B, const float* C0, float* D) {
  const int lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  unsigned a[4], b[2];
  for (int rr = 0; rr < 4; ++rr) {
    unsigned w = 0;
    for (int h = 0; h < 2; ++h) {
      const int i = rr * 2 + h;
      const int row = (i < 2 || (i >= 4 && i < 6)) ? g : g + 8;
      const int k = tig * 2 + (i & 1) + (i >= 4 ? 8 : 0);
      w |= (unsigned)A[row * 16 + k] << (16 * h);
    }
    a[rr] = w;
  }
  for (int rr = 0; rr < 2; ++rr) {
    unsigned w = 0;
    for (int h = 0; h < 2; ++h) {
      const int i = rr * 2 + h;
      const int k = tig * 2 + (i & 1) + (i >= 2 ? 8 : 0);
      w |= (unsigned)B[k * 8 + g] << (16 * h);
    }
    b[rr] = w;
  }
  float c[4];
  for (int i = 0; i < 4; ++i) c[i] = C0[(i < 2 ? g : g + 8) * 8 + tig * 2 + (i & 1)];
  if (BF16) volta::mma_bf16(c, a, b[0], b[1]); else volta::mma_f16(c, a, b[0], b[1]);
  for (int i = 0; i < 4; ++i) D[(i < 2 ? g : g + 8) * 8 + tig * 2 + (i & 1)] = c[i];
}

static uint16_t to_bf16(float f) { unsigned u; std::memcpy(&u, &f, 4); return (uint16_t)(u >> 16); }

template <bool BF16>
static bool test_mma16() {
  const float vals[] = {-2.f, -1.5f, -1.f, -0.5f, -0.25f, 0.f, 0.25f, 0.5f, 1.f, 1.5f, 2.f, 3.f};
  std::mt19937 rng(BF16 ? 777 : 555);
  std::vector<uint16_t> A(16 * 16), B(16 * 8);
  std::vector<float> C0(16 * 8), D(16 * 8), ref(16 * 8), Af(16 * 16), Bf(16 * 8);
  auto pick = [&] { return vals[rng() % 12]; };
  for (int i = 0; i < 16 * 16; ++i) { const float f = pick(); Af[i] = f; A[i] = BF16 ? to_bf16(f) : (uint16_t)__half_as_ushort(__float2half(f)); }
  for (int i = 0; i < 16 * 8; ++i) { const float f = pick(); Bf[i] = f; B[i] = BF16 ? to_bf16(f) : (uint16_t)__half_as_ushort(__float2half(f)); }
  for (int i = 0; i < 16 * 8; ++i) C0[i] = pick();
  for (int r = 0; r < 16; ++r)
    for (int c = 0; c < 8; ++c) {
      float s = C0[r * 8 + c];
      for (int k = 0; k < 16; ++k) s += Af[r * 16 + k] * Bf[k * 8 + c];
      ref[r * 8 + c] = s;
    }
  uint16_t *dA, *dB; float *dC, *dD;
  cudaMalloc(&dA, A.size() * 2); cudaMalloc(&dB, B.size() * 2); cudaMalloc(&dC, C0.size() * 4); cudaMalloc(&dD, D.size() * 4);
  cudaMemcpy(dA, A.data(), A.size() * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B.data(), B.size() * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dC, C0.data(), C0.size() * 4, cudaMemcpyHostToDevice);
  mma16_kernel<BF16><<<1, 32>>>(dA, dB, dC, dD);
  const cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { std::printf("  cuda error: %s\n", cudaGetErrorString(e)); return false; }
  cudaMemcpy(D.data(), dD, D.size() * 4, cudaMemcpyDeviceToHost);
  cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dD);
  for (int i = 0; i < 16 * 8; ++i) if (std::fabs(D[i] - ref[i]) > 1e-3f) return false;
  return true;
}

// ---------- ldmatrix x4 / x4.trans ----------

template <bool TRANS>
__global__ void ldm_kernel(const uint16_t* in, unsigned* out) {
  __shared__ __align__(16) uint16_t sm[4 * 64];
  for (int i = threadIdx.x; i < 4 * 64; i += blockDim.x) sm[i] = in[i];
  __syncthreads();
  const int lane = threadIdx.x & 31;
  unsigned r[4];
  const uint16_t* p = sm + (lane >> 3) * 64 + (lane & 7) * 8;
  if (TRANS) volta::ldm_x4_trans(r, p); else volta::ldm_x4(r, p);
  for (int t = 0; t < 4; ++t) out[lane * 4 + t] = r[t];
}

static bool test_ldm(bool trans) {
  std::vector<uint16_t> in(4 * 64);
  for (int t = 0; t < 4; ++t)
    for (int r = 0; r < 8; ++r)
      for (int c = 0; c < 8; ++c) in[t * 64 + r * 8 + c] = (uint16_t)(t * 512 + r * 64 + c * 7 + 1);
  uint16_t* dIn; unsigned* dOut;
  std::vector<unsigned> out(32 * 4);
  cudaMalloc(&dIn, in.size() * 2); cudaMalloc(&dOut, out.size() * 4);
  cudaMemcpy(dIn, in.data(), in.size() * 2, cudaMemcpyHostToDevice);
  if (trans) ldm_kernel<true><<<1, 32>>>(dIn, dOut); else ldm_kernel<false><<<1, 32>>>(dIn, dOut);
  const cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { std::printf("  cuda error: %s\n", cudaGetErrorString(e)); return false; }
  cudaMemcpy(out.data(), dOut, out.size() * 4, cudaMemcpyDeviceToHost);
  cudaFree(dIn); cudaFree(dOut);
  for (int lane = 0; lane < 32; ++lane) {
    const int gi = lane >> 2, tig = lane & 3;
    for (int t = 0; t < 4; ++t) {
      const uint16_t lo = trans ? in[t * 64 + (2 * tig) * 8 + gi] : in[t * 64 + gi * 8 + 2 * tig];
      const uint16_t hi = trans ? in[t * 64 + (2 * tig + 1) * 8 + gi] : in[t * 64 + gi * 8 + 2 * tig + 1];
      if (out[lane * 4 + t] != ((unsigned)lo | ((unsigned)hi << 16))) return false;
    }
  }
  return true;
}

// ---------- cp.async fallback ----------

__global__ void cp_kernel(const uint4* in16, const uint2* in8, const unsigned* in4, unsigned* out) {
  __shared__ __align__(16) uint4 s16; __shared__ __align__(8) uint2 s8; __shared__ unsigned s4;
  if (threadIdx.x == 0) { volta::cp16(&s16, in16); volta::cp8(&s8, in8); volta::cp4(&s4, in4); volta::cp_commit(); }
  volta::cp_wait<0>();
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned* p = reinterpret_cast<const unsigned*>(&s16);
    unsigned s = s4 + s8.x + s8.y;
    for (int i = 0; i < 4; ++i) s += p[i];
    out[0] = s;
  }
}

static bool test_cp() {
  uint4 a = make_uint4(1, 2, 3, 4); uint2 b = make_uint2(5, 6); unsigned c = 7, out = 0;
  uint4* dA; uint2* dB; unsigned *dC, *dO;
  cudaMalloc(&dA, 16); cudaMalloc(&dB, 8); cudaMalloc(&dC, 4); cudaMalloc(&dO, 4);
  cudaMemcpy(dA, &a, 16, cudaMemcpyHostToDevice); cudaMemcpy(dB, &b, 8, cudaMemcpyHostToDevice);
  cudaMemcpy(dC, &c, 4, cudaMemcpyHostToDevice);
  cp_kernel<<<1, 32>>>(dA, dB, dC, dO);
  const cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { std::printf("  cuda error: %s\n", cudaGetErrorString(e)); return false; }
  cudaMemcpy(&out, dO, 4, cudaMemcpyDeviceToHost);
  cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dO);
  return out == 1 + 2 + 3 + 4 + 5 + 6 + 7;
}

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  cudaDeviceProp p{};
  if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) { std::printf("no CUDA device\n"); return 2; }
  std::printf("device: %s (compute capability %d.%d)\n", p.name, p.major, p.minor);
  report("mma_s8", test_mma32(false));
  report("mma_u8s8", test_mma32(true));
  report("mma_bf16", test_mma16<true>());
  report("mma_f16", test_mma16<false>());
  report("ldmatrix_x4", test_ldm(false));
  report("ldmatrix_x4t", test_ldm(true));
  report("cp_async", test_cp());
  std::printf(failures ? "FAILED (%d)\n" : "ALL PASS\n", failures);
  return failures ? 1 : 0;
}
