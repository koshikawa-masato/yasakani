// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
//
// Volta (sm_70, e.g. Tesla V100) fallbacks for the GPU instructions the engine's kernels use that Volta does not have:
//
//   * cp.async (needs sm_80)          -> an ordinary synchronous copy through shared memory; commit is a no-op and
//                                        wait is a warp synchronisation. The copies are issued by the same threads
//                                        and in the same order as the cp.async ones, and every cross-warp reader in
//                                        the engine already passes a __syncthreads() after the wait, so the data is
//                                        visible exactly where the cp.async version guarantees it.
//   * mma.sync s8 / u8 x s8 m16n8k32 (needs sm_75) -> __dp4a (sm_61+) over fragments exchanged with warp shuffles.
//                                        The fragment layouts are the ones in the PTX ISA manual (warp-level matrix
//                                        fragments for mma.m16n8k32 with .s8 / .u8), so the integer result is bit by
//                                        bit the same as the tensor-core instruction: integer sums do not depend
//                                        on the order they are added in.
//   * mma.sync bf16 / f16 m16n8k16 (needs sm_75 / sm_80) -> warp shuffles plus scalar FMA in fp32. The products of
//                                        two bf16 / f16 values are exact in fp32; only the summation order
//                                        differs from the tensor cores, so attention scores can differ in the
//                                        last bits. Nothing else in the engine depends on those bits.
//   * ldmatrix .x4 and .x4.trans (needs sm_75) -> each thread loads the same 16 bytes it would have addressed, and
//                                        warp shuffles redistribute them into the mma fragment layout (for .trans,
//                                        the 8x8 tile is transposed during the redistribution, as the instruction
//                                        specifies).
//
// The functions here are only compiled into device code for the architectures that need them; code built for
// sm_86 and newer never sees them. Correctness of the fragment shuffles is covered by tools/volta_test/.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace yskn {
namespace volta {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800

// ---- cp.async (sm_80+) -----------------------------------------------------

__device__ __forceinline__ void cp16(void* smem, const void* gmem) {
  *reinterpret_cast<uint4*>(smem) = *reinterpret_cast<const uint4*>(gmem);
}
__device__ __forceinline__ void cp8(void* smem, const void* gmem) {
  *reinterpret_cast<uint2*>(smem) = *reinterpret_cast<const uint2*>(gmem);
}
__device__ __forceinline__ void cp4(void* smem, const void* gmem) {
  *reinterpret_cast<unsigned*>(smem) = *reinterpret_cast<const unsigned*>(gmem);
}
__device__ __forceinline__ void cp_commit() {}
// cp.async.wait_group orders the issuing thread's own copies, which here have already completed; what a reader
// still needs is for the threads of its warp that wrote the shared data to have executed their copies. Sync the
// threads that reach the wait together (the active mask, not the full warp: a wait reached divergently must not
// wait for threads that took another path).
template <int N> __device__ __forceinline__ void cp_wait() { __syncwarp(__activemask()); }
__device__ __forceinline__ void cp_wait_all() { __syncwarp(__activemask()); }

// A cp.async addressed with a 32-bit shared-memory address instead of a pointer (q35_g6.cu).
__device__ __forceinline__ void cp16_saddr(unsigned saddr, const void* gmem) {
  *reinterpret_cast<uint4*>(__cvta_shared_to_generic(saddr)) = *reinterpret_cast<const uint4*>(gmem);
}

#endif  // __CUDA_ARCH__ < 800

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 750

// ---- mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 ------------------------
//
// Fragment layout (PTX ISA, mma.m16n8k32, .s8): with groupID = lane >> 2 and tig = lane & 3, a thread's A fragment
// a[0..15] (four .b32 registers, bytes low to high) covers rows groupID (a[0..3], a[8..11]) and groupID + 8 (the
// rest), at k = tig * 4 + (i & 3), plus 16 for i >= 8. Its B fragment b[0..7] (two registers) covers the single
// column groupID, at k = tig * 4 + (i & 3), plus 16 for i >= 4. Its accumulator covers (groupID, tig * 2),
// (groupID, tig * 2 + 1), (groupID + 8, tig * 2), (groupID + 8, tig * 2 + 1). For one output the A values for a
// group of four consecutive k sit in one register of one thread of the row's group, and the B values for the same
// k group in one register of one thread of the column's group, both packed in k order: exactly a __dp4a operand
// pair. Eight shuffles per operand and eight __dp4a per output rebuild the instruction.

__device__ __forceinline__ void mma_s8(int* d, const unsigned* a, unsigned b0, unsigned b1) {
  const int lane = (int)(threadIdx.x & 31), g = lane >> 2, tig = lane & 3;
  const unsigned b[2] = {b0, b1};
  int acc0 = d[0], acc1 = d[1], acc2 = d[2], acc3 = d[3];
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    const int src_a = g * 4 + (q & 3);
    const unsigned a_lo = __shfl_sync(0xffffffffu, a[q >= 4 ? 2 : 0], src_a);   // row groupID,     k = 4q .. 4q+3
    const unsigned a_hi = __shfl_sync(0xffffffffu, a[q >= 4 ? 3 : 1], src_a);   // row groupID + 8, k = 4q .. 4q+3
    const unsigned bb0 = __shfl_sync(0xffffffffu, b[q >= 4 ? 1 : 0], (tig * 2) * 4 + (q & 3));        // column tig*2
    const unsigned bb1 = __shfl_sync(0xffffffffu, b[q >= 4 ? 1 : 0], (tig * 2 + 1) * 4 + (q & 3));    // column tig*2+1
    acc0 = __dp4a((int)a_lo, (int)bb0, acc0);
    acc1 = __dp4a((int)a_lo, (int)bb1, acc1);
    acc2 = __dp4a((int)a_hi, (int)bb0, acc2);
    acc3 = __dp4a((int)a_hi, (int)bb1, acc3);
  }
  d[0] = acc0; d[1] = acc1; d[2] = acc2; d[3] = acc3;
}

// ---- mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 ------------------------
//
// Same layout, but A is unsigned. Writing an unsigned byte as its signed twin plus 256 where its top bit is set:
// d += dp4a(A as s8, B) + 256 * (sum of the B bytes whose A byte has the top bit set). The correction is computed
// byte by byte from the same shuffled registers the __dp4a path uses.

__device__ __forceinline__ void mma_u8s8(int* d, const unsigned* a, unsigned b0, unsigned b1) {
  const int lane = (int)(threadIdx.x & 31), g = lane >> 2, tig = lane & 3;
  const unsigned b[2] = {b0, b1};
  int acc0 = d[0], acc1 = d[1], acc2 = d[2], acc3 = d[3];
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    const int src_a = g * 4 + (q & 3);
    const unsigned a_lo = __shfl_sync(0xffffffffu, a[q >= 4 ? 2 : 0], src_a);
    const unsigned a_hi = __shfl_sync(0xffffffffu, a[q >= 4 ? 3 : 1], src_a);
    const unsigned bb0 = __shfl_sync(0xffffffffu, b[q >= 4 ? 1 : 0], (tig * 2) * 4 + (q & 3));
    const unsigned bb1 = __shfl_sync(0xffffffffu, b[q >= 4 ? 1 : 0], (tig * 2 + 1) * 4 + (q & 3));
    acc0 = __dp4a((int)a_lo, (int)bb0, acc0);
    acc1 = __dp4a((int)a_lo, (int)bb1, acc1);
    acc2 = __dp4a((int)a_hi, (int)bb0, acc2);
    acc3 = __dp4a((int)a_hi, (int)bb1, acc3);
    int s00 = 0, s01 = 0, s10 = 0, s11 = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int b_0 = (int)(int8_t)(bb0 >> (8 * j)), b_1 = (int)(int8_t)(bb1 >> (8 * j));
      if ((a_lo >> (8 * j)) & 0x80u) { s00 += b_0; s01 += b_1; }
      if ((a_hi >> (8 * j)) & 0x80u) { s10 += b_0; s11 += b_1; }
    }
    acc0 += 256 * s00; acc1 += 256 * s01; acc2 += 256 * s10; acc3 += 256 * s11;
  }
  d[0] = acc0; d[1] = acc1; d[2] = acc2; d[3] = acc3;
}

// ---- mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 and .f16 -----------
//
// Fragment layout (PTX ISA, mma.m16n8k16, .f16 / .bf16): a thread's A fragment a[0..7] (four .b32 registers, two
// halves each, low first) covers rows groupID (a[0], a[1], a[4], a[5]) and groupID + 8 (the rest) at
// k = tig * 2 + (i & 1), plus 8 for i >= 4. Its B fragment b[0..3] (two registers) covers the single column
// groupID at k = tig * 2 + (i & 1), plus 8 for i >= 2. The accumulator layout is the same as for the s8 shape.
// For a pair of consecutive k the A values of one row sit in one register of one thread of the row's group and
// the B values in one register of one thread of the column's group, so eight shuffle rounds rebuild it.

__device__ __forceinline__ float2 cvt_bf16(unsigned r) {
  return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&r));
}
__device__ __forceinline__ float2 cvt_f16(unsigned r) {
  return __half22float2(*reinterpret_cast<const __half2*>(&r));
}

template <bool BF16>
__device__ __forceinline__ void mma_16(float* c, const unsigned* a, unsigned b0, unsigned b1) {
  const int lane = (int)(threadIdx.x & 31), g = lane >> 2, tig = lane & 3;
  const unsigned b[2] = {b0, b1};
  float acc0 = c[0], acc1 = c[1], acc2 = c[2], acc3 = c[3];
#pragma unroll
  for (int p = 0; p < 8; ++p) {
    const int src_a = g * 4 + (p & 3);
    const unsigned a_lo = __shfl_sync(0xffffffffu, a[p >= 4 ? 2 : 0], src_a);   // row groupID,     k = 2p, 2p+1
    const unsigned a_hi = __shfl_sync(0xffffffffu, a[p >= 4 ? 3 : 1], src_a);   // row groupID + 8, k = 2p, 2p+1
    const unsigned bb0 = __shfl_sync(0xffffffffu, b[p >= 4 ? 1 : 0], (tig * 2) * 4 + (p & 3));
    const unsigned bb1 = __shfl_sync(0xffffffffu, b[p >= 4 ? 1 : 0], (tig * 2 + 1) * 4 + (p & 3));
    const float2 av_lo = BF16 ? cvt_bf16(a_lo) : cvt_f16(a_lo), av_hi = BF16 ? cvt_bf16(a_hi) : cvt_f16(a_hi);
    const float2 bv0 = BF16 ? cvt_bf16(bb0) : cvt_f16(bb0), bv1 = BF16 ? cvt_bf16(bb1) : cvt_f16(bb1);
    acc0 = __fmaf_rn(av_lo.x, bv0.x, __fmaf_rn(av_lo.y, bv0.y, acc0));
    acc1 = __fmaf_rn(av_lo.x, bv1.x, __fmaf_rn(av_lo.y, bv1.y, acc1));
    acc2 = __fmaf_rn(av_hi.x, bv0.x, __fmaf_rn(av_hi.y, bv0.y, acc2));
    acc3 = __fmaf_rn(av_hi.x, bv1.x, __fmaf_rn(av_hi.y, bv1.y, acc3));
  }
  c[0] = acc0; c[1] = acc1; c[2] = acc2; c[3] = acc3;
}
__device__ __forceinline__ void mma_bf16(float* c, const unsigned* a, unsigned b0, unsigned b1) { mma_16<true>(c, a, b0, b1); }
__device__ __forceinline__ void mma_f16(float* c, const unsigned* a, unsigned b0, unsigned b1) { mma_16<false>(c, a, b0, b1); }

// ---- ldmatrix.sync.aligned.m8n8.x4.shared.b16 --------------------------------
//
// The 32 threads each address one 16-byte row: thread q addresses row (q & 7) of tile (q >> 3). The instruction
// redistributes the rows so that thread L ends with, in register t, the two .b16 elements of tile t at row
// (L >> 2) and columns 2 * (L & 3), 2 * (L & 3) + 1: word (L & 3) of the row addressed by thread 8t + (L >> 2).

__device__ __forceinline__ void ldm_x4(unsigned* r, const void* p) {
  const uint4 chunk = *reinterpret_cast<const uint4*>(p);
  const int lane = (int)(threadIdx.x & 31), gi = lane >> 2, tig = lane & 3;
  const unsigned w[4] = {chunk.x, chunk.y, chunk.z, chunk.w};
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    unsigned v = 0;
#pragma unroll
    for (int ww = 0; ww < 4; ++ww) {
      const unsigned s = __shfl_sync(0xffffffffu, w[ww], 8 * t + gi);
      if (ww == tig) v = s;
    }
    r[t] = v;
  }
}

// ---- ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 --------------------------
//
// As above, but each 8x8 tile is transposed first: register t of thread L holds tile t's elements
// (2 * (L & 3), L >> 2) and (2 * (L & 3) + 1, L >> 2) of the stored tile, i.e. element (L >> 2) of the rows
// addressed by threads 8t + 2 * (L & 3) and 8t + 2 * (L & 3) + 1.

__device__ __forceinline__ void ldm_x4_trans(unsigned* r, const void* p) {
  const uint4 chunk = *reinterpret_cast<const uint4*>(p);
  const int lane = (int)(threadIdx.x & 31), gi = lane >> 2, tig = lane & 3;
  const unsigned w[4] = {chunk.x, chunk.y, chunk.z, chunk.w};
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    unsigned lo = 0, hi = 0;
#pragma unroll
    for (int ww = 0; ww < 4; ++ww) {
      const unsigned s0 = __shfl_sync(0xffffffffu, w[ww], 8 * t + tig * 2);
      const unsigned s1 = __shfl_sync(0xffffffffu, w[ww], 8 * t + tig * 2 + 1);
      if (ww == (gi >> 1)) {
        lo = (gi & 1) ? (s0 >> 16) : (s0 & 0xffffu);
        hi = (gi & 1) ? (s1 >> 16) : (s1 & 0xffffu);
      }
    }
    r[t] = lo | (hi << 16);
  }
}

#endif  // __CUDA_ARCH__ < 750

}  // namespace volta
}  // namespace yskn
