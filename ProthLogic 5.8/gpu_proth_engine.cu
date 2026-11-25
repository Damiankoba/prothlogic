// Copyright (C) Damian Koba.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// GPU Proth engine (API and the scaled form: gpu_proth_engine.h). Errors throw instead of exit, CUDA Graphs (16
// squarings per launch: the host thread does almost nothing), waits on a blocking event (the thread sleeps instead of
// spinning a core the CPU workers need).
//
// The number lives on the card as signed digits d_t of b bits (value = sum d_t 2^(t*b), t < L/2), NEARLY balanced:
// |d_t| <= 2^(b-1) + 1 - never brought to the canonical form, so no carry chain anywhere:
//  1. pack: d_t mod p for 3-4 primes p < 2^31 (zero above L/2: the product of two halves fits the cyclic length L)
//  2. forward NTT, pointwise square (or product of two transforms), inverse NTT
//  3. per coefficient: scale, CRT (Garner) -> exact signed coefficient c_m (int128)
//  4. E_m = sum of the b-bit parts of c_m, c_(m-1), ... landing at position m  (|E_m| < K*2^b)
//  5. split at bit n (n = m0*b + off): S = H - k*L == -k*S_full (mod N)  (k*2^n == -1)
//  6. q = round(S / N) from the digits around bit n (Horner from the top: the top digits cancel), S -= q*N: |S| ~< N/2
//  7. neighbour passes (2-3, from the digit bound): d_t = bal(S_t) + carry(S_(t-1)) -> nearly balanced again
// A digit of index T != 0 forces |value| >= ~2^(T*b - 1), so the digits never spread above the value's size.
// Bound of the convolution: |c_m| <= L/2 * (2^(b-1)+2)^2 < P/2 (product of the primes) - worst case, no statistics.

#include "gpu_proth_engine.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <mutex>
#include <thread>
#include <atomic>
#include <chrono>
#include <map>
#include <array>
#include <exception>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    char m_[256]; std::snprintf(m_, sizeof(m_), "CUDA error %s at %s:%d", cudaGetErrorString(e_), __FILE__, __LINE__); \
    throw std::runtime_error(m_); } } while (0)

typedef __int128 i128;
typedef unsigned __int128 u128;

namespace {

struct Pr { uint32_t p, pinv; };
__constant__ Pr c_pr[4];
__constant__ uint32_t c_inv[4][4];    // Garner: p_j^-1 mod p_i (j < i), Montgomery form
__constant__ uint32_t c_sc[25][4];    // [lg L][prime]: R^2 / L mod p - undoes the Montgomery product (R^-1) and the L of the inverse NTT
__constant__ uint32_t c_r23[4][2];    // R^2, R^3 mod p: residues of 64-bit digits without a 64-bit division

const int LG_MIN = 12, LG_MAX = 24;   // k_block needs L >= 2^12 (2^12 per block); 754974721 = 45*2^24 + 1 limits the length
const uint32_t PRIMES[4] = { 2013265921u, 1811939329u, 754974721u, 469762049u };   // c*2^k+1, k = 27, 26, 24, 26
const unsigned GRAPH_SQ = 16;         // squarings per CUDA Graph launch

__device__ __forceinline__ uint32_t mmul(uint32_t a, uint32_t b, uint32_t p, uint32_t pinv) {   // a*b/R mod p, a, b < p
    const uint64_t t = (uint64_t)a * b;
    const uint32_t m = (uint32_t)t * pinv;
    const uint32_t u = (uint32_t)((t + (uint64_t)m * p) >> 32);
    return u >= p ? u - p : u;
}
__device__ __forceinline__ uint32_t madd(uint32_t a, uint32_t b, uint32_t p) { const uint32_t s = a + b; return s >= p ? s - p : s; }
__device__ __forceinline__ uint32_t msub(uint32_t a, uint32_t b, uint32_t p) { return a >= b ? a - b : a + p - b; }
__device__ __forceinline__ void dif(uint32_t& x, uint32_t& y, uint32_t w, uint32_t p, uint32_t pv) {
    const uint32_t u = x, v = y;
    x = madd(u, v, p);
    y = mmul(msub(u, v, p), w, p, pv);
}
__device__ __forceinline__ void dit(uint32_t& x, uint32_t& y, uint32_t w, uint32_t p, uint32_t pv) {
    const uint32_t t = mmul(y, w, p, pv);
    const uint32_t u = x;
    x = madd(u, t, p);
    y = msub(u, t, p);
}

// ---- NTT kernels: array c of np uses prime c, a[c*L + i], tw[c*L + h + j] = w_2h^j * R (itw: inverse roots)
__global__ void k_dif2(uint32_t* a, const uint32_t* tw, int lgL, int lgh, int np) {
    const size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t L = (size_t)1 << lgL, h = (size_t)1 << lgh, per = L >> 1;
    if (t >= per * np) return;
    const int c = (int)(t >> (lgL - 1));
    const size_t r = t & (per - 1), j = r & (h - 1);
    uint32_t* x = a + c * L + ((r >> lgh) << (lgh + 1)) + j;
    uint32_t u = x[0], v = x[h];
    dif(u, v, tw[c * L + h + j], c_pr[c].p, c_pr[c].pinv);
    x[0] = u; x[h] = v;
}
__global__ void k_dit2(uint32_t* a, const uint32_t* itw, int lgL, int lgh, int np) {
    const size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t L = (size_t)1 << lgL, h = (size_t)1 << lgh, per = L >> 1;
    if (t >= per * np) return;
    const int c = (int)(t >> (lgL - 1));
    const size_t r = t & (per - 1), j = r & (h - 1);
    uint32_t* x = a + c * L + ((r >> lgh) << (lgh + 1)) + j;
    uint32_t u = x[0], v = x[h];
    dit(u, v, itw[c * L + h + j], c_pr[c].p, c_pr[c].pinv);
    x[0] = u; x[h] = v;
}
// 3 levels in registers: v[i] = x[i*q] of a group of 8q, levels h = 4q, 2q, q (DIF) or q, 2q, 4q (DIT); T = the
// table of the prime, j = position in the group (< q).
// The 7 twiddles of the group: w[0..3] level 4q at j + i*q, w[4..5] level 2q at j + i*q, w[6] level q at j. tw8: from
// the one entry w_8q^j (w_8q^(j+iq) = w_8q^j * w_8^i, w_4q^(j+iq) = w_8q^(2j) * w_4^i, w_2q^j = w_8q^(4j); w_8^i = T[4+i],
// w_4 = T[3]): 6 products instead of 6 loads - the outer passes of 2^21 are bound by memory (25 MB per array, the table
// as big as the data). k_block: the table (tw8t).
__device__ __forceinline__ void tw8x(uint32_t* w, uint32_t x, const uint32_t* T, uint32_t p, uint32_t pv) {   // from x = w_8q^j
    w[0] = x;
    w[1] = mmul(x, T[5], p, pv);
    w[2] = mmul(x, T[6], p, pv);
    w[3] = mmul(x, T[7], p, pv);
    w[4] = mmul(x, x, p, pv);
    w[5] = mmul(w[4], T[3], p, pv);
    w[6] = mmul(w[4], w[4], p, pv);
}
__device__ __forceinline__ void tw8(uint32_t* w, const uint32_t* T, unsigned q, unsigned j, uint32_t p, uint32_t pv) {
    tw8x(w, T[4 * q + j], T, p, pv);
}
__device__ __forceinline__ void tw8t(uint32_t* w, const uint32_t* T, unsigned q, unsigned j) {
#pragma unroll
    for (int i = 0; i < 4; ++i) w[i] = T[4 * q + j + i * q];
    w[4] = T[2 * q + j];
    w[5] = T[3 * q + j];
    w[6] = T[q + j];
}
__device__ __forceinline__ void dif8w_tail(uint32_t* v, const uint32_t* w, uint32_t p, uint32_t pv) {   // levels 2q, q
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        dif(v[i], v[i + 2], w[4 + i], p, pv);
        dif(v[i + 4], v[i + 6], w[4 + i], p, pv);
    }
#pragma unroll
    for (int i = 0; i < 8; i += 2) dif(v[i], v[i + 1], w[6], p, pv);
}
__device__ __forceinline__ void dif8w(uint32_t* v, const uint32_t* w, uint32_t p, uint32_t pv) {
#pragma unroll
    for (int i = 0; i < 4; ++i) dif(v[i], v[i + 4], w[i], p, pv);
    dif8w_tail(v, w, p, pv);
}
__device__ __forceinline__ void dit8w(uint32_t* v, const uint32_t* w, uint32_t p, uint32_t pv) {
#pragma unroll
    for (int i = 0; i < 8; i += 2) dit(v[i], v[i + 1], w[6], p, pv);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        dit(v[i], v[i + 2], w[4 + i], p, pv);
        dit(v[i + 4], v[i + 6], w[4 + i], p, pv);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) dit(v[i], v[i + 4], w[i], p, pv);
}
__device__ __forceinline__ void dif8r(uint32_t* v, const uint32_t* T, unsigned q, unsigned j, uint32_t p, uint32_t pv) {
    uint32_t w[7];
    tw8t(w, T, q, j);
    dif8w(v, w, p, pv);
}
__device__ __forceinline__ void dit8r(uint32_t* v, const uint32_t* T, unsigned q, unsigned j, uint32_t p, uint32_t pv) {
    uint32_t w[7];
    tw8t(w, T, q, j);
    dit8w(v, w, p, pv);
}
__global__ void k_dif8(uint32_t* a, const uint32_t* tw, int lgL, int lgq, int np) {
    const size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t L = (size_t)1 << lgL, q = (size_t)1 << lgq, per = L >> 3;
    if (t >= per * np) return;
    const int c = (int)(t >> (lgL - 3));
    const size_t r = t & (per - 1), j = r & (q - 1);
    const uint32_t p = c_pr[c].p, pv = c_pr[c].pinv;
    uint32_t* x = a + c * L + ((r >> lgq) << (lgq + 3)) + j;
    uint32_t v[8], w[7];
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = x[i * q];
    tw8(w, tw + c * L, (unsigned)q, (unsigned)j, p, pv);
    dif8w(v, w, p, pv);
#pragma unroll
    for (int i = 0; i < 8; ++i) x[i * q] = v[i];
}
__global__ void k_dit8(uint32_t* a, const uint32_t* itw, int lgL, int lgq, int np) {
    const size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t L = (size_t)1 << lgL, q = (size_t)1 << lgq, per = L >> 3;
    if (t >= per * np) return;
    const int c = (int)(t >> (lgL - 3));
    const size_t r = t & (per - 1), j = r & (q - 1);
    const uint32_t p = c_pr[c].p, pv = c_pr[c].pinv;
    uint32_t* x = a + c * L + ((r >> lgq) << (lgq + 3)) + j;
    uint32_t v[8], w[7];
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = x[i * q];
    tw8(w, itw + c * L, (unsigned)q, (unsigned)j, p, pv);
    dit8w(v, w, p, pv);
#pragma unroll
    for (int i = 0; i < 8; ++i) x[i * q] = v[i];
}
// The lowest 12 levels of 4096 elements: 512 threads x 8 elements, radix-8 passes (strides 512, 64, 8, 1) through shared
// memory padded with a word per 32 (no bank conflicts for the strides 512, 64 and 1). The last forward pass, the square
// and the first inverse pass work on the same 8 elements in registers. MODE 0: forward + square + inverse,
// 1: forward only, 2: inverse only (the product of two values: both forward, k_pmul, inverse).
// Montgomery products; a minimum of 3 blocks per SM was the fastest register limit (2^24 x 4).
template <bool INV> __device__ __forceinline__ void r8(uint32_t* v, const uint32_t* T, unsigned q, unsigned j, uint32_t p,
                                                       uint32_t pv) {
    if (INV) dit8r(v, T, q, j, p, pv);
    else dif8r(v, T, q, j, p, pv);
}
__device__ __forceinline__ unsigned spad(unsigned i) { return i + (i >> 5); }
template <bool INV> __device__ __forceinline__ void spass(uint32_t* s, uint32_t* v, const uint32_t* T, unsigned lgq, unsigned r,
                                                          uint32_t p, uint32_t pv) {
    const unsigned q = 1u << lgq, j = r & (q - 1), base = ((r >> lgq) << (lgq + 3)) + j;
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = s[spad(base + i * q)];
    r8<INV>(v, T, q, j, p, pv);
#pragma unroll
    for (int i = 0; i < 8; ++i) s[spad(base + i * q)] = v[i];
    __syncthreads();
}
template <int MODE> __global__ void __launch_bounds__(512, 3) k_block(uint32_t* a, const uint32_t* tw, const uint32_t* itw, int lgL) {
    __shared__ uint32_t s[4096 + 128];
    const size_t L = (size_t)1 << lgL;
    const unsigned nb = (unsigned)(L >> 12);
    const unsigned c = blockIdx.x / nb, b = blockIdx.x % nb;
    const uint32_t p = c_pr[c].p, pv = c_pr[c].pinv;
    const uint32_t* T = tw + c * L;
    const uint32_t* IT = itw + c * L;
    uint32_t* g = a + c * L + ((size_t)b << 12);
    const unsigned r = threadIdx.x;
    uint4* g8 = reinterpret_cast<uint4*>(g + 8 * r);
    uint32_t v[8];
    if (MODE != 2) {
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = g[r + 512 * i];
        r8<false>(v, T, 512, r, p, pv);
#pragma unroll
        for (int i = 0; i < 8; ++i) s[spad(r + 512 * i)] = v[i];
        __syncthreads();
        spass<false>(s, v, T, 6, r, p, pv);
        spass<false>(s, v, T, 3, r, p, pv);
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = s[spad(8 * r + i)];
        r8<false>(v, T, 1, 0, p, pv);
    }
    if (MODE == 1) {
        g8[0] = make_uint4(v[0], v[1], v[2], v[3]);
        g8[1] = make_uint4(v[4], v[5], v[6], v[7]);
        return;
    }
    if (MODE == 2) {
        const uint4 x = g8[0], y = g8[1];
        v[0] = x.x; v[1] = x.y; v[2] = x.z; v[3] = x.w;
        v[4] = y.x; v[5] = y.y; v[6] = y.z; v[7] = y.w;
    }
    if (MODE == 0) {
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = mmul(v[i], v[i], p, pv);
    }
    r8<true>(v, IT, 1, 0, p, pv);
#pragma unroll
    for (int i = 0; i < 8; ++i) s[spad(8 * r + i)] = v[i];
    __syncthreads();
    spass<true>(s, v, IT, 3, r, p, pv);
    spass<true>(s, v, IT, 6, r, p, pv);
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = s[spad(r + 512 * i)];
    r8<true>(v, IT, 512, r, p, pv);
#pragma unroll
    for (int i = 0; i < 8; ++i) g[r + 512 * i] = v[i];
}
__global__ void k_pmul(uint32_t* a, const uint32_t* b, int lgL, int np) {
    const size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (t >= ((size_t)np << lgL)) return;
    const int c = (int)(t >> lgL);
    a[t] = mmul(a[t], b[t], c_pr[c].p, c_pr[c].pinv);
}

// 1. digits (first L/2) -> residues of every prime. d mod p without a 64-bit division (slow on the card):
// |d| = xh*2^32 + xl, R*|d| = xl*R + xh*R^2 = mmul(xl, R^2) + mmul(xh, R^3), then one mmul by 1 divides by R
__device__ __forceinline__ uint32_t res(int64_t d, int c) {
    const uint32_t p = c_pr[c].p, pv = c_pr[c].pinv;
    const uint64_t x = d < 0 ? 0 - (uint64_t)d : (uint64_t)d;
    const uint32_t y = madd(mmul((uint32_t)x, c_r23[c][0], p, pv), mmul((uint32_t)(x >> 32), c_r23[c][1], p, pv), p);
    const uint32_t r = mmul(y, 1, p, pv);
    return (d < 0 && r) ? p - r : r;
}
__global__ void k_pack(const int64_t* d, uint32_t* a, int lgL, int np) {
    const size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t L = (size_t)1 << lgL;
    if (t >= L * np) return;
    const int c = (int)(t >> lgL);
    const size_t j = t & (L - 1);
    a[t] = (j < (L >> 1)) ? res(d[j], c) : 0;
}
// pack + the first forward pass (levels L/2, L/4, L/8) in one: the upper half of the input is zero, so level L/2 is
// x = u, y = u*w
__global__ void k_dif8_first(const int64_t* d, uint32_t* a, const uint32_t* tw, int lgL, int np) {
    const size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t L = (size_t)1 << lgL, q = L >> 3;
    if (t >= q * np) return;
    const int c = (int)(t >> (lgL - 3));
    const unsigned j = (unsigned)(t & (q - 1));
    const uint32_t p = c_pr[c].p, pv = c_pr[c].pinv;
    uint32_t v[8], w[7];
    tw8(w, tw + c * L, (unsigned)q, j, p, pv);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[i] = res(d[j + i * q], c);
        v[i + 4] = mmul(v[i], w[i], p, pv);
    }
    dif8w_tail(v, w, p, pv);
    uint32_t* x = a + c * L + j;
#pragma unroll
    for (int i = 0; i < 8; ++i) x[i * q] = v[i];
}
// 6 levels in one pass over memory (groups of 64q, q >= 4096): 256 threads = 32 columns j x 8 parts, two radix-8 steps
// through shared memory (64 rows x 33). DIF: levels 32q, 16q, 8q on the elements part + 8m (stride 8q, position
// j + part*q), then 4q, 2q, q on 8*part + i (stride q, position j). FIRST: the first pass (64q = L) reads the digits,
// the upper half zero.
template <bool FIRST> __global__ void __launch_bounds__(256) k_dif64(const int64_t* d, uint32_t* a, const uint32_t* tw, int lgL,
                                                                     int lgq) {
    __shared__ uint32_t s[64 * 33];
    const size_t L = (size_t)1 << lgL, q = (size_t)1 << lgq;
    const unsigned per = (unsigned)(L >> 11), cpg = (unsigned)(q >> 5);
    const unsigned c = blockIdx.x / per, rem = blockIdx.x % per;
    const unsigned col = threadIdx.x & 31, part = threadIdx.x >> 5;
    const unsigned j = (rem % cpg) * 32 + col, uq = (unsigned)q;
    const uint32_t p = c_pr[c].p, pv = c_pr[c].pinv;
    const uint32_t* T = tw + c * L;
    uint32_t* x = a + c * L + (size_t)(rem / cpg) * 64 * q + j;
    uint32_t v[8], w[7];
    if (FIRST) {
#pragma unroll
        for (int m = 0; m < 4; ++m) v[m] = res(d[(part + 8 * m) * q + j], c);
#pragma unroll
        for (int m = 4; m < 8; ++m) v[m] = 0;
    }
    else {
#pragma unroll
        for (int m = 0; m < 8; ++m) v[m] = x[(part + 8 * m) * q];
    }
    tw8(w, T, 8 * uq, j + part * uq, p, pv);
    dif8w(v, w, p, pv);
#pragma unroll
    for (int m = 0; m < 8; ++m) s[(part + 8 * m) * 33 + col] = v[m];
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = s[(8 * part + i) * 33 + col];
    tw8(w, T, uq, j, p, pv);
    dif8w(v, w, p, pv);
#pragma unroll
    for (int i = 0; i < 8; ++i) x[(8 * part + i) * q] = v[i];
}
// the inverse of k_dif64: levels q, 2q, 4q on 8*part + i, then 8q, 16q, 32q on part + 8m
__global__ void __launch_bounds__(256) k_dit64(uint32_t* a, const uint32_t* itw, int lgL, int lgq) {
    __shared__ uint32_t s[64 * 33];
    const size_t L = (size_t)1 << lgL, q = (size_t)1 << lgq;
    const unsigned per = (unsigned)(L >> 11), cpg = (unsigned)(q >> 5);
    const unsigned c = blockIdx.x / per, rem = blockIdx.x % per;
    const unsigned col = threadIdx.x & 31, part = threadIdx.x >> 5;
    const unsigned j = (rem % cpg) * 32 + col, uq = (unsigned)q;
    const uint32_t p = c_pr[c].p, pv = c_pr[c].pinv;
    const uint32_t* T = itw + c * L;
    uint32_t* x = a + c * L + (size_t)(rem / cpg) * 64 * q + j;
    uint32_t v[8], w[7];
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = x[(8 * part + i) * q];
    tw8(w, T, uq, j, p, pv);
    dit8w(v, w, p, pv);
#pragma unroll
    for (int i = 0; i < 8; ++i) s[(8 * part + i) * 33 + col] = v[i];
    __syncthreads();
#pragma unroll
    for (int m = 0; m < 8; ++m) v[m] = s[(part + 8 * m) * 33 + col];
    tw8(w, T, 8 * uq, j + part * uq, p, pv);
    dit8w(v, w, p, pv);
#pragma unroll
    for (int m = 0; m < 8; ++m) x[(part + 8 * m) * q] = v[m];
}

// bal = the remainder in [-2^(b-1), 2^(b-1)), g = the carry
__device__ __forceinline__ int64_t bal(int64_t x, int b, int64_t& g) {
    const int64_t half = (int64_t)1 << (b - 1);
    const int64_t v = ((x + half) & (((int64_t)1 << b) - 1)) - half;
    g = (x - v) >> b;
    return v;
}
__global__ void k_poke(int64_t* d) { d[1] += 16; }

// ---- steps 3-7 in two kernels
// 3: scale + CRT (Garner) -> exact signed coefficient m (the reduction t_j mod p_i by subtraction: t_j < 2^31, p_i > 2^28)
template <int NP> __device__ __forceinline__ i128 crt_at(const uint32_t* a, size_t L, size_t m, int lgL, u128 P) {
    uint32_t t[NP];
#pragma unroll
    for (int i = 0; i < NP; ++i) {
        const uint32_t p = c_pr[i].p, pv = c_pr[i].pinv;
        uint32_t u = mmul(a[i * L + m], c_sc[lgL][i], p, pv);
#pragma unroll
        for (int j = 0; j < i; ++j) {
            uint32_t x = t[j];
            while (x >= p) x -= p;
            u = mmul(msub(u, x, p), c_inv[i][j], p, pv);
        }
        t[i] = u;
    }
    u128 x = t[NP - 1];
#pragma unroll
    for (int i = NP - 2; i >= 0; --i) x = x * c_pr[i].p + t[i];
    return (x > (P >> 1)) ? -(i128)(P - x) : (i128)x;
}
// 3+4: the coefficients m-K+1 .. m of the block's 256 positions into shared memory (K-1 of them computed twice), E_m
// from there: no round trip of the int128 coefficients through memory. K <= 33.
template <int NP> __global__ void __launch_bounds__(256) k_crtE(const uint32_t* a, int64_t* E, int lgL, size_t lenE, int b,
                                                                int K, uint64_t P_lo, uint64_t P_hi) {
    __shared__ uint64_t slo[256 + 32];
    __shared__ int64_t shi[256 + 32];
    const size_t L = (size_t)1 << lgL, mb = (size_t)blockIdx.x * 256;
    const u128 P = ((u128)P_hi << 64) | P_lo;
    for (int i = threadIdx.x; i < 256 + K - 1; i += 256) {   // i <-> coefficient mb - (K-1) + i
        const long long j = (long long)mb - (K - 1) + i;
        i128 c = 0;
        if (j >= 0 && (size_t)j < L) c = crt_at<NP>(a, L, (size_t)j, lgL, P);
        slo[i] = (uint64_t)c;
        shi[i] = (int64_t)(c >> 64);
    }
    __syncthreads();
    const size_t m = mb + threadIdx.x;
    if (m >= lenE) return;
    const i128 mask = ((i128)1 << b) - 1;
    int64_t acc = 0;
    for (int t = 0; t < K; ++t) {
        const int i = threadIdx.x + (K - 1) - t;   // coefficient m - t
        const i128 c = (i128)(((u128)(uint64_t)shi[i] << 64) | slo[i]);
        i128 part = c >> (t * b);
        if (t < K - 1) part &= mask;
        acc += (int64_t)part;
    }
    E[m] = acc;
}
// The LAST inverse radix-64 pass (levels L/64 .. L/2, k_dit64 with q = L/64) + CRT + E in one kernel (when np*L leaves
// the L2, gpu_create): the residues of the np primes do not go back to memory - minus a write and a read of np*L words.
// A block = 32 columns j with the threads of k_dit64 (lane = column, warp = part), all primes in turn; a thread has the
// same 8 positions (rows part + 8m) in every prime, so Garner runs incrementally in registers. E_m needs c_m .. c_(m-H),
// H = K - 1: within the warp by shuffles (lane = column), the lanes < H from H halo columns before the group (extra warps
// of 4 columns x 8 parts; for the first group the end of the row above). The block of the last group also writes
// E_L .. E_(L+K-1). The loads of the next prime are issued before the work on the current one (in flight across its
// barriers: 2 blocks of 288 threads per SM at 86 registers hide little latency otherwise). Min. 4 blocks per SM for the
// register limit (48 registers, a small spill) was the fastest; 5 x 320 threads no longer fit an SM.
const int FUSE_HMAX = 8;   // the largest K - 1 of the fused kernel
template <int NP> __global__ void __launch_bounds__(256 + 32 * (FUSE_HMAX / 4), 4) k_dit64E(const uint32_t* a,
        int64_t* E, const uint32_t* itw, int lgL, size_t lenE, int b, int K, uint64_t P_lo, uint64_t P_hi) {
    const int SW = 32 + FUSE_HMAX + 1;   // exchange row: halo slots 0..7, the group 8..39 (odd stride)
    __shared__ uint32_t s[64 * SW];
    __shared__ uint64_t hlo[64][FUSE_HMAX];
    __shared__ int64_t hhi[64][FUSE_HMAX];
    __shared__ uint64_t tlo[32];
    __shared__ int64_t thi[32];
    const size_t L = (size_t)1 << lgL;
    const unsigned q = (unsigned)(L >> 6), H = (unsigned)K - 1, tid = threadIdx.x, j0 = blockIdx.x * 32;
    const bool grp = tid < 256;
    unsigned col, part, X, j;   // X = exchange slot, j = column in memory
    if (grp) { col = tid & 31; part = tid >> 5; X = FUSE_HMAX + col; j = j0 + col; }
    else {
        const unsigned lt = tid - 256;
        col = 4 * (lt >> 5) + (lt & 3); part = (lt & 31) >> 2;
        X = col; j = (j0 + q - H + col) & (q - 1);
    }
    const bool act = grp || col < H;
    uint32_t t[NP][8], vn[8];
    if (act) {
#pragma unroll
        for (int i = 0; i < 8; ++i) vn[i] = a[j + (size_t)(8 * part + i) * q];
    }
#pragma unroll
    for (int c = 0; c < NP; ++c) {
        const uint32_t p = c_pr[c].p, pv = c_pr[c].pinv;
        const uint32_t* T = itw + c * L;
        uint32_t v[8], w[7];
        if (act) {
#pragma unroll
            for (int i = 0; i < 8; ++i) v[i] = vn[i];
            if (c + 1 < NP) {   // the next prime in flight
                const uint32_t* x = a + (c + 1) * L + j;
#pragma unroll
                for (int i = 0; i < 8; ++i) vn[i] = x[(size_t)(8 * part + i) * q];
            }
            tw8(w, T, q, j, p, pv);
            dit8w(v, w, p, pv);
#pragma unroll
            for (int i = 0; i < 8; ++i) s[(8 * part + i) * SW + X] = v[i];
        }
        __syncthreads();
        if (act) {
#pragma unroll
            for (int m = 0; m < 8; ++m) v[m] = s[(part + 8 * m) * SW + X];
            tw8(w, T, 8 * q, j + part * q, p, pv);
            dit8w(v, w, p, pv);
#pragma unroll
            for (int m = 0; m < 8; ++m) {   // Garner as crt_at
                uint32_t u = mmul(v[m], c_sc[lgL][c], p, pv);
#pragma unroll
                for (int i = 0; i < c; ++i) {
                    uint32_t y = t[i][m];
                    while (y >= p) y -= p;
                    u = mmul(msub(u, y, p), c_inv[c][i], p, pv);
                }
                t[c][m] = u;
            }
        }
        __syncthreads();
    }
    const u128 P = ((u128)P_hi << 64) | P_lo;
    uint64_t lo[8];
    int64_t hi[8];
#pragma unroll
    for (int m = 0; m < 8; ++m) {
        u128 x = t[NP - 1][m];
#pragma unroll
        for (int i = NP - 2; i >= 0; --i) x = x * c_pr[i].p + t[i][m];
        const i128 cf = (x > (P >> 1)) ? -(i128)(P - x) : (i128)x;
        lo[m] = (uint64_t)cf;
        hi[m] = (int64_t)(cf >> 64);
    }
    if (!grp && act) {
#pragma unroll
        for (int m = 0; m < 8; ++m) {
            unsigned row = part + 8 * m;
            uint64_t l = lo[m];
            int64_t h = hi[m];
            if (j0 == 0) {   // the halo of the first group: the end of the row above (none above row 0)
                if (row == 63) { l = 0; h = 0; }
                row = (row + 1) & 63;
            }
            hlo[row][col] = l;
            hhi[row][col] = h;
        }
    }
    const bool last = (j0 + 32 == q);
    if (last && grp && part == 7) { tlo[col] = lo[7]; thi[col] = hi[7]; }   // row 63
    __syncthreads();
    if (!grp) return;
    const i128 mask = ((i128)1 << b) - 1;
#pragma unroll
    for (int m = 0; m < 8; ++m) {
        const unsigned row = part + 8 * m;
        int64_t acc = 0;
        for (int tt = 0; tt < K; ++tt) {   // coefficient of column col - tt
            uint64_t l = __shfl_up_sync(0xffffffffu, lo[m], tt);
            int64_t h = __shfl_up_sync(0xffffffffu, hi[m], tt);
            if (col < (unsigned)tt) { const unsigned hc = H + col - tt; l = hlo[row][hc]; h = hhi[row][hc]; }
            const i128 cc = (i128)(((u128)(uint64_t)h << 64) | l);
            i128 pt = cc >> (tt * b);
            if (tt < K - 1) pt &= mask;
            acc += (int64_t)pt;
        }
        E[j + (size_t)row * q] = acc;
    }
    if (last && tid < (unsigned)K) {   // E_(L+u): the parts of c_(L+u-tt) = row 63, column q-1-(tt-u-1)
        const unsigned u = tid;
        int64_t acc = 0;
        for (int tt = u + 1; tt < K; ++tt) {
            const unsigned ci = 31 - (tt - u - 1);
            const i128 cc = (i128)(((u128)(uint64_t)thi[ci] << 64) | tlo[ci]);
            i128 pt = cc >> (tt * b);
            if (tt < K - 1) pt &= mask;
            acc += (int64_t)pt;
        }
        if (L + u < lenE) E[L + u] = acc;
    }
}
// 5: S_t = H_t - k*L_t: H = S_full >> n in b-bit digits aligned at bit n (each from E_(m0+t) >> off and the low off bits
// of E_(m0+t+1)), L = S_full mod 2^n (E_t below m0, the low off bits of E_m0)
__device__ __forceinline__ int64_t S_at(const int64_t* E, size_t lenE, size_t t, size_t m0, int off, int b, int64_t k) {
    const size_t i = m0 + t;
    const int64_t e0 = (i < lenE) ? E[i] : 0;
    const int64_t e1 = (i + 1 < lenE) ? E[i + 1] : 0;
    const uint64_t mo = off ? ((1ULL << off) - 1) : 0;
    const int64_t h = (e0 >> off) + (int64_t)(((uint64_t)e1 & mo) << (b - off));
    int64_t l = 0;
    if (t < m0) l = E[t];
    else if (t == m0) l = (int64_t)((uint64_t)E[m0] & mo);
    return h - k * l;
}
// 6: q = round(S / N) from the digits around bit n, S -= q*N = q*k*2^n + q: the window of S (<= 44 digits, loaded in
// parallel), into qc: q (at digit 0), the corrections at m0, m0+1, m0+2. One block of 64 threads: the serial double Horner
// (slow FP64 on GeForce) runs once, not in every block. Horner from the TOP digit down: the top digits of S cancel (2^b,
// 2^b - 1, -1 from the signed top parts of E), summed from the bottom the small significant part is lost next to them
// (15.5M: S/N came out 0). The window: qlo digits below m0 (the part below is < 0.05 in units of 2^n, Plan) and 24 above.
__global__ void k_qc(const int64_t* E, int64_t* qc, size_t L, size_t lenE, size_t m0, int off, int b, int64_t k, int qlo) {
    __shared__ int64_t sw[64];
    const unsigned tid = threadIdx.x;
    const size_t t0 = (m0 >= (size_t)qlo) ? m0 - qlo : 0, t1 = (L < m0 + 24) ? L : m0 + 24;
    const unsigned nw = (unsigned)(t1 - t0);
    if (tid < nw) sw[tid] = S_at(E, lenE, t0 + tid, m0, off, b, k);
    __syncthreads();
    if (tid != 0) return;
    const double B = ldexp(1.0, b);
    double acc = 0.0;
    for (unsigned i = nw; i-- > 0;) acc = acc * B + (double)sw[i];
    const double x = ldexp(acc, (int)((long long)t0 - (long long)m0) * b - off);
    const int64_t q = llround(x / (double)k);
    i128 v = (i128)q * k * ((i128)1 << off);   // q*k*2^off at digit m0 (= q*k*2^n)
    const i128 mask = ((i128)1 << b) - 1;
    for (int u = 0; u < 2; ++u) {
        const i128 lo = v & mask;
        qc[1 + u] = (int64_t)lo;
        v = (v - lo) >> b;
    }
    qc[3] = (int64_t)v;
    qc[0] = q;
}
// 5+7: S of the block's 256 digits and the npass digits below them with the q correction, the npass neighbour passes
// (out_t = bal(in_t) + carry(in_(t-1))) in shared memory. npass <= 16.
__global__ void __launch_bounds__(256) k_Sqn(const int64_t* E, const int64_t* qc, int64_t* out, size_t L, size_t lenE, size_t m0,
                                             int off, int b, int64_t k, int npass) {
    __shared__ int64_t sv[256 + 16];    // digit base - npass + i
    const unsigned tid = threadIdx.x;
    const size_t base = (size_t)blockIdx.x * 256;
    const unsigned H = (unsigned)npass, n = 256 + H;
    for (unsigned i = tid; i < n; i += 256) {
        const long long x = (long long)base - (long long)H + i;
        int64_t s = 0;
        if (x >= 0 && (size_t)x < L) {
            s = S_at(E, lenE, (size_t)x, m0, off, b, k);
            if (x == 0) s -= qc[0];
            if ((size_t)x >= m0 && (size_t)x < m0 + 3) s -= qc[1 + ((size_t)x - m0)];
        }
        sv[i] = s;
    }
    __syncthreads();
    for (unsigned ps = 1; ps <= H; ++ps) {   // after pass ps the digits i >= ps are right
        int64_t nv[2] = { 0, 0 };
        for (unsigned r = 0; r < 2; ++r) {
            const unsigned i = tid + 256 * r;
            if (i >= ps && i < n) {
                int64_t g, gp;
                nv[r] = bal(sv[i], b, g);
                bal(sv[i - 1], b, gp);
                nv[r] += gp;
            }
        }
        __syncthreads();
        for (unsigned r = 0; r < 2; ++r) {
            const unsigned i = tid + 256 * r;
            if (i >= ps && i < n) sv[i] = nv[r];
        }
        __syncthreads();
    }
    if (base + tid < L) out[base + tid] = sv[H + tid];
}

// ---- host
uint64_t pw(uint64_t a, uint64_t e, uint64_t m) {
    uint64_t r = 1 % m;
    a %= m;
    while (e) { if (e & 1) r = (u128)r * a % m; a = (u128)a * a % m; e >>= 1; }
    return r;
}
unsigned grid(size_t threads) { return (unsigned)((threads + 255) / 256); }

void mpz_set_i128(mpz_t r, i128 v) {
    const u128 a = v < 0 ? (u128)(-v) : (u128)v;
    mpz_set_ui(r, (unsigned long)(uint64_t)(a >> 64));
    mpz_mul_2exp(r, r, 64);
    mpz_add_ui(r, r, (unsigned long)(uint64_t)a);
    if (v < 0) mpz_neg(r, r);
}
// r = sum v[m] * 2^(m*b): one host carry chain, then mpz_import - O(len)
void digits_to_mpz(mpz_t r, const int64_t* v, size_t len, int b) {
    std::vector<uint64_t> w(len * (size_t)b / 64 + 2, 0);
    const i128 mask = ((i128)1 << b) - 1;
    i128 carry = 0;
    for (size_t m = 0; m < len; ++m) {
        const i128 s = (i128)v[m] + carry;
        const uint64_t dg = (uint64_t)(s & mask);
        carry = s >> b;
        const size_t bit = m * (size_t)b, i = bit >> 6;
        const int sh = (int)(bit & 63);
        w[i] |= dg << sh;
        if (sh + b > 64) w[i + 1] |= dg >> (64 - sh);
    }
    mpz_import(r, w.size(), -1, sizeof(uint64_t), 0, 0, w.data());
    if (carry != 0) {
        mpz_t c;
        mpz_init(c);
        mpz_set_i128(c, carry);
        mpz_mul_2exp(c, c, len * (size_t)b);
        mpz_add(r, r, c);
        mpz_clear(c);
    }
}
// 0 <= v -> balanced b-bit digits d[0 .. ndig) (the rest of d[0 .. len) zero); false = v does not fit
bool mpz_to_digits(int64_t* d, size_t len, const mpz_t v, int b, size_t ndig) {
    std::vector<uint64_t> w((mpz_sizeinbase(v, 2) + 63) / 64 + 2, 0);
    size_t cnt = 0;
    mpz_export(w.data(), &cnt, -1, sizeof(uint64_t), 0, 0, v);
    std::fill(d, d + len, 0);
    const int64_t half = (int64_t)1 << (b - 1);
    const uint64_t mask = ((uint64_t)1 << b) - 1;
    int64_t carry = 0;
    for (size_t t = 0; t < ndig; ++t) {
        const size_t bit = t * (size_t)b, i = bit >> 6;
        const int sh = (int)(bit & 63);
        uint64_t x = (i < w.size()) ? (w[i] >> sh) : 0;
        if (sh + b > 64 && i + 1 < w.size()) x |= w[i + 1] << (64 - sh);
        int64_t f = (int64_t)(x & mask) + carry;
        carry = 0;
        if (f >= half) { f -= (int64_t)1 << b; carry = 1; }
        d[t] = f;
    }
    return carry == 0 && mpz_sizeinbase(v, 2) <= ndig * (size_t)b;
}

bool g_const_done[64] = {};   // constants per device (the primes do not depend on n, k or L)
std::mutex g_const_mutex;     // engines of several card workers are created from their own threads

void init_constants(int device) {
    std::lock_guard<std::mutex> lk(g_const_mutex);
    if (device < 0 || device >= 64 || g_const_done[device]) return;
    Pr pr[4];
    uint32_t inv[4][4] = {}, sc[25][4] = {}, r23[4][2] = {};
    for (int i = 0; i < 4; ++i) {
        const uint64_t p = PRIMES[i], R = (1ull << 32) % p;
        uint32_t x = 1;
        for (int it = 0; it < 5; ++it) x *= 2 - (uint32_t)p * x;
        pr[i] = { (uint32_t)p, (uint32_t)(0u - x) };
        r23[i][0] = (uint32_t)(R * R % p);
        r23[i][1] = (uint32_t)(R * R % p * R % p);
        for (int j = 0; j < i; ++j) inv[i][j] = (uint32_t)(pw(PRIMES[j] % p, p - 2, p) * R % p);
        for (int lg = LG_MIN; lg <= LG_MAX; ++lg)
            sc[lg][i] = (uint32_t)((u128)R * R % p * pw(((uint64_t)1 << lg) % p, p - 2, p) % p);
    }
    CK(cudaMemcpyToSymbol(c_pr, pr, sizeof(pr)));
    CK(cudaMemcpyToSymbol(c_inv, inv, sizeof(inv)));
    CK(cudaMemcpyToSymbol(c_sc, sc, sizeof(sc)));
    CK(cudaMemcpyToSymbol(c_r23, r23, sizeof(r23)));
    g_const_done[device] = true;
}

struct Plan { int lgL = 0, np = 0, b = 0, K = 0, npass = 0, qlo = 6; size_t L = 0, m0 = 0; int off = 0; };

// Smallest np * L: digits of b bits over L/2 - 16 positions hold x < N (and the slack above for H), the worst case of a
// coefficient |c| <= L/2 * (2^(b-1) + 2)^2 < P/2 (exact form), the products k*E within int64.
bool make_plan(unsigned n, uint64_t k, Plan& pl) {
    const int d = 64 - __builtin_clzll(k);
    double best = 1e300;
    for (int lg = LG_MIN; lg <= LG_MAX; ++lg) {
        const size_t L = (size_t)1 << lg, half = L >> 1;
        const int b = (int)(((size_t)n + 3 * (size_t)d + (half - 17)) / (half - 16));
        // b >= 4 (small n: 2^12 x 4-bit digits from n ~ 8000); the q window of k_q (24 digits above bit n) must hold S
        if (b < 4 || b > 40 || d + b + 8 > 62 || 24 * b < 2 * d + 16) continue;
        double sum = 0;
        for (int np = 1; np <= 4; ++np) {
            sum += std::log2((double)PRIMES[np - 1]);
            const double cbits = (lg - 1) + 2.0 * std::log2(std::ldexp(1.0, b - 1) + 2.0);   // log2 of the |c| bound
            if (cbits < sum - 1.0 - 0.01) {
                const double cost = (double)np * L;
                if (cost < best) {
                    best = cost;
                    pl.lgL = lg; pl.np = np; pl.b = b; pl.L = L;
                    pl.K = ((int)std::floor(sum) + 2 + b - 1) / b;
                    pl.m0 = n / b; pl.off = (int)(n % b);
                }
                break;
            }
        }
    }
    if (best >= 1e300) return false;
    // digits of S below the q window: |S_t| < 2^(d+b+4) each, so the part below qlo digits under bit n is < 2^(d+b+4-qlo*b)
    // in units of 2^n: qlo*b >= d + b + 9 keeps it < 0.05 (|error of q| < 0.05/k); 6 for the tested 26..32-bit digits
    pl.qlo = std::max(6, (d + 9) / pl.b + 2);
    // passes until the carry is at most 1: |S_t| <= (k*K + K + 3) * 2^b, one pass: bound -> 2^(b-1) + carry
    const double B = std::ldexp(1.0, pl.b), half = B / 2;
    double bound = ((double)k * pl.K + pl.K + 3) * B;
    for (;;) {
        const double g = std::floor((bound + half) / B);
        bound = half + g;
        ++pl.npass;
        if (g <= 1) break;
    }
    return pl.K - 1 <= 32 && pl.npass <= 16 && pl.qlo + 24 <= 64;   // k_crtE / k_Sqn / k_qc (not reached: b >= 4)
}

}  // namespace

struct HybEngine;   // the hybrid engine (gpu_hybrid.cuh, default; LOTR_GPU_HYB=0 = classic)

struct GpuEngine {
    HybEngine* hy = nullptr;                 // set: the hybrid engine does the work (only the stream, slots, host, N here)
    unsigned n = 0;
    uint64_t k = 0;
    int device = 0;
    Plan pl;
    size_t L = 0, lenE = 0, top_max = 0;
    u128 P = 1;
    bool graphs = true;
    bool fuse = false;                       // the last inverse radix-64 pass + CRT + E in k_dit64E
    cudaStream_t st = nullptr;
    cudaEvent_t ev = nullptr;
    uint32_t *a = nullptr, *b2 = nullptr, *tw = nullptr, *itw = nullptr;
    int64_t *E = nullptr, *qc = nullptr;
    int64_t* slot[GPU_SLOTS] = {};
    cudaGraphExec_t gexec[GPU_SLOTS] = {};   // GRAPH_SQ squarings of the slot (captured at the first use)
    int64_t* host = nullptr;                 // pinned staging buffer, L digits
    mpz_t N;
    // LOTR_GPU_PROFILE=1: events between the stages of every squaring (no graphs, a sync per squaring), a line of
    // averages every 10000 squarings
    bool prof = false;
    cudaEvent_t pev[10] = {};
    double pt[9] = {};
    unsigned long long pcount = 0;
};

#include "gpu_hybrid.cuh"

namespace {

void mark(GpuEngine* g, int i) { if (g->prof) CK(cudaEventRecord(g->pev[i], g->st)); }
// digits d -> residues in a, forward levels h = L/2 .. 4096: radix-64 passes (6 levels), radix-8, radix-2; the first pass
// from 2^15 packs itself (the profile "pack" is then 0, "fwd" holds it)
void fwd_outer(GpuEngine* g, const int64_t* d, uint32_t* a) {
    const int lgL = g->pl.lgL, np = g->pl.np;
    const size_t L = g->L;
    const unsigned g64 = (unsigned)((L >> 11) * np);
    int lgh = lgL - 1;
    mark(g, 1);
    if (lgh - 5 >= 12) {
        k_dif64<true><<<g64, 256, 0, g->st>>>(d, a, g->tw, lgL, lgh - 5);
        lgh -= 6;
    }
    else if (lgh - 2 >= 12) {
        k_dif8_first<<<grid((L >> 3) * np), 256, 0, g->st>>>(d, a, g->tw, lgL, np);
        lgh -= 3;
    }
    else
        k_pack<<<grid(L * np), 256, 0, g->st>>>(d, a, lgL, np);
    while (lgh >= 12) {
        if (lgh - 5 >= 12) { k_dif64<false><<<g64, 256, 0, g->st>>>(nullptr, a, g->tw, lgL, lgh - 5); lgh -= 6; }
        else if (lgh - 2 >= 12) { k_dif8<<<grid((L >> 3) * np), 256, 0, g->st>>>(a, g->tw, lgL, lgh - 2, np); lgh -= 3; }
        else { k_dif2<<<grid((L >> 1) * np), 256, 0, g->st>>>(a, g->tw, lgL, lgh, np); lgh -= 1; }
    }
}
void inv_outer(GpuEngine* g, uint32_t* a) {   // inverse levels h = 4096 .. L/2
    const int lgL = g->pl.lgL, np = g->pl.np;
    const size_t L = g->L;
    // fuse: the top 6 levels L/64 .. L/2 go to k_dit64E in finish (the levels are independent stages, any grouping in
    // increasing order is the same transform)
    const int top = g->fuse ? lgL - 7 : lgL - 1;
    int lgh = 12;
    while (lgh <= top) {
        if (lgh + 5 <= top) {
            k_dit64<<<(unsigned)((L >> 11) * np), 256, 0, g->st>>>(a, g->itw, lgL, lgh);
            lgh += 6;
        }
        else if (lgh + 2 <= top) { k_dit8<<<grid((L >> 3) * np), 256, 0, g->st>>>(a, g->itw, lgL, lgh, np); lgh += 3; }
        else { k_dit2<<<grid((L >> 1) * np), 256, 0, g->st>>>(a, g->itw, lgL, lgh, np); lgh += 1; }
    }
}
void dit64E(GpuEngine* g) {   // the fused last inverse pass + CRT + E (g->fuse)
    const Plan& pl = g->pl;
    const uint64_t plo = (uint64_t)g->P, phi = (uint64_t)(g->P >> 64);
    const unsigned gr = (unsigned)(g->L >> 11), th = 256 + 32 * ((pl.K - 1 + 3) / 4);
    switch (pl.np) {
    case 1: k_dit64E<1><<<gr, th, 0, g->st>>>(g->a, g->E, g->itw, pl.lgL, g->lenE, pl.b, pl.K, plo, phi); break;
    case 2: k_dit64E<2><<<gr, th, 0, g->st>>>(g->a, g->E, g->itw, pl.lgL, g->lenE, pl.b, pl.K, plo, phi); break;
    case 3: k_dit64E<3><<<gr, th, 0, g->st>>>(g->a, g->E, g->itw, pl.lgL, g->lenE, pl.b, pl.K, plo, phi); break;
    default: k_dit64E<4><<<gr, th, 0, g->st>>>(g->a, g->E, g->itw, pl.lgL, g->lenE, pl.b, pl.K, plo, phi); break;
    }
}
void crtE(GpuEngine* g) {
    const Plan& pl = g->pl;
    const uint64_t plo = (uint64_t)g->P, phi = (uint64_t)(g->P >> 64);
    const unsigned gr = grid(g->lenE);
    switch (pl.np) {
    case 1: k_crtE<1><<<gr, 256, 0, g->st>>>(g->a, g->E, pl.lgL, g->lenE, pl.b, pl.K, plo, phi); break;
    case 2: k_crtE<2><<<gr, 256, 0, g->st>>>(g->a, g->E, pl.lgL, g->lenE, pl.b, pl.K, plo, phi); break;
    case 3: k_crtE<3><<<gr, 256, 0, g->st>>>(g->a, g->E, pl.lgL, g->lenE, pl.b, pl.K, plo, phi); break;
    default: k_crtE<4><<<gr, 256, 0, g->st>>>(g->a, g->E, pl.lgL, g->lenE, pl.b, pl.K, plo, phi); break;
    }
}
template <int MODE> void block(GpuEngine* g, uint32_t* a) {
    k_block<MODE><<<(unsigned)((g->L >> 12) * g->pl.np), 512, 0, g->st>>>(a, g->tw, g->itw, g->pl.lgL);
}
// the transform in g->a (already inverse) -> -k*S mod N, nearly balanced, in out
// (profile: "crtE" = crt+E, with fuse the last inverse pass + CRT + E; "q" = k_qc, "S+norm" = k_Sqn)
void finish(GpuEngine* g, int64_t* out) {
    const Plan& pl = g->pl;
    const size_t L = g->L;
    if (g->fuse) dit64E(g);
    else crtE(g);
    mark(g, 5);
    mark(g, 6);
    mark(g, 7);
    k_qc<<<1, 64, 0, g->st>>>(g->E, g->qc, L, g->lenE, pl.m0, pl.off, pl.b, (int64_t)g->k, pl.qlo);
    mark(g, 8);
    // only the lower half: the digits >= L/2 are zero (top_max < L/2), k_dif64<true> does not read them, and the slot
    // keeps the zeros of gpu_set / cudaMemset (2^24 x 4: 64 MB less writing per iteration)
    k_Sqn<<<grid(L / 2), 256, 0, g->st>>>(g->E, g->qc, out, L, g->lenE, pl.m0, pl.off, pl.b, (int64_t)g->k, pl.npass);
    mark(g, 9);
}
void prof_step(GpuEngine* g) {
    static const char* const cnames[9] = { "pack", "fwd", "block", "inv", "crtE", "-", "-", "q", "S+norm" };
    static const char* const hnames[9] = { "pack", "fwd", "block", "inv", "crtE+norm", "-", "-", "-", "-" };
    const char* const* names = g->hy ? hnames : cnames;
    CK(cudaEventSynchronize(g->pev[9]));
    for (int i = 0; i < 9; ++i) {
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, g->pev[i], g->pev[i + 1]));
        g->pt[i] += ms;
    }
    if (++g->pcount % 10000) return;
    char buf[512];
    int len = std::snprintf(buf, sizeof(buf), "[GPU PROFILE] us/sq:");
    double sum = 0;
    for (int i = 0; i < 9; ++i) {
        const double us = g->pt[i] * 1000.0 / 10000;
        sum += us;
        len += std::snprintf(buf + len, sizeof(buf) - len, " %s %.2f", names[i], us);
        g->pt[i] = 0;
    }
    std::snprintf(buf + len, sizeof(buf) - len, " | sum %.2f (%llu sq)", sum, g->pcount);
    std::printf("\n%s\n", buf);
    std::fflush(stdout);
}
void square_once(GpuEngine* g, int slot) {
    if (g->hy) { hyb_square_once(g, slot); return; }
    mark(g, 0);
    fwd_outer(g, g->slot[slot], g->a);
    mark(g, 2);
    block<0>(g, g->a);
    mark(g, 3);
    inv_outer(g, g->a);
    mark(g, 4);
    finish(g, g->slot[slot]);
    if (g->prof) prof_step(g);
}
void release(GpuEngine* g) {   // no throwing: also the cleanup after an error
    cudaSetDevice(g->device);
    if (g->st) cudaStreamSynchronize(g->st);
    if (g->hy) hyb_free(g);
    for (int s = 0; s < GPU_SLOTS; ++s) {
        if (g->gexec[s]) cudaGraphExecDestroy(g->gexec[s]);
        if (g->slot[s]) cudaFree(g->slot[s]);
    }
    for (void* p : { (void*)g->a, (void*)g->b2, (void*)g->tw, (void*)g->itw, (void*)g->E, (void*)g->qc })
        if (p) cudaFree(p);
    if (g->host) cudaFreeHost(g->host);
    for (cudaEvent_t e : g->pev)
        if (e) cudaEventDestroy(e);
    if (g->ev) cudaEventDestroy(g->ev);
    if (g->st) cudaStreamDestroy(g->st);
    mpz_clear(g->N);
    delete g;
}

// LOTR_GPU_COLBENCH=1: the kernels of the classic square timed alone against a coalesced copy of the same bytes (the
// bandwidth ceiling). The contents of a / b2 are garbage, timing only; GB/s = a read + write of a.
__global__ void k_copy(const uint4* src, uint4* dst, size_t n) {
    for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < n; t += (size_t)gridDim.x * blockDim.x) dst[t] = src[t];
}
template <typename F> double bench_us(GpuEngine* g, F launch) {   // us per launch, 10 launches after 2 warm-up ones
    for (int i = 0; i < 2; ++i) launch();
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0, g->st));
    for (int i = 0; i < 10; ++i) launch();
    CK(cudaEventRecord(e1, g->st));
    CK(cudaEventSynchronize(e1));
    CK(cudaGetLastError());
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return ms * 100.0;
}
void bench_line(const GpuEngine* g, const char* name, double us, const char* extra = "") {
    const double bytes = 8.0 * (double)g->L * g->pl.np;
    std::printf("    %-30s %9.1f us %7.1f GB/s%s\n", name, us, bytes / (us * 1e3), extra);
}
template <typename K> void kern_extra(char* buf, size_t len, K kern, int threads) {   // registers, spill, blocks per SM
    cudaFuncAttributes fa;
    int nb = 0;
    CK(cudaFuncGetAttributes(&fa, kern));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kern, threads, 0));
    std::snprintf(buf, len, "  (%d regs, %zu B spill, %d blocks/SM)", fa.numRegs, (size_t)fa.localSizeBytes, nb);
}
void col_bench(GpuEngine* g) {
    const int lgL = g->pl.lgL, np = g->pl.np;
    const size_t L = g->L;
    char extra[80];
    std::printf("[GPU COLBENCH] NTT 2^%d x %d primes: us per pass\n", lgL, np);
    bench_line(g, "copy (coalesced)", bench_us(g, [&] {
        k_copy<<<1024, 256, 0, g->st>>>((const uint4*)g->a, (uint4*)g->b2, L * np / 4); }));
    kern_extra(extra, sizeof(extra), k_block<0>, 512);
    bench_line(g, "k_block<0>", bench_us(g, [&] { block<0>(g, g->a); }), extra);
    const unsigned g64 = (unsigned)((L >> 11) * np);
    if (lgL - 6 >= 12)
        bench_line(g, "k_dif64<true> (first)", bench_us(g, [&] {
            k_dif64<true><<<g64, 256, 0, g->st>>>(g->slot[0], g->a, g->tw, lgL, lgL - 6); }));
    if (lgL >= 18) {
        bench_line(g, "k_dif64 q=4096", bench_us(g, [&] {
            k_dif64<false><<<g64, 256, 0, g->st>>>(nullptr, g->a, g->tw, lgL, 12); }));
        bench_line(g, "k_dit64 q=4096", bench_us(g, [&] {
            k_dit64<<<g64, 256, 0, g->st>>>(g->a, g->itw, lgL, 12); }));
        bench_line(g, "k_dit64 q=L/64 (last)", bench_us(g, [&] {
            k_dit64<<<g64, 256, 0, g->st>>>(g->a, g->itw, lgL, lgL - 6); }));
        bench_line(g, "k_crtE", bench_us(g, [&] { crtE(g); }));
        if (g->pl.K >= 1 && g->pl.K - 1 <= FUSE_HMAX) {
            const int th = 256 + 32 * ((g->pl.K - 1 + 3) / 4);
            switch (np) {
            case 1: kern_extra(extra, sizeof(extra), k_dit64E<1>, th); break;
            case 2: kern_extra(extra, sizeof(extra), k_dit64E<2>, th); break;
            case 3: kern_extra(extra, sizeof(extra), k_dit64E<3>, th); break;
            default: kern_extra(extra, sizeof(extra), k_dit64E<4>, th); break;
            }
            bench_line(g, "k_dit64E (last+CRT+E)", bench_us(g, [&] { dit64E(g); }), extra);
        }
    }
    std::fflush(stdout);
}

}  // namespace

bool gpu_probe(int device, std::string& info) {
    int cnt = 0;
    if (cudaGetDeviceCount(&cnt) != cudaSuccess || cnt <= device || device < 0) {
        info = "no CUDA device " + std::to_string(device);
        return false;
    }
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) { info = "CUDA device " + std::to_string(device) + " unusable"; return false; }
    info = prop.name;
    return true;
}

// the engine for n, k (gpu_create and gpu_streams): the hybrid one from 2^14 complex, below it only when the classic one
// has no plan (measured: the hybrid engine is slower than the classic NTT at 2^12-2^13 complex, faster from 2^14).
// LOTR_GPU_HYB=0: always classic, =1: always hybrid (when it has a plan)
bool hyb_pick(unsigned n, uint64_t k, HybEngine& h) {
    const char* hyb = std::getenv("LOTR_GPU_HYB");
    if ((hyb && *hyb == '0') || !hyb_plan(n, k, h)) return false;
    if (h.lgM >= 14 || (hyb && *hyb == '1')) return true;
    Plan pl;
    return !make_plan(n, k, pl);
}

// how many times more candidates this card holds at once than the RTX 5060 Ti the gpu_streams rules were measured on
// (36 SMs, 32 MB L2): the smaller of the SM and L2 ratios, at least 1 (e.g. RTX 5070 Ti x1.5, RTX 4090 x2.25)
static double gpu_scale() {
    static const double s = [] {
        int dev = 0, sm = 0, l2 = 0;
        if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess
            || cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev) != cudaSuccess || sm <= 0 || l2 <= 0) return 1.0;
        return std::max(1.0, std::min(sm / 36.0, l2 / (double)(32 << 20)));
    }();
    return s;
}

int gpu_streams(unsigned n, uint64_t k) {
    if (const char* e = std::getenv("LOTR_GPU_STREAMS")) {
        const int v = std::atoi(e);
        if (v >= 1 && v <= 16) return v;
    }
    // the rules below x gpu_scale (5060 Ti 3 -> 5070 Ti 5, 4090 7); 1 stays 1 (that working set already leaves the L2)
    auto scaled = [](int base) { return base > 1 ? std::min(16, (int)std::lround(base * gpu_scale())) : 1; };
    // the hybrid engine (as gpu_create): its own plan; measured results per hour at 2^16 x 3: best at 3 at once (four
    // working sets miss the L2), 2^16 x 4: 3, 2^17 x 3: 2, 2^17 x 4 and up: 1
    HybEngine hp;
    if (hyb_pick(n, k, hp)) {
        const size_t v = (size_t)hp.np * hp.M;
        return scaled(v <= ((size_t)1 << 18) ? 3 : v <= ((size_t)3 << 17) ? 2 : 1);
    }
    Plan pl;
    if (!make_plan(n, k, pl)) return 1;
    return scaled((size_t)pl.np * pl.L <= ((size_t)1 << 20) ? 2 : 1);   // the size of the working set, not L alone
}

static GpuEngine* cls_create(unsigned n, uint64_t k, int device, std::string& info, const Plan& pl);

GpuEngine* gpu_create(unsigned n, uint64_t k, int device, std::string& info) {
    // the hybrid engine (gpu_hybrid.cuh) by default when it has a plan for this n, k (from 2^14 complex: hyb_pick); else
    // the classic one below. LOTR_GPU_HYB=0 forces the classic engine.
    HybEngine hp;
    if (hyb_pick(n, k, hp)) {
        if (GpuEngine* g = hyb_create(n, k, device, info)) return g;
    }
    Plan pl;
    if (!make_plan(n, k, pl)) { info = "no NTT plan for this n, k"; return nullptr; }
    // large k (no automatic hybrid plan): the classic engine against WIDE 7, once per plan shape (cls_w7_cand, hyb_w7_duel;
    // a WIDE 7 that won is taken by hyb_pick above for the next candidates)
    HybEngine w7;
    HybShape key;
    if (cls_w7_cand(n, k, pl.lgL, pl.np, w7, key)) {
        std::lock_guard<std::mutex> lk(g_w7_bench_mtx);
        const int kept = hyb_w7_get(key);
        if (kept == 1) return hyb_create_p(n, k, device, info, w7);
        if (kept < 0) {
            const auto t_start = std::chrono::steady_clock::now();
            std::string info_a;
            GpuEngine* ga = cls_create(n, k, device, info_a, pl);
            char la[64];
            std::snprintf(la, sizeof(la), "classic NTT 2^%d x %d (%d-bit)", pl.lgL, pl.np, pl.b);
            return hyb_w7_duel(ga, info_a, la, n, k, device, info, w7, key, t_start);
        }
    }
    return cls_create(n, k, device, info, pl);
}

// the classic engine (no hybrid plan, LOTR_GPU_HYB=0, or the loser of the WIDE 7 bench)
static GpuEngine* cls_create(unsigned n, uint64_t k, int device, std::string& info, const Plan& pl) {
    CK(cudaSetDevice(device));
    init_constants(device);
    GpuEngine* g = new GpuEngine;
    g->n = n; g->k = k; g->device = device; g->pl = pl;
    g->L = pl.L;
    g->lenE = pl.L + pl.K;
    g->top_max = ((size_t)n + 64 - __builtin_clzll(k)) / pl.b + 1;   // nearly balanced: |x| >= 2^(top*b - 2)
    g->prof = (std::getenv("LOTR_GPU_PROFILE") != nullptr);
    g->graphs = (std::getenv("LOTR_GPU_NOGRAPH") == nullptr) && !g->prof;
    // the fused last inverse pass + CRT + E (q = L/64 >= 4096: from 2^18, K - 1 <= FUSE_HMAX), only when the np*L words
    // leave the 32 MB L2 (np*L >= 2^23): it helps at 2^24 x 4 (-9%) but costs time while the data still fits the L2
    g->fuse = pl.lgL >= 18 && pl.K >= 1 && pl.K - 1 <= FUSE_HMAX && (size_t)pl.np * pl.L >= ((size_t)1 << 23);
    for (int i = 0; i < pl.np; ++i) g->P *= PRIMES[i];
    mpz_init(g->N);
    mpz_set_ui(g->N, (unsigned long)k);
    mpz_mul_2exp(g->N, g->N, n);
    mpz_add_ui(g->N, g->N, 1);
    try {
        const size_t L = g->L;
        const int np = pl.np;
        std::vector<uint32_t> tw(np * L), itw(np * L);
        for (int i = 0; i < np; ++i) {
            const uint64_t p = PRIMES[i], R = (1ull << 32) % p;
            uint64_t z = 2;
            while (pw(z, (p - 1) / 2, p) != p - 1) ++z;
            const uint64_t om = pw(z, (p - 1) / L, p), iom = pw(om, p - 2, p);
            tw[i * L] = itw[i * L] = (uint32_t)R;
            for (size_t h = 1; h < L; h <<= 1) {
                const uint64_t bb = pw(om, L / (2 * h), p), ib = pw(iom, L / (2 * h), p);
                uint64_t u = 1, v = 1;
                for (size_t j = 0; j < h; ++j) {
                    tw[i * L + h + j] = (uint32_t)(u * R % p);
                    itw[i * L + h + j] = (uint32_t)(v * R % p);
                    u = u * bb % p; v = v * ib % p;
                }
            }
        }
        CK(cudaStreamCreateWithFlags(&g->st, cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&g->ev, cudaEventBlockingSync | cudaEventDisableTiming));
        if (g->prof)
            for (cudaEvent_t& e : g->pev) CK(cudaEventCreateWithFlags(&e, cudaEventBlockingSync));
        CK(cudaMallocHost(&g->host, sizeof(int64_t) * L));
        CK(cudaMalloc(&g->a, sizeof(uint32_t) * L * np));
        CK(cudaMalloc(&g->b2, sizeof(uint32_t) * L * np));
        CK(cudaMalloc(&g->tw, sizeof(uint32_t) * L * np));
        CK(cudaMalloc(&g->itw, sizeof(uint32_t) * L * np));
        CK(cudaMalloc(&g->E, sizeof(int64_t) * g->lenE));
        CK(cudaMalloc(&g->qc, sizeof(int64_t) * 4));
        for (int s = 0; s < GPU_SLOTS; ++s) {
            CK(cudaMalloc(&g->slot[s], sizeof(int64_t) * L));
            CK(cudaMemsetAsync(g->slot[s], 0, sizeof(int64_t) * L, g->st));
        }
        CK(cudaMemcpyAsync(g->tw, tw.data(), sizeof(uint32_t) * L * np, cudaMemcpyHostToDevice, g->st));
        CK(cudaMemcpyAsync(g->itw, itw.data(), sizeof(uint32_t) * L * np, cudaMemcpyHostToDevice, g->st));
        gpu_sync(g);
        if (std::getenv("LOTR_GPU_COLBENCH")) col_bench(g);
    }
    catch (...) {
        release(g);
        throw;
    }
    char buf[256];
    std::snprintf(buf, sizeof(buf), "NTT 2^%d x %d primes, %d-bit digits, %d carry passes%s%s", pl.lgL, pl.np, pl.b, pl.npass,
                  g->fuse ? " (inv+CRT fused)" : "", g->prof ? " (profile, no CUDA Graphs)" : g->graphs ? "" : " (no CUDA Graphs)");
    info = buf;
    return g;
}

void gpu_destroy(GpuEngine* g) {
    if (g) release(g);
}

void gpu_sync(GpuEngine* g) {
    CK(cudaEventRecord(g->ev, g->st));
    CK(cudaEventSynchronize(g->ev));
    CK(cudaGetLastError());
}

void gpu_set(GpuEngine* g, int slot, const mpz_t v) {
    if (g->hy) { hyb_set(g, slot, v); return; }
    gpu_sync(g);   // the staging buffer may still be in use
    if (!mpz_to_digits(g->host, g->L, v, g->pl.b, g->L / 2)) throw std::runtime_error("gpu_set: the value does not fit the digits");
    CK(cudaMemcpyAsync(g->slot[slot], g->host, sizeof(int64_t) * g->L, cudaMemcpyHostToDevice, g->st));
    gpu_sync(g);
}

bool gpu_get(GpuEngine* g, int slot, mpz_t v) {
    if (g->hy) return hyb_get(g, slot, v);
    CK(cudaMemcpyAsync(g->host, g->slot[slot], sizeof(int64_t) * g->L, cudaMemcpyDeviceToHost, g->st));
    gpu_sync(g);
    const int64_t lim = ((int64_t)1 << (g->pl.b - 1)) + 1;
    bool ok = true;
    for (size_t t = 0; t < g->L; ++t) {
        const int64_t x = g->host[t];
        if (x > lim || x < -lim || (x != 0 && t > g->top_max)) ok = false;
    }
    digits_to_mpz(v, g->host, g->L, g->pl.b);
    mpz_mod(v, v, g->N);
    return ok;
}

void gpu_copy(GpuEngine* g, int dst, int src) {
    const size_t db = (g->hy && g->hy->wide) ? 16 : sizeof(int64_t);   // (the wide digits of the hybrid engine: 16 B)
    if (dst != src) CK(cudaMemcpyAsync(g->slot[dst], g->slot[src], db * g->L, cudaMemcpyDeviceToDevice, g->st));
}

void gpu_square(GpuEngine* g, int slot, unsigned count) {
    if (g->graphs) {
        while (count >= GRAPH_SQ) {
            if (!g->gexec[slot]) {
                cudaGraph_t graph;
                CK(cudaStreamBeginCapture(g->st, cudaStreamCaptureModeThreadLocal));
                for (unsigned i = 0; i < GRAPH_SQ; ++i) square_once(g, slot);
                CK(cudaStreamEndCapture(g->st, &graph));
                hyb_graph_l2(g, graph);
                const cudaError_t e = cudaGraphInstantiate(&g->gexec[slot], graph, 0);
                cudaGraphDestroy(graph);
                CK(e);
            }
            CK(cudaGraphLaunch(g->gexec[slot], g->st));
            count -= GRAPH_SQ;
        }
    }
    for (unsigned i = 0; i < count; ++i) square_once(g, slot);
}

void gpu_mul(GpuEngine* g, int dst, int x, int y) {
    if (g->hy) { hyb_mul(g, dst, x, y); return; }
    if (x == y) {
        fwd_outer(g, g->slot[x], g->a);
        block<0>(g, g->a);
    }
    else {
        fwd_outer(g, g->slot[x], g->a);
        block<1>(g, g->a);
        fwd_outer(g, g->slot[y], g->b2);
        block<1>(g, g->b2);
        k_pmul<<<grid(g->L * g->pl.np), 256, 0, g->st>>>(g->a, g->b2, g->pl.lgL, g->pl.np);
        block<2>(g, g->a);
    }
    inv_outer(g, g->a);
    finish(g, g->slot[dst]);
}

void gpu_poke(GpuEngine* g, int slot) { k_poke<<<1, 1, 0, g->st>>>(g->slot[slot]); }

// How many candidates at once, measured (the rule gpu_streams is only a guess on cards other than the one it was
// measured on). A quick test before a batch: s = 1, 2, ... engines of this n, k square side by side, each on its own
// thread and stream as the workers do, ~0.4 s per step; the squarings per second of all of them are compared. A new best
// needs +1 %; the test stops two steps after the last best, at 16 or at the number of candidates, or when the card has
// less than a tenth of its memory free (the next engine might not fit) - the best s is taken. LOTR_GPU_STREAMS forces the
// count without the test, LOTR_GPU_AUTOTUNE=0 keeps the rule. The result is kept per n, k (cli and dispatch both ask).
int gpu_streams_auto(unsigned n, uint64_t k, size_t cands) {
    const int rule = gpu_streams(n, k);
    if (cands <= 1 || std::getenv("LOTR_GPU_STREAMS")) return rule;
    const char* at = std::getenv("LOTR_GPU_AUTOTUNE");
    if (at && *at == '0') return rule;
    static std::mutex mtx;
    static std::map<std::pair<unsigned, uint64_t>, int> kept;
    std::lock_guard<std::mutex> lk(mtx);
    const auto key = std::make_pair(n, k);
    const auto it = kept.find(key);
    if (it != kept.end()) return it->second;

    const auto t_start = std::chrono::steady_clock::now();
    const int smax = (int)std::min<size_t>(16, cands);
    std::vector<GpuEngine*> eng;
    std::vector<double> rate(1, 0.0);   // rate[s]: squarings per second of s engines together
    int best = 0;
    mpz_t v;
    mpz_init_set_ui(v, 3);
    try {
        double r1 = 0;   // the first engine alone, from a short timed run: sets the length of the steps
        for (int s = 1; s <= smax; ++s) {
            size_t fr = 0, tot = 0;
            if (s > 1 && (cudaMemGetInfo(&fr, &tot) != cudaSuccess || fr < tot / 10)) break;
            std::string info;
            GpuEngine* g = nullptr;
            try { g = gpu_create(n, k, 0, info); } catch (const std::exception&) { g = nullptr; }
            if (!g) break;
            eng.push_back(g);
            gpu_set(g, GPU_X, v);
            gpu_square(g, GPU_X, 2 * GRAPH_SQ);   // warm-up: the CUDA Graph is built here
            gpu_sync(g);
            if (s == 1) {
                const auto t0 = std::chrono::steady_clock::now();
                gpu_square(g, GPU_X, 4 * GRAPH_SQ);
                gpu_sync(g);
                r1 = 4.0 * GRAPH_SQ / std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            }
            const double r = std::max(r1, rate[best]);
            const unsigned per = std::max(2u, (unsigned)(0.4 * r / s / GRAPH_SQ + 0.5)) * GRAPH_SQ;
            std::atomic<int> ready{ 0 };
            std::atomic<bool> go{ false };
            std::vector<std::exception_ptr> errs(s);
            std::vector<std::thread> th;
            for (int i = 0; i < s; ++i)
                th.emplace_back([&, i] {
                    try {
                        ++ready;
                        while (!go) std::this_thread::yield();
                        gpu_square(eng[i], GPU_X, per);
                        gpu_sync(eng[i]);
                    }
                    catch (...) { errs[i] = std::current_exception(); }
                });
            while (ready < s) std::this_thread::yield();
            const auto t0 = std::chrono::steady_clock::now();
            go = true;
            for (std::thread& t : th) t.join();
            const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            for (const std::exception_ptr& e : errs)
                if (e) std::rethrow_exception(e);
            rate.push_back((double)s * per / sec);
            if (best == 0 || rate[s] > rate[best] * 1.01) best = s;
            else if (s - best >= 2) break;
        }
    }
    catch (const std::exception& e) {
        std::printf("[GPU] quick test of candidates at once failed (%s) - the rule: %d\n", e.what(), rule);
        best = 0;
    }
    for (GpuEngine* g : eng) gpu_destroy(g);
    mpz_clear(v);
    const int res = best > 0 ? best : rule;
    if (best > 0) {
        std::string line;
        char buf[48];
        for (size_t s = 1; s < rate.size(); ++s) {
            std::snprintf(buf, sizeof(buf), "%s%zu: %.2f", s > 1 ? ", " : "", s, rate[s] / rate[1]);
            line += buf;
        }
        std::printf("[GPU] quick test, candidates at once (speed vs 1): %s -> %d (rule %d, %.1f s)\n", line.c_str(), res,
                    rule, std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count());
        std::fflush(stdout);
    }
    kept[key] = res;
    return res;
}
