// Copyright (C) Damian Koba.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// HYBRID ENGINE (the default; LOTR_GPU_HYB=0 = classic; included by gpu_proth_engine.cu after struct GpuEngine). The
// arithmetic of the classic engine (32-bit Montgomery, radix-64 / radix-8 outer passes, k_block, Garner) without its zero
// padding: the primes are p = c*2^e - 1 (p = 3 mod 4) and the NTT runs in GF(p^2) = F_p[i], i^2 = -1 (complex numbers
// mod p). Checked against a Python model (30 squarings = GMP, a whole Proth test).
// N = k*2^n + 1, K = k*2^n, d = bit length of k, L real digits at positions e_t = ceil(t*(n+d)/L) (sizes Bm or Bm+1 bits,
// e_L = n + d). Squaring = weighted negacyclic convolution of length L:
//   u_t = a_t * d_t, a_t = 2^(e_t) * rho^(-t), rho^L = K in F_p (exists when k is a square mod p: 2 is one, p = 7 mod 8),
//   negacyclic length L = cyclic length M = L/2 in GF(p^2): z_j = theta^j (u_j + i u_(j+M)), theta^M = i (order 2L, L | p+1),
//   w_t = k * c_t / a_t are integers (|w_t| < P/2) and W = sum w_t 2^(e_t) == k * Y^2 (mod N)  [the wrap terms carry
//   2^(n+d+...) / K = 2^(...)/k, times k: integers]. The code multiplies by -k: W == -k * Y^2, the scaled form of the API.
// Weights without tables of L entries: a_t = tau^(s_t) * sigma^t with tau^L = 2, sigma^L = 2^d / k, s_t = (-t*(n+d)) mod L,
// so z_j = (theta*sigma)^j (tau^(s_j) d_j + i sigma^M tau^(s_(j+M)) d_(j+M)): two-level tables (2048 x hi) for tau^s and
// (theta*sigma)^j, one product each.
// Carry: w_t in parts at the digits t, t+1, .. t+H-1 (E_m); the parts at positions L+v (v < H-1) wrap: their value
// Wtop * 2^(n+d), Wtop * 2^d = Q*k + R (long division by k over the H-1 positions) == -Q + R*2^n: -q_v at the
// digit v, R*2^(n - e_(L-1)) at the top digit. Then npass neighbour passes; the TOP digit keeps its carry (no wrap of
// the carry chain): it holds up to Dt, the other digits are nearly balanced |d_t| <= 2^(b_t - 1) + 1. All of it (E,
// fold, passes) in k_hcrtE with halos.
// Bound: |w_t| <= 2^(d+2) * ((L-2) D^2 + 2 D Dt + Dt^2) < P/2 (worst case, as the classic engine).
// Pipeline: pack inside the first outer pass, radix-16 / column passes, the lowest levels in k_hblock16, unpack + CRT +
// fold + neighbour passes inside k_hcrtE (k_hcrtW for the wide digits).

#include <dlfcn.h>   // libnvidia-ml at run time (the energy lines of COLBENCH)

struct HP {                      // the primes of one engine (depend on k): in device memory, one pointer per kernel
    // 8 primes; the wide digits use 6-7
    uint32_t p[8], pv[8];        // p, -p^-1 mod 2^32
    uint32_t r2[8], r3[8];       // R^2, R^3 mod p (residues of 64-bit digits)
    uint32_t inv[8][8];          // Garner: p_j^-1 mod p_i (j < i), Montgomery form
    uint32_t cw[8];              // tau^-D x R, D = (M/16 * (n+d)) mod L: the pack step of k_hcolf<true> (rows + 64)
    uint32_t ck[8];              // sigma^M tau^-D1 x R, D1 = (M * (n+d)) mod L: tau^s_(j+M) sigma^M / tau^s_j (x 2 when s_j < D1)
    uint32_t cu[8];              // sigma^-M tau^D1 x R: the same for the unpack weights (hw2 UW, / 2 when s_j < D1)
    uint32_t c52[8];             // 2^52 mod p (hredc2: the wide digits)
    uint32_t c84[8];   // 2^84 mod p (hredc2: digits of 84 bits and more)
};
struct H4 { uint64_t w[4]; };   // P of the wide digits (6-8 primes of 31 bits)
// the CRT by a fractional estimate (hcrtf): c_i x R, 1 / p_i, P / p_i and P in words of 32 bits
struct HF { uint32_t ci[8]; double ip[8]; uint32_t Q[8][8]; uint32_t P[8]; };

struct HybEngine {
    int lgL = 0, lgM = 0, np = 0, B = 0, Bm = 0, H = 0, npass = 0, d = 0, shn = 0;
    size_t L = 0, M = 0;
    uint64_t nd = 0;             // n + d = e_L
    uint64_t kinv = 0;           // floor((2^64 - 1) / k) (the fold in k_hcrtE)
    bool col = false;            // the column pass (when it fits: 10 outer levels, 64 KiB shared)
    // the column pass: k_hcolf radix-16 as hr16f, k_hcoli prime inner with hdit16v (the measured best of each)
    // UC: with the column pass the unpack factor of column jj = j mod q (q = M / 1024 columns) multiplies the column in
    // k_hcoli (folded into the powers of its first stage), k_hcrtE takes only the row factor (DG 6, hw2 UM 3)
    int nsm = 0;                 // SMs (the persistent grid of the column pass)
    double Dt = 0;               // bound of the top digit
    u128 P = 1;
    HP hp{};
    HP* dhp = nullptr;
    uint2 *a = nullptr, *b2 = nullptr, *tw = nullptr, *itw = nullptr;
    uint2 *pkh = nullptr, *pkl = nullptr;   // (theta*sigma)^j, two levels
    uint32_t *tl = nullptr, *th = nullptr, *thm = nullptr;                  // tau^s (low half, high half x sigma^M); th, thm no R
    uint2 *ht16 = nullptr, *iht16 = nullptr;   // k_hblock16 TT: the twiddles of the stages q = 256, 16 (hr16t), 4096 per prime
    uint2 *ucb = nullptr, *ucq = nullptr;   // UC: (theta*sigma)^-qr F tau^-s0(qr) per row r, (theta*sigma)^-jj tau^(r_jj) per column
    uint2 *ub = nullptr, *uq = nullptr;   // hw2 UB: (theta*sigma)^-jb k R / M tau^-s_jb per block jb = 256 b, (theta*sigma)^-t tau^(r_t)
    // the full pack weights (small transforms: np M <= 2^20): pwa[c*M + j] = tau^s_j (theta*sigma)^j,
    // pwb = i sigma^M tau^s_(j+M) (theta*sigma)^j, both x R^2 (the digits as d / R, hredc): 2 reductions + 4 products a
    // complex value instead of 14 products and 6 scattered loads
    uint2 *pwa = nullptr, *pwb = nullptr;
    // the full unpack weights (the same sizes): uf[c*M + j] = {A0, B0, A1, B1}, w re = (a.x A0 + a.y B0) / R,
    // im = (a.x A1 + a.y B1) / R (hdot): the UB factor ub x uq, cu and the halvings hw / hv in one entry, 2 reductions a
    // position instead of 9 products (k_hcrtE DG 7, hw2 UM 4)
    uint4* uf = nullptr;
    // LOTR_GPU_L2P=<MB> (experiment): a part of a pinned in the L2 (persisting lines) between the kernels: a is read or
    // written 6 times a square (colf, block x 2, coli x 2, crtE)
    cudaAccessPolicyWindow l2w{};
    bool l2on = false;
    // the wide digits (LOTR_GPU_WIDE=7 (the worst case) / 6 (statistical, GEC as the guard)): ~80-bit digits in 16 B
    // (lo, hi), half the L of the 40-bit plan and 6-7 primes; k_hblock11 (2048) or k_hblock16, the column pass at 2^21,
    // k_hcrtW (w in 4 words, the CRT of hcrtf). crtf: the CRT of hcrtf in k_hcrtE (LOTR_GPU_CRTF=1)
    int wide = 0;
    bool crtf = false;
    // LOTR_GPU_L2ORD (the column pass): block + coli one prime after the other, 1 in one stream (hyb_blkinv_l2),
    // 2 overlapped in two streams (hyb_blkinv_l2o: st2, the events evf / evb / evc), 3 coli -> CRT in chunks (hyb_colicrt),
    // 0 = all primes in one launch each. Without the variable: the faster of 0 and 2, timed once per shape
    // (hyb_l2ord_pick, with CUDA Graphs)
    int l2ord = 0;
    cudaStream_t st2 = nullptr;
    cudaEvent_t evf = nullptr, evb[8] = {}, evc[8] = {};
    uint2 *ht11 = nullptr, *iht11 = nullptr;
    H4 P4{};
    HF hf{};
    // the rows of the column pass, HC_LGR (1024) or HC9_LGR (512: the wide digits at 2^21 with k_hblock16, 2 blocks per
    // SM; LOTR_GPU_COL9=0 the 1024-row pass + k_hblock11); cgf / cgi the persistent grids of the 512-row kernels (SMs x
    // blocks per SM)
    int lgR = 10;
    unsigned cgf = 0, cgi = 0;
    // LOTR_GPU_L2ORD=3 (hyb_colicrt): coli -> CRT in cch chunks of columns (LOTR_GPU_CCH, 4 / 8 / 16), cpo the prime as
    // the outer tile index of k_hcoli9 there (LOTR_GPU_CPO=0: inner); evk the events of the chunks
    int cch = 8, cpo = 1;
    cudaEvent_t evk[17] = {};
};

namespace {

// the 31-bit primes c*2^e - 1 with e >= 23, largest first
const uint32_t HYB_PRIMES[] = { 2147483647u, 2063597567u, 2046820351u, 1962934271u, 1912602623u, 1887436799u, 1795162111u,
                                1769996287u, 1669332991u, 1635778559u, 1560281087u, 1518338047u, 1493172223u, 1468006399u,
                                1392508927u, 1384120319u, 1283457023u, 1031798783u, 931135487u, 763363327u, 738197503u,
                                654311423u, 578813951u };

__host__ __device__ __forceinline__ uint64_t hpos(uint64_t x, uint64_t nd, int lgL) {   // e_x = ceil(x * nd / L)
    return (x * nd + (((uint64_t)1 << lgL) - 1)) >> lgL;
}

// ---- GF(p^2) on the card: uint2 {re, im}; a twiddle w is stored x R (Montgomery), cmul(a, w) = a*w
__device__ __forceinline__ uint2 cadd(uint2 a, uint2 b, uint32_t p) { return make_uint2(madd(a.x, b.x, p), madd(a.y, b.y, p)); }
__device__ __forceinline__ uint2 csub(uint2 a, uint2 b, uint32_t p) { return make_uint2(msub(a.x, b.x, p), msub(a.y, b.y, p)); }
// (x A + y B) / R mod p with one reduction: x, y, A < p < 2^31, B <= p, the sum < 2 p^2 < p R
__device__ __forceinline__ uint32_t hdot(uint32_t x, uint32_t A, uint32_t y, uint32_t B, uint32_t p, uint32_t pv) {
    const uint64_t t = (uint64_t)x * A + (uint64_t)y * B;
    const uint32_t m = (uint32_t)t * pv;
    const uint32_t u = (uint32_t)((t + (uint64_t)m * p) >> 32);
    return u >= p ? u - p : u;
}
// re = (a.x w.x + a.y (p - w.y)) / R, im = (a.x w.y + a.y w.x) / R by hdot: 4 products and 2 reductions instead of
// Karatsuba's 3 products, 3 reductions and 5 additions (the same canonical values, bit for bit)
__device__ __forceinline__ uint2 cmul(uint2 a, uint2 w, uint32_t p, uint32_t pv) {
    return make_uint2(hdot(a.x, w.x, a.y, p - w.y, p, pv), hdot(a.x, w.y, a.y, w.x, p, pv));
}
__device__ __forceinline__ uint2 csq(uint2 a, uint32_t p, uint32_t pv) {   // a^2 / R
    const uint32_t t = mmul(a.x, a.y, p, pv);
    return make_uint2(mmul(madd(a.x, a.y, p), msub(a.x, a.y, p), p, pv), madd(t, t, p));
}
__device__ __forceinline__ void cdif(uint2& x, uint2& y, uint2 w, uint32_t p, uint32_t pv) {
    const uint2 u = x, v = y;
    x = cadd(u, v, p);
    y = cmul(csub(u, v, p), w, p, pv);
}
__device__ __forceinline__ void cdit(uint2& x, uint2& y, uint2 w, uint32_t p, uint32_t pv) {
    const uint2 t = cmul(y, w, p, pv), u = x;
    x = cadd(u, t, p);
    y = csub(u, t, p);
}
// the 7 twiddles of a radix-8 group (as tw8x): from the one entry w_8q^j
__device__ __forceinline__ void ctw8(uint2* w, const uint2* T, unsigned q, unsigned j, uint32_t p, uint32_t pv) {
    const uint2 x = T[4 * q + j];
    w[0] = x;
    w[1] = cmul(x, T[5], p, pv);
    w[2] = cmul(x, T[6], p, pv);
    w[3] = cmul(x, T[7], p, pv);
    w[4] = cmul(x, x, p, pv);
    w[5] = cmul(w[4], T[3], p, pv);
    w[6] = cmul(w[4], w[4], p, pv);
}
__device__ __forceinline__ void cdif8w(uint2* v, const uint2* w, uint32_t p, uint32_t pv) {
#pragma unroll
    for (int i = 0; i < 4; ++i) cdif(v[i], v[i + 4], w[i], p, pv);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        cdif(v[i], v[i + 2], w[4 + i], p, pv);
        cdif(v[i + 4], v[i + 6], w[4 + i], p, pv);
    }
#pragma unroll
    for (int i = 0; i < 8; i += 2) cdif(v[i], v[i + 1], w[6], p, pv);
}
__device__ __forceinline__ void cdit8w(uint2* v, const uint2* w, uint32_t p, uint32_t pv) {
#pragma unroll
    for (int i = 0; i < 8; i += 2) cdit(v[i], v[i + 1], w[6], p, pv);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        cdit(v[i], v[i + 2], w[4 + i], p, pv);
        cdit(v[i + 4], v[i + 6], w[4 + i], p, pv);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) cdit(v[i], v[i + 4], w[i], p, pv);
}
// d * R mod p (Montgomery form, as res() without its last product: the pack tables th / thm hold no R instead)
__device__ __forceinline__ uint32_t hresR(int64_t d, uint32_t p, uint32_t pv, uint32_t r2, uint32_t r3) {
    const uint64_t x = d < 0 ? 0 - (uint64_t)d : (uint64_t)d;
    const uint32_t y = madd(mmul((uint32_t)x, r2, p, pv), mmul((uint32_t)(x >> 32), r3, p, pv), p);
    return (d < 0 && y) ? p - y : y;
}
// d / R mod p: one Montgomery reduction of (hi mod p) * 2^32 + lo (< p R), c32 = 2^32 mod p; |d| < p 2^32 (digits < 2^59)
__device__ __forceinline__ uint32_t hredc(int64_t d, uint32_t p, uint32_t pv, uint32_t c32) {
    const int32_t hi = (int32_t)(d >> 32);
    const uint64_t t = (uint64_t)(hi < 0 ? (uint32_t)(hi + (int32_t)p) : (uint32_t)hi) * c32 + (uint32_t)d;
    const uint32_t m = (uint32_t)t * pv;
    const uint32_t u = (uint32_t)((t + (uint64_t)m * p) >> 32);
    return u >= p ? u - p : u;
}
// d / R mod p of a wide digit |v| < 2^115 (16 B: lo, hi): v = hi 2^52 + lo52, lo52 / R by one reduction (< p + 2^20),
// |hi| = a 2^32 + b: hi 2^52 / R = b c52 / R + a c84 / R by two mmul (c52 = 2^52, c84 = 2^84 mod p; mmul: a < 2^32
// with b < p is enough). Digits of 84-96 bits (WIDE plans of 10-40M digits) need the c84 part.
__device__ __forceinline__ uint32_t hredc2(longlong2 v, uint32_t p, uint32_t pv, uint32_t c52, uint32_t c84) {
    const uint64_t lo = (uint64_t)v.x, t = lo & ((1ull << 52) - 1);
    const int64_t hi = (int64_t)(((uint64_t)v.y << 12) | (lo >> 52));
    const uint32_t m = (uint32_t)t * pv;
    uint32_t u = (uint32_t)((t + (uint64_t)m * p) >> 32);
    u = u >= p ? u - p : u;
    const uint64_t ah = hi < 0 ? 0 - (uint64_t)hi : (uint64_t)hi;
    uint32_t y = madd(mmul((uint32_t)ah, c52, p, pv), mmul((uint32_t)(ah >> 32), c84, p, pv), p);
    if (hi < 0 && y) y = p - y;
    return madd(u, y, p);
}
// ---- kernels; array c of np: a[c*M + j] (complex), the tables of prime c at c * (their size)
struct HPK { const uint32_t *tl, *th, *thm; const uint2 *pkh, *pkl; uint64_t ndm; const uint2 *wa, *wb; };   // pack tables, ndm = (n+d) mod L; wa, wb: the full weights (or null)
struct HUK { uint64_t ndm; const uint2 *ub, *uq, *ucb; int lgq; const uint4* uf; };   // unpack tables
// digits -> the weighted, twisted residue z_j of prime c (z_j above). WIDE: the digits 16 B (longlong2),
// d / R by hredc2, d R = (d / R) R^3 / R
template <bool WIDE = false> __device__ __forceinline__ uint2 hpackel(const int64_t* d, size_t j, unsigned c, const HP* hp,
                                                                      int lgL, const HPK& pk) {
    const size_t L = (size_t)1 << lgL, M = L >> 1, LH = L >> 11, MH = M >> 11;
    const uint32_t p = hp->p[c], pv = hp->pv[c], r2 = hp->r2[c], r3 = hp->r3[c];
    const longlong2* dw = reinterpret_cast<const longlong2*>(d);
    if (pk.wa) {   // the full weights: d_j / R * pwa + d_(j+M) / R * pwb
        const uint32_t c32 = mmul(r2, 1, p, pv);
        const uint2 A = pk.wa[c * M + j], B = pk.wb[c * M + j];
        const uint32_t y0 = WIDE ? hredc2(dw[j], p, pv, hp->c52[c], hp->c84[c]) : hredc(d[j], p, pv, c32);
        const uint32_t y1 = WIDE ? hredc2(dw[j + M], p, pv, hp->c52[c], hp->c84[c]) : hredc(d[j + M], p, pv, c32);
        return make_uint2(hdot(y0, A.x, y1, B.x, p, pv), hdot(y0, A.y, y1, B.y, p, pv));
    }
    const size_t s0 = (L - ((j * pk.ndm) & (L - 1))) & (L - 1), s1 = (L - (((j + M) * pk.ndm) & (L - 1))) & (L - 1);
    const uint32_t* TL = pk.tl + c * 2048;
    // (d R) * (tau^s plain: th without R, tl x R) = d tau^s
    const uint32_t dr0 = WIDE ? mmul(hredc2(dw[j], p, pv, hp->c52[c], hp->c84[c]), r3, p, pv) : hresR(d[j], p, pv, r2, r3);
    const uint32_t dr1 = WIDE ? mmul(hredc2(dw[j + M], p, pv, hp->c52[c], hp->c84[c]), r3, p, pv) : hresR(d[j + M], p, pv, r2, r3);
    const uint32_t x0 = mmul(dr0, mmul(pk.th[c * LH + (s0 >> 11)], TL[s0 & 2047], p, pv), p, pv);
    const uint32_t x1 = mmul(dr1, mmul(pk.thm[c * LH + (s1 >> 11)], TL[s1 & 2047], p, pv), p, pv);
    return cmul(make_uint2(x0, x1), cmul(pk.pkh[c * MH + (j >> 11)], pk.pkl[c * 2048 + (j & 2047)], p, pv), p, pv);
}
// pack alone (when the first outer pass cannot do it: radix-8 / radix-2 first, or no outer pass)
template <bool WIDE = false> __global__ void k_hpack(const int64_t* d, uint2* a, const HP* hp, int lgL, int np, HPK pk) {
    const size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const int lgM = lgL - 1;
    if (t >= ((size_t)np << lgM)) return;
    a[t] = hpackel<WIDE>(d, t & (((size_t)1 << lgM) - 1), (unsigned)(t >> lgM), hp, lgL, pk);
}
// 6 levels in one pass (as k_dif64 / k_dit64, complex; shared memory as two planes); FIRST: packs the digits itself
template <bool FIRST, bool WIDE = false> __global__ void __launch_bounds__(256) k_hdif64(uint2* a, const uint2* tw, int lgM,
                                                                                         int lgq, const HP* hp,
                                                                                         const int64_t* d, HPK pk) {
    __shared__ uint32_t sx[64 * 33], sy[64 * 33];
    const size_t M = (size_t)1 << lgM, q = (size_t)1 << lgq;
    const unsigned per = (unsigned)(M >> 11), cpg = (unsigned)(q >> 5);
    const unsigned c = blockIdx.x / per, rem = blockIdx.x % per;
    const unsigned col = threadIdx.x & 31, part = threadIdx.x >> 5;
    const unsigned j = (rem % cpg) * 32 + col, uq = (unsigned)q;
    const uint32_t p = hp->p[c], pv = hp->pv[c];
    const uint2* T = tw + c * M;
    const size_t g0 = (size_t)(rem / cpg) * 64 * q + j;
    uint2* x = a + c * M + g0;
    uint2 v[8], w[7];
    if (FIRST) {
#pragma unroll
        for (int m = 0; m < 8; ++m) v[m] = hpackel<WIDE>(d, g0 + (part + 8 * m) * q, c, hp, lgM + 1, pk);
    }
    else {
#pragma unroll
        for (int m = 0; m < 8; ++m) v[m] = x[(part + 8 * m) * q];
    }
    ctw8(w, T, 8 * uq, j + part * uq, p, pv);
    cdif8w(v, w, p, pv);
#pragma unroll
    for (int m = 0; m < 8; ++m) { sx[(part + 8 * m) * 33 + col] = v[m].x; sy[(part + 8 * m) * 33 + col] = v[m].y; }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = make_uint2(sx[(8 * part + i) * 33 + col], sy[(8 * part + i) * 33 + col]);
    ctw8(w, T, uq, j, p, pv);
    cdif8w(v, w, p, pv);
#pragma unroll
    for (int i = 0; i < 8; ++i) x[(8 * part + i) * q] = v[i];
}
__global__ void __launch_bounds__(256) k_hdit64(uint2* a, const uint2* itw, int lgM, int lgq, const HP* hp) {
    __shared__ uint32_t sx[64 * 33], sy[64 * 33];
    const size_t M = (size_t)1 << lgM, q = (size_t)1 << lgq;
    const unsigned per = (unsigned)(M >> 11), cpg = (unsigned)(q >> 5);
    const unsigned c = blockIdx.x / per, rem = blockIdx.x % per;
    const unsigned col = threadIdx.x & 31, part = threadIdx.x >> 5;
    const unsigned j = (rem % cpg) * 32 + col, uq = (unsigned)q;
    const uint32_t p = hp->p[c], pv = hp->pv[c];
    const uint2* T = itw + c * M;
    uint2* x = a + c * M + (size_t)(rem / cpg) * 64 * q + j;
    uint2 v[8], w[7];
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = x[(8 * part + i) * q];
    ctw8(w, T, uq, j, p, pv);
    cdit8w(v, w, p, pv);
#pragma unroll
    for (int i = 0; i < 8; ++i) { sx[(8 * part + i) * 33 + col] = v[i].x; sy[(8 * part + i) * 33 + col] = v[i].y; }
    __syncthreads();
#pragma unroll
    for (int m = 0; m < 8; ++m) v[m] = make_uint2(sx[(part + 8 * m) * 33 + col], sy[(part + 8 * m) * 33 + col]);
    ctw8(w, T, 8 * uq, j + part * uq, p, pv);
    cdit8w(v, w, p, pv);
#pragma unroll
    for (int m = 0; m < 8; ++m) x[(part + 8 * m) * q] = v[m];
}
// FIRST: packs the digits itself in the first outer pass of every radix (one kernel less a square)
template <bool FIRST, bool WIDE = false> __global__ void k_hdif8(uint2* a, const uint2* tw, int lgM, int lgq, int np, const HP* hp,
                                              const int64_t* d, HPK pk) {
    const size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t M = (size_t)1 << lgM, q = (size_t)1 << lgq, per = M >> 3;
    if (t >= per * np) return;
    const unsigned c = (unsigned)(t >> (lgM - 3));
    const size_t r = t & (per - 1), j = r & (q - 1), g0 = ((r >> lgq) << (lgq + 3)) + j;
    const uint32_t p = hp->p[c], pv = hp->pv[c];
    uint2* x = a + c * M + g0;
    uint2 v[8], w[7];
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = FIRST ? hpackel<WIDE>(d, g0 + i * q, c, hp, lgM + 1, pk) : x[i * q];
    ctw8(w, tw + c * M, (unsigned)q, (unsigned)j, p, pv);
    cdif8w(v, w, p, pv);
#pragma unroll
    for (int i = 0; i < 8; ++i) x[i * q] = v[i];
}
__global__ void k_hdit8(uint2* a, const uint2* itw, int lgM, int lgq, int np, const HP* hp) {
    const size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t M = (size_t)1 << lgM, q = (size_t)1 << lgq, per = M >> 3;
    if (t >= per * np) return;
    const unsigned c = (unsigned)(t >> (lgM - 3));
    const size_t r = t & (per - 1), j = r & (q - 1);
    const uint32_t p = hp->p[c], pv = hp->pv[c];
    uint2* x = a + c * M + ((r >> lgq) << (lgq + 3)) + j;
    uint2 v[8], w[7];
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = x[i * q];
    ctw8(w, itw + c * M, (unsigned)q, (unsigned)j, p, pv);
    cdit8w(v, w, p, pv);
#pragma unroll
    for (int i = 0; i < 8; ++i) x[i * q] = v[i];
}
template <bool FIRST, bool WIDE = false> __global__ void k_hdif2(uint2* a, const uint2* tw, int lgM, int lgh, int np, const HP* hp,
                                              const int64_t* d, HPK pk) {
    const size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t M = (size_t)1 << lgM, h = (size_t)1 << lgh, per = M >> 1;
    if (t >= per * np) return;
    const unsigned c = (unsigned)(t >> (lgM - 1));
    const size_t r = t & (per - 1), j = r & (h - 1), g0 = ((r >> lgh) << (lgh + 1)) + j;
    uint2* x = a + c * M + g0;
    uint2 u = FIRST ? hpackel<WIDE>(d, g0, c, hp, lgM + 1, pk) : x[0], v = FIRST ? hpackel<WIDE>(d, g0 + h, c, hp, lgM + 1, pk) : x[h];
    cdif(u, v, tw[c * M + h + j], hp->p[c], hp->pv[c]);
    x[0] = u; x[h] = v;
}
__global__ void k_hdit2(uint2* a, const uint2* itw, int lgM, int lgh, int np, const HP* hp) {
    const size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t M = (size_t)1 << lgM, h = (size_t)1 << lgh, per = M >> 1;
    if (t >= per * np) return;
    const unsigned c = (unsigned)(t >> (lgM - 1));
    const size_t r = t & (per - 1), j = r & (h - 1);
    uint2* x = a + c * M + ((r >> lgh) << (lgh + 1)) + j;
    uint2 u = x[0], v = x[h];
    cdit(u, v, itw[c * M + h + j], hp->p[c], hp->pv[c]);
    x[0] = u; x[h] = v;
}
// 2 levels in registers (2^14: one pass instead of two radix-2 passes and the pack alone): 4 values at stride q,
// DIF levels 2q, q (T[h + j] = w_2h^j: w_4q^(j+q) = T[3q + j]). FIRST: packs the digits itself.
template <bool FIRST, bool WIDE = false> __global__ void k_hdif4(uint2* a, const uint2* tw, int lgM, int lgq, int np, const HP* hp,
                                              const int64_t* d, HPK pk) {
    const size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t M = (size_t)1 << lgM, q = (size_t)1 << lgq, per = M >> 2;
    if (t >= per * np) return;
    const unsigned c = (unsigned)(t >> (lgM - 2));
    const size_t r = t & (per - 1), j = r & (q - 1), g0 = ((r >> lgq) << (lgq + 2)) + j;
    const uint32_t p = hp->p[c], pv = hp->pv[c];
    const uint2* T = tw + c * M;
    uint2* x = a + c * M + g0;
    uint2 v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = FIRST ? hpackel<WIDE>(d, g0 + i * q, c, hp, lgM + 1, pk) : x[i * q];
    cdif(v[0], v[2], T[2 * q + j], p, pv);
    cdif(v[1], v[3], T[3 * q + j], p, pv);
    const uint2 w = T[q + j];
    cdif(v[0], v[1], w, p, pv);
    cdif(v[2], v[3], w, p, pv);
#pragma unroll
    for (int i = 0; i < 4; ++i) x[i * q] = v[i];
}
__global__ void k_hdit4(uint2* a, const uint2* itw, int lgM, int lgq, int np, const HP* hp) {
    const size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t M = (size_t)1 << lgM, q = (size_t)1 << lgq, per = M >> 2;
    if (t >= per * np) return;
    const unsigned c = (unsigned)(t >> (lgM - 2));
    const size_t r = t & (per - 1), j = r & (q - 1);
    const uint32_t p = hp->p[c], pv = hp->pv[c];
    const uint2* T = itw + c * M;
    uint2* x = a + c * M + ((r >> lgq) << (lgq + 2)) + j;
    uint2 v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = x[i * q];
    const uint2 w = T[q + j];
    cdit(v[0], v[1], w, p, pv);
    cdit(v[2], v[3], w, p, pv);
    cdit(v[0], v[2], T[2 * q + j], p, pv);
    cdit(v[1], v[3], T[3 * q + j], p, pv);
#pragma unroll
    for (int i = 0; i < 4; ++i) x[i * q] = v[i];
}
// 4 levels in registers (16 values at stride q, no shared memory): DIF levels 8q, 4q, 2q, q. The twiddles from the one
// entry x = w_16q^j: w_16q^(j+iq) = x w_16^i, w_8q^(j+iq) = x^2 w_8^i, w_4q^(j+iq) = x^4 w_4^i, w_2q^j = x^8
// (w_16^i = T[8+i], w_8^i = T[4+i], w_4 = T[3]). FIRST: packs the digits itself.
__device__ __forceinline__ void ctw16(uint2* x, const uint2* T, unsigned q, unsigned j, uint32_t p, uint32_t pv) {
    x[0] = T[8 * q + j];
    x[1] = cmul(x[0], x[0], p, pv);
    x[2] = cmul(x[1], x[1], p, pv);
    x[3] = cmul(x[2], x[2], p, pv);
}
// the radix-16 group (16 values at stride q, position j < q) in registers, DIF / DIT
__device__ __forceinline__ void hdif16v(uint2* v, const uint2* T, unsigned q, unsigned j, uint32_t p, uint32_t pv) {
    uint2 xw[4];
    ctw16(xw, T, q, j, p, pv);
#pragma unroll
    for (int i = 0; i < 8; ++i) cdif(v[i], v[i + 8], i ? cmul(xw[0], T[8 + i], p, pv) : xw[0], p, pv);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint2 w = i ? cmul(xw[1], T[4 + i], p, pv) : xw[1];
        cdif(v[i], v[i + 4], w, p, pv);
        cdif(v[i + 8], v[i + 12], w, p, pv);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint2 w = i ? cmul(xw[2], T[3], p, pv) : xw[2];
#pragma unroll
        for (int b = 0; b < 16; b += 4) cdif(v[b + i], v[b + i + 2], w, p, pv);
    }
#pragma unroll
    for (int i = 0; i < 16; i += 2) cdif(v[i], v[i + 1], xw[3], p, pv);
}
__device__ __forceinline__ void hdit16v(uint2* v, const uint2* T, unsigned q, unsigned j, uint32_t p, uint32_t pv) {
    uint2 xw[4];
    ctw16(xw, T, q, j, p, pv);
#pragma unroll
    for (int i = 0; i < 16; i += 2) cdit(v[i], v[i + 1], xw[3], p, pv);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint2 w = i ? cmul(xw[2], T[3], p, pv) : xw[2];
#pragma unroll
        for (int b = 0; b < 16; b += 4) cdit(v[b + i], v[b + i + 2], w, p, pv);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint2 w = i ? cmul(xw[1], T[4 + i], p, pv) : xw[1];
        cdit(v[i], v[i + 4], w, p, pv);
        cdit(v[i + 8], v[i + 12], w, p, pv);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) cdit(v[i], v[i + 8], i ? cmul(xw[0], T[8 + i], p, pv) : xw[0], p, pv);
}
template <bool FIRST, bool WIDE = false> __global__ void __launch_bounds__(256) k_hdif16(uint2* a, const uint2* tw, int lgM, int lgq, int np,
                                                                       const HP* hp, const int64_t* d, HPK pk) {
    const size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t M = (size_t)1 << lgM, q = (size_t)1 << lgq, per = M >> 4;
    if (t >= per * np) return;
    const unsigned c = (unsigned)(t >> (lgM - 4));
    const size_t r = t & (per - 1), j = r & (q - 1), g0 = ((r >> lgq) << (lgq + 4)) + j;
    const uint32_t p = hp->p[c], pv = hp->pv[c];
    uint2* x = a + c * M + g0;
    uint2 v[16];
    if (FIRST) {
#pragma unroll
        for (int i = 0; i < 16; ++i) v[i] = hpackel<WIDE>(d, g0 + i * q, c, hp, lgM + 1, pk);
    }
    else {
#pragma unroll
        for (int i = 0; i < 16; ++i) v[i] = x[i * q];
    }
    hdif16v(v, tw + c * M, (unsigned)q, (unsigned)j, p, pv);
#pragma unroll
    for (int i = 0; i < 16; ++i) x[i * q] = v[i];
}
__global__ void __launch_bounds__(256) k_hdit16(uint2* a, const uint2* itw, int lgM, int lgq, int np, const HP* hp) {
    const size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t M = (size_t)1 << lgM, q = (size_t)1 << lgq, per = M >> 4;
    if (t >= per * np) return;
    const unsigned c = (unsigned)(t >> (lgM - 4));
    const size_t r = t & (per - 1), j = r & (q - 1), g0 = ((r >> lgq) << (lgq + 4)) + j;
    const uint32_t p = hp->p[c], pv = hp->pv[c];
    uint2* x = a + c * M + g0;
    uint2 v[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) v[i] = x[i * q];
    hdit16v(v, itw + c * M, (unsigned)q, (unsigned)j, p, pv);
#pragma unroll
    for (int i = 0; i < 16; ++i) x[i * q] = v[i];
}
// ONE column pass over the 10 outer levels (2^22: instead of radix-64 + radix-16, a read + write of the array less per
// transform). A tile = C = 8 complex columns jj = j0 .. j0+7 x R = 1024 rows at stride q = M/1024 of one prime: 64 KiB
// shared (two planes, rows swizzled: no bank conflicts) -> one block of 512 threads per SM, persistent (grid = the SMs,
// tiles in turn): the next tile goes into L2 (prefetch.global.L2, no registers) while this one is computed. Thread =
// (col, g), g < 64. Forward (DIF): radix-16 in registers straight from memory (rows g + 64i: row half-spans 512 .. 64),
// radix-8 in shared (row stride 8: 32 .. 8; groups g, g+64), radix-8 (stride 1: 4 .. 1) straight to memory; inverse
// (DIT) the same in reverse order (any grouping of the levels in order is the same transform). FIRST: packs the digits
// itself.
__device__ __forceinline__ void hpw8(uint2* w, uint2 x, uint32_t p, uint32_t pv);   // (below, at hr8w)
__device__ __forceinline__ void hpw8c(uint2* w, uint2 x, uint2 cf, uint32_t p, uint32_t pv);
template <bool INV> __device__ __forceinline__ void hr8w(uint2* v, const uint2* T, const uint2* w, uint32_t p, uint32_t pv);
template <bool INV> __device__ __forceinline__ void hr16f(uint2* v, const uint2* T, unsigned q, unsigned j, uint32_t p,
                                                          uint32_t pv);   // (below, at k_hblock16)
// the radix-8 stages: the two groups of a thread (gi = g, g + 64) have the same a (gi & 7 = g & 7; stage 3: j = jj), so
// the powers a^m are made once (hpw8) for both (k_hcolf stage 3 stores to global memory between them: the compiler could
// not reuse them; 15 products per group)
// k_hcolf: the radix-16 stage as hr16f (111 products per 16 values instead of hdif16v's 138); k_hcoli: the prime as the
// inner tile index (as k_hcolf), the radix-16 stage as hdit16v (faster there than hr16f)
constexpr int HC_C = 8, HC_LGR = 10, HC_R = 1 << HC_LGR, HC_N = HC_R * HC_C, HC_T = HC_N / 16;   // 512 threads
constexpr size_t HC_SHM = 2 * HC_N * sizeof(uint32_t);   // 64 KiB dynamic (cudaFuncAttributeMaxDynamicSharedMemorySize)
__device__ __forceinline__ unsigned hcsw(unsigned row, unsigned col) {   // a warp = 4 rows x 8 columns: 32 banks
    return (row ^ ((row >> 3) & (32 / HC_C - 1))) * HC_C + col;
}
__device__ __forceinline__ void l2pf(const void* x) { asm volatile("prefetch.global.L2 [%0];" ::"l"(x)); }
__device__ __forceinline__ void hc_rd8(const uint32_t* sx, const uint32_t* sy, uint2* u, unsigned r0, unsigned s, unsigned col) {
#pragma unroll
    for (int i = 0; i < 8; ++i) { const unsigned k = hcsw(r0 + i * s, col); u[i] = make_uint2(sx[k], sy[k]); }
}
__device__ __forceinline__ void hc_wr8(uint32_t* sx, uint32_t* sy, const uint2* u, unsigned r0, unsigned s, unsigned col) {
#pragma unroll
    for (int i = 0; i < 8; ++i) { const unsigned k = hcsw(r0 + i * s, col); sx[k] = u[i].x; sy[k] = u[i].y; }
}
// WIDE: the digits 16 B each (longlong2: lo, hi), hredc2 instead of hredc. CM (COLBENCH only): 0 the kernel, 1 the
// memory + shared memory pattern alone (no arithmetic), 2 the pack without the butterflies. With 1 block per SM the
// arithmetic does not overlap the loads (the reason for the 512-row k_hcolf9 below).
template <bool WIDE, int CM = 0> __global__ void __launch_bounds__(HC_T, 1) k_hcolf(uint2* a, const uint2* tw, int lgM, int np,
                                                                                     const HP* hp, const int64_t* d, HPK pk) {
    const longlong2* dw = reinterpret_cast<const longlong2*>(d);
    extern __shared__ uint32_t hcs[];
    uint32_t *sx = hcs, *sy = hcs + HC_N;
    const size_t M = (size_t)1 << lgM, q = M >> HC_LGR;
    const unsigned uq = (unsigned)q, per = uq / HC_C, unp = (unsigned)np, ntl = per * unp;
    const unsigned col = threadIdx.x % HC_C, g = threadIdx.x / HC_C;
    // the prime is the INNER tile index: the np blocks of the same columns run side by side, so with FIRST the digits
    // (64 MB at 2^23, more than the L2) come from DRAM once, not np times
    for (unsigned tile = blockIdx.x; tile < ntl; tile += gridDim.x) {
        const unsigned c = tile % unp, jj = (tile / unp) * HC_C + col, nt = tile + gridDim.x;
        if (nt < ntl) {
            const size_t nj = (nt / unp) * HC_C + col;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const size_t y = nj + (size_t)(g + 64 * i) * q;
                if (WIDE) { l2pf(dw + y); l2pf(dw + y + M); }
                else { l2pf(d + y); l2pf(d + y + M); }
            }
        }
        const uint32_t p = hp->p[c], pv = hp->pv[c];
        const uint2* T = tw + (size_t)c * M;
        uint2* x = a + (size_t)c * M + jj;
        uint2 v[16];
        {   // the pack of rows g + 64i (complex j + i*M/16) with the weights by steps, not hpackel per row (6 table
                       // loads from L2 each, 1 block per SM cannot hide them). One complex weight W = tau^s (theta sigma)^j R^2:
                       // sigma^M tau^s_(j+M) = tau^s kappa (hp->ck, x 2 when s < D1), the digits as d / R (hredc); step W x S tau^-D,
                       // S = (theta sigma)^(M/16), x 2 when s wraps below 0 (tau^L = 2). 2 reductions + 7 products per complex instead of
                       // 14 products (bit for bit the same as hpackel per row)
            const size_t L = M << 1, LH = L >> 11, MH = M >> 11, j = jj + (size_t)g * q, ul = L - 1;
            const unsigned D = (unsigned)(((M >> 4) * pk.ndm) & ul), D1 = (unsigned)((M * pk.ndm) & ul);
            unsigned s0 = (unsigned)((L - ((j * pk.ndm) & ul)) & ul);
            const uint32_t* TL = pk.tl + c * 2048;
            const uint32_t w0 = mmul(mmul(pk.th[c * LH + (s0 >> 11)], TL[s0 & 2047], p, pv), hp->r3[c], p, pv);
            const uint2 z = cmul(pk.pkh[c * MH + (j >> 11)], pk.pkl[c * 2048 + (j & 2047)], p, pv);
            const uint2 S = pk.pkh[c * MH + ((M >> 4) >> 11)];
            const uint32_t td = hp->cw[c], k1 = hp->ck[c], k2 = madd(k1, k1, p), c32 = mmul(hp->r2[c], 1, p, pv), c52 = hp->c52[c], c84 = hp->c84[c];
            const uint2 SD = make_uint2(mmul(S.x, td, p, pv), mmul(S.y, td, p, pv));
            uint2 W = make_uint2(mmul(z.x, w0, p, pv), mmul(z.y, w0, p, pv));
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const size_t y = j + (size_t)i * (M >> 4);
                if (CM == 1) {   // (the same loads, no arithmetic)
                    v[i] = WIDE ? make_uint2((uint32_t)dw[y].x ^ (uint32_t)dw[y].y, (uint32_t)dw[y + M].x ^ (uint32_t)dw[y + M].y)
                                : make_uint2((uint32_t)d[y], (uint32_t)d[y + M]);
                    continue;
                }
                const uint32_t r0 = WIDE ? hredc2(dw[y], p, pv, c52, c84) : hredc(d[y], p, pv, c32);
                const uint32_t r1 = WIDE ? hredc2(dw[y + M], p, pv, c52, c84) : hredc(d[y + M], p, pv, c32);
                const uint32_t x0 = r0, x1 = mmul(r1, s0 < D1 ? k2 : k1, p, pv);
                v[i] = cmul(make_uint2(x0, x1), W, p, pv);
                if (i < 15) {
                    W = cmul(W, SD, p, pv);
                    if (s0 < D) W = make_uint2(madd(W.x, W.x, p), madd(W.y, W.y, p));
                    s0 = (s0 + (unsigned)L - D) & (unsigned)ul;
                }
            }
        }
        if (CM == 0) hr16f<false>(v, T, 64 * uq, jj + g * uq, p, pv);
#pragma unroll
        for (int i = 0; i < 16; ++i) { const unsigned k = hcsw(g + 64 * i, col); sx[k] = v[i].x; sy[k] = v[i].y; }
        __syncthreads();
        uint2 w[8];
        if (CM == 0) hpw8(w, T[32 * uq + jj + (g & 7) * uq], p, pv);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const unsigned gi = g + 64 * h, r0 = (gi >> 3) * 64 + (gi & 7);
            hc_rd8(sx, sy, v, r0, 8, col);
            if (CM == 0) hr8w<false>(v, T, w, p, pv);
            hc_wr8(sx, sy, v, r0, 8, col);
        }
        __syncthreads();
        if (CM == 0) hpw8(w, T[4 * uq + jj], p, pv);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const unsigned r0 = 8 * (g + 64 * h);
            hc_rd8(sx, sy, v, r0, 1, col);
            if (CM == 0) hr8w<false>(v, T, w, p, pv);
#pragma unroll
            for (int i = 0; i < 8; ++i) x[(size_t)(r0 + i) * q] = v[i];
        }
        __syncthreads();   // (the next tile writes the shared planes)
    }
}
// The column pass with 512 rows (9 levels) for the wide digits at 2^21, with k_hblock16 (12 levels) instead of
// k_hblock11: a tile of 8 columns x 512 rows = 32 KB, 256 threads, 2 blocks per SM, so the butterflies of one block
// overlap the loads of the other (1024 rows: 1 block per SM, the butterflies on top of the memory time). Stages:
// radix-16 rows g + 32i (hr16f, q = 32 uq; the pack as k_hcolf, the same step M/16), radix-4 rows r0 + 8i,
// r0 = (gi >> 3) 32 + (gi & 7) (as k_hdif4, q = 8 uq, j the same for the 4 groups of a thread), radix-8 rows 8 gi + i
// (hr8w) to memory; k_hcoli9 the same in reverse. The same radix-2 network, so the values bit for bit as the 1024-row
// pass + k_hblock11 (32 banks for every warp access with hcsw)
constexpr int HC9_LGR = 9, HC9_R = 1 << HC9_LGR, HC9_N = HC9_R * HC_C, HC9_T = HC9_N / 16;   // 256 threads
constexpr size_t HC9_SHM = 2 * HC9_N * sizeof(uint32_t);   // 32 KiB dynamic
// COLBENCH variants: CM 0 the kernel, 1 the memory + shared pattern alone, 2 the pack without the butterflies, 3 as 1
// without the __syncthreads (timing only, the shared data races), 4 the loads and the stores alone (no shared memory:
// rows g + 32i straight back); PF the L2 prefetch of the tile PF launches ahead (0 none, 1 the next tile - the kernel)
// TP: the twiddles of the next tile into the L2 too (T[256 uq + j], T[8/16/24 uq + j4], T[4 uq + jj]: the top levels,
// 16 MB a prime, not in the L2 - otherwise 5 scattered DRAM loads a thread at the start of the butterflies)
template <bool WIDE, int CM = 0, int PF = 1, int TP = 1> __global__ void __launch_bounds__(HC9_T, 2) k_hcolf9(uint2* a, const uint2* tw,
                                                                                                              int lgM, int np,
                                                                                                              const HP* hp,
                                                                                                              const int64_t* d, HPK pk) {
    constexpr bool NOAR = CM == 1 || CM == 3 || CM == 4;   // (no arithmetic)
    const longlong2* dw = reinterpret_cast<const longlong2*>(d);
    extern __shared__ uint32_t hcs[];
    uint32_t *sx = hcs, *sy = hcs + HC9_N;
    const size_t M = (size_t)1 << lgM, q = M >> HC9_LGR;
    const unsigned uq = (unsigned)q, per = uq / HC_C, unp = (unsigned)np, ntl = per * unp;
    const unsigned col = threadIdx.x % HC_C, g = threadIdx.x / HC_C;   // g < 32
    for (unsigned tile = blockIdx.x; tile < ntl; tile += gridDim.x) {
        const unsigned c = tile % unp, jj = (tile / unp) * HC_C + col, nt = tile + PF * gridDim.x;
        if (PF > 0 && nt < ntl) {
            const size_t nj = (nt / unp) * HC_C + col;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const size_t y = nj + (size_t)(g + 32 * i) * q;
                if (WIDE) { l2pf(dw + y); l2pf(dw + y + M); }
                else { l2pf(d + y); l2pf(d + y + M); }
            }
            if (TP && CM == 0) {
                const uint2* Tn = tw + (size_t)(nt % unp) * M + nj;
                const size_t j4 = (size_t)(g & 7) * q;
                l2pf(Tn + 256 * q + (size_t)g * q);
                l2pf(Tn + 8 * q + j4);
                l2pf(Tn + 16 * q + j4);
                l2pf(Tn + 24 * q + j4);
                l2pf(Tn + 4 * q);
            }
        }
        const uint32_t p = hp->p[c], pv = hp->pv[c];
        const uint2* T = tw + (size_t)c * M;
        uint2* x = a + (size_t)c * M + jj;
        uint2 v[16];
        {   // the pack of rows g + 32i = complex j + i M/16 (as k_hcolf, rows g + 64i there: the same step)
            const size_t L = M << 1, LH = L >> 11, MH = M >> 11, j = jj + (size_t)g * q, ul = L - 1;
            const unsigned D = (unsigned)(((M >> 4) * pk.ndm) & ul), D1 = (unsigned)((M * pk.ndm) & ul);
            unsigned s0 = (unsigned)((L - ((j * pk.ndm) & ul)) & ul);
            const uint32_t* TL = pk.tl + c * 2048;
            const uint32_t w0 = mmul(mmul(pk.th[c * LH + (s0 >> 11)], TL[s0 & 2047], p, pv), hp->r3[c], p, pv);
            const uint2 z = cmul(pk.pkh[c * MH + (j >> 11)], pk.pkl[c * 2048 + (j & 2047)], p, pv);
            const uint2 S = pk.pkh[c * MH + ((M >> 4) >> 11)];
            const uint32_t td = hp->cw[c], k1 = hp->ck[c], k2 = madd(k1, k1, p), c32 = mmul(hp->r2[c], 1, p, pv), c52 = hp->c52[c], c84 = hp->c84[c];
            const uint2 SD = make_uint2(mmul(S.x, td, p, pv), mmul(S.y, td, p, pv));
            uint2 W = make_uint2(mmul(z.x, w0, p, pv), mmul(z.y, w0, p, pv));
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const size_t y = j + (size_t)i * (M >> 4);
                if (NOAR) {   // (the same loads, no arithmetic)
                    v[i] = WIDE ? make_uint2((uint32_t)dw[y].x ^ (uint32_t)dw[y].y, (uint32_t)dw[y + M].x ^ (uint32_t)dw[y + M].y)
                                : make_uint2((uint32_t)d[y], (uint32_t)d[y + M]);
                    continue;
                }
                const uint32_t r0 = WIDE ? hredc2(dw[y], p, pv, c52, c84) : hredc(d[y], p, pv, c32);
                const uint32_t r1 = WIDE ? hredc2(dw[y + M], p, pv, c52, c84) : hredc(d[y + M], p, pv, c32);
                const uint32_t x0 = r0, x1 = mmul(r1, s0 < D1 ? k2 : k1, p, pv);
                v[i] = cmul(make_uint2(x0, x1), W, p, pv);
                if (i < 15) {
                    W = cmul(W, SD, p, pv);
                    if (s0 < D) W = make_uint2(madd(W.x, W.x, p), madd(W.y, W.y, p));
                    s0 = (s0 + (unsigned)L - D) & (unsigned)ul;
                }
            }
        }
        if (CM == 4) {   // (COLBENCH: the loads and the stores alone)
#pragma unroll
            for (int i = 0; i < 16; ++i) x[(size_t)(g + 32 * i) * q] = v[i];
            continue;
        }
        if (CM == 0) hr16f<false>(v, T, 32 * uq, jj + g * uq, p, pv);
#pragma unroll
        for (int i = 0; i < 16; ++i) { const unsigned k = hcsw(g + 32 * i, col); sx[k] = v[i].x; sy[k] = v[i].y; }
        if (CM != 3) __syncthreads();
        {   // radix-4: rows r0 + 8i, the levels 16 and 8 rows (as k_hdif4 with q = 8 uq)
            const unsigned q4 = 8 * uq, j4 = jj + (g & 7) * uq;
            uint2 t1 = make_uint2(0, 0), t2 = t1, t3 = t1;
            if (CM == 0) { t1 = T[q4 + j4]; t2 = T[2 * q4 + j4]; t3 = T[3 * q4 + j4]; }
#pragma unroll
            for (int h = 0; h < 4; ++h) {
                const unsigned gi = g + 32 * h, r0 = (gi >> 3) * 32 + (gi & 7);
                uint2 u[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) { const unsigned k = hcsw(r0 + 8 * i, col); u[i] = make_uint2(sx[k], sy[k]); }
                if (CM == 0) {
                    cdif(u[0], u[2], t2, p, pv);
                    cdif(u[1], u[3], t3, p, pv);
                    cdif(u[0], u[1], t1, p, pv);
                    cdif(u[2], u[3], t1, p, pv);
                }
#pragma unroll
                for (int i = 0; i < 4; ++i) { const unsigned k = hcsw(r0 + 8 * i, col); sx[k] = u[i].x; sy[k] = u[i].y; }
            }
        }
        if (CM != 3) __syncthreads();
        uint2 w[8];
        if (CM == 0) hpw8(w, T[4 * uq + jj], p, pv);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const unsigned r0 = 8 * (g + 32 * h);
            hc_rd8(sx, sy, v, r0, 1, col);
            if (CM == 0) hr8w<false>(v, T, w, p, pv);
#pragma unroll
            for (int i = 0; i < 8; ++i) x[(size_t)(r0 + i) * q] = v[i];
        }
        if (CM != 3) __syncthreads();   // (the next tile writes the shared planes)
    }
}
// gr0, ngr (LOTR_GPU_L2ORD=3): only the column groups gr0 .. gr0 + ngr - 1 (8 columns each; ngr 0 = all)
template <int TP = 1> __global__ void __launch_bounds__(HC9_T, 2) k_hcoli9(uint2* a, const uint2* itw, int lgM, int np,
                                                                           const HP* hp, const uint2* ucq, unsigned c0,
                                                                           unsigned gr0 = 0, unsigned ngr = 0, unsigned po = 0) {
    // po (LOTR_GPU_L2ORD=3): the prime as the OUTER tile index (one prime after the other: the coli of the chunks; 0 = inner)
    extern __shared__ uint32_t hcs[];
    uint32_t *sx = hcs, *sy = hcs + HC9_N;
    const size_t M = (size_t)1 << lgM, q = M >> HC9_LGR;
    const unsigned uq = (unsigned)q, per = ngr ? ngr : uq / HC_C, unp = (unsigned)np, ntl = per * unp;
    const unsigned col = threadIdx.x % HC_C, g = threadIdx.x / HC_C;   // g < 32
    for (unsigned tile = blockIdx.x; tile < ntl; tile += gridDim.x) {
        const unsigned nt = tile + gridDim.x;
        const unsigned c = c0 + (po ? tile / per : tile % unp), jj = (gr0 + (po ? tile % per : tile / unp)) * HC_C + col;
        if (nt < ntl) {
            const size_t nc = c0 + (po ? nt / per : nt % unp), nj = (gr0 + (po ? nt % per : nt / unp)) * HC_C + col;
#pragma unroll
            for (int i = 0; i < 16; ++i) l2pf(a + nc * M + nj + (size_t)(8 * (g + 32 * (i >> 3)) + (i & 7)) * q);
            if (TP) {   // (the twiddles and the column factor of the next tile, as k_hcolf9)
                const uint2* Tn = itw + nc * M + nj;
                const size_t j4 = (size_t)(g & 7) * q;
                l2pf(Tn + 4 * q);
                l2pf(Tn + 8 * q + j4);
                l2pf(Tn + 16 * q + j4);
                l2pf(Tn + 24 * q + j4);
                l2pf(Tn + 256 * q + (size_t)g * q);
                l2pf(ucq + nc * q + nj);
            }
        }
        const uint32_t p = hp->p[c], pv = hp->pv[c];
        const uint2* T = itw + (size_t)c * M;
        uint2* x = a + (size_t)c * M + jj;
        uint2 v[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) v[i] = x[(size_t)(8 * (g + 32 * (i >> 3)) + (i & 7)) * q];   // rows 8 gi + i
        uint2 w[8];
        const uint2 cf = ucq[c * uq + jj];
        hpw8c(w, T[4 * uq + jj], cf, p, pv);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            v[8 * h] = cmul(v[8 * h], w[0], p, pv);
            hr8w<true>(v + 8 * h, T, w, p, pv);
            hc_wr8(sx, sy, v + 8 * h, 8 * (g + 32 * h), 1, col);
        }
        __syncthreads();
        {   // radix-4 inverse: rows r0 + 8i, the levels 8 and 16 rows (as k_hdit4 with q = 8 uq)
            const unsigned q4 = 8 * uq, j4 = jj + (g & 7) * uq;
            const uint2 t1 = T[q4 + j4], t2 = T[2 * q4 + j4], t3 = T[3 * q4 + j4];
#pragma unroll
            for (int h = 0; h < 4; ++h) {
                const unsigned gi = g + 32 * h, r0 = (gi >> 3) * 32 + (gi & 7);
                uint2 u[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) { const unsigned k = hcsw(r0 + 8 * i, col); u[i] = make_uint2(sx[k], sy[k]); }
                cdit(u[0], u[1], t1, p, pv);
                cdit(u[2], u[3], t1, p, pv);
                cdit(u[0], u[2], t2, p, pv);
                cdit(u[1], u[3], t3, p, pv);
#pragma unroll
                for (int i = 0; i < 4; ++i) { const unsigned k = hcsw(r0 + 8 * i, col); sx[k] = u[i].x; sy[k] = u[i].y; }
            }
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 16; ++i) { const unsigned k = hcsw(g + 32 * i, col); v[i] = make_uint2(sx[k], sy[k]); }
        hdit16v(v, T, 32 * uq, jj + g * uq, p, pv);
#pragma unroll
        for (int i = 0; i < 16; ++i) x[(size_t)(g + 32 * i) * q] = v[i];
        __syncthreads();
    }
}
// UC: column jj times ucq[jj] (the column factor of the unpack, HybEngine::col; linear: the inputs instead of the
// outputs), folded into the powers of the first stage: w[0] = C, w[m] = C a^m (21 products instead of 15) + v[0],
// v[8]: 12 products per 16 values here, 3 per value less in k_hcrtE (DG 6)
// np primes from c0 (np = 1 for LOTR_GPU_L2ORD, else all primes from 0)
__global__ void __launch_bounds__(HC_T, 1) k_hcoli(uint2* a, const uint2* itw, int lgM, int np, const HP* hp,
                                                   const uint2* ucq, unsigned c0) {
    extern __shared__ uint32_t hcs[];
    uint32_t *sx = hcs, *sy = hcs + HC_N;
    const size_t M = (size_t)1 << lgM, q = M >> HC_LGR;
    const unsigned uq = (unsigned)q, per = uq / HC_C, unp = (unsigned)np, ntl = per * unp;
    const unsigned col = threadIdx.x % HC_C, g = threadIdx.x / HC_C;
    for (unsigned tile = blockIdx.x; tile < ntl; tile += gridDim.x) {
        const unsigned c = c0 + tile % unp, jj = (tile / unp) * HC_C + col, nt = tile + gridDim.x;
        if (nt < ntl) {
            const size_t nc = c0 + nt % unp, nj = (nt / unp) * HC_C + col;
#pragma unroll
            for (int i = 0; i < 16; ++i) l2pf(a + nc * M + nj + (size_t)(8 * (g + 64 * (i >> 3)) + (i & 7)) * q);
        }
        const uint32_t p = hp->p[c], pv = hp->pv[c];
        const uint2* T = itw + (size_t)c * M;
        uint2* x = a + (size_t)c * M + jj;
        uint2 v[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) v[i] = x[(size_t)(8 * (g + 64 * (i >> 3)) + (i & 7)) * q];   // rows 8 gi + i
        uint2 w[8];
        const uint2 cf = ucq[c * uq + jj];
        hpw8c(w, T[4 * uq + jj], cf, p, pv);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            v[8 * h] = cmul(v[8 * h], w[0], p, pv);
            hr8w<true>(v + 8 * h, T, w, p, pv);
            hc_wr8(sx, sy, v + 8 * h, 8 * (g + 64 * h), 1, col);
        }
        __syncthreads();
        hpw8(w, T[32 * uq + jj + (g & 7) * uq], p, pv);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const unsigned gi = g + 64 * h, r0 = (gi >> 3) * 64 + (gi & 7);
            hc_rd8(sx, sy, v, r0, 8, col);
            hr8w<true>(v, T, w, p, pv);
            hc_wr8(sx, sy, v, r0, 8, col);
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 16; ++i) { const unsigned k = hcsw(g + 64 * i, col); v[i] = make_uint2(sx[k], sy[k]); }
        hdit16v(v, T, 64 * uq, jj + g * uq, p, pv);
#pragma unroll
        for (int i = 0; i < 16; ++i) x[(size_t)(g + 64 * i) * q] = v[i];
        __syncthreads();
    }
}
// the lowest 12 levels of 4096 complex values per block (as k_block; shared memory as two planes, spad: no bank
// conflicts), MODE 0 forward + square + inverse, 1 forward only, 2 inverse only
// the lowest radix-8 group (q = 1, j = 0): the twiddles are 1, +-i (free: forward w_4 = theta^M = i, inverse -i), w_8
// and w_8^3 = (s, +-s) (a square root of +-i: 2 products instead of 3) - 4 products instead of 36
__device__ __forceinline__ uint2 cmul8(uint2 a, uint2 w, uint32_t p, uint32_t pv) {
    if (w.y == w.x) return make_uint2(mmul(msub(a.x, a.y, p), w.x, p, pv), mmul(madd(a.x, a.y, p), w.x, p, pv));
    return make_uint2(mmul(madd(a.x, a.y, p), w.x, p, pv), mmul(msub(a.y, a.x, p), w.x, p, pv));
}
__device__ __forceinline__ uint2 cmuli(uint2 a, bool neg, uint32_t p) {   // a * i, a * (-i) when neg
    return neg ? make_uint2(a.y, msub(0, a.x, p)) : make_uint2(msub(0, a.y, p), a.x);
}
template <bool INV> __device__ __forceinline__ uint2 hw8m(uint2 a, int i, uint2 w1, uint2 w3, uint32_t p, uint32_t pv) {
    return i == 0 ? a : i == 1 ? cmul8(a, w1, p, pv) : i == 2 ? cmuli(a, INV, p) : cmul8(a, w3, p, pv);   // a * w_8^i
}
template <bool INV> __device__ __forceinline__ void hr8low(uint2* v, const uint2* T, uint32_t p, uint32_t pv) {
    const uint2 w1 = T[5], w3 = T[7];
    if (!INV) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint2 u = v[i], x = v[i + 4];
            v[i] = cadd(u, x, p);
            v[i + 4] = hw8m<INV>(csub(u, x, p), i, w1, w3, p, pv);
        }
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int b = 0; b < 8; b += 4) {
                const uint2 u = v[b + i], x = v[b + i + 2], y = csub(u, x, p);
                v[b + i] = cadd(u, x, p);
                v[b + i + 2] = i ? cmuli(y, INV, p) : y;
            }
#pragma unroll
        for (int i = 0; i < 8; i += 2) {
            const uint2 u = v[i], x = v[i + 1];
            v[i] = cadd(u, x, p);
            v[i + 1] = csub(u, x, p);
        }
    }
    else {
#pragma unroll
        for (int i = 0; i < 8; i += 2) {
            const uint2 u = v[i], t = v[i + 1];
            v[i] = cadd(u, t, p);
            v[i + 1] = csub(u, t, p);
        }
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int b = 0; b < 8; b += 4) {
                const uint2 u = v[b + i], t = i ? cmuli(v[b + i + 2], INV, p) : v[b + i + 2];
                v[b + i] = cadd(u, t, p);
                v[b + i + 2] = csub(u, t, p);
            }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint2 u = v[i], t = hw8m<INV>(v[i + 4], i, w1, w3, p, pv);
            v[i] = cadd(u, t, p);
            v[i + 4] = csub(u, t, p);
        }
    }
}
// a radix-8 group at (q, j) (k_hblock and the column pass): hr8low (the group at j = 0) and then the output i times
// a^rev3(i), a = w_8q^j (inverse: the inputs first) = the same transform: 4 + 7*3 products instead of 12*3. a^m: powers
// of the one entry (hpw8, 15 products; ctw8 takes 18)
__device__ __forceinline__ void hpw8(uint2* w, uint2 x, uint32_t p, uint32_t pv) {   // w[m] = x^m, m = 1 .. 7 (15 products)
    w[1] = x;
    w[2] = csq(w[1], p, pv);
    w[4] = csq(w[2], p, pv);
    w[3] = cmul(w[1], w[2], p, pv);
    w[5] = cmul(w[1], w[4], p, pv);
    w[6] = csq(w[3], p, pv);
    w[7] = cmul(w[3], w[4], p, pv);
}
__device__ __forceinline__ void hpw8c(uint2* w, uint2 x, uint2 cf, uint32_t p, uint32_t pv) {   // w[m] = cf x^m, m = 0 .. 7
    w[0] = cf;
#pragma unroll
    for (int m = 1; m < 8; ++m) w[m] = cmul(w[m - 1], x, p, pv);
}
template <bool INV> __device__ __forceinline__ void hr8w(uint2* v, const uint2* T, const uint2* w, uint32_t p, uint32_t pv) {
    if (!INV) hr8low<false>(v, T, p, pv);   // (the powers w[m] = a^m made by hpw8)
#pragma unroll
    for (int i = 1; i < 8; ++i) v[i] = cmul(v[i], w[((i & 1) << 2) | (i & 2) | (i >> 2)], p, pv);
    if (INV) hr8low<true>(v, T, p, pv);
}
// k_hblock16: the same 12 levels as 3 x radix-16 (256 threads x 16 values: q = 256, 16, 1): 2 shared round trips and
// 2 __syncthreads per direction instead of 3, ~22 products per complex instead of 30, and no padding (swizzle hsw16),
// so 3 blocks of 32 KB fit on an SM (12288 values in flight instead of 8192). Equal to 12 radix-2 levels both ways;
// hsw16 is a bijection without bank conflicts for the 3 access patterns of a warp.
// hr16low: the group at j = 0 (level h = 8 with w16^i, the even i as w8^(i/2) by hw8m, then 2 x hr8low; 24 products)
template <bool INV> __device__ __forceinline__ void hr16low(uint2* v, const uint2* T, uint32_t p, uint32_t pv) {
    const uint2 w1 = T[5], w3 = T[7];
    if (INV) { hr8low<true>(v, T, p, pv); hr8low<true>(v + 8, T, p, pv); }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (!INV) {
            const uint2 u = v[i], x = v[i + 8], y = csub(u, x, p);
            v[i] = cadd(u, x, p);
            v[i + 8] = (i & 1) ? cmul(y, T[8 + i], p, pv) : hw8m<INV>(y, i >> 1, w1, w3, p, pv);
        }
        else {
            const uint2 u = v[i], t = (i & 1) ? cmul(v[i + 8], T[8 + i], p, pv) : hw8m<INV>(v[i + 8], i >> 1, w1, w3, p, pv);
            v[i] = cadd(u, t, p);
            v[i + 8] = csub(u, t, p);
        }
    }
    if (!INV) { hr8low<false>(v, T, p, pv); hr8low<false>(v + 8, T, p, pv); }
}
// a radix-16 group from a table (k_hblock16): W[(i - 1) s] = w_16q^(rev4(i) j) - by T[8q + ..] the entry of output i
// is at the stride rev4(i) * 8 B over the lanes (q = 256: ~32 sectors per warp load, 15 per stage), here contiguous
// (HybEngine::ht16: [(i - 1) 256 + j] for q = 256, [3840 + (i - 1) 16 + j] for q = 16, 4096 per prime)
template <bool INV> __device__ __forceinline__ void hr16t(uint2* v, const uint2* T, const uint2* W, unsigned s, uint32_t p,
                                                          uint32_t pv) {
    if (!INV) hr16low<false>(v, T, p, pv);
#pragma unroll
    for (int i = 1; i < 16; ++i) v[i] = cmul(v[i], W[(i - 1) * s], p, pv);
    if (INV) hr16low<true>(v, T, p, pv);
}
// the same from the one entry a = T[8q + j], its powers in turn (v[rev4(m)] x a^m, m = 1 .. 15; 2 registers, not 15
// loads: the column pass, where T[8q + ..] are the top levels, 16 MB per prime, not in L2): 24 + 14*3 + 15*3 products
template <bool INV> __device__ __forceinline__ void hr16f(uint2* v, const uint2* T, unsigned q, unsigned j, uint32_t p,
                                                          uint32_t pv) {
    const uint2 x = T[8 * q + j];
    if (!INV) hr16low<false>(v, T, p, pv);
    uint2 w = x;
#pragma unroll
    for (int m = 1; m < 16; ++m) {
        const int i = ((m & 1) << 3) | ((m & 2) << 1) | ((m & 4) >> 1) | (m >> 3);
        v[i] = cmul(v[i], w, p, pv);
        if (m < 15) w = cmul(w, x, p, pv);
    }
    if (INV) hr16low<true>(v, T, p, pv);
}
// bank = low 5 bits ^ (bits 5..8, bit 8 again at 4): rows r + 256i, (r >> 4) 256 + (r & 15) + 16i and 16r + i all hit
// 32 banks per warp
__device__ __forceinline__ unsigned hsw16(unsigned x) { return x ^ (((x >> 5) & 15) | ((x >> 4) & 16)); }
__device__ __forceinline__ void hs16_rd(const uint32_t* sx, const uint32_t* sy, uint2* v, unsigned b, unsigned s) {
#pragma unroll
    for (int i = 0; i < 16; ++i) { const unsigned k = hsw16(b + i * s); v[i] = make_uint2(sx[k], sy[k]); }
}
__device__ __forceinline__ void hs16_wr(uint32_t* sx, uint32_t* sy, const uint2* v, unsigned b, unsigned s) {
#pragma unroll
    for (int i = 0; i < 16; ++i) { const unsigned k = hsw16(b + i * s); sx[k] = v[i].x; sy[k] = v[i].y; }
}
// the twiddles of the stages q = 256 and 16 from ht / iht (hr16t);
// b0: the first block (one prime a launch for LOTR_GPU_L2ORD, else 0)
template <int MODE> __global__ void __launch_bounds__(256, 3) k_hblock16(uint2* a, const uint2* tw,
                                                                                  const uint2* itw, int lgM, const HP* hp,
                                                                                  unsigned pfd, const uint2* ht,
                                                                                  const uint2* iht, unsigned b0) {
    __shared__ uint32_t sx[4096], sy[4096];
    const size_t M = (size_t)1 << lgM;
    const unsigned nb = (unsigned)(M >> 12), bi = b0 + blockIdx.x;
    if (pfd && blockIdx.x + pfd < gridDim.x) {   // 256 lines of 128 B (as k_hblock)
        const unsigned bn = bi + pfd;
        l2pf(a + (size_t)(bn / nb) * M + ((size_t)(bn % nb) << 12) + threadIdx.x * 16);
    }
    const unsigned c = bi / nb, b = bi % nb;
    const uint32_t p = hp->p[c], pv = hp->pv[c];
    const uint2* T = tw + c * M;
    const uint2* IT = itw + c * M;
    uint2* g = a + c * M + ((size_t)b << 12);
    const unsigned r = threadIdx.x, j2 = r & 15, b2 = ((r >> 4) << 8) + j2;
    const uint2* HT = ht + c * 4096;
    const uint2* IHT = iht + c * 4096;
    uint2 v[16];
    if (MODE != 2) {
#pragma unroll
        for (int i = 0; i < 16; ++i) v[i] = g[r + 256 * i];
        hr16t<false>(v, T, HT + r, 256, p, pv);
        hs16_wr(sx, sy, v, r, 256);
        __syncthreads();
        hs16_rd(sx, sy, v, b2, 16);
        hr16t<false>(v, T, HT + 3840 + j2, 16, p, pv);
        hs16_wr(sx, sy, v, b2, 16);
        __syncthreads();
        hs16_rd(sx, sy, v, 16 * r, 1);
        hr16low<false>(v, T, p, pv);
    }
    if (MODE == 1) {
#pragma unroll
        for (int i = 0; i < 16; ++i) g[16 * r + i] = v[i];
        return;
    }
    if (MODE == 2) {
#pragma unroll
        for (int i = 0; i < 16; ++i) v[i] = g[16 * r + i];
    }
    if (MODE == 0) {
#pragma unroll
        for (int i = 0; i < 16; ++i) v[i] = csq(v[i], p, pv);
    }
    hr16low<true>(v, IT, p, pv);
    hs16_wr(sx, sy, v, 16 * r, 1);   // (the own 16 values of stage 3: no barrier needed before)
    __syncthreads();
    hs16_rd(sx, sy, v, b2, 16);
    hr16t<true>(v, IT, IHT + 3840 + j2, 16, p, pv);
    hs16_wr(sx, sy, v, b2, 16);
    __syncthreads();
    hs16_rd(sx, sy, v, r, 256);
    hr16t<true>(v, IT, IHT + r, 256, p, pv);
#pragma unroll
    for (int i = 0; i < 16; ++i) g[r + 256 * i] = v[i];
}
// k_hblock11 (the wide digits: 2^21 complex = 10 column levels + 11 here): 2048 values a transform as radix-16
// (q = 128), radix-16 (q = 8) and 2 x radix-8 (q = 1); 128 threads x 16 values, two transforms a block of 256 (32 KB, as
// k_hblock16). Tables as ht16: [(i - 1) 128 + j] for q = 128, [1920 + (i - 1) 8 + j] for q = 8 (2048 per prime).
// hsw11: bank = low 5 bits ^ bits 5..8 ^ (bits 7..8 at 3): the rows r + 128i, (r >> 3) 128 + (r & 7) + 8i and 16r + i
// of a warp all hit 32 banks (a bijection: only the low 5 bits change, from the bits above)
__device__ __forceinline__ unsigned hsw11(unsigned x) { return x ^ ((x >> 5) & 15) ^ (((x >> 7) & 3) << 3); }
__device__ __forceinline__ void hs11_rd(const uint32_t* sx, const uint32_t* sy, uint2* v, unsigned b, unsigned s) {
#pragma unroll
    for (int i = 0; i < 16; ++i) { const unsigned k = hsw11(b + i * s); v[i] = make_uint2(sx[k], sy[k]); }
}
__device__ __forceinline__ void hs11_wr(uint32_t* sx, uint32_t* sy, const uint2* v, unsigned b, unsigned s) {
#pragma unroll
    for (int i = 0; i < 16; ++i) { const unsigned k = hsw11(b + i * s); sx[k] = v[i].x; sy[k] = v[i].y; }
}
template <int MODE> __global__ void __launch_bounds__(256, 3) k_hblock11(uint2* a, const uint2* tw, const uint2* itw,
                                                                         int lgM, const HP* hp, unsigned pfd,
                                                                         const uint2* ht, const uint2* iht, unsigned b0) {
    __shared__ uint32_t sxa[4096], sya[4096];
    const size_t M = (size_t)1 << lgM;
    const unsigned nb = (unsigned)(M >> 11), hb = threadIdx.x >> 7, r = threadIdx.x & 127, tb = (b0 + blockIdx.x) * 2 + hb;
    if (pfd && blockIdx.x + pfd < gridDim.x) {   // 2 x 2048 values = 2 x 128 lines of 128 B
        const unsigned bn = (b0 + blockIdx.x + pfd) * 2 + hb;
        l2pf(a + (size_t)(bn / nb) * M + ((size_t)(bn % nb) << 11) + r * 16);
    }
    uint32_t *sx = sxa + hb * 2048, *sy = sya + hb * 2048;
    const unsigned c = tb / nb, b = tb % nb;
    const uint32_t p = hp->p[c], pv = hp->pv[c];
    const uint2* T = tw + c * M;
    const uint2* IT = itw + c * M;
    uint2* g = a + c * M + ((size_t)b << 11);
    const unsigned j2 = r & 7, b2 = ((r >> 3) << 7) + j2;
    const uint2* HT = ht + c * 2048;
    const uint2* IHT = iht + c * 2048;
    uint2 v[16];
    if (MODE != 2) {
#pragma unroll
        for (int i = 0; i < 16; ++i) v[i] = g[r + 128 * i];
        hr16t<false>(v, T, HT + r, 128, p, pv);
        hs11_wr(sx, sy, v, r, 128);
        __syncthreads();
        hs11_rd(sx, sy, v, b2, 8);
        hr16t<false>(v, T, HT + 1920 + j2, 8, p, pv);
        hs11_wr(sx, sy, v, b2, 8);
        __syncthreads();
        hs11_rd(sx, sy, v, 16 * r, 1);
        hr8low<false>(v, T, p, pv);
        hr8low<false>(v + 8, T, p, pv);
    }
    if (MODE == 1) {
#pragma unroll
        for (int i = 0; i < 16; ++i) g[16 * r + i] = v[i];
        return;
    }
    if (MODE == 2) {
#pragma unroll
        for (int i = 0; i < 16; ++i) v[i] = g[16 * r + i];
    }
    if (MODE == 0) {
#pragma unroll
        for (int i = 0; i < 16; ++i) v[i] = csq(v[i], p, pv);
    }
    hr8low<true>(v, IT, p, pv);
    hr8low<true>(v + 8, IT, p, pv);
    hs11_wr(sx, sy, v, 16 * r, 1);   // (the own 16 values of stage 3: no barrier needed before)
    __syncthreads();
    hs11_rd(sx, sy, v, b2, 8);
    hr16t<true>(v, IT, IHT + 1920 + j2, 8, p, pv);
    hs11_wr(sx, sy, v, b2, 8);
    __syncthreads();
    hs11_rd(sx, sy, v, r, 128);
    hr16t<true>(v, IT, IHT + r, 128, p, pv);
#pragma unroll
    for (int i = 0; i < 16; ++i) g[r + 128 * i] = v[i];
}
__global__ void k_hpmul(uint2* a, const uint2* b, int lgM, int np, const HP* hp) {   // a*b / R (as k_pmul)
    const size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (t >= ((size_t)np << lgM)) return;
    const unsigned c = (unsigned)(t >> lgM);
    a[t] = cmul(a[t], b[t], hp->p[c], hp->pv[c]);
}
// unpack + CRT of complex position j (the unpack fused into the CRT): w_j (re) and w_(j+M) (im) = -k * c / a (untwist,
// unweight, k / M and the R of the square; HUK tables), Garner over the np primes
__device__ __forceinline__ uint32_t mhalf(uint32_t x, uint32_t p) { return (x + ((x & 1) ? p : 0)) >> 1; }   // x / 2 mod p
// UW: the im weight from the re one, F sigma^-M tau^-s1 = (F tau^-s0) kappa', kappa' = hp->cu (/ 2 when s0 < D1).
// UB: j = jb + t (jb = j & ~255): F tau^-s0 = F tau^-s0(jb) tau^(r_t) (/ 2 when s0(jb) < r_t, tau^L = 2),
// r_t = t (n+d) mod L: the twist and the weight from ub[jb >> 8] (one address per block of k_hcrtE) x uq[t] (contiguous);
// 7 products per prime instead of 10. UM: 2 = UB, 3 = UC (the split at q = 2^lgq, the column factor already applied by
// k_hcoli<.., UC>: only ucb[j >> lgq], one product less per prime), 4 = the full table uf (HybEngine::uf, the UB values:
// == UB bit for bit). The Garner digits t0 (re), t1 (im) are kept apart from the sum (also for the wide digits' 4-word
// sum); GARNER false: only the unpacked residues u_i, for the CRT of hcrtf
template <int NP, int UM = 2, bool GARNER = true> __device__ __forceinline__ void hgarner(const uint2* a, size_t j,
                                                                                         const HP* hp, int lgL,
                                                                                         const HUK& uk, uint32_t* t0,
                                                                                         uint32_t* t1) {
    const size_t L = (size_t)1 << lgL, M = L >> 1;
    const size_t s0 = (L - ((j * uk.ndm) & (L - 1))) & (L - 1);
    const bool hv = s0 < ((M * uk.ndm) & (L - 1));
    const uint32_t ul = (uint32_t)(L - 1), nm = (uint32_t)uk.ndm;   // (UB: mod L in 32 bits, L <= 2^32)
    const size_t sm = UM == 3 ? ((size_t)1 << uk.lgq) - 1 : 255;
    const uint32_t sb = (0u - (uint32_t)(j & ~sm) * nm) & ul, rt = ((uint32_t)(j & sm) * nm) & ul;
    const bool hw = sb < rt;
#pragma unroll
    for (int i = 0; i < NP; ++i) {
        const uint32_t p = hp->p[i], pv = hp->pv[i];
        uint2 z;
        if (UM == 4) z = a[i * M + j];
        else if (UM == 3) z = cmul(a[i * M + j], uk.ucb[i * (M >> uk.lgq) + (j >> uk.lgq)], p, pv);
        else {
            const uint2 tw = cmul(uk.ub[i * (M >> 8) + (j >> 8)], uk.uq[i * 256 + (j & 255)], p, pv);
            z = cmul(a[i * M + j], tw, p, pv);
        }
        uint32_t u0, u1;
        if (UM == 4) {
            const uint4 f = uk.uf[i * M + j];
            u0 = hdot(z.x, f.x, z.y, f.y, p, pv);
            u1 = hdot(z.x, f.z, z.y, f.w, p, pv);
        }
        else {
            u0 = hw ? mhalf(z.x, p) : z.x;
            u1 = mmul(z.y, hp->cu[i], p, pv);
            if (hw) u1 = mhalf(u1, p);
            if (hv) u1 = mhalf(u1, p);
        }
#pragma unroll
        for (int jj = 0; jj < (GARNER ? i : 0); ++jj) {
            uint32_t x0 = t0[jj], x1 = t1[jj];
            while (x0 >= p) x0 -= p;
            while (x1 >= p) x1 -= p;
            u0 = mmul(msub(u0, x0, p), hp->inv[i][jj], p, pv);
            u1 = mmul(msub(u1, x1, p), hp->inv[i][jj], p, pv);
        }
        t0[i] = u0;
        t1[i] = u1;
    }
}
// the CRT by a fractional estimate (instead of Garner's NP(NP-1)/2 steps and the Horner sum): y_i = u_i c_i mod p_i
// (c_i = (P/p_i)^-1 mod p_i), w = sum y_i P/p_i - k P with k = round(sum y_i / p_i) in double: w comes centered
// (|w| / P far below 0.5, the double error ~2^-49; exact up to |w| = 0.49 P); 32 x 32 -> 64 products over NL words
// of 32 bits; x = w in 4 words of 64 bits (two's complement)
template <int NP> __device__ __forceinline__ void hcrtf(const uint32_t* u, const HP* hp, const HF& F, uint64_t* x) {
    constexpr int NL = (31 * NP + 3) / 32 + 1;   // sum y_i P/p_i < NP P < 2^(31 NP + 3)
    uint32_t y[NP], acc[NL];
    double f = 0;
#pragma unroll
    for (int i = 0; i < NP; ++i) {
        y[i] = mmul(u[i], F.ci[i], hp->p[i], hp->pv[i]);
        f = fma((double)y[i], F.ip[i], f);
    }
#pragma unroll
    for (int l = 0; l < NL; ++l) acc[l] = 0;
#pragma unroll
    for (int i = 0; i < NP; ++i) {
        uint64_t c = 0;
#pragma unroll
        for (int l = 0; l < NL; ++l) {
            const uint64_t t = (uint64_t)y[i] * F.Q[i][l] + acc[l] + c;
            acc[l] = (uint32_t)t;
            c = t >> 32;
        }
    }
    const uint32_t kq = __double2uint_rn(f);
    uint32_t c = 0, bw = 0;
#pragma unroll
    for (int l = 0; l < NL; ++l) {   // acc -= kq P
        const uint64_t m = (uint64_t)kq * F.P[l] + c;
        c = (uint32_t)(m >> 32);
        const uint64_t dd = (uint64_t)acc[l] - (uint32_t)m - bw;
        acc[l] = (uint32_t)dd;
        bw = (uint32_t)(dd >> 63);
    }
    const uint32_t sg = (acc[NL - 1] >> 31) ? 0xFFFFFFFFu : 0;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint32_t lo = 2 * l < NL ? acc[2 * l] : sg, hi = 2 * l + 1 < NL ? acc[2 * l + 1] : sg;
        x[l] = ((uint64_t)hi << 32) | lo;
    }
}
template <int NP, int UM = 2> __device__ __forceinline__ void hw2(const uint2* a, size_t j, const HP* hp, int lgL,
                                                                  const HUK& uk, u128 P, i128& w0, i128& w1) {
    uint32_t t0[NP], t1[NP];
    hgarner<NP, UM>(a, j, hp, lgL, uk, t0, t1);
    u128 x0 = t0[NP - 1], x1 = t1[NP - 1];
#pragma unroll
    for (int i = NP - 2; i >= 0; --i) { x0 = x0 * hp->p[i] + t0[i]; x1 = x1 * hp->p[i] + t1[i]; }
    w0 = (x0 > (P >> 1)) ? -(i128)(P - x0) : (i128)x0;
    w1 = (x1 > (P >> 1)) ? -(i128)(P - x1) : (i128)x1;
}
// part u of w_t: the bits of the digit t+u (the last part, u = H-1: everything above, signed). |w| < 2^126: a shift of
// 127 gives the sign bits, as any longer one would.
__device__ __forceinline__ int64_t hpart(i128 w, uint64_t t, int u, int H, uint64_t nd, int lgL) {
    const uint64_t eu = hpos(t + u, nd, lgL);
    const uint64_t s = eu - hpos(t, nd, lgL);
    const int sh = (int)(s < 127 ? s : 127);
    if (u == H - 1) return (int64_t)(w >> sh);
    const int b = (int)(hpos(t + u + 1, nd, lgL) - eu);
    return (int64_t)((w >> sh) & (((i128)1 << b) - 1));
}
// the same from the two 64-bit words of w (hi:lo, two's complement), not i128 shifts:
// top = (int64)(w >> sh), else the low b bits of w >> sh (b < 64), 0 <= sh <= 127
__device__ __forceinline__ int64_t hbits(uint64_t lo, int64_t hi, int sh, int b, bool top) {
    const uint64_t r = sh >= 64 ? (uint64_t)(hi >> (sh - 64)) : sh == 0 ? lo : (lo >> sh) | ((uint64_t)hi << (64 - sh));
    return top ? (int64_t)r : (int64_t)(r & ((1ull << b) - 1));
}
// the wrap: Wtop = sum Pw[v] 2^(e_v) at 2^(n+d); Wtop*2^d = Q*k + R by long division over the positions v = H-2 .. 0
// -> q_v (subtracted at the digit v), R*2^(n - e_(L-1)) (added to the top digit)
// floor(a / k), r = a - q*k in [0, k): 16-bit chunks from the top, rem*2^16 + chunk < 2^64 (k < 2^48), the quotient of a
// chunk by Barrett (kinv = floor((2^64-1) / k), at most 2 corrections; a bit loop is much slower).
__device__ i128 hfloordiv(i128 a, uint64_t k, uint64_t kinv, i128& r) {
    const bool neg = a < 0;
    const u128 x = neg ? (u128)(-a) : (u128)a;
    int top = 128;
    while (top > 0 && (x >> (top - 16)) == 0) top -= 16;
    u128 q = 0;
    uint64_t rr = 0;
    for (int s = top - 16; s >= 0; s -= 16) {
        const uint64_t cur = (rr << 16) | (uint64_t)((x >> s) & 0xFFFF);
        uint64_t qq = __umul64hi(cur, kinv);
        uint64_t rm = cur - qq * k;
        while (rm >= k) { rm -= k; ++qq; }
        rr = rm;
        q = (q << 16) | qq;
    }
    if (!neg) { r = (i128)rr; return (i128)q; }
    if (rr == 0) { r = 0; return -(i128)q; }
    r = (i128)(k - rr);
    return -(i128)q - 1;
}
// unpack + CRT + carry in one kernel (E does not go through memory). A block = the complex positions j = mb .. mb+255:
// the runs of digits mb.. (re, row 0) and M+mb.. (im, row 1). w with hs = H-1+npass halo positions below each run (the
// extra warp, in parallel; the halo of the im run of block 0 is the top of the re run, the halo of the re run of block 0
// the top of the im run = the wrap), E for the run and npass halo digits (E_m = sum of the parts landing at m), the fold
// in block 0 (-q_v at the digits v < H-1) and in the last block (R*2^shn at the top digit), both from the wrap parts
// Pw[v] of their own w (positions L+v: the same division in both), then npass neighbour passes in shared memory: after
// pass ps the entries e >= ps are right. The top digit L-1 keeps its carry. out = the final digits (the same values as
// E + fold + global passes).
// pfd > 0: the input of the block pfd further into L2 first (as k_hblock: the block phases leave DRAM idle)
// DG: 5 = hw2 UB, 6 = hw2 UC (after k_hcoli), 7 = hw2 with the full table uf
// (the body as a device function: the kernels below differ only in __launch_bounds__ - a min-blocks argument on all of
// them changed the register choice of the others)
// MT 1 (LOTR_GPU_CRTF=1): the CRT of hcrtf instead of Garner (|w| < 2^126: the low two words)
template <int NP, int DG, int MT> __device__ __forceinline__ void hcrtE_body(const uint2* a, int64_t* out, const HP* hp,
                                                                             int lgL, uint64_t nd, int H, int npass,
                                                                             uint64_t P_lo, uint64_t P_hi, const HF& F,
                                                                             HUK uk, uint64_t k, uint64_t kinv, int d,
                                                                             int shn, unsigned pfd) {
    __shared__ uint64_t slo[2][256 + 32];   // w of the digit (run base) - hs + i
    __shared__ int64_t shi[2][256 + 32];
    __shared__ int64_t sv[2][256 + 16];     // S of the digit (run base) - npass + e
    __shared__ uint64_t sp[2][256 + 32];   // e_x of the digit (run base) - hs + i, i <= 256 + hs (once, not
                                            // 6 hpos per digit in the parts and 6 in the passes)
    __shared__ int64_t sq[8];               // the fold: q_v (v < H-1), R*2^shn at [H-1]
    const size_t L = (size_t)1 << lgL, M = L >> 1, mb = (size_t)blockIdx.x * 256;
    if (pfd && threadIdx.x < 16 * NP && blockIdx.x + pfd < gridDim.x)   // NP runs of 2 KB = 16 lines of 128 B each
        l2pf(a + (threadIdx.x >> 4) * M + (size_t)(blockIdx.x + pfd) * 256 + (threadIdx.x & 15) * 16);
    const u128 P = ((u128)P_hi << 64) | P_lo;
    const unsigned tid = threadIdx.x, h1 = (unsigned)H - 1, pn = (unsigned)npass, hs = h1 + pn, ne = 256 + pn;
    const bool first = (mb == 0), last = (mb + 256 >= M);
    auto w2 = [&](size_t j, i128& x0, i128& x1) {
        constexpr int UM = DG == 7 ? 4 : DG == 6 ? 3 : 2;
        if (MT == 0) hw2<NP, UM>(a, j, hp, lgL, uk, P, x0, x1);
        else {
            uint32_t t0[NP], t1[NP];
            uint64_t y0[4], y1[4];
            hgarner<NP, UM, false>(a, j, hp, lgL, uk, t0, t1);
            hcrtf<NP>(t0, hp, F, y0);
            hcrtf<NP>(t1, hp, F, y1);
            x0 = (i128)(((u128)y0[1] << 64) | y0[0]);
            x1 = (i128)(((u128)y1[1] << 64) | y1[0]);
        }
    };
    if (tid < 256) {
        i128 w0, w1;
        w2(mb + tid, w0, w1);
        slo[0][hs + tid] = (uint64_t)w0; shi[0][hs + tid] = (int64_t)(w0 >> 64);
        slo[1][hs + tid] = (uint64_t)w1; shi[1][hs + tid] = (int64_t)(w1 >> 64);
    }
    else if (tid - 256 < hs) {   // halo digit mb - hs + i (re), M + mb - hs + i (im)
        const unsigned i = tid - 256;
        const long long jh = (long long)mb - hs + i;
        i128 w0, w1;
        if (jh >= 0) w2((size_t)jh, w0, w1);
        else w2((size_t)((long long)M + jh), w1, w0);   // block 0: re -> im run, im (L + jh) -> re run
        slo[0][i] = (uint64_t)w0; shi[0][i] = (int64_t)(w0 >> 64);
        slo[1][i] = (uint64_t)w1; shi[1][i] = (int64_t)(w1 >> 64);
    }
    const unsigned np1 = 257 + hs;   // the positions: digit (run base) - hs + i (below 0: unused)
    for (unsigned x = tid; x < 2 * np1; x += 288) {
        const unsigned hf = x >= np1 ? 1 : 0, i = x - hf * np1;
        const long long t = (long long)(hf * M + mb) - (long long)hs + i;
        sp[hf][i] = t >= 0 ? hpos((uint64_t)t, nd, lgL) : 0;
    }
    __syncthreads();
    for (unsigned x = tid; x < 2 * ne; x += 288) {   // the part of w_(m-u) at digit m: bits e_m - e_(m-u) .. + b_m (u = H-1: all above)
        const unsigned hf = x >= ne ? 1 : 0, e = x - hf * ne;
        const long long m = (long long)(hf * M + mb) - (long long)pn + e;
        int64_t acc = 0;
        if (m >= 0) {   // (block 0, re run: no digits below 0, the wrap goes through the fold)
            const uint64_t em = sp[hf][e + h1];   // (digit m = run base - hs + e + h1)
            const int bm = (int)(sp[hf][e + h1 + 1] - em);
            for (int u = 0; u < H && u <= m; ++u) {
                const unsigned i = e + h1 - u;
                const uint64_t s = em - sp[hf][i];
                acc += hbits(slo[hf][i], shi[hf][i], (int)(s < 127 ? s : 127), bm, u == H - 1);
            }
        }
        sv[hf][e] = acc;
    }
    if ((first || last) && tid == 0) {   // Pw[v] = the parts of w_t, t = L+v-u (u = v+1 .. H-1), at position L+v
        i128 rem = 0;
        for (int v = H - 2; v >= 0; --v) {
            int64_t pw = 0;
            for (int u = v + 1; u < H; ++u) {
                const size_t t = L + v - u;
                const unsigned r = first ? 0 : 1, i = first ? (unsigned)(t - (L - hs)) : (unsigned)(t - (M + mb) + hs);
                const i128 w = (i128)(((u128)(uint64_t)shi[r][i] << 64) | slo[r][i]);
                pw += hpart(w, t, u, H, nd, lgL);
            }
            const int bv = (int)(hpos(v + 1, nd, lgL) - hpos(v, nd, lgL));
            const i128 cur = rem * ((i128)1 << bv) + (i128)pw * ((i128)1 << d);
            i128 rr;
            sq[v] = (int64_t)hfloordiv(cur, k, kinv, rr);
            rem = rr;
        }
        sq[h1] = (int64_t)(rem * ((i128)1 << shn));
    }
    __syncthreads();
    if (first && tid < h1) sv[0][pn + tid] -= sq[tid];
    if (last && tid == 0) sv[1][pn + 255] += sq[h1];   // the digit L-1
    __syncthreads();
    for (unsigned ps = 1; ps <= pn; ++ps) {
        int64_t nv[2] = { 0, 0 };
        bool ok[2] = { false, false };
        for (unsigned r = 0; r < 2; ++r) {
            const unsigned x = tid + 288 * r;
            if (x >= 2 * ne) continue;
            const unsigned hf = x >= ne ? 1 : 0, e = x - hf * ne;
            const long long m = (long long)(hf * M + mb) - (long long)pn + e;
            if (e < ps || m < 0) continue;
            int64_t g, gp;
            const unsigned ki = e + h1;   // sp index of digit m
            const int bx = (int)(sp[hf][ki + 1] - sp[hf][ki]);
            nv[r] = ((size_t)m == L - 1) ? sv[hf][e] : bal(sv[hf][e], bx, g);
            if (m >= 1) {
                const int bp = (int)(sp[hf][ki] - sp[hf][ki - 1]);
                bal(sv[hf][e - 1], bp, gp);
                nv[r] += gp;
            }
            ok[r] = true;
        }
        __syncthreads();
        for (unsigned r = 0; r < 2; ++r) {
            const unsigned x = tid + 288 * r;
            if (ok[r]) sv[x >= ne ? 1 : 0][x - (x >= ne ? ne : 0)] = nv[r];
        }
        __syncthreads();
    }
    if (tid < 256) {
        out[mb + tid] = sv[0][pn + tid];
        out[M + mb + tid] = sv[1][pn + tid];
    }
}
template <int NP, int DG = 5, int MT = 0> __global__ void __launch_bounds__(288) k_hcrtE(const uint2* a, int64_t* out,
                                                                                       const HP* hp, int lgL, uint64_t nd,
                                                                                       int H, int npass, uint64_t P_lo,
                                                                                       uint64_t P_hi, HF F, HUK uk,
                                                                                       uint64_t k, uint64_t kinv, int d,
                                                                                       int shn, unsigned pfd) {
    hcrtE_body<NP, DG, MT>(a, out, hp, lgL, nd, H, npass, P_lo, P_hi, F, uk, k, kinv, d, shn, pfd);
}

// ---- WIDE DIGITS (100M: 2^22 digits of ~80 bits and 6-7 primes instead of 2^23 of 40 bits and 4): w in 4 words of
// 64 bits (two's complement, |w| < 2^217), the parts, E, the fold and the passes in i128 (E ~2^82, the fold ~2^98 at
// k ~ 2^17), the digits 16 B (lo, hi; 100M: |d| <= 2^79 + 1, the top < 2^84; B up to 96 bits, the top up to Dt < 2^112,
// hredc2 takes |v| < 2^115)
// x = sum t_i (p_0 .. p_(i-1)) by Horner from the top, then the nearer of x and x - P: x - P when x + (x - P) >= 0
template <int NP> __device__ __forceinline__ void hgsum4(const uint32_t* t, const HP* hp, const H4& P, uint64_t* x) {
    x[0] = t[NP - 1];
    x[1] = x[2] = x[3] = 0;
#pragma unroll
    for (int i = NP - 2; i >= 0; --i) {
        const uint64_t m = hp->p[i];
        uint64_t c = t[i];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint64_t lo = x[l] * m, hi = __umul64hi(x[l], m), s = lo + c;
            c = hi + (s < lo);
            x[l] = s;
        }
    }
    uint64_t y[4], br = 0;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint64_t dd = x[l] - P.w[l], b1 = x[l] < P.w[l];
        y[l] = dd - br;
        br = b1 | (dd < br);
    }
    uint64_t cy = 0, top = 0;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint64_t s = x[l] + y[l], s2 = s + cy;
        cy = (s < x[l]) | (s2 < s);
        top = s2;
    }
    if ((int64_t)top >= 0) {
#pragma unroll
        for (int l = 0; l < 4; ++l) x[l] = y[l];
    }
}
// bits sh .. sh+b-1 of w (b < 128), or with top everything from sh up, signed (|w >> sh| < 2^127); 0 <= sh <= 255
__device__ __forceinline__ i128 hbits4(const uint64_t* w, unsigned sh, int b, bool top) {
    const uint64_t sg = (uint64_t)((int64_t)w[3] >> 63);
    const unsigned li = sh >> 6, bt = sh & 63;
    uint64_t z[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) z[i] = li + i < 4 ? w[li + i] : sg;
    const uint64_t r0 = bt ? (z[0] >> bt) | (z[1] << (64 - bt)) : z[0];
    const uint64_t r1 = bt ? (z[1] >> bt) | (z[2] << (64 - bt)) : z[1];
    const u128 r = ((u128)r1 << 64) | r0;
    return top ? (i128)r : (i128)(r & (((u128)1 << b) - 1));
}
__device__ __forceinline__ i128 hpart4(const uint64_t* w, uint64_t t, int u, int H, uint64_t nd, int lgL) {
    const uint64_t eu = hpos(t + u, nd, lgL), s = eu - hpos(t, nd, lgL);
    const unsigned sh = (unsigned)(s < 255 ? s : 255);
    if (u == H - 1) return hbits4(w, sh, 0, true);
    return hbits4(w, sh, (int)(hpos(t + u + 1, nd, lgL) - eu), false);
}
__device__ __forceinline__ i128 bal128(i128 x, int b, i128& g) {
    const i128 half = (i128)1 << (b - 1);
    const i128 v = ((x + half) & (((i128)1 << b) - 1)) - half;
    g = (x - v) >> b;
    return v;
}
// k_hcrtE for the wide digits: the same phases, w in 4 words; MT 0: Garner + hgsum4, 1: hcrtf; UM as hw2 (3 = UC after
// k_hcoli, 4 = the table uf, 2 = UB)
// CM (COLBENCH only): 1 the loads of the 7 primes and the stores alone, 2 + the unpack and the CRT (no parts, no carry
// passes: w straight out), 0 the kernel
template <int NP, int MT, int UM = 3, int CM = 0> __global__ void __launch_bounds__(288) k_hcrtW(const uint2* a, longlong2* out, const HP* hp,
                                                                         int lgL, uint64_t nd, int H, int npass, H4 P,
                                                                         HF F, HUK uk, uint64_t k, uint64_t kinv, int d,
                                                                         int shn, unsigned pfd, unsigned mq = 0,
                                                                         unsigned ms0 = 0, unsigned mns = 0) {
    __shared__ uint64_t sw[2][256 + 32][4];   // w of the digit (run base) - hs + i
    __shared__ i128 sv[2][256 + 16];          // S of the digit (run base) - npass + e
    __shared__ uint64_t sp[2][256 + 32];      // e_x of the digit (run base) - hs + i
    __shared__ i128 sq[8];                    // the fold: q_v (v < H-1), R*2^shn at [H-1]
    // block b -> run base mb; mns = 0: b x 256 (all positions), else the segments ms0 .. ms0 + mns - 1 (256 positions each)
    // of every row of mq positions (the coli -> CRT chunks through the L2, LOTR_GPU_L2ORD=3)
    auto mapb = [&](unsigned b) -> size_t {
        return mns ? (size_t)(b / mns) * mq + (size_t)(ms0 + b % mns) * 256 : (size_t)b * 256;
    };
    const size_t L = (size_t)1 << lgL, M = L >> 1, mb = mapb(blockIdx.x);
    if (pfd && threadIdx.x < 16 * NP && blockIdx.x + pfd < gridDim.x)
        l2pf(a + (threadIdx.x >> 4) * M + mapb(blockIdx.x + pfd) + (threadIdx.x & 15) * 16);
    const unsigned tid = threadIdx.x, h1 = (unsigned)H - 1, pn = (unsigned)npass, hs = h1 + pn, ne = 256 + pn;
    const bool first = (mb == 0), last = (mb + 256 >= M);
    auto w2 = [&](size_t j, uint64_t* x0, uint64_t* x1) {
        if (CM == 1) {   // (the loads alone)
            uint64_t s0 = 0, s1 = 0;
#pragma unroll
            for (int i = 0; i < NP; ++i) { const uint2 z = a[i * M + j]; s0 ^= z.x; s1 ^= z.y; }
            x0[0] = s0; x0[1] = x0[2] = x0[3] = 0;
            x1[0] = s1; x1[1] = x1[2] = x1[3] = 0;
            return;
        }
        uint32_t t0[NP], t1[NP];
        hgarner<NP, UM, MT == 0>(a, j, hp, lgL, uk, t0, t1);
        if (MT == 0) { hgsum4<NP>(t0, hp, P, x0); hgsum4<NP>(t1, hp, P, x1); }
        else { hcrtf<NP>(t0, hp, F, x0); hcrtf<NP>(t1, hp, F, x1); }
    };
    if (tid < 256) w2(mb + tid, sw[0][hs + tid], sw[1][hs + tid]);
    else if (tid - 256 < hs) {   // halo digit mb - hs + i (re), M + mb - hs + i (im)
        const unsigned i = tid - 256;
        const long long jh = (long long)mb - hs + i;
        if (jh >= 0) w2((size_t)jh, sw[0][i], sw[1][i]);
        else w2((size_t)((long long)M + jh), sw[1][i], sw[0][i]);   // block 0: re -> im run, im (L + jh) -> re run
    }
    if (CM != 0) {   // (COLBENCH: w straight out, no parts and no carry passes)
        __syncthreads();
        if (tid < 256) {
            out[mb + tid] = make_longlong2((long long)sw[0][hs + tid][0], (long long)sw[0][hs + tid][1]);
            out[M + mb + tid] = make_longlong2((long long)sw[1][hs + tid][0], (long long)sw[1][hs + tid][1]);
        }
        return;
    }
    const unsigned np1 = 257 + hs;
    for (unsigned x = tid; x < 2 * np1; x += 288) {
        const unsigned hf = x >= np1 ? 1 : 0, i = x - hf * np1;
        const long long t = (long long)(hf * M + mb) - (long long)hs + i;
        sp[hf][i] = t >= 0 ? hpos((uint64_t)t, nd, lgL) : 0;
    }
    __syncthreads();
    for (unsigned x = tid; x < 2 * ne; x += 288) {
        const unsigned hf = x >= ne ? 1 : 0, e = x - hf * ne;
        const long long m = (long long)(hf * M + mb) - (long long)pn + e;
        i128 acc = 0;
        if (m >= 0) {
            const uint64_t em = sp[hf][e + h1];
            const int bm = (int)(sp[hf][e + h1 + 1] - em);
            for (int u = 0; u < H && u <= m; ++u) {
                const unsigned i = e + h1 - u;
                const uint64_t s = em - sp[hf][i];
                acc += hbits4(sw[hf][i], (unsigned)(s < 255 ? s : 255), bm, u == H - 1);
            }
        }
        sv[hf][e] = acc;
    }
    if ((first || last) && tid == 0) {
        i128 rem = 0;
        for (int v = H - 2; v >= 0; --v) {
            i128 pw = 0;
            for (int u = v + 1; u < H; ++u) {
                const size_t t = L + v - u;
                const unsigned r = first ? 0 : 1, i = first ? (unsigned)(t - (L - hs)) : (unsigned)(t - (M + mb) + hs);
                pw += hpart4(sw[r][i], t, u, H, nd, lgL);
            }
            const int bv = (int)(hpos(v + 1, nd, lgL) - hpos(v, nd, lgL));
            const i128 cur = rem * ((i128)1 << bv) + pw * ((i128)1 << d);
            i128 rr;
            sq[v] = hfloordiv(cur, k, kinv, rr);
            rem = rr;
        }
        sq[h1] = rem * ((i128)1 << shn);
    }
    __syncthreads();
    if (first && tid < h1) sv[0][pn + tid] -= sq[tid];
    if (last && tid == 0) sv[1][pn + 255] += sq[h1];
    __syncthreads();
    for (unsigned ps = 1; ps <= pn; ++ps) {
        i128 nv[2] = { 0, 0 };
        bool ok[2] = { false, false };
        for (unsigned r = 0; r < 2; ++r) {
            const unsigned x = tid + 288 * r;
            if (x >= 2 * ne) continue;
            const unsigned hf = x >= ne ? 1 : 0, e = x - hf * ne;
            const long long m = (long long)(hf * M + mb) - (long long)pn + e;
            if (e < ps || m < 0) continue;
            i128 g, gp;
            const unsigned ki = e + h1;
            const int bx = (int)(sp[hf][ki + 1] - sp[hf][ki]);
            nv[r] = ((size_t)m == L - 1) ? sv[hf][e] : bal128(sv[hf][e], bx, g);
            if (m >= 1) {
                const int bp = (int)(sp[hf][ki] - sp[hf][ki - 1]);
                bal128(sv[hf][e - 1], bp, gp);
                nv[r] += gp;
            }
            ok[r] = true;
        }
        __syncthreads();
        for (unsigned r = 0; r < 2; ++r) {
            const unsigned x = tid + 288 * r;
            if (ok[r]) sv[x >= ne ? 1 : 0][x - (x >= ne ? ne : 0)] = nv[r];
        }
        __syncthreads();
    }
    if (tid < 256) {
        const i128 v0 = sv[0][pn + tid], v1 = sv[1][pn + tid];
        out[mb + tid] = make_longlong2((long long)(uint64_t)v0, (long long)(v0 >> 64));
        out[M + mb + tid] = make_longlong2((long long)(uint64_t)v1, (long long)(v1 >> 64));
    }
}

// ---- host
void release(GpuEngine* g);
void prof_step(GpuEngine* g);
void hyb_bench(GpuEngine* g);
int hyb_l2ord_pick(GpuEngine* g);
void hmark(GpuEngine* g, int i) { if (g->prof) CK(cudaEventRecord(g->pev[i], g->st)); }

struct C2 { uint64_t x, y; };
uint64_t hmulm(uint64_t a, uint64_t b, uint64_t p) { return (uint64_t)((u128)a * b % p); }
C2 c2mul(C2 a, C2 b, uint64_t p) {
    return { (uint64_t)(((u128)a.x * b.x + (u128)a.y * ((p - b.y) % p)) % p), (uint64_t)(((u128)a.x * b.y + (u128)a.y * b.x) % p) };
}
C2 c2pow(C2 a, uint64_t e, uint64_t p) {
    C2 r{ 1, 0 };
    while (e) { if (e & 1) r = c2mul(r, a, p); a = c2mul(a, a, p); e >>= 1; }
    return r;
}
uint64_t hinvm(uint64_t a, uint64_t m) {   // a^-1 mod m (gcd 1), m < 2^62
    int64_t t = 0, nt = 1, r = (int64_t)m, nr = (int64_t)(a % m);
    while (nr) {
        const int64_t q = r / nr;
        int64_t x = t - q * nt; t = nt; nt = x;
        x = r - q * nr; r = nr; nr = x;
    }
    return (uint64_t)(t < 0 ? t + (int64_t)m : t);
}

// the plan: the smallest np * L with the worst case of |w_t| < P/2 (primes: e >= lgL, k a square mod p)
// wide 7 / 6: the wide digits with exactly that many primes - 7 by the worst case, 6 by the typical size of w
// (max |w| ~ 2^(d+2+2(B-1)+lgL/2+3), 4 bits of margin below P/2; GEC as the guard); B <= 96, the fold d + B <= 110
// (i128; WIDE 7: d + B <= 120), the block of 2048 (lgM >= 12)
// The fold of k_hcrtW: cur = rem 2^bv + pw 2^d with rem < k < 2^d, bv <= B and pw = H - 1 parts below 2^B (the top
// part too: H is chosen so that (H-1) Bm >= sum - B), so |cur| < (H + 2) 2^(d+B) <= 2^(d+B+3.4) < 2^127 for
// d + B <= 120 (H <= 8). WIDE 7 goes through its bench (values compared with the other engine) and the GEC.
// Example: a 35-bit k (Voyager) keeps 2^15 complex x 7 up to n = 5308381 (81-bit digits) instead of 4915165.
bool hyb_plan_w(unsigned n, uint64_t k, HybEngine& h, int wide) {
    const int d = 64 - __builtin_clzll(k);
    const uint64_t nd = (uint64_t)n + d;
    if (wide != 0 && wide != 6 && wide != 7) return false;
    double best = 1e300;
    for (int lg = 13; lg <= 24; ++lg) {
        const uint64_t L = (uint64_t)1 << lg;
        const int Bm = (int)(nd / L), B = (int)((nd + L - 1) / L);
        if (wide ? (Bm < 8 || B > 96 || d > Bm || d + B > (wide == 7 ? 120 : 110) || d > 47) : (Bm < 8 || B > 48 || d > Bm || d + B > 58 || d > 47))
            continue;
        std::vector<uint32_t> ps;
        for (uint32_t p : HYB_PRIMES)
            if (__builtin_ctzll((uint64_t)p + 1) >= lg && k % p != 0 && pw(k % p, (p - 1) / 2, p) == 1) ps.push_back(p);
        if (wide) {
            if ((int)ps.size() < wide) continue;
            double sum = 0;
            for (int i = 0; i < wide; ++i) sum += std::log2((double)ps[i]);
            const int H = std::max(2, (int)std::ceil((sum - B) / Bm) + 1);
            if (H > 8) continue;
            const double D = std::ldexp(1.0, B - 1) + 1, Dt = (3.0 * H + 2) * std::ldexp(1.0, B) + std::ldexp(1.0, B + 1);
            // WIDE 7 (the worst case, only through its bench) up to Dt < 2^112 (hredc2 takes |v| < 2^115; the i128 of k_hcrtW:
            // B <= 96, d + B <= 120 above); WIDE 6 (the automatic plans, no bench) keeps Dt < 2^84 and d + B <= 110
            if (Dt >= std::ldexp(1.0, wide == 7 ? 112 : 84)) continue;
            const double wb = (d + 2) + std::log2((double)(L - 2) * D * D + 2 * D * Dt + Dt * Dt);
            const bool ok = wide == 6 ? (d + 2) + 2.0 * (B - 1) + 0.5 * lg + 3 + 4 < sum - 1 : wb + 1 < sum - 0.01;
            const double cost = (double)wide * L;
            if (ok && cost < best) {
                best = cost;
                h.lgL = lg; h.lgM = lg - 1; h.np = wide; h.B = B; h.Bm = Bm; h.H = H; h.d = d; h.nd = nd;
                h.L = L; h.M = L >> 1; h.Dt = Dt; h.wide = wide;
                for (int i = 0; i < wide; ++i) h.hp.p[i] = ps[i];
            }
            continue;
        }
        double sum = 0;
        for (int np = 1; np <= 4 && np <= (int)ps.size(); ++np) {
            sum += std::log2((double)ps[np - 1]);
            if (np < 2) continue;
            const int H = std::max(2, (int)std::ceil((sum - B) / Bm) + 1);
            if (H > 8) continue;
            const double D = std::ldexp(1.0, B - 1) + 1, b0 = (3.0 * H + 2) * std::ldexp(1.0, B), Dt = b0 + std::ldexp(1.0, B + 1);
            const double wb = (d + 2) + std::log2((double)(L - 2) * D * D + 2 * D * Dt + Dt * Dt);
            if (wb + 1 < sum - 0.01) {
                const double cost = (double)np * L;
                if (cost < best) {
                    best = cost;
                    h.lgL = lg; h.lgM = lg - 1; h.np = np; h.B = B; h.Bm = Bm; h.H = H; h.d = d; h.nd = nd;
                    h.L = L; h.M = L >> 1; h.Dt = Dt;
                    for (int i = 0; i < np; ++i) h.hp.p[i] = ps[i];
                }
                break;
            }
        }
    }
    if (best >= 1e300) return false;
    const double Bs = std::ldexp(1.0, h.Bm), half = Bs / 2;   // passes until the carry is at most 1
    double bound = (3.0 * h.H + 2) * std::ldexp(1.0, h.B);
    h.npass = 0;
    for (;;) {
        const double g = std::floor((bound + half) / Bs);
        bound = std::ldexp(1.0, h.B - 1) + g;
        ++h.npass;
        if (g <= 1) break;
    }
    h.shn = (int)((uint64_t)n - hpos(h.L - 1, h.nd, h.lgL));
    h.kinv = ~(uint64_t)0 / k;
    return h.npass <= 16 && h.shn >= 0;
}
// the wide digits with 6 primes by default from 2^18 complex of the 40-bit plan, when they move less data (np M):
// measured 21-30% faster from 15M to 100M bits (100M: the checkpoint bit for bit equal in all engines); at 2^16 x 4 the
// arithmetic, not the memory, is the limit and they do not help. LOTR_GPU_WIDE=0: never, =6 / =7: always (when there
// is a plan)
bool hyb_plan_auto(unsigned n, uint64_t k, HybEngine& h) {
    if (!hyb_plan_w(n, k, h, 0)) return false;
    if (h.lgM >= 18) {
        HybEngine w;
        if (hyb_plan_w(n, k, w, 6) && (size_t)w.np * w.M < (size_t)h.np * h.M) h = w;
    }
    return true;
}

// The WIDE 7 bench (e.g. 100M k=585: automatic 2^22 x 4 40-bit vs WIDE 7 2^21 x 7 80-bit worst case, -6%): from 1M
// digits, when the 7-prime plan moves less data than the automatic one (np M), hyb_create times both on the card and
// checks that they give the same value; WIDE 7 is taken when it agrees and is at least 1% faster. The result is kept
// per plan shape for the whole run (the next candidates of that shape skip the bench). LOTR_GPU_WIDE forces a plan
// without the bench, LOTR_GPU_WIDEBENCH=0 = the automatic plan alone.
typedef std::array<int, 4> HybShape;   // automatic lgM, np, wide; WIDE 7 lgM
std::mutex g_w7_mtx;
std::map<HybShape, int> g_w7_kept;      // 1 = WIDE 7 won, 0 = the automatic plan stays
bool hyb_w7_cand(unsigned n, uint64_t k, const HybEngine& h, HybEngine& w, HybShape& key) {
    const char* wb = std::getenv("LOTR_GPU_WIDEBENCH");
    if ((wb && *wb == '0') || h.wide == 7 || h.nd < 3321925) return false;   // 3321925 bits ~ 1 000 000 digits
    if (!hyb_plan_w(n, k, w, 7) || (size_t)w.np * w.M >= (size_t)h.np * h.M) return false;
    key = HybShape{ h.lgM, h.np, h.wide, w.lgM };
    return true;
}
int hyb_w7_get(const HybShape& key) {   // -1: not benched yet
    std::lock_guard<std::mutex> lk(g_w7_mtx);
    const auto it = g_w7_kept.find(key);
    return it == g_w7_kept.end() ? -1 : it->second;
}
// The classic engine against WIDE 7: large k has no automatic hybrid plan (d > Bm of the 40-bit digits), so it takes
// the classic NTT; WIDE 7 can still win (e.g. a 35-bit k at n ~2.1M: 2^14 x 7 75-bit digits ~1.6x faster than the
// classic 2^18 x 2, the same RES64). With no digit-count limit, from 2^14 complex (the hybrid engine's range), when
// WIDE 7 moves fewer bytes (7 M x 8 B against np L x 4 B); gpu_create times both (hyb_w7_duel). Key: classic lgL, np,
// -1, WIDE 7 lgM.
bool cls_w7_cand(unsigned n, uint64_t k, int lgL, int np, HybEngine& w, HybShape& key) {
    const char* wb = std::getenv("LOTR_GPU_WIDEBENCH");
    const char* hy = std::getenv("LOTR_GPU_HYB");
    if ((wb && *wb == '0') || (hy && *hy == '0') || std::getenv("LOTR_GPU_WIDE")) return false;
    if (!hyb_plan_w(n, k, w, 7) || w.lgM < 14 || (size_t)7 * w.M * 8 >= ((size_t)np << lgL) * 4) return false;
    key = HybShape{ lgL, np, -1, w.lgM };
    return true;
}

bool hyb_plan(unsigned n, uint64_t k, HybEngine& h) {
    if (const char* we = std::getenv("LOTR_GPU_WIDE")) return hyb_plan_w(n, k, h, std::atoi(we));
    if (!hyb_plan_auto(n, k, h)) {   // no automatic hybrid plan: WIDE 7 only after it won against the classic engine
        Plan pl;
        HybEngine w;
        HybShape key;
        if (make_plan(n, k, pl) && cls_w7_cand(n, k, pl.lgL, pl.np, w, key) && hyb_w7_get(key) == 1) {
            h = w;
            return true;
        }
        return false;
    }
    HybEngine w;
    HybShape key;
    if (hyb_w7_cand(n, k, h, w, key) && hyb_w7_get(key) == 1) h = w;
    return true;
}

void hyb_free(GpuEngine* g) {
    HybEngine* h = g->hy;
    for (void* p : { (void*)h->dhp, (void*)h->a, (void*)h->b2, (void*)h->tw, (void*)h->itw, (void*)h->pkh, (void*)h->pkl,
                     (void*)h->tl, (void*)h->th, (void*)h->thm, (void*)h->ub, (void*)h->uq, (void*)h->ht16, (void*)h->iht16, (void*)h->ucb,
                     (void*)h->ucq, (void*)h->pwa, (void*)h->pwb, (void*)h->uf, (void*)h->ht11, (void*)h->iht11 })
        if (p) cudaFree(p);
    if (h->st2) cudaStreamSynchronize(h->st2);
    if (h->evf) cudaEventDestroy(h->evf);
    for (int c = 0; c < 8; ++c) {
        if (h->evb[c]) cudaEventDestroy(h->evb[c]);
        if (h->evc[c]) cudaEventDestroy(h->evc[c]);
    }
    for (cudaEvent_t e : h->evk)
        if (e) cudaEventDestroy(e);
    if (h->st2) cudaStreamDestroy(h->st2);
    delete h;
    g->hy = nullptr;
}

template <typename T> void hup(T*& dst, const std::vector<T>& v, cudaStream_t st) {
    CK(cudaMalloc(&dst, sizeof(T) * v.size()));
    CK(cudaMemcpyAsync(dst, v.data(), sizeof(T) * v.size(), cudaMemcpyHostToDevice, st));
}

GpuEngine* hyb_create_p(unsigned n, uint64_t k, int device, std::string& info, const HybEngine& plan) {
    CK(cudaSetDevice(device));
    GpuEngine* g = new GpuEngine;
    g->n = n; g->k = k; g->device = device;
    g->hy = new HybEngine(plan);
    HybEngine& h = *g->hy;
    g->L = h.L;
    g->prof = (std::getenv("LOTR_GPU_PROFILE") != nullptr);
    g->graphs = (std::getenv("LOTR_GPU_NOGRAPH") == nullptr) && !g->prof;
    mpz_init(g->N);
    mpz_set_ui(g->N, (unsigned long)k);
    mpz_mul_2exp(g->N, g->N, n);
    mpz_add_ui(g->N, g->N, 1);
    try {
        const int np = h.np;
        const size_t L = h.L, M = h.M, LH = L >> 11, MH = M >> 11;
        for (int i = 0; i < np; ++i) {   // constants
            const uint64_t p = h.hp.p[i], R = (1ull << 32) % p;
            uint32_t x = 1;
            for (int it = 0; it < 5; ++it) x *= 2 - (uint32_t)p * x;
            h.hp.pv[i] = 0u - x;
            h.hp.r2[i] = (uint32_t)(R * R % p);
            h.hp.r3[i] = (uint32_t)(R * R % p * R % p);
            h.hp.c52[i] = (uint32_t)((1ull << 52) % p);
            h.hp.c84[i] = (uint32_t)(((u128)1 << 84) % p);
            for (int j = 0; j < i; ++j) h.hp.inv[i][j] = (uint32_t)(pw(h.hp.p[j] % p, p - 2, p) * R % p);
            if (np <= 4) h.P *= p;
        }
        {   // hcrtf: c_i x R, 1 / p_i, P / p_i and P in words of 32 bits; P4 (the wide digits) in words of 64
            auto mulw = [](uint32_t* x, uint32_t m) {
                uint64_t c = 0;
                for (int l = 0; l < 8; ++l) { c += (uint64_t)x[l] * m; x[l] = (uint32_t)c; c >>= 32; }
            };
            h.hf.P[0] = 1;
            for (int i = 0; i < np; ++i) mulw(h.hf.P, h.hp.p[i]);
            for (int i = 0; i < np; ++i) {
                const uint64_t p = h.hp.p[i], R = (1ull << 32) % p;
                uint64_t qm = 1;
                h.hf.Q[i][0] = 1;
                for (int j = 0; j < np; ++j)
                    if (j != i) { mulw(h.hf.Q[i], h.hp.p[j]); qm = qm * h.hp.p[j] % p; }
                h.hf.ci[i] = (uint32_t)(pw(qm, p - 2, p) * R % p);
                h.hf.ip[i] = 1.0 / (double)p;
            }
            for (int l = 0; l < 4; ++l) h.P4.w[l] = ((uint64_t)h.hf.P[2 * l + 1] << 32) | h.hf.P[2 * l];
        }
        h.crtf = !h.wide && std::getenv("LOTR_GPU_CRTF") != nullptr;
        std::vector<uint2> tw(np * M), itw(np * M), pkh(np * MH), pkl(np * 2048);
        std::vector<uint32_t> tl(np * 2048), th(np * LH), thm(np * LH);
        std::vector<uint2> ub(np * (M >> 8)), uq(np * 256), ht16(np * 4096), iht16(np * 4096);
        std::vector<uint2> ht11(h.wide ? np * 2048 : 0), iht11(h.wide ? np * 2048 : 0);
        {   // 512 rows + k_hblock16 for the wide digits at 2^21 (LOTR_GPU_COL9=0: 1024 rows + k_hblock11)
            const char* e9 = std::getenv("LOTR_GPU_COL9");
            h.lgR = (h.wide && h.lgM - 12 == HC9_LGR && !(e9 && e9[0] == '0')) ? HC9_LGR : HC_LGR;
        }
        const size_t cq = M >> h.lgR;   // (UC: the columns of the column pass)
        std::vector<uint2> ucb(np * (M / cq)), ucq(np * cq);
        const bool pkt = (size_t)np * M <= ((size_t)1 << 20);
        std::vector<uint2> pwa(pkt ? np * M : 0), pwb(pkt ? np * M : 0);
        const bool upt = (size_t)np * M <= ((size_t)1 << 20);
        std::vector<uint4> uf(upt ? np * M : 0);
        for (int c = 0; c < np; ++c) {
            const uint64_t p = h.hp.p[c], R = (1ull << 32) % p;
            const int e = __builtin_ctzll(p + 1);
            auto toR = [&](uint64_t v) { return (uint32_t)hmulm(v, R, p); };
            auto c2R = [&](C2 v) { return make_uint2(toR(v.x), toR(v.y)); };
            C2 g2{ 0, 0 };   // an element of order 2^(e+1) (the 2-part of GF(p^2)*), then theta of order 2L with theta^M = i
            for (uint64_t z = 1;; ++z) {
                g2 = c2pow(C2{ 1, z }, (p * p - 1) >> (e + 1), p);
                const C2 t = c2pow(g2, (uint64_t)1 << e, p);
                if (!(t.x == 1 && t.y == 0)) break;
            }
            C2 theta = c2pow(g2, (uint64_t)1 << (e - h.lgL), p);
            C2 tm = c2pow(theta, M, p);
            if (!(tm.x == 0 && tm.y == 1)) theta = c2pow(theta, 2 * L - 1, p);
            tm = c2pow(theta, M, p);
            if (!(tm.x == 0 && tm.y == 1)) throw std::runtime_error("hybrid: no theta with theta^M = i");
            const C2 om = c2pow(theta, 4, p), iom = c2pow(om, M - 1, p);
            tw[c * M] = itw[c * M] = make_uint2((uint32_t)R, 0);
            for (size_t hh = 1; hh < M; hh <<= 1) {
                const C2 bb = c2pow(om, M / (2 * hh), p), ib = c2pow(iom, M / (2 * hh), p);
                C2 u{ 1, 0 }, v{ 1, 0 };
                for (size_t j = 0; j < hh; ++j) {
                    tw[c * M + hh + j] = c2R(u);
                    itw[c * M + hh + j] = c2R(v);
                    u = c2mul(u, bb, p);
                    v = c2mul(v, ib, p);
                }
            }
            for (unsigned i = 1; i < 16; ++i) {   // k_hblock16: w_16q^(rev4(i) j) of the stages q = 256 and 16
                const unsigned ri = ((i & 1) << 3) | ((i & 2) << 1) | ((i & 4) >> 1) | (i >> 3);
                for (unsigned q : { 256u, 16u })
                    for (unsigned j = 0; j < q; ++j) {
                        const unsigned x = ri * j;
                        const size_t o = c * 4096 + (q == 256 ? (i - 1) * 256 + j : 3840 + (i - 1) * 16 + j);
                        const bool lo = x < 8 * q;
                        const uint2 t = tw[c * M + (lo ? 8 * q + x : x)], it = itw[c * M + (lo ? 8 * q + x : x)];
                        auto ng = [&](uint2 w) { return lo ? w : make_uint2(w.x ? (uint32_t)p - w.x : 0, w.y ? (uint32_t)p - w.y : 0); };
                        ht16[o] = ng(t);
                        iht16[o] = ng(it);
                    }
                if (h.wide)   // k_hblock11: the same for the stages q = 128 and 8 ([(i - 1) 128 + j], [1920 + (i - 1) 8 + j])
                    for (unsigned q : { 128u, 8u })
                        for (unsigned j = 0; j < q; ++j) {
                            const unsigned x = ri * j;
                            const size_t o = c * 2048 + (q == 128 ? (i - 1) * 128 + j : 1920 + (i - 1) * 8 + j);
                            const bool lo = x < 8 * q;
                            const uint2 t = tw[c * M + (lo ? 8 * q + x : x)], it = itw[c * M + (lo ? 8 * q + x : x)];
                            auto ng = [&](uint2 w) { return lo ? w : make_uint2(w.x ? (uint32_t)p - w.x : 0, w.y ? (uint32_t)p - w.y : 0); };
                            ht11[o] = ng(t);
                            iht11[o] = ng(it);
                        }
            }
            const uint64_t m2 = (p - 1) / 2, ui = hinvm(L % m2, m2);   // L-th roots in the squares (odd order m2)
            const uint64_t tau = pw(2, ui, p), xs = hmulm(pw(2, h.d, p), pw(k % p, p - 2, p), p), sig = pw(xs, ui, p);
            if (pw(tau, L, p) != 2 || pw(sig, L, p) != xs) throw std::runtime_error("hybrid: no L-th root (tau / sigma)");
            const C2 ts{ hmulm(theta.x, sig, p), hmulm(theta.y, sig, p) }, its = c2pow(ts, p * p - 2, p);
            const C2 ts11 = c2pow(ts, 2048, p);
            C2 u{ 1, 0 }, v{ 1, 0 };
            for (size_t y = 0; y < 2048; ++y) { pkl[c * 2048 + y] = c2R(u); u = c2mul(u, ts, p); }
            u = C2{ 1, 0 };
            for (size_t x = 0; x < MH; ++x) { pkh[c * MH + x] = c2R(u); u = c2mul(u, ts11, p); }
            const uint64_t itau = pw(tau, p - 2, p), sigM = pw(sig, M, p), isigM = pw(sigM, p - 2, p);
            h.hp.cw[c] = toR(pw(itau, ((M >> 4) * (h.nd & (L - 1))) & (L - 1), p));   // (as D in k_hcolf<true>)
            h.hp.ck[c] = toR(hmulm(sigM, pw(itau, (M * (h.nd & (L - 1))) & (L - 1), p), p));   // (as D1 there)
            h.hp.cu[c] = toR(hmulm(isigM, pw(tau, (M * (h.nd & (L - 1))) & (L - 1), p), p));
            if (pkt) {   // pwa / pwb (HybEngine): tau^s from plain halves as th x tl, (theta*sigma)^j step by step
                const uint64_t R2 = R * R % p, ndm = h.nd & (L - 1), t11 = pw(tau, 2048, p);
                std::vector<uint64_t> tlo(2048), thi(LH);
                uint64_t x = 1;
                for (size_t y = 0; y < 2048; ++y) { tlo[y] = x; x = hmulm(x, tau, p); }
                x = 1;
                for (size_t y = 0; y < LH; ++y) { thi[y] = x; x = hmulm(x, t11, p); }
                C2 u{ 1, 0 };
                for (size_t j = 0; j < M; ++j) {
                    const size_t s0 = (L - ((j * ndm) & (L - 1))) & (L - 1), s1 = (L - (((j + M) * ndm) & (L - 1))) & (L - 1);
                    const uint64_t w0 = hmulm(hmulm(thi[s0 >> 11], tlo[s0 & 2047], p), R2, p);
                    const uint64_t w1 = hmulm(hmulm(hmulm(thi[s1 >> 11], tlo[s1 & 2047], p), sigM, p), R2, p);
                    pwa[c * M + j] = make_uint2((uint32_t)hmulm(u.x, w0, p), (uint32_t)hmulm(u.y, w0, p));
                    const uint64_t by = hmulm(u.y, w1, p);   // i (w1 u) = (-w1 u.y, w1 u.x)
                    pwb[c * M + j] = make_uint2((uint32_t)((p - by) % p), (uint32_t)hmulm(u.x, w1, p));
                    u = c2mul(u, ts, p);
                }
            }
            // -k * R / M: w_t = -k c_t / a_t, so W == -k * Y^2 (the scaled form of the engine API)
            const uint64_t F = hmulm(hmulm((p - k % p) % p, R, p), pw(M % p, p - 2, p), p);
            uint64_t a1 = 1;
            for (size_t y = 0; y < 2048; ++y) { tl[c * 2048 + y] = toR(a1); a1 = hmulm(a1, tau, p); }
            const uint64_t t11 = pw(tau, 2048, p);
            a1 = 1;
            for (size_t x = 0; x < LH; ++x) {
                th[c * LH + x] = (uint32_t)a1;                   // plain (no R): the digit comes as d R (hresR)
                thm[c * LH + x] = (uint32_t)hmulm(sigM, a1, p);
                a1 = hmulm(a1, t11, p);
            }
            // hw2 UB: ub[b] = (theta*sigma)^-256b F tau^-s0(256b), uq[t] = (theta*sigma)^-t tau^(r_t), r_t = t (n+d) mod L
            const uint64_t ndm = h.nd & (L - 1);
            const C2 its8 = c2pow(its, 256, p);
            v = C2{ 1, 0 };
            for (size_t b = 0; b < (M >> 8); ++b) {
                const uint64_t s = hmulm(F, pw(itau, (L - ((256 * b * ndm) & (L - 1))) & (L - 1), p), p);
                ub[c * (M >> 8) + b] = c2R(C2{ hmulm(v.x, s, p), hmulm(v.y, s, p) });
                v = c2mul(v, its8, p);
            }
            v = C2{ 1, 0 };
            for (size_t t = 0; t < 256; ++t) {
                const uint64_t s = pw(tau, (t * ndm) & (L - 1), p);
                uq[c * 256 + t] = c2R(C2{ hmulm(v.x, s, p), hmulm(v.y, s, p) });
                v = c2mul(v, its, p);
            }
            // UC: the same split at q = M / 1024 (the columns of the column pass): ucb[r] (row jb = q r), ucq[jj]
            const C2 itsq = c2pow(its, cq, p);
            v = C2{ 1, 0 };
            for (size_t r = 0; r < (M / cq); ++r) {
                const uint64_t s = hmulm(F, pw(itau, (L - ((cq * r * ndm) & (L - 1))) & (L - 1), p), p);
                ucb[c * (M / cq) + r] = c2R(C2{ hmulm(v.x, s, p), hmulm(v.y, s, p) });
                v = c2mul(v, itsq, p);
            }
            v = C2{ 1, 0 };
            for (size_t t = 0; t < cq; ++t) {
                const uint64_t s = pw(tau, (t * ndm) & (L - 1), p);
                ucq[c * cq + t] = c2R(C2{ hmulm(v.x, s, p), hmulm(v.y, s, p) });
                v = c2mul(v, its, p);
            }
            if (upt) {   // uf (HybEngine) from the UB values: tw = ub uq / R, re = Re(a tw / R) h0, im = Im(a tw / R) cu / R h1
                const uint64_t iR = pw(R, p - 2, p), i2 = (p + 1) / 2, cuR = hmulm(h.hp.cu[c], iR, p), sv = (M * ndm) & (L - 1);
                for (size_t j = 0; j < M; ++j) {
                    const uint2 b = ub[c * (M >> 8) + (j >> 8)], q = uq[c * 256 + (j & 255)];
                    const uint64_t tx = hmulm((hmulm(b.x, q.x, p) + p - hmulm(b.y, q.y, p)) % p, iR, p);
                    const uint64_t ty = hmulm((hmulm(b.x, q.y, p) + hmulm(b.y, q.x, p)) % p, iR, p);
                    const uint64_t sb = (L - (((j & ~(size_t)255) * ndm) & (L - 1))) & (L - 1), rt = ((j & 255) * ndm) & (L - 1);
                    const uint64_t s0 = (L - ((j * ndm) & (L - 1))) & (L - 1);
                    const uint64_t h0 = sb < rt ? i2 : 1, h1 = hmulm(h0, s0 < sv ? i2 : 1, p), c1 = hmulm(cuR, h1, p);
                    uf[c * M + j] = make_uint4((uint32_t)hmulm(tx, h0, p), (uint32_t)hmulm((p - ty) % p, h0, p),
                                               (uint32_t)hmulm(ty, c1, p), (uint32_t)hmulm(tx, c1, p));
                }
            }
        }
        CK(cudaStreamCreateWithFlags(&g->st, cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&g->ev, cudaEventBlockingSync | cudaEventDisableTiming));
        if (g->prof)
            for (cudaEvent_t& ev : g->pev) CK(cudaEventCreateWithFlags(&ev, cudaEventBlockingSync));
        CK(cudaMallocHost(&g->host, (h.wide ? 16 : 8) * L));   // (the wide digits: 16 B a digit)
        CK(cudaMalloc(&h.dhp, sizeof(HP)));
        CK(cudaMemcpyAsync(h.dhp, &h.hp, sizeof(HP), cudaMemcpyHostToDevice, g->st));
        CK(cudaMalloc(&h.a, sizeof(uint2) * M * np));
        CK(cudaMalloc(&h.b2, sizeof(uint2) * M * np));
        hup(h.tw, tw, g->st); hup(h.itw, itw, g->st);
        hup(h.pkh, pkh, g->st); hup(h.pkl, pkl, g->st);
        hup(h.tl, tl, g->st); hup(h.th, th, g->st); hup(h.thm, thm, g->st);
        hup(h.ub, ub, g->st); hup(h.uq, uq, g->st); hup(h.ht16, ht16, g->st); hup(h.iht16, iht16, g->st);
        if (h.wide) { hup(h.ht11, ht11, g->st); hup(h.iht11, iht11, g->st); }
        hup(h.ucb, ucb, g->st); hup(h.ucq, ucq, g->st);
        if (pkt) { hup(h.pwa, pwa, g->st); hup(h.pwb, pwb, g->st); }
        if (upt) hup(h.uf, uf, g->st);
        for (int s = 0; s < GPU_SLOTS; ++s) {
            CK(cudaMalloc(&g->slot[s], (h.wide ? 16 : 8) * L));
            CK(cudaMemsetAsync(g->slot[s], 0, (h.wide ? 16 : 8) * L, g->st));
        }
        int optin = 0;
        CK(cudaDeviceGetAttribute(&h.nsm, cudaDevAttrMultiProcessorCount, device));
        CK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
        if (h.lgR == HC9_LGR) {   // 512 rows (32 KB, 2 blocks per SM: the largest shared carveout)
            CK(cudaFuncSetAttribute(k_hcolf9<true>, cudaFuncAttributePreferredSharedMemoryCarveout, (int)cudaSharedmemCarveoutMaxShared));
            CK(cudaFuncSetAttribute(k_hcoli9<1>, cudaFuncAttributePreferredSharedMemoryCarveout, (int)cudaSharedmemCarveoutMaxShared));
            int of = 0, oi = 0;
            CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&of, k_hcolf9<true>, HC9_T, HC9_SHM));
            CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&oi, k_hcoli9<1>, HC9_T, HC9_SHM));
            h.cgf = (unsigned)(h.nsm * std::max(of, 1));
            h.cgi = (unsigned)(h.nsm * std::max(oi, 1));
            h.col = true;
            if (const char* e = std::getenv("LOTR_GPU_CCH")) {   // (LOTR_GPU_L2ORD=3: 2^(9 - ...) column groups a chunk, at least 32)
                const int v = std::atoi(e);
                if (v == 2 || v == 4 || v == 8 || v == 16) h.cch = v;
            }
            if (const char* e = std::getenv("LOTR_GPU_CPO")) h.cpo = e[0] == '0' ? 0 : 1;
            for (int c = 0; c < 17; ++c) CK(cudaEventCreateWithFlags(&h.evk[c], cudaEventDisableTiming));
        }
        else if (h.lgM - (h.wide ? 11 : 12) == HC_LGR && (size_t)optin >= HC_SHM) {
            CK(cudaFuncSetAttribute(k_hcolf<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)HC_SHM));
            CK(cudaFuncSetAttribute(k_hcolf<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)HC_SHM));
            CK(cudaFuncSetAttribute(k_hcoli, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)HC_SHM));
            h.col = true;
        }
        if (h.col) {
            const char* e = std::getenv("LOTR_GPU_L2ORD");
            // with CUDA Graphs timed at the end of the creation (hyb_l2ord_pick: 2 wins on a card with a small L2 and few SMs,
            // 0 on a large one, where the tails of 7 small launches leave many SMs idle); without the graphs the cross-stream
            // waits go through the host: 0
            h.l2ord = e ? std::max(0, std::min(h.lgR == HC9_LGR ? 3 : 2, std::atoi(e))) : g->graphs && !g->prof ? -1 : 0;
            CK(cudaStreamCreateWithFlags(&h.st2, cudaStreamNonBlocking));   // (hyb_blkinv_l2o, also COLBENCH)
            CK(cudaEventCreateWithFlags(&h.evf, cudaEventDisableTiming));
            for (int c = 0; c < 8; ++c) {
                CK(cudaEventCreateWithFlags(&h.evb[c], cudaEventDisableTiming));
                CK(cudaEventCreateWithFlags(&h.evc[c], cudaEventDisableTiming));
            }
        }
        if (const char* e = std::getenv("LOTR_GPU_L2P")) {
            int mx = 0, mw = 0;
            CK(cudaDeviceGetAttribute(&mx, cudaDevAttrMaxPersistingL2CacheSize, device));
            CK(cudaDeviceGetAttribute(&mw, cudaDevAttrMaxAccessPolicyWindowSize, device));
            const size_t ps = std::min((size_t)(std::atof(e) * 1048576.0), (size_t)mx);
            const size_t wb = std::min(sizeof(uint2) * M * np, (size_t)mw);
            if (ps > 0 && wb > 0) {
                CK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, ps));
                h.l2w.base_ptr = h.a;
                h.l2w.num_bytes = wb;
                h.l2w.hitRatio = (float)std::min(1.0, (double)ps / (double)wb);
                h.l2w.hitProp = cudaAccessPropertyPersisting;
                h.l2w.missProp = cudaAccessPropertyNormal;
                cudaStreamAttrValue av{};
                av.accessPolicyWindow = h.l2w;
                CK(cudaStreamSetAttribute(g->st, cudaStreamAttributeAccessPolicyWindow, &av));
                h.l2on = true;
            }
            std::printf("[GPU] L2 persisting %.1f MB (card max %.1f MB), window %.1f MB of the array (%.1f MB, window max %.1f MB), hit ratio %.3f\n",
                        ps / 1048576.0, mx / 1048576.0, wb / 1048576.0, sizeof(uint2) * M * np / 1048576.0, mw / 1048576.0,
                        h.l2on ? h.l2w.hitRatio : 0.0f);
        }
        gpu_sync(g);
        if (h.l2ord < 0) h.l2ord = hyb_l2ord_pick(g);
        if (std::getenv("LOTR_GPU_COLBENCH")) hyb_bench(g);
    }
    catch (...) {
        release(g);
        throw;
    }
    char buf[256];
    std::snprintf(buf, sizeof(buf), "HYBRID NTT 2^%d complex (2^%d digits) x %d primes, %d-bit digits, %d carry passes%s%s, %s block%s%s%s%s",
                  h.lgM, h.lgL, h.np, h.B, h.npass, h.col ? (h.lgR == HC9_LGR ? ", column pass 512 rows" : ", column pass") : "",
                  h.l2ord == 3 ? " (coli -> CRT through the L2 in chunks)" : h.l2ord == 2 ? " (L2 order by prime, 2 streams)"
                  : h.l2ord ? " (L2 order by prime)" : "",
                  h.lgR == HC9_LGR ? (h.wide == 7 ? "WIDE (worst case), radix-16" : "WIDE (statistical), radix-16")
                  : h.wide == 7 ? "WIDE (worst case), radix-16/8" : h.wide == 6 ? "WIDE (statistical), radix-16/8" : "radix-16",
                  h.pwa ? ", pack table" : "", h.uf && !h.col ? ", unpack table" : "",
                  h.l2on ? ", L2 persisting" : h.crtf ? ", fractional CRT" : "",
                  g->prof ? " (profile, no CUDA Graphs)" : g->graphs ? "" : " (no CUDA Graphs)");
    info = buf;
    return g;
}

// the WIDE 7 bench (see hyb_w7_cand, cls_w7_cand): engine ga (its plan named la, already created) against a new WIDE 7
// engine, both from the same value, two timed rounds each (the better counts), then the same number of squarings on
// both must give the same value mod N. Returns the winner (info = its plan), the other one is destroyed.
std::mutex g_w7_bench_mtx;   // one bench at a time; the other threads of the same shape wait for its result
GpuEngine* hyb_w7_duel(GpuEngine* ga, const std::string& info_a, const std::string& la, unsigned n, uint64_t k, int device,
                       std::string& info, const HybEngine& w7, const HybShape& key,
                       std::chrono::steady_clock::time_point t_start) {
    std::string info_b, why;
    GpuEngine* gb = nullptr;
    double ta = 1e300, tb = 1e300;
    bool same = false;
    unsigned per = 0;
    mpz_t va, vb;
    mpz_init_set_ui(va, 3);
    mpz_init(vb);
    try {
        gb = hyb_create_p(n, k, device, info_b, w7);
        gpu_set(ga, GPU_X, va);
        gpu_set(gb, GPU_X, va);
        auto timed = [](GpuEngine* g, unsigned cnt) {
            const auto t0 = std::chrono::steady_clock::now();
            gpu_square(g, GPU_X, cnt);
            gpu_sync(g);
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        };
        // warm-up (the CUDA Graphs are built here); the first one also sets the length of a round (~0.3 s)
        const double tw = timed(ga, 4 * GRAPH_SQ) / (4 * GRAPH_SQ);
        timed(gb, 4 * GRAPH_SQ);
        per = std::max(2u, (unsigned)(0.3 / std::max(tw, 1e-6) / GRAPH_SQ + 0.5)) * GRAPH_SQ;
        for (int r = 0; r < 2; ++r) {
            ta = std::min(ta, timed(ga, per) / per);
            tb = std::min(tb, timed(gb, per) / per);
        }
        const bool oka = gpu_get(ga, GPU_X, va), okb = gpu_get(gb, GPU_X, vb);
        same = oka && okb && mpz_cmp(va, vb) == 0;
        if (!same) why = "the two plans gave different values";
    }
    catch (const std::exception& e) { why = e.what(); }
    mpz_clear(va);
    mpz_clear(vb);
    const bool use7 = same && tb < ta * 0.99;
    char buf[320];
    if (gb && why.empty())
        std::snprintf(buf, sizeof(buf), "%s %.4f ms/iter vs WIDE 7 2^%d x 7 %.4f ms/iter (%+.1f%%), values equal",
                      la.c_str(), ta * 1e3, w7.lgM, tb * 1e3, (tb / ta - 1) * 100);
    else
        std::snprintf(buf, sizeof(buf), "WIDE 7 2^%d x 7 not usable (%s)", w7.lgM, why.c_str());
    std::printf("[GPU] WIDE 7 bench: %s -> %s (%.1f s)\n", buf, use7 ? "WIDE 7" : "automatic plan",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count());
    std::fflush(stdout);
    {
        std::lock_guard<std::mutex> lk(g_w7_mtx);
        g_w7_kept[key] = use7 ? 1 : 0;
    }
    if (use7) {
        gpu_destroy(ga);
        info = info_b;
        return gb;
    }
    if (gb) gpu_destroy(gb);
    info = info_a;
    return ga;
}
GpuEngine* hyb_w7_bench(unsigned n, uint64_t k, int device, std::string& info, const HybEngine& plan,
                        const HybEngine& w7, const HybShape& key) {
    const auto t_start = std::chrono::steady_clock::now();
    std::string info_a;
    GpuEngine* ga = hyb_create_p(n, k, device, info_a, plan);   // (an error here goes up as before the bench)
    char la[64];
    std::snprintf(la, sizeof(la), "2^%d x %d %s", plan.lgM, plan.np, plan.wide ? "80-bit" : "40-bit");
    return hyb_w7_duel(ga, info_a, la, n, k, device, info, w7, key, t_start);
}

// the engine for n, k: LOTR_GPU_WIDE forces the plan; else the automatic one, or WIDE 7 by its bench (hyb_w7_cand) -
// one bench at a time, the other threads of the same shape wait for its result. No automatic plan (large k): only
// WIDE 7 that already won against the classic engine (hyb_plan; the bench itself is in gpu_create)
GpuEngine* hyb_create(unsigned n, uint64_t k, int device, std::string& info) {
    HybEngine plan, w7;
    if (std::getenv("LOTR_GPU_WIDE")) {
        if (!hyb_plan(n, k, plan)) return nullptr;
        return hyb_create_p(n, k, device, info, plan);
    }
    if (!hyb_plan_auto(n, k, plan)) {
        if (!hyb_plan(n, k, plan)) return nullptr;
        return hyb_create_p(n, k, device, info, plan);
    }
    HybShape key;
    if (!hyb_w7_cand(n, k, plan, w7, key)) return hyb_create_p(n, k, device, info, plan);
    std::lock_guard<std::mutex> lk(g_w7_bench_mtx);
    const int kept = hyb_w7_get(key);
    if (kept >= 0) return hyb_create_p(n, k, device, info, kept ? w7 : plan);
    return hyb_w7_bench(n, k, device, info, plan, w7, key);
}

// the outer passes (levels per pass, top first) for the lgM - 12 levels above k_hblock: the fewest passes, radix-64 /
// 16 / 8 / 4 / 2 (2^22: 6 + 4, two passes instead of 6 + 3 + 1; 2^16: one radix-16 pass; 2^14: one radix-4 pass)
int hyb_passes(int rem, int* s) {
    int n = 0;
    while (rem > 0) {
        const int x = (rem == 4 || rem == 5 || rem == 8) ? 4 : (rem >= 6) ? 6 : (rem == 3) ? 3 : (rem == 2) ? 2 : 1;
        s[n++] = x;
        rem -= x;
    }
    return n;
}
// digits -> transform (forward levels M/2 .. the block); the first outer pass packs the digits itself (every radix;
// k_hpack alone only when there is no outer pass). The wide digits: the block k_hblock11 (2048), so the outer passes
// cover lgM - 11 levels (the column pass at 2^21)
// k_hcoli or k_hcoli9 (512 rows): np primes from c0 in stream st
void hcoli_l(GpuEngine* g, uint2* a, int np, unsigned c0, cudaStream_t st) {
    HybEngine& h = *g->hy;
    if (h.lgR == HC9_LGR) k_hcoli9<1><<<h.cgi, HC9_T, HC9_SHM, st>>>(a, h.itw, h.lgM, np, h.dhp, h.ucq, c0);
    else k_hcoli<<<h.nsm, HC_T, HC_SHM, st>>>(a, h.itw, h.lgM, np, h.dhp, h.ucq, c0);
}
template <bool WIDE> void hyb_fwd_t(GpuEngine* g, const int64_t* d, uint2* a) {
    HybEngine& h = *g->hy;
    const int np = h.np, lgM = h.lgM;
    const size_t M = h.M;
    const HPK pk{ h.tl, h.th, h.thm, h.pkh, h.pkl, h.nd & (h.L - 1), h.pwa, h.pwb };
    if (h.col) {
        if (h.lgR == HC9_LGR) k_hcolf9<WIDE><<<h.cgf, HC9_T, HC9_SHM, g->st>>>(a, h.tw, lgM, np, h.dhp, d, pk);
        else k_hcolf<WIDE><<<h.nsm, HC_T, HC_SHM, g->st>>>(a, h.tw, lgM, np, h.dhp, d, pk);
        hmark(g, 1);
        return;
    }
    int s[16];
    const int ns = hyb_passes(lgM - (WIDE ? 11 : 12), s);
    if (ns == 0) k_hpack<WIDE><<<grid(M * np), 256, 0, g->st>>>(d, a, h.dhp, h.lgL, np, pk);
    hmark(g, 1);
    int lgh = lgM - 1;
    for (int i = 0; i < ns; ++i) {
        const bool first = i == 0;
        switch (s[i]) {
        case 6:
            if (first) k_hdif64<true, WIDE><<<(unsigned)((M >> 11) * np), 256, 0, g->st>>>(a, h.tw, lgM, lgh - 5, h.dhp, d, pk);
            else k_hdif64<false><<<(unsigned)((M >> 11) * np), 256, 0, g->st>>>(a, h.tw, lgM, lgh - 5, h.dhp, d, pk);
            break;
        case 4:
            if (first) k_hdif16<true, WIDE><<<grid((M >> 4) * np), 256, 0, g->st>>>(a, h.tw, lgM, lgh - 3, np, h.dhp, d, pk);
            else k_hdif16<false><<<grid((M >> 4) * np), 256, 0, g->st>>>(a, h.tw, lgM, lgh - 3, np, h.dhp, d, pk);
            break;
        case 3:
            if (first) k_hdif8<true, WIDE><<<grid((M >> 3) * np), 256, 0, g->st>>>(a, h.tw, lgM, lgh - 2, np, h.dhp, d, pk);
            else k_hdif8<false><<<grid((M >> 3) * np), 256, 0, g->st>>>(a, h.tw, lgM, lgh - 2, np, h.dhp, d, pk);
            break;
        case 2:
            if (first) k_hdif4<true, WIDE><<<grid((M >> 2) * np), 256, 0, g->st>>>(a, h.tw, lgM, lgh - 1, np, h.dhp, d, pk);
            else k_hdif4<false><<<grid((M >> 2) * np), 256, 0, g->st>>>(a, h.tw, lgM, lgh - 1, np, h.dhp, d, pk);
            break;
        default:
            if (first) k_hdif2<true, WIDE><<<grid((M >> 1) * np), 256, 0, g->st>>>(a, h.tw, lgM, lgh, np, h.dhp, d, pk);
            else k_hdif2<false><<<grid((M >> 1) * np), 256, 0, g->st>>>(a, h.tw, lgM, lgh, np, h.dhp, d, pk);
            break;
        }
        lgh -= s[i];
    }
}
void hyb_fwd(GpuEngine* g, const int64_t* d, uint2* a) {
    if (g->hy->wide) hyb_fwd_t<true>(g, d, a);
    else hyb_fwd_t<false>(g, d, a);
}
void hyb_inv(GpuEngine* g, uint2* a) {   // inverse levels of the outer passes, the passes of hyb_fwd in reverse order
    HybEngine& h = *g->hy;
    const int np = h.np, lgM = h.lgM;
    const size_t M = h.M;
    if (h.col) {
        hcoli_l(g, a, np, 0, g->st);
        return;
    }
    const int lb = h.wide ? 11 : 12;   // the levels of the block
    int s[16];
    const int ns = hyb_passes(lgM - lb, s);
    int lgh = lb;
    for (int i = ns - 1; i >= 0; --i) {
        switch (s[i]) {
        case 6: k_hdit64<<<(unsigned)((M >> 11) * np), 256, 0, g->st>>>(a, h.itw, lgM, lgh, h.dhp); break;
        case 4: k_hdit16<<<grid((M >> 4) * np), 256, 0, g->st>>>(a, h.itw, lgM, lgh, np, h.dhp); break;
        case 3: k_hdit8<<<grid((M >> 3) * np), 256, 0, g->st>>>(a, h.itw, lgM, lgh, np, h.dhp); break;
        case 2: k_hdit4<<<grid((M >> 2) * np), 256, 0, g->st>>>(a, h.itw, lgM, lgh, np, h.dhp); break;
        default: k_hdit2<<<grid((M >> 1) * np), 256, 0, g->st>>>(a, h.itw, lgM, lgh, np, h.dhp); break;
        }
        lgh += s[i];
    }
}
// the L2 prefetch distance of k_hblock16 / k_hcrtE: the blocks resident on the card (SMs x blocks per SM)
template <typename K> unsigned hpfd(const HybEngine& h, K kern, int threads) {
    int nb = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kern, threads, 0));
    return (unsigned)(h.nsm * (nb > 0 ? nb : 1));
}
// k_hblock16 MB 3 with the twiddle tables; the wide digits: k_hblock11 (two transforms of 2048 a block);
// c >= 0: only the prime c (LOTR_GPU_L2ORD)
template <int MODE> void hyb_block(GpuEngine* g, uint2* a, int c = -1) {
    HybEngine& h = *g->hy;
    const unsigned nbp = (unsigned)(h.M >> 12);   // the blocks of one prime (both kernels: 4096 values a block)
    const unsigned gb = c < 0 ? nbp * (unsigned)h.np : nbp, b0 = c < 0 ? 0u : nbp * (unsigned)c;
    if (h.wide && h.lgR != HC9_LGR) {   // (512 rows: k_hblock16 below)
        static const bool c11 = [] {
            CK(cudaFuncSetAttribute(k_hblock11<MODE>, cudaFuncAttributePreferredSharedMemoryCarveout, (int)cudaSharedmemCarveoutMaxShared));
            return true;
        }();
        (void)c11;
        static const unsigned pf11 = hpfd(h, k_hblock11<MODE>, 256);
        k_hblock11<MODE><<<gb, 256, 0, g->st>>>(a, h.tw, h.itw, h.lgM, h.dhp, pf11, h.ht11, h.iht11, b0);
        return;
    }
    static const bool c16 = [] {   // 3 x 32 KB per SM: the largest shared carveout (before the occupancy of hpfd)
        CK(cudaFuncSetAttribute(k_hblock16<MODE>, cudaFuncAttributePreferredSharedMemoryCarveout, (int)cudaSharedmemCarveoutMaxShared));
        return true;
    }();
    (void)c16;
    static const unsigned pf = hpfd(h, k_hblock16<MODE>, 256);
    k_hblock16<MODE><<<gb, 256, 0, g->st>>>(a, h.tw, h.itw, h.lgM, h.dhp, pf, h.ht16, h.iht16, b0);
}
// LOTR_GPU_L2ORD=1 (the column pass): block + coli one prime after the other instead of block (all primes) then coli
// (all): one prime is 16 MB at 2^21 (the wide digits, 100M), inside a 32 MB L2, so coli reads what the block just wrote
// from the L2 (cache blocking). The same arithmetic bit for bit, only the order of the launches; the profile counts coli
// in "block"
void hyb_blkinv_l2(GpuEngine* g, uint2* a) {
    HybEngine& h = *g->hy;
    for (int c = 0; c < h.np; ++c) {
        hyb_block<0>(g, a, c);
        hcoli_l(g, a, 1, (unsigned)c, g->st);
    }
}
// LOTR_GPU_L2ORD=2: the same order, overlapped (the split into np small launches alone costs the empty last wave of
// each launch): the blocks in g->st, the colis in h.st2, coli(c) after block(c), block(c) after coli(c - 2) - so
// block(c + 1) fills the tail of coli(c) on the free SMs and at most 2 primes (32 MB) are in flight. Events: a fork
// from and a join to g->st (also inside a graph capture). More primes in flight measured the same.
void hyb_blkinv_l2o(GpuEngine* g, uint2* a) {
    HybEngine& h = *g->hy;
    CK(cudaEventRecord(h.evf, g->st));
    CK(cudaStreamWaitEvent(h.st2, h.evf, 0));
    for (int c = 0; c < h.np; ++c) {
        if (c >= 2) CK(cudaStreamWaitEvent(g->st, h.evc[c - 2], 0));
        hyb_block<0>(g, a, c);
        CK(cudaEventRecord(h.evb[c], g->st));
        CK(cudaStreamWaitEvent(h.st2, h.evb[c], 0));
        hcoli_l(g, a, 1, (unsigned)c, h.st2);
        CK(cudaEventRecord(h.evc[c], h.st2));
    }
    CK(cudaStreamWaitEvent(g->st, h.evc[h.np - 1], 0));
}
template <int NP, int DG, int MT> void hcrtE_l(GpuEngine* g, int64_t* out) {   // blocks of 256 complex positions = 512 digits
    HybEngine& h = *g->hy;
    const HUK uk{ h.nd & (h.L - 1), h.ub, h.uq, h.ucb, h.lgM - h.lgR, h.uf };
    const uint64_t plo = (uint64_t)h.P, phi = (uint64_t)(h.P >> 64), k = (uint64_t)g->k;
    const unsigned gm = (unsigned)(h.M >> 8);
    static const unsigned pf = hpfd(h, k_hcrtE<NP, DG, MT>, 288);
    k_hcrtE<NP, DG, MT><<<gm, 288, 0, g->st>>>(h.a, out, h.dhp, h.lgL, h.nd, h.H, h.npass, plo, phi, h.hf, uk, k, h.kinv,
                                               h.d, h.shn, pf);
}
template <int NP, int UM> void hcrtW_l(GpuEngine* g, int64_t* out) {   // the wide digits (the CRT of hcrtf)
    HybEngine& h = *g->hy;
    const HUK uk{ h.nd & (h.L - 1), h.ub, h.uq, h.ucb, h.lgM - h.lgR, h.uf };
    static const unsigned pf = hpfd(h, k_hcrtW<NP, 1, UM>, 288);
    k_hcrtW<NP, 1, UM><<<(unsigned)(h.M >> 8), 288, 0, g->st>>>(h.a, reinterpret_cast<longlong2*>(out), h.dhp, h.lgL, h.nd,
                                                                h.H, h.npass, h.P4, h.hf, uk, (uint64_t)g->k, h.kinv, h.d,
                                                                h.shn, pf);
}
// k_hcrtW (UC, the column pass) on the segments ms0 .. ms0 + mns - 1 of every row (q positions), in st
template <int NP> void hcrtW_seg(GpuEngine* g, int64_t* out, cudaStream_t st, unsigned ms0, unsigned mns) {
    HybEngine& h = *g->hy;
    const HUK uk{ h.nd & (h.L - 1), h.ub, h.uq, h.ucb, h.lgM - h.lgR, h.uf };
    static const unsigned pf = hpfd(h, k_hcrtW<NP, 1, 3>, 288);
    const unsigned rows = 1u << h.lgR, q = (unsigned)(h.M >> h.lgR);
    k_hcrtW<NP, 1, 3><<<rows * mns, 288, 0, st>>>(h.a, reinterpret_cast<longlong2*>(out), h.dhp, h.lgL, h.nd, h.H, h.npass, h.P4,
                                                  h.hf, uk, (uint64_t)g->k, h.kinv, h.d, h.shn, pf, q, ms0, mns);
}
void hcrtW_seg_np(GpuEngine* g, int64_t* out, cudaStream_t st, unsigned ms0, unsigned mns) {
    if (g->hy->np == 7) hcrtW_seg<7>(g, out, st, ms0, mns);
    else hcrtW_seg<6>(g, out, st, ms0, mns);
}
// LOTR_GPU_L2ORD=3 (512 rows): the inverse column pass and the CRT + carries in chunks of columns, so the CRT reads the
// 7 primes from the L2 instead of DRAM: k_hcoli9 on the column groups of chunk c (all primes) in g->st, k_hcrtW of the
// same columns in h.st2 after it (and coli of chunk c + 1 meanwhile). A run of 256 positions needs the hs positions
// before it: in the same row (the chunks go left to right) except the first segment of a row (the end of the row above,
// the last chunk; block 0 wraps to the end): those 2^lgR segments last, after the join. The same arithmetic bit for bit,
// only the order of the launches. Uses ~1% less energy, but measured slower than order 2: kept as an option.
void hyb_colicrt(GpuEngine* g, int64_t* out) {
    HybEngine& h = *g->hy;
    const unsigned ngr = (unsigned)(h.M >> h.lgR) / HC_C, nch = (unsigned)h.cch, cg = ngr / nch, spc = cg * HC_C / 256;
    CK(cudaEventRecord(h.evf, g->st));
    CK(cudaStreamWaitEvent(h.st2, h.evf, 0));
    for (unsigned c = 0; c < nch; ++c) {
        k_hcoli9<1><<<h.cgi, HC9_T, HC9_SHM, g->st>>>(h.a, h.itw, h.lgM, h.np, h.dhp, h.ucq, 0, c * cg, cg, (unsigned)h.cpo);
        CK(cudaEventRecord(h.evk[c], g->st));
        CK(cudaStreamWaitEvent(h.st2, h.evk[c], 0));
        const unsigned s0 = c * spc + (c == 0 ? 1 : 0), ns = spc - (c == 0 ? 1 : 0);
        if (ns) hcrtW_seg_np(g, out, h.st2, s0, ns);
    }
    CK(cudaEventRecord(h.evk[nch], h.st2));
    CK(cudaStreamWaitEvent(g->st, h.evk[nch], 0));
    hcrtW_seg_np(g, out, g->st, 0, 1);   // the first segment of every row
}
// COLBENCH: k_hcrtW of the column pass apart - the loads and the stores, + the unpack and the CRT, + the parts and the
// carry passes; the occupancy, and the same with the largest shared carveout
template <typename F> double hbench_us(GpuEngine* g, F launch);   // (below, with COLBENCH)
template <int NP> void hcrtw_parts(GpuEngine* g) {
    HybEngine& h = *g->hy;
    const HUK uk{ h.nd & (h.L - 1), h.ub, h.uq, h.ucb, h.lgM - h.lgR, h.uf };
    longlong2* o = reinterpret_cast<longlong2*>(g->slot[1]);
    using KW = void (*)(const uint2*, longlong2*, const HP*, int, uint64_t, int, int, H4, HF, HUK, uint64_t, uint64_t, int, int, unsigned,
                        unsigned, unsigned, unsigned);
    auto run = [&](KW kern, int* occ) {
        int nb = 0;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kern, 288, 0));
        if (occ) *occ = nb;
        const unsigned pf = (unsigned)(h.nsm * std::max(nb, 1));
        return hbench_us(g, [&] {
            kern<<<(unsigned)(h.M >> 8), 288, 0, g->st>>>(h.a, o, h.dhp, h.lgL, h.nd, h.H, h.npass, h.P4, h.hf, uk, (uint64_t)g->k,
                                                          h.kinv, h.d, h.shn, pf, 0u, 0u, 0u);
        });
    };
    int o0 = 0, om = 0;
    const double c1 = run(k_hcrtW<NP, 1, 3, 1>, nullptr), c2 = run(k_hcrtW<NP, 1, 3, 2>, nullptr), c0 = run(k_hcrtW<NP, 1, 3, 0>, &o0);
    CK(cudaFuncSetAttribute(k_hcrtW<NP, 1, 3, 0>, cudaFuncAttributePreferredSharedMemoryCarveout, (int)cudaSharedmemCarveoutMaxShared));
    const double cm = run(k_hcrtW<NP, 1, 3, 0>, &om);
    CK(cudaFuncSetAttribute(k_hcrtW<NP, 1, 3, 0>, cudaFuncAttributePreferredSharedMemoryCarveout, (int)cudaSharedmemCarveoutDefault));
    cudaFuncAttributes fa;
    CK(cudaFuncGetAttributes(&fa, k_hcrtW<NP, 1, 3, 0>));
    std::printf("    crtW parts: loads + stores alone %.1f us, + unpack and CRT %.1f (+%.1f), + parts and carries %.1f (+%.1f);"
                " %d regs, %zu B spill, %zu B shared, %d blocks/SM (the largest shared carveout: %d blocks/SM, %.1f us)\n",
                c1, c2, c2 - c1, c0, c0 - c2, fa.numRegs, (size_t)fa.localSizeBytes, (size_t)fa.sharedSizeBytes, o0, om, cm);
}
template <int NP, int MT> void hcrtE_np(GpuEngine* g, int64_t* out, int x) {
    if (x == 7) hcrtE_l<NP, 7, MT>(g, out);
    else if (x == 6) hcrtE_l<NP, 6, MT>(g, out);
    else hcrtE_l<NP, 5, MT>(g, out);
}
// 6 (UC) after k_hcoli (the column pass), else 7 with the table uf, else 5 (UB); the wide digits: k_hcrtW, 6 or 7 primes
void hcrtE(GpuEngine* g, int64_t* out) {
    HybEngine& h = *g->hy;
    const int x = h.col ? 6 : h.uf ? 7 : 5;
    if (h.wide) {
        const int um = x == 6 ? 3 : x == 7 ? 4 : 2;
        if (h.np == 7) { if (um == 3) hcrtW_l<7, 3>(g, out); else if (um == 4) hcrtW_l<7, 4>(g, out); else hcrtW_l<7, 2>(g, out); }
        else { if (um == 3) hcrtW_l<6, 3>(g, out); else if (um == 4) hcrtW_l<6, 4>(g, out); else hcrtW_l<6, 2>(g, out); }
        return;
    }
    switch (h.np) {
    case 2: if (h.crtf) hcrtE_np<2, 1>(g, out, x); else hcrtE_np<2, 0>(g, out, x); break;
    case 3: if (h.crtf) hcrtE_np<3, 1>(g, out, x); else hcrtE_np<3, 0>(g, out, x); break;
    default: if (h.crtf) hcrtE_np<4, 1>(g, out, x); else hcrtE_np<4, 0>(g, out, x); break;
    }
}
// the inverse transform in h.a -> -k * value^2 mod N (top digit wide, the others nearly balanced) in out
void hyb_finish(GpuEngine* g, int64_t* out) {
    hcrtE(g, out);
    for (int i = 5; i <= 9; ++i) hmark(g, i);
}
void hyb_colicrt(GpuEngine* g, int64_t* out);
void hyb_invfin(GpuEngine* g, int64_t* out) {   // the inverse transform, then unpack + CRT + carries
    if (g->hy->l2ord == 3) { hyb_colicrt(g, out); return; }   // (coli -> CRT in chunks through the L2)
    hyb_inv(g, g->hy->a);
    hmark(g, 4);
    hyb_finish(g, out);
}
// LOTR_GPU_COLBENCH=1 (hybrid engine): every kernel of the hybrid square timed alone (garbage data, timing only),
// us per launch (10 after 2 warm-up) and GB/s of a read + write of the complex array (8 B x M x np); the ceiling of a
// pass is the classic coalesced copy of the same bytes
template <typename F> double hbench_us(GpuEngine* g, F launch) {
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
// The energy: a long test runs at the card's power limit (the clocks drop below those of the short benches above), so
// the speed of a test = the power limit / the joules of a square. NVML (libnvidia-ml loaded at run time, no build
// change) has a total energy counter in mJ that needs no admin rights (unlike the profiler counters).
// henergy: the launches for ~2 s after 0.5 s of warm-up, sustained.
struct HNvml {
    bool ok = false;
    void* dev = nullptr;
    int (*energy)(void*, unsigned long long*) = nullptr;
    int (*clock)(void*, int, unsigned*) = nullptr;
};
HNvml& hnvml(int device) {
    static HNvml s;
    static bool tried = false;
    if (tried) return s;
    tried = true;
    void* lib = dlopen("libnvidia-ml.so.1", RTLD_NOW);
    if (!lib) return s;
    auto init = (int (*)())dlsym(lib, "nvmlInit_v2");
    auto byp = (int (*)(const char*, void**))dlsym(lib, "nvmlDeviceGetHandleByPciBusId_v2");
    s.energy = (int (*)(void*, unsigned long long*))dlsym(lib, "nvmlDeviceGetTotalEnergyConsumption");
    s.clock = (int (*)(void*, int, unsigned*))dlsym(lib, "nvmlDeviceGetClockInfo");
    if (!init || !byp || !s.energy || !s.clock || init() != 0) return s;
    char bus[64] = {};
    if (cudaDeviceGetPCIBusId(bus, (int)sizeof(bus), device) != cudaSuccess || byp(bus, &s.dev) != 0) return s;
    unsigned long long e = 0;
    s.ok = s.energy(s.dev, &e) == 0;
    return s;
}
struct HEn { double us, mj, w; unsigned mhz; };   // a launch: us, mJ; the mean power, the SM clock at the end
template <typename F> HEn henergy(GpuEngine* g, F launch, int batch = 16, double secs = 2.0) {
    HNvml& nv = hnvml(g->device);
    using clk = std::chrono::steady_clock;
    auto run_for = [&](double s) {
        long n = 0;
        const auto t0 = clk::now();
        while (std::chrono::duration<double>(clk::now() - t0).count() < s) {
            for (int i = 0; i < batch; ++i) launch();
            n += batch;
            CK(cudaStreamSynchronize(g->st));
        }
        return n;
    };
    run_for(0.5);
    unsigned long long e0 = 0, e1 = 0;
    if (nv.ok) nv.energy(nv.dev, &e0);
    const auto t0 = clk::now();
    const long n = run_for(secs);
    const double dt = std::chrono::duration<double>(clk::now() - t0).count();
    if (nv.ok) nv.energy(nv.dev, &e1);
    unsigned mhz = 0;
    if (nv.ok) nv.clock(nv.dev, 1, &mhz);   // (NVML_CLOCK_SM)
    return { dt * 1e6 / (double)n, nv.ok ? (double)(e1 - e0) / (double)n : 0.0, nv.ok ? (double)(e1 - e0) / 1000.0 / dt : 0.0, mhz };
}
template <typename K> void hbench_line(GpuEngine* g, const char* name, K kern, int threads, double us, size_t shm = 0) {
    cudaFuncAttributes fa;
    int nb = 0;
    CK(cudaFuncGetAttributes(&fa, kern));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kern, threads, shm));
    const double bytes = 16.0 * (double)g->hy->M * g->hy->np;
    std::printf("    %-30s %9.1f us %7.1f GB/s  (%d regs, %zu B spill, %d blocks/SM)\n", name, us, bytes / (us * 1e3),
                fa.numRegs, (size_t)fa.localSizeBytes, nb);
}
// the ceilings of the card: products a second without memory (8 independent chains a thread, OP 0 mmul, 1 hdot, 2 cmul,
// 3 the dot mod 2^31 - 1 without Montgomery: what a Mersenne prime would cost), a copy of the array (in the L2 at the
// small sizes) and an empty kernel inside a CUDA Graph (the cost of one kernel boundary)
__device__ __forceinline__ uint32_t m31dot(uint32_t x, uint32_t A, uint32_t y, uint32_t B) {   // (x A + y B) mod 2^31 - 1
    const uint64_t t = (uint64_t)x * A + (uint64_t)y * B;
    const uint64_t r = (t & 0x7FFFFFFFu) + (t >> 31);
    const uint32_t u = (uint32_t)((r & 0x7FFFFFFFu) + (r >> 31));
    return u >= 0x7FFFFFFFu ? u - 0x7FFFFFFFu : u;
}
template <int OP> __global__ void __launch_bounds__(256) k_mbar(uint32_t* out, unsigned iters, const HP* hp) {
    const uint32_t p = hp->p[0], pv = hp->pv[0];
    uint2 x[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) x[i] = make_uint2((threadIdx.x * 8u + i + blockIdx.x * 77u) % p, (threadIdx.x * 3u + i * 5u + 1u) % p);
    for (unsigned it = 0; it < iters; ++it) {
        uint2 y[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {   // (both halves change every round: no product is loop invariant)
            const uint2 b = x[(i + 3) & 7];
            if (OP == 0) y[i] = make_uint2(mmul(x[i].x, b.x, p, pv), x[(i + 5) & 7].x);
            else if (OP == 1) y[i] = make_uint2(hdot(x[i].x, b.x, x[i].y, b.y, p, pv), x[(i + 5) & 7].x);
            else if (OP == 2) y[i] = cmul(x[i], b, p, pv);
            else y[i] = make_uint2(m31dot(x[i].x, b.x, x[i].y, b.y), x[(i + 5) & 7].x);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) x[i] = y[i];
    }
    uint32_t s = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) s ^= x[i].x ^ x[i].y;
    if (s == 0x9E3779B9u) out[0] = s;   // (keeps the chains)
}
__global__ void __launch_bounds__(256) k_mbcopy(const uint4* s, uint4* d, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) d[i] = s[i];
}
__global__ void k_mbnop(uint32_t* out) { if (threadIdx.x == 1023u) out[0] = 0; }
void hyb_square_once(GpuEngine* g, int slot);
void hyb_graph_l2(GpuEngine* g, cudaGraph_t gr);
// one square timed inside a CUDA Graph of 16 (as gpu_square), us a square (the values of slot 0 change)
double hyb_sq_graph_us(GpuEngine* g) {
    cudaGraph_t gr;
    cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(g->st, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < 16; ++i) hyb_square_once(g, 0);
    CK(cudaStreamEndCapture(g->st, &gr));
    hyb_graph_l2(g, gr);
    CK(cudaGraphInstantiate(&ge, gr, 0));
    const double us = hbench_us(g, [&] { CK(cudaGraphLaunch(ge, g->st)); }) / 16;
    cudaGraphExecDestroy(ge);
    cudaGraphDestroy(gr);
    return us;
}
// The L2 order bench: the column pass (100M) with LOTR_GPU_L2ORD unset and CUDA Graphs on: the square with all primes
// in one launch (0) vs prime by prime in 2 streams (2), two rounds each (the better counts), the faster one is kept per
// shape (2^lgM, primes, wide) - a card with a 32 MB L2 and 36 SMs gains with 2, one with 64 MB and 84 SMs loses ~3%.
// The same arithmetic bit for bit, only the order of the launches. A line "[GPU] L2 order bench: ..."
std::mutex g_l2o_mtx;
std::map<int, int> g_l2o_kept;
int hyb_l2ord_pick(GpuEngine* g) {
    HybEngine& h = *g->hy;
    const int key = (h.lgM << 8) | (h.np << 4) | h.wide;
    std::lock_guard<std::mutex> lk(g_l2o_mtx);
    const auto it = g_l2o_kept.find(key);
    if (it != g_l2o_kept.end()) return it->second;
    const auto t_start = std::chrono::steady_clock::now();
    double t0 = 1e300, t2 = 1e300, t3 = 1e300;
    const bool c3 = h.lgR == HC9_LGR;   // (coli -> CRT through the L2 in chunks, 512 rows only)
    for (int r = 0; r < 2; ++r) {
        h.l2ord = 0;
        t0 = std::min(t0, hyb_sq_graph_us(g));
        h.l2ord = 2;
        t2 = std::min(t2, hyb_sq_graph_us(g));
        if (c3) { h.l2ord = 3; t3 = std::min(t3, hyb_sq_graph_us(g)); }
    }
    const int pick = (t3 < t2 && t3 < t0) ? 3 : t2 < t0 ? 2 : 0;
    std::printf("[GPU] L2 order bench: all primes %.1f us vs prime by prime %.1f us", t0, t2);
    if (c3) std::printf(" vs coli -> CRT in %d chunks %.1f us", h.cch, t3);
    std::printf(" a square -> %s (%.1f s)\n", pick == 3 ? "coli -> CRT chunks" : pick ? "prime by prime" : "all primes",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count());
    std::fflush(stdout);
    g_l2o_kept[key] = pick;
    return pick;
}
void hyb_ceil(GpuEngine* g) {
    HybEngine& h = *g->hy;
    uint32_t* out = (uint32_t*)h.b2;
    const unsigned nb = (unsigned)h.nsm * 8, it = 1024;
    const double ops = (double)nb * 256 * 8 * it * 1e-3;   // / us -> G a second
    const double r0 = ops / hbench_us(g, [&] { k_mbar<0><<<nb, 256, 0, g->st>>>(out, it, h.dhp); });
    const double r1 = ops / hbench_us(g, [&] { k_mbar<1><<<nb, 256, 0, g->st>>>(out, it, h.dhp); });
    const double r2 = ops / hbench_us(g, [&] { k_mbar<2><<<nb, 256, 0, g->st>>>(out, it, h.dhp); });
    const double r3 = ops / hbench_us(g, [&] { k_mbar<3><<<nb, 256, 0, g->st>>>(out, it, h.dhp); });
    std::printf("    card: mmul %.0f G/s, hdot %.0f G/s, cmul %.0f G/s, M31 dot %.0f G/s (no memory)\n", r0, r1, r2, r3);
    const double cu = hbench_us(g, [&] {
        k_mbcopy<<<nb, 256, 0, g->st>>>((const uint4*)h.a, (uint4*)h.b2, h.M * h.np / 2); });
    std::printf("    %-30s %9.1f us %7.1f GB/s  (%.2f MB, read + write)\n", "copy of the array", cu,
                16.0 * h.M * h.np / (cu * 1e3), 8.0 * h.M * h.np / 1048576.0);
    for (unsigned gd : { grid((h.M >> 4) * h.np), 1u }) {   // 32 empty kernels in a graph: the grid of a radix-16 pass, 1 block
        cudaGraph_t gr;
        cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(g->st, cudaStreamCaptureModeThreadLocal));
        for (int i = 0; i < 32; ++i) k_mbnop<<<gd, 256, 0, g->st>>>(out);
        CK(cudaStreamEndCapture(g->st, &gr));
        CK(cudaGraphInstantiate(&ge, gr, 0));
        const double us = hbench_us(g, [&] { CK(cudaGraphLaunch(ge, g->st)); }) / 32;
        cudaGraphExecDestroy(ge);
        cudaGraphDestroy(gr);
        std::printf("    %-30s %9.2f us  (%u blocks)\n", "empty kernel in a CUDA Graph", us, gd);
    }
    // the square: its parts alone, the whole in the stream, and in a graph of 16 (as gpu_square: the speed of a test)
    int s[16];
    const int ns = h.col ? 1 : hyb_passes(h.lgM - (h.wide ? 11 : 12), s);
    const double fw = hbench_us(g, [&] { hyb_fwd(g, g->slot[0], h.a); });
    const double bl = hbench_us(g, [&] { hyb_block<0>(g, h.a); });
    const double iv = hbench_us(g, [&] { hyb_inv(g, h.a); });
    const double cr = hbench_us(g, [&] { hcrtE(g, g->slot[1]); });
    const double sq = hbench_us(g, [&] { hyb_square_once(g, 0); });
    std::printf("    square: fwd %.1f + block %.1f + inv %.1f + crtE %.1f = %.1f us (%d kernels); in the stream %.1f us",
                fw, bl, iv, cr, fw + bl + iv + cr, 2 * ns + 2 + (ns == 0 ? 1 : 0), sq);
    if (!g->prof) {
        cudaGraph_t gr;
        cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(g->st, cudaStreamCaptureModeThreadLocal));
        for (int i = 0; i < 16; ++i) hyb_square_once(g, 0);
        CK(cudaStreamEndCapture(g->st, &gr));
        CK(cudaGraphInstantiate(&ge, gr, 0));
        const double us = hbench_us(g, [&] { CK(cudaGraphLaunch(ge, g->st)); }) / 16;
        cudaGraphExecDestroy(ge);
        cudaGraphDestroy(gr);
        std::printf(", in a CUDA Graph %.1f us", us);
    }
    std::printf("\n");
    if (h.col) {   // LOTR_GPU_L2ORD: block + coli prime by prime (through the L2) vs all primes in one launch each
        const double ba = hbench_us(g, [&] { hyb_block<0>(g, h.a); hyb_inv(g, h.a); });
        const double bp = hbench_us(g, [&] { hyb_blkinv_l2(g, h.a); });
        std::printf("    block + coli: all primes %.1f us, prime by prime (L2 order) %.1f us (%+.1f%%)%s\n", ba, bp,
                    100.0 * (bp / ba - 1.0), h.l2ord ? ", L2 order in use" : "");
        // the split alone (the tail of np small launches, no L2 reuse: all blocks first, then all colis): the gain of
        // the L2 = this sum - the prime by prime time above
        const double b1 = hbench_us(g, [&] { for (int c = 0; c < h.np; ++c) hyb_block<0>(g, h.a, c); });
        const double c1 = hbench_us(g, [&] {
            for (int c = 0; c < h.np; ++c) hcoli_l(g, h.a, 1, (unsigned)c, g->st);
        });
        std::printf("    split alone: block %d x 1 prime %.1f us, coli %d x 1 prime %.1f us = %.1f us -> the L2 gain %.1f us (%.1f%%)\n",
                    h.np, b1, h.np, c1, b1 + c1, b1 + c1 - bp, 100.0 * (b1 + c1 - bp) / (b1 + c1));
        const double bo = hbench_us(g, [&] { hyb_blkinv_l2o(g, h.a); });
        std::printf("    prime by prime, 2 streams (LOTR_GPU_L2ORD=2) %.1f us (%+.1f%% vs all primes)\n", bo, 100.0 * (bo / ba - 1.0));
        if (!g->prof) {   // the whole square in a graph of 16 with each order (as gpu_square)
            const int keep = h.l2ord;
            for (int o = 0; o <= (h.lgR == HC9_LGR ? 3 : 2); ++o) {
                h.l2ord = o;
                std::printf("%s LOTR_GPU_L2ORD=%d %.1f", o ? "," : "    square in a CUDA Graph:", o, hyb_sq_graph_us(g));
            }
            std::printf(" us (in use: %d)\n", keep);
            h.l2ord = keep;
        }
        // Measured on a 64 MB L2: nothing passes usefully from kernel to kernel through the L2 (colf right after crtW, crtW
        // right after block + coli: no faster than after a flush), and pinning the state digits in the L2 is slower.
        if (h.wide && h.lgR == HC9_LGR) {   // the same for k_hcolf9 (512 rows, 2 blocks per SM)
            const HPK pk{ h.tl, h.th, h.thm, h.pkh, h.pkl, h.nd & (h.L - 1), h.pwa, h.pwb };
            const int64_t* d = g->slot[0];
            using K9 = void (*)(uint2*, const uint2*, int, int, const HP*, const int64_t*, HPK);
            auto t9 = [&](K9 kern) {   // (the same carveout and grid as the kernel)
                CK(cudaFuncSetAttribute(kern, cudaFuncAttributePreferredSharedMemoryCarveout, (int)cudaSharedmemCarveoutMaxShared));
                return hbench_us(g, [&] { kern<<<h.cgf, HC9_T, HC9_SHM, g->st>>>(h.a, h.tw, h.lgM, h.np, h.dhp, d, pk); });
            };
            const double f0 = t9(k_hcolf9<true, 0>), f2 = t9(k_hcolf9<true, 2>), f1 = t9(k_hcolf9<true, 1>);
            // where the rest goes: without the barriers, without shared memory, without / with a deeper L2 prefetch
            const double f3 = t9(k_hcolf9<true, 3>), f4 = t9(k_hcolf9<true, 4>), f1p0 = t9(k_hcolf9<true, 1, 0>);
            const double f1p2 = t9(k_hcolf9<true, 1, 2>), f0p0 = t9(k_hcolf9<true, 0, 0>), f0p2 = t9(k_hcolf9<true, 0, 2>);
            std::printf("    colf9 parts: loads + stores alone (no shared) %.1f us; + shared %.1f without barriers / %.1f with (the barriers"
                        " %+.1f); L2 prefetch none / next / 2 ahead: memory %.1f / %.1f / %.1f, kernel %.1f / %.1f / %.1f us\n",
                        f4, f3, f1, f1 - f3, f1p0, f1, f1p2, f0p0, f0, f0p2);
            // the twiddles of the next tile prefetched into the L2 (TP 1, the kernels) or not (TP 0)
            const double f0t0 = t9(k_hcolf9<true, 0, 1, 0>);
            CK(cudaFuncSetAttribute(k_hcoli9<0>, cudaFuncAttributePreferredSharedMemoryCarveout, (int)cudaSharedmemCarveoutMaxShared));
            double ci[2], cp[2];
            for (int t = 0; t < 2; ++t) {
                auto ki = t ? k_hcoli9<1> : k_hcoli9<0>;
                ci[t] = hbench_us(g, [&] { ki<<<h.cgi, HC9_T, HC9_SHM, g->st>>>(h.a, h.itw, h.lgM, h.np, h.dhp, h.ucq, 0, 0, 0, 0); });
                cp[t] = hbench_us(g, [&] {
                    for (int c = 0; c < h.np; ++c)
                        ki<<<h.cgi, HC9_T, HC9_SHM, g->st>>>(h.a, h.itw, h.lgM, 1, h.dhp, h.ucq, (unsigned)c, 0, 0, 0);
                });
            }
            std::printf("    twiddles prefetched into the L2: colf9 without %.1f / with %.1f us; coli9 all primes %.1f / %.1f,"
                        " %d x 1 prime %.1f / %.1f us\n", f0t0, f0, ci[0], ci[1], h.np, cp[0], cp[1]);
            if (h.np == 7) hcrtw_parts<7>(g);
            else hcrtw_parts<6>(g);
            // the energy, sustained: what a square and every kernel costs in joules at the power limit
            if (!hnvml(g->device).ok) std::printf("    energy: NVML not available\n");
            else {
                auto sqg = [&](int o) {   // the square in a graph of 16 (as gpu_square) with the order o
                    const int keep = h.l2ord;
                    h.l2ord = o;
                    cudaGraph_t gr;
                    cudaGraphExec_t ge;
                    CK(cudaStreamBeginCapture(g->st, cudaStreamCaptureModeThreadLocal));
                    for (int i = 0; i < 16; ++i) hyb_square_once(g, 0);
                    CK(cudaStreamEndCapture(g->st, &gr));
                    hyb_graph_l2(g, gr);
                    CK(cudaGraphInstantiate(&ge, gr, 0));
                    HEn e = henergy(g, [&] { CK(cudaGraphLaunch(ge, g->st)); }, 2);
                    cudaGraphExecDestroy(ge);
                    cudaGraphDestroy(gr);
                    h.l2ord = keep;
                    e.us /= 16;
                    e.mj /= 16;
                    return e;
                };
                std::printf("    energy, sustained 2 s each (NVML): the square in a CUDA Graph");
                for (int o : { 0, 2, 3 }) {
                    const HEn e = sqg(o);
                    std::printf("%s L2ORD=%d %.1f us %.1f mJ (%.0f W, %u MHz)", o ? "," : "", o, e.us, e.mj, e.w, e.mhz);
                }
                std::printf("\n");
                std::fflush(stdout);
                const int keep = h.l2ord;
                h.l2ord = 0;   // (hyb_inv: coli9 of all primes in one launch)
                const HEn ef = henergy(g, [&] { hyb_fwd(g, g->slot[0], h.a); });
                const HEn eb = henergy(g, [&] { hyb_block<0>(g, h.a); });
                const HEn ei = henergy(g, [&] { hyb_inv(g, h.a); });
                const HEn ec = henergy(g, [&] { hcrtE(g, g->slot[1]); });
                const HEn ex = henergy(g, [&] { hyb_colicrt(g, g->slot[1]); });
                h.l2ord = keep;
                std::printf("    energy per kernel (us / mJ / W): colf9 %.1f / %.1f / %.0f, block %.1f / %.1f / %.0f, coli9 %.1f / %.1f / %.0f,"
                            " crtW %.1f / %.1f / %.0f; coli -> CRT in %d chunks %.1f / %.1f / %.0f (coli9 + crtW %.1f / %.1f)\n",
                            ef.us, ef.mj, ef.w, eb.us, eb.mj, eb.w, ei.us, ei.mj, ei.w, ec.us, ec.mj, ec.w, h.cch, ex.us, ex.mj, ex.w,
                            ei.us + ec.us, ei.mj + ec.mj);
            }
            cudaFuncAttributes f0a, f2a, fia;
            CK(cudaFuncGetAttributes(&f0a, k_hcolf9<true, 0>));
            CK(cudaFuncGetAttributes(&f2a, k_hcolf9<true, 2>));
            CK(cudaFuncGetAttributes(&fia, k_hcoli9<1>));
            std::printf("    colf9 apart (512 rows, %u + %u blocks = %u / %u per SM): memory + shared alone %.1f us, + pack %.1f (+%.1f),"
                        " + butterflies %.1f (+%.1f); regs colf9 %d / %d (pack only) / coli9 %d, spill %zu / %zu / %zu B\n",
                        h.cgf, h.cgi, h.cgf / (unsigned)h.nsm, h.cgi / (unsigned)h.nsm, f1, f2, f2 - f1, f0, f0 - f2, f0a.numRegs,
                        f2a.numRegs, fia.numRegs, (size_t)f0a.localSizeBytes, (size_t)f2a.localSizeBytes, (size_t)fia.localSizeBytes);
        }
        else if (h.wide) {   // k_hcolf apart: the memory + shared pattern alone, + the pack, + the butterflies
            static const bool cm = [] {
                CK(cudaFuncSetAttribute(k_hcolf<true, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)HC_SHM));
                CK(cudaFuncSetAttribute(k_hcolf<true, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)HC_SHM));
                return true;
            }();
            (void)cm;
            const HPK pk{ h.tl, h.th, h.thm, h.pkh, h.pkl, h.nd & (h.L - 1), h.pwa, h.pwb };
            const int64_t* d = g->slot[0];
            const double f0 = hbench_us(g, [&] { k_hcolf<true, 0><<<h.nsm, HC_T, HC_SHM, g->st>>>(h.a, h.tw, h.lgM, h.np, h.dhp, d, pk); });
            const double f2 = hbench_us(g, [&] { k_hcolf<true, 2><<<h.nsm, HC_T, HC_SHM, g->st>>>(h.a, h.tw, h.lgM, h.np, h.dhp, d, pk); });
            const double f1 = hbench_us(g, [&] { k_hcolf<true, 1><<<h.nsm, HC_T, HC_SHM, g->st>>>(h.a, h.tw, h.lgM, h.np, h.dhp, d, pk); });
            cudaFuncAttributes f0a, f2a;
            CK(cudaFuncGetAttributes(&f0a, k_hcolf<true, 0>));
            CK(cudaFuncGetAttributes(&f2a, k_hcolf<true, 2>));
            std::printf("    colf apart: memory + shared alone %.1f us, + pack %.1f (+%.1f), + butterflies %.1f (+%.1f); %.0f MB moved"
                        " (digits 16 B read, 8 B x %d primes written); regs %d / %d (pack only), spill %zu / %zu B\n",
                        f1, f2, f2 - f1, f0, f0 - f2, (16.0 * h.L + 8.0 * h.M * h.np) / 1048576.0, h.np, f0a.numRegs,
                        f2a.numRegs, (size_t)f0a.localSizeBytes, (size_t)f2a.localSizeBytes);
        }
    }
}
// the wide digits (timing only): at the column-pass sizes (100M), 2^(lgM-1) complex x 7 and x 6 primes (the primes of
// the plan for k, garbage tables and data < 2^29), colf<WIDE> + block11 + coli + crtW
__global__ void k_mbfill(uint32_t* x, size_t n, uint32_t salt) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t v = (uint32_t)i * 2654435761u ^ salt;
        v ^= v >> 15;
        v *= 2246822519u;
        v ^= v >> 13;
        x[i] = v & 0x1FFFFFFFu;
    }
}
void hyb_wbench(GpuEngine* g) {
    const HybEngine& h = *g->hy;
    const int lgL = h.lgL - 1, lgM = lgL - 1;
    const size_t L = (size_t)1 << lgL, M = L >> 1, cq = M >> HC_LGR;
    const uint64_t k = g->k, nd = h.nd;
    std::vector<uint32_t> ps;
    for (uint32_t p : HYB_PRIMES)
        if (__builtin_ctzll((uint64_t)p + 1) >= lgL && k % p != 0 && pw(k % p, (p - 1) / 2, p) == 1) ps.push_back(p);
    const unsigned nb = (unsigned)h.nsm * 8;
    auto fill = [&](void* x, size_t bytes, uint32_t salt) { k_mbfill<<<nb, 256, 0, g->st>>>((uint32_t*)x, bytes / 4, salt); };
    for (int np : { 7, 6 }) {
        if ((int)ps.size() < np) { std::printf("    wide 2^%d x %d: only %zu primes for k\n", lgM, np, ps.size()); continue; }
        HP hp{};
        double s = 0;
        for (int i = 0; i < np; ++i) {
            const uint64_t p = ps[i], R = (1ull << 32) % p;
            uint32_t x = 1;
            for (int it = 0; it < 5; ++it) x *= 2 - (uint32_t)p * x;
            hp.p[i] = (uint32_t)p; hp.pv[i] = 0u - x;
            hp.r2[i] = (uint32_t)(R * R % p); hp.r3[i] = (uint32_t)(R * R % p * R % p);
            hp.c52[i] = (uint32_t)((1ull << 52) % p);
            hp.c84[i] = (uint32_t)(((u128)1 << 84) % p);
            hp.cw[i] = hp.ck[i] = hp.cu[i] = hp.r2[i];
            for (int j = 0; j < i; ++j) hp.inv[i][j] = (uint32_t)(pw(hp.p[j] % p, p - 2, p) * R % p);
            s += std::log2((double)p);
        }
        const int Bm = (int)(nd / L), B = (int)((nd + L - 1) / L), H = std::max(2, (int)std::ceil((s - B) / Bm) + 1);
        int npass = 0;
        for (double bound = (3.0 * H + 2) * std::ldexp(1.0, B);;) {
            const double gg = std::floor((bound + std::ldexp(1.0, Bm - 1)) / std::ldexp(1.0, Bm));
            bound = std::ldexp(1.0, B - 1) + gg;
            ++npass;
            if (gg <= 1) break;
        }
        H4 P{ { 1, 0, 0, 0 } };
        for (int i = 0; i < np; ++i) {
            u128 c = 0;
            for (int l = 0; l < 4; ++l) { c += (u128)P.w[l] * hp.p[i]; P.w[l] = (uint64_t)c; c >>= 64; }
        }
        HF F{};   // hcrtf: c_i x R, 1 / p_i, P / p_i and P in words of 32 bits
        auto mulw = [](uint32_t* x, uint32_t m) {
            uint64_t c = 0;
            for (int l = 0; l < 8; ++l) { c += (uint64_t)x[l] * m; x[l] = (uint32_t)c; c >>= 32; }
        };
        F.P[0] = 1;
        for (int i = 0; i < np; ++i) mulw(F.P, hp.p[i]);
        for (int i = 0; i < np; ++i) {
            const uint64_t p = hp.p[i], R = (1ull << 32) % p;
            uint64_t qm = 1;
            F.Q[i][0] = 1;
            for (int j = 0; j < np; ++j)
                if (j != i) { mulw(F.Q[i], hp.p[j]); qm = qm * hp.p[j] % p; }
            F.ci[i] = (uint32_t)(pw(qm, p - 2, p) * R % p);
            F.ip[i] = 1.0 / (double)p;
        }
        const int shn = (int)(g->n - hpos(L - 1, nd, lgL));
        HP* dhp = nullptr;
        uint2 *a = nullptr, *tw = nullptr, *itw = nullptr, *pkh = nullptr, *pkl = nullptr, *ht = nullptr, *iht = nullptr;
        uint2 *ucb = nullptr, *ucq = nullptr;
        uint32_t *tl = nullptr, *th = nullptr, *thm = nullptr;
        longlong2* dg = nullptr;
        CK(cudaMalloc(&dhp, sizeof(HP)));
        CK(cudaMemcpyAsync(dhp, &hp, sizeof(HP), cudaMemcpyHostToDevice, g->st));
        const size_t LH = L >> 11, MH = M >> 11;
        CK(cudaMalloc(&a, sizeof(uint2) * M * np)); fill(a, sizeof(uint2) * M * np, 1);
        CK(cudaMalloc(&tw, sizeof(uint2) * M * np)); fill(tw, sizeof(uint2) * M * np, 2);
        CK(cudaMalloc(&itw, sizeof(uint2) * M * np)); fill(itw, sizeof(uint2) * M * np, 3);
        CK(cudaMalloc(&dg, sizeof(longlong2) * L)); fill(dg, sizeof(longlong2) * L, 4);
        CK(cudaMalloc(&pkh, sizeof(uint2) * MH * np)); fill(pkh, sizeof(uint2) * MH * np, 5);
        CK(cudaMalloc(&pkl, sizeof(uint2) * 2048 * np)); fill(pkl, sizeof(uint2) * 2048 * np, 6);
        CK(cudaMalloc(&tl, sizeof(uint32_t) * 2048 * np)); fill(tl, sizeof(uint32_t) * 2048 * np, 7);
        CK(cudaMalloc(&th, sizeof(uint32_t) * LH * np)); fill(th, sizeof(uint32_t) * LH * np, 8);
        CK(cudaMalloc(&thm, sizeof(uint32_t) * LH * np)); fill(thm, sizeof(uint32_t) * LH * np, 9);
        CK(cudaMalloc(&ht, sizeof(uint2) * 2048 * np)); fill(ht, sizeof(uint2) * 2048 * np, 10);
        CK(cudaMalloc(&iht, sizeof(uint2) * 2048 * np)); fill(iht, sizeof(uint2) * 2048 * np, 11);
        CK(cudaMalloc(&ucb, sizeof(uint2) * (M / cq) * np)); fill(ucb, sizeof(uint2) * (M / cq) * np, 12);
        CK(cudaMalloc(&ucq, sizeof(uint2) * cq * np)); fill(ucq, sizeof(uint2) * cq * np, 13);
        const HPK pk{ tl, th, thm, pkh, pkl, nd & (L - 1), nullptr, nullptr };
        const HUK uk{ nd & (L - 1), nullptr, nullptr, ucb, lgM - HC_LGR, nullptr };
        static const bool c11 = [] {
            CK(cudaFuncSetAttribute(k_hblock11<0>, cudaFuncAttributePreferredSharedMemoryCarveout, (int)cudaSharedmemCarveoutMaxShared));
            return true;
        }();
        (void)c11;
        const void* kw0 = np == 7 ? (const void*)k_hcrtW<7, 0> : (const void*)k_hcrtW<6, 0>;
        const void* kw1 = np == 7 ? (const void*)k_hcrtW<7, 1> : (const void*)k_hcrtW<6, 1>;
        int ob = 0, oc = 0, of = 0;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ob, k_hblock11<0>, 256, 0));
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&oc, kw0, 288, 0));
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&of, kw1, 288, 0));
        const unsigned pfb = (unsigned)(h.nsm * std::max(ob, 1)), pfc = (unsigned)(h.nsm * std::max(oc, 1));
        const unsigned pff = (unsigned)(h.nsm * std::max(of, 1));
        const double cf = hbench_us(g, [&] {
            k_hcolf<true><<<h.nsm, HC_T, HC_SHM, g->st>>>(a, tw, lgM, np, dhp, (const int64_t*)dg, pk); });
        const double bl = hbench_us(g, [&] {
            k_hblock11<0><<<(unsigned)((M >> 11) * np / 2), 256, 0, g->st>>>(a, tw, itw, lgM, dhp, pfb, ht, iht, 0); });
        const double ci = hbench_us(g, [&] { k_hcoli<<<h.nsm, HC_T, HC_SHM, g->st>>>(a, itw, lgM, np, dhp, ucq, 0); });
        const unsigned gm = (unsigned)(M >> 8);
        const double cr = hbench_us(g, [&] {
            if (np == 7) k_hcrtW<7, 0><<<gm, 288, 0, g->st>>>(a, dg, dhp, lgL, nd, H, npass, P, F, uk, k, h.kinv, h.d, shn, pfc);
            else k_hcrtW<6, 0><<<gm, 288, 0, g->st>>>(a, dg, dhp, lgL, nd, H, npass, P, F, uk, k, h.kinv, h.d, shn, pfc);
        });
        const double crf = hbench_us(g, [&] {
            if (np == 7) k_hcrtW<7, 1><<<gm, 288, 0, g->st>>>(a, dg, dhp, lgL, nd, H, npass, P, F, uk, k, h.kinv, h.d, shn, pff);
            else k_hcrtW<6, 1><<<gm, 288, 0, g->st>>>(a, dg, dhp, lgL, nd, H, npass, P, F, uk, k, h.kinv, h.d, shn, pff);
        });
        cudaFuncAttributes fb, fc, ff;
        CK(cudaFuncGetAttributes(&fb, k_hblock11<0>));
        CK(cudaFuncGetAttributes(&fc, kw0));
        CK(cudaFuncGetAttributes(&ff, kw1));
        std::printf("    wide 2^%d x %d (%d-bit digits, H %d, %d passes, array %.1f MB): colf %.1f + block11 %.1f + coli %.1f + crtW %.1f"
                    " = %.1f us; crtW fractional %.1f -> %.1f us\n", lgM, np, B, H, npass, 8.0 * M * np / 1048576.0, cf, bl, ci,
                    cr, cf + bl + ci + cr, crf, cf + bl + ci + crf);
        std::printf("      (block11 %d regs %zu B spill %d/SM, crtW Garner %d regs %zu B spill %d/SM, fractional %d regs %zu B spill %d/SM)\n",
                    fb.numRegs, (size_t)fb.localSizeBytes, ob, fc.numRegs, (size_t)fc.localSizeBytes, oc, ff.numRegs,
                    (size_t)ff.localSizeBytes, of);
        for (void* x : { (void*)dhp, (void*)a, (void*)tw, (void*)itw, (void*)dg, (void*)pkh, (void*)pkl, (void*)tl, (void*)th,
                         (void*)thm, (void*)ht, (void*)iht, (void*)ucb, (void*)ucq })
            cudaFree(x);
    }
}
void hyb_bench(GpuEngine* g) {
    HybEngine& h = *g->hy;
    const int np = h.np, lgM = h.lgM;
    const size_t M = h.M;
    const int64_t* d = g->slot[0];
    uint2* a = h.a;
    const HPK pk{ h.tl, h.th, h.thm, h.pkh, h.pkl, h.nd & (h.L - 1), h.pwa, h.pwb };
    std::printf("[GPU COLBENCH] HYBRID 2^%d complex x %d primes: us per launch%s%s%s%s\n", lgM, np, h.pwa ? " (pack table)" : "",
                h.uf ? " (unpack table)" : "", h.wide ? " (wide digits)" : "", h.crtf ? " (fractional CRT)" : "");
    if (h.wide) {   // (the kernels below are those of the 40-bit digits: only the square as a whole)
        hyb_ceil(g);
        std::fflush(stdout);
        return;
    }
    hbench_line(g, "k_hpack (alone)", k_hpack<false>, 256,
                hbench_us(g, [&] { k_hpack<false><<<grid(M * np), 256, 0, g->st>>>(d, a, h.dhp, h.lgL, np, pk); }));
    if (lgM - 6 >= 12) {
        const unsigned g64 = (unsigned)((M >> 11) * np);
        hbench_line(g, "k_hdif64<true> (packs)", k_hdif64<true>, 256, hbench_us(g, [&] {
            k_hdif64<true><<<g64, 256, 0, g->st>>>(a, h.tw, lgM, lgM - 6, h.dhp, d, pk); }));
        hbench_line(g, "k_hdif64 q=M/64", k_hdif64<false>, 256, hbench_us(g, [&] {
            k_hdif64<false><<<g64, 256, 0, g->st>>>(a, h.tw, lgM, lgM - 6, h.dhp, d, pk); }));
        hbench_line(g, "k_hdit64 q=M/64", k_hdit64, 256, hbench_us(g, [&] {
            k_hdit64<<<g64, 256, 0, g->st>>>(a, h.itw, lgM, lgM - 6, h.dhp); }));
    }
    if (lgM - 4 >= 12) {
        hbench_line(g, "k_hdif16<true> q=4096 (packs)", k_hdif16<true>, 256, hbench_us(g, [&] {
            k_hdif16<true><<<grid((M >> 4) * np), 256, 0, g->st>>>(a, h.tw, lgM, 12, np, h.dhp, d, pk); }));
        hbench_line(g, "k_hdif16 q=4096", k_hdif16<false>, 256, hbench_us(g, [&] {
            k_hdif16<false><<<grid((M >> 4) * np), 256, 0, g->st>>>(a, h.tw, lgM, 12, np, h.dhp, d, pk); }));
        hbench_line(g, "k_hdit16 q=4096", k_hdit16, 256, hbench_us(g, [&] {
            k_hdit16<<<grid((M >> 4) * np), 256, 0, g->st>>>(a, h.itw, lgM, 12, np, h.dhp); }));
    }
    if (h.col) {   // the column pass (one block per SM, persistent)
        hbench_line(g, "k_hcolf (packs, hr16f)", k_hcolf<false>, HC_T, hbench_us(g, [&] {
            k_hcolf<false><<<h.nsm, HC_T, HC_SHM, g->st>>>(a, h.tw, lgM, np, h.dhp, d, pk); }), HC_SHM);
        hbench_line(g, "k_hcoli (prime inner, UC)", k_hcoli, HC_T, hbench_us(g, [&] {
            k_hcoli<<<h.nsm, HC_T, HC_SHM, g->st>>>(a, h.itw, lgM, np, h.dhp, h.ucq, 0); }), HC_SHM);
    }
    hbench_line(g, "k_hblock16<0> MB 3 tw tables", k_hblock16<0>, 256, hbench_us(g, [&] { hyb_block<0>(g, a); }));
    const double us = hbench_us(g, [&] { hcrtE(g, g->slot[1]); });
    const int x = h.col ? 6 : h.uf ? 7 : 5;
    const char* cn = x == 6 ? "k_hcrtE UC (row; col in coli)" : x == 7 ? "k_hcrtE unpack table" : "k_hcrtE UB (block x lane)";
    switch (np) {   // (the kernel only for its registers and blocks per SM)
    case 2: hbench_line(g, cn, x == 7 ? k_hcrtE<2, 7> : x == 6 ? k_hcrtE<2, 6> : k_hcrtE<2, 5>, 288, us); break;
    case 3: hbench_line(g, cn, x == 7 ? k_hcrtE<3, 7> : x == 6 ? k_hcrtE<3, 6> : k_hcrtE<3, 5>, 288, us); break;
    default: hbench_line(g, cn, x == 7 ? k_hcrtE<4, 7> : x == 6 ? k_hcrtE<4, 6> : k_hcrtE<4, 5>, 288, us); break;
    }
    hyb_ceil(g);
    if (h.col && !h.wide) hyb_wbench(g);
    std::fflush(stdout);
}
// LOTR_GPU_L2P: the window on every kernel node of a captured graph (gpu_square; the stream attribute is not carried into
// a capture reliably)
void hyb_graph_l2(GpuEngine* g, cudaGraph_t gr) {
    if (!g->hy || !g->hy->l2on) return;
    size_t n = 0;
    CK(cudaGraphGetNodes(gr, nullptr, &n));
    std::vector<cudaGraphNode_t> nodes(n);
    CK(cudaGraphGetNodes(gr, nodes.data(), &n));
    cudaKernelNodeAttrValue av{};
    av.accessPolicyWindow = g->hy->l2w;
    for (cudaGraphNode_t nd : nodes) {
        cudaGraphNodeType t;
        CK(cudaGraphNodeGetType(nd, &t));
        if (t == cudaGraphNodeTypeKernel) CK(cudaGraphKernelNodeSetAttribute(nd, cudaKernelNodeAttributeAccessPolicyWindow, &av));
    }
}
void hyb_square_once(GpuEngine* g, int slot) {
    HybEngine& h = *g->hy;
    hmark(g, 0);
    hyb_fwd(g, g->slot[slot], h.a);
    hmark(g, 2);
    if (h.l2ord == 3) {   // (coli -> CRT in chunks)
        hyb_block<0>(g, h.a);
        hmark(g, 3);
        hmark(g, 4);
        hyb_colicrt(g, g->slot[slot]);
        for (int i = 5; i <= 9; ++i) hmark(g, i);
    }
    else if (h.l2ord) {
        if (h.l2ord == 2) hyb_blkinv_l2o(g, h.a);
        else hyb_blkinv_l2(g, h.a);
        hmark(g, 3);
        hmark(g, 4);
        hyb_finish(g, g->slot[slot]);
    }
    else {
        hyb_block<0>(g, h.a);
        hmark(g, 3);
        hyb_invfin(g, g->slot[slot]);
    }
    if (g->prof) prof_step(g);
}
void hyb_mul(GpuEngine* g, int dst, int x, int y) {
    HybEngine& h = *g->hy;
    if (x == y) {
        hyb_fwd(g, g->slot[x], h.a);
        hyb_block<0>(g, h.a);
    }
    else {
        hyb_fwd(g, g->slot[x], h.a);
        hyb_block<1>(g, h.a);
        hyb_fwd(g, g->slot[y], h.b2);
        hyb_block<1>(g, h.b2);
        k_hpmul<<<grid(h.M * h.np), 256, 0, g->st>>>(h.a, h.b2, h.lgM, h.np, h.dhp);
        hyb_block<2>(g, h.a);
    }
    hyb_invfin(g, g->slot[dst]);
}

// host digits <-> values: positions e_t, the top digit unbalanced
void hyb_set(GpuEngine* g, int slot, const mpz_t v) {
    HybEngine& h = *g->hy;
    gpu_sync(g);   // the staging buffer may still be in use
    if (mpz_sgn(v) < 0 || mpz_sizeinbase(v, 2) > h.nd) throw std::runtime_error("gpu_set: the value does not fit the digits");
    std::vector<uint64_t> w((mpz_sizeinbase(v, 2) + 63) / 64 + 3, 0);
    size_t cnt = 0;
    mpz_export(w.data(), &cnt, -1, sizeof(uint64_t), 0, 0, v);
    auto word = [&](size_t i) { return i < w.size() ? w[i] : (uint64_t)0; };
    i128 carry = 0;
    for (size_t t = 0; t < h.L; ++t) {   // (digits up to 96 bits, i128; the wide digits as lo, hi)
        const uint64_t e0 = hpos(t, h.nd, h.lgL);
        const int b = (int)(hpos(t + 1, h.nd, h.lgL) - e0);
        const size_t i = e0 >> 6;
        const int sh = (int)(e0 & 63);
        u128 x = (((u128)word(i + 1) << 64) | word(i)) >> sh;
        if (sh) x |= (u128)word(i + 2) << (128 - sh);
        i128 f = (i128)(x & (((u128)1 << b) - 1)) + carry;
        carry = 0;
        if (t + 1 < h.L && f >= ((i128)1 << (b - 1))) { f -= (i128)1 << b; carry = 1; }
        if (h.wide) { g->host[2 * t] = (int64_t)(uint64_t)f; g->host[2 * t + 1] = (int64_t)(f >> 64); }
        else g->host[t] = (int64_t)f;
    }
    CK(cudaMemcpyAsync(g->slot[slot], g->host, (h.wide ? 16 : 8) * h.L, cudaMemcpyHostToDevice, g->st));
    gpu_sync(g);
}
bool hyb_get(GpuEngine* g, int slot, mpz_t v) {
    HybEngine& h = *g->hy;
    CK(cudaMemcpyAsync(g->host, g->slot[slot], (h.wide ? 16 : 8) * h.L, cudaMemcpyDeviceToHost, g->st));
    gpu_sync(g);
    bool ok = true;
    std::vector<uint64_t> w((size_t)(h.nd / 64) + 4, 0);
    i128 carry = 0;
    for (size_t t = 0; t < h.L; ++t) {
        const uint64_t e0 = hpos(t, h.nd, h.lgL);
        const int b = (int)(hpos(t + 1, h.nd, h.lgL) - e0);
        const i128 x = h.wide ? (i128)(((u128)(uint64_t)g->host[2 * t + 1] << 64) | (uint64_t)g->host[2 * t])
                              : (i128)g->host[t];
        const double lim = (t + 1 < h.L) ? std::ldexp(1.0, b - 1) + 1 : h.Dt;
        if ((double)x > lim || (double)x < -lim) ok = false;
        const i128 s = x + carry;
        const u128 dg = (u128)(s & (((i128)1 << b) - 1));
        carry = s >> b;
        const size_t i = e0 >> 6;
        const int sh = (int)(e0 & 63);
        w[i] |= (uint64_t)(dg << sh);
        const u128 rest = dg >> (64 - sh);   // (sh = 0: 64 - 0 = 64, the word above)
        w[i + 1] |= (uint64_t)rest;
        w[i + 2] |= (uint64_t)(rest >> 64);
    }
    mpz_import(v, w.size(), -1, sizeof(uint64_t), 0, 0, w.data());
    if (carry != 0) {
        mpz_t c;
        mpz_init(c);
        mpz_set_i128(c, carry);
        mpz_mul_2exp(c, c, h.nd);
        mpz_add(v, v, c);
        mpz_clear(c);
    }
    mpz_mod(v, v, g->N);
    return ok;
}

}  // namespace
