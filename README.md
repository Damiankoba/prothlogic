# ProthLogic 
an experimental program for sieving and testing Proth numbers (sources of v5.3 and v5.7 are open; v5.8 will be published after its first Top 5000 prime)”
CPU/GPU tool for sieving and testing the primality of Proth numbers.
Sources used to build the program:
Article "What Every Programmer should know about memory",
"mersenneforum.org"
"Wikipedia"
"AI models"
To build my program I use the rule: don't copy, but inspire.

This project uses the MPIR / GMP library for multiple precision arithmetic.
The included headers (gmp.h, gmpxx.h) are part of the MPIR Library and are distributed under the terms of the GNU Lesser General Public License (LGPL).


I am an independent creator of ProthLogic using AI models for coding, an open-source mathematical software program designed to test the primality of Proth numbers using both CPU and GPU resources.

What I'm building
Our development focuses heavily on high-performance computing and low-level hardware optimizations. I write code that utilizes CPU cache retention, OpenMP multi-threading, and AVX-512 vector instructions to push modern hardware to its absolute limits and accelerate complex mathematical calculations.



## ProthLogic 5.8 (in development – not released yet)

Version 5.8 adds a full **CUDA GPU engine** next to the AVX-512 CPU engine.
The source code will be published after the first ProthLogic prime enters the Top 5000 list.
Version 5.8 is focused on Proth numbers tested on GPU hardware.

**What's new**
- **GPU primality testing (CUDA)** for k·2^n+1: exact integer NTT arithmetic (no floating-point rounding errors), from small n up to 100,000,000-digit numbers
- **Automatic plan selection**: at start-up the program benchmarks several transform layouts on the user's card and picks the fastest one
- **Batch mode**: several candidates tested at once on one GPU, so even small numbers keep the card fully loaded
- **GPU sieve** for large candidate lists
- **Safe checkpoints**: data is flushed to disk, and damaged checkpoint files are detected and reported
- **Gerbicz-Li error checking on GPU**: every segment of 65,536 squarings is verified; on a mismatch the test rolls back to the last verified state
- **CPU ↔ GPU checkpoints**: a test started on the CPU can continue on the card and back
- - **Hybrid CPU + GPU mode**: a list of candidates from a file is tested on the CPU (AVX-512) and the GPU at the same time, from one shared queue – each engine takes the next candidate as soon as it is free
- Works on NVIDIA RTX 20, 40 and 50 series (tested: 2080 Ti, 4070, 5060, 5060 Ti, 5080, 5090)

**Results so far**
- Same card (RTX 5080), same 18 test numbers: **2.87× faster than proth20** overall
- Residues (RES64) of composite numbers match proth20 exactly (independent verification)
- 100,000,000-digit Proth test (585·2^332192797+1) completed on GPU
- Full numbers and performance tests: see `ProthLogic GPU Benchmark` and `one light-day Voyager 1`

### How the GPU engines work

Both engines compute the squarings of a Proth test (a^((N-1)/2) mod N, N = k·2^n+1) with **exact integer
arithmetic** – a Number Theoretic Transform (NTT) modulo several 31-bit primes instead of a floating-point FFT,
so there are no rounding errors to watch. The results for each prime are combined with the Chinese Remainder
Theorem, then the carries are propagated.

**Classic GPU engine**
- The number is split into small digits; each squaring is a cyclic convolution with zero padding
  (the transform is twice as long as the number), followed by reduction modulo N.
- 2–4 primes, 32-bit Montgomery arithmetic, large-radix passes that keep the data in on-chip memory.
- Simple and robust; used for small n and as a reference for the hybrid engine.

**Hybrid NTT engine** (default from ~800k bits)
- **No zero padding**: a weighted (negacyclic) convolution performs the reduction modulo k·2^n+1 inside the
  transform itself, with digits of variable size – the transform is half as long as in the classic engine.
- The transform runs over **complex numbers modulo p** (GF(p²), primes p ≡ 3 mod 4), which halves the length again.
- Wide digits (up to ~80 bits) with 6–7 primes for very large numbers: fewer, bigger digits mean less memory traffic.
- At start-up the program measures several layouts on the user's card and keeps the fastest one.
- Every segment is verified with the Gerbicz-Li check; results match the classic engine and other programs bit for bit.


----------------

## Version ProthLogic v5
1. Continuing to optimize time ms/iter, actual for tested N=3322774 time is 1.35 ms/iter and for N=3099097 time is 1.40 ms/iter. both numbers are were tested at the same time with taskset-c 0-5/6-10 and working with 4 threads each test. done
2. Implementation in progress: radix for AVX2(very slow for now, not supported by program yet) ?
3. Operation Fusion (Eliminate unnecessary RAM reads) new function unpack_karatsuba_overlap_add_vbmi done
4. Testing FMA instruction interleaving in functions like pass_radix5_seq or fft_stockham_radix4_blocked for Bailey2D 
5. Testing Factorizer and Mixed-Radix Architecture: Bailey2D and Radix8 working on FFT Len $2^n$, Factorizer allows me to use an array size that is, for example, a combination of factors 2, 3, and 5, for now is much slower than Bailey2D with radix4 and Radix8 and radix 2 implementation for Bailey 2D engine. done
6. Batch mode: I noticed that testing in batch mode runs threads sequentially, and autoconfig for larger numbers allows testing of one number due to a cache conflict, while the AMD Ryzen 9 9900X3D processor architecture allows testing of two numbers simultaneously, assigning them to separate L3 CCD0 and CCD1 chiplet. With the taskset -c 0-4/6-10 command. The next step will be to rebuild the batch mode and autoconfig file for processor analysis and adapt the test to separate L3 memories if possible without data conflicts and clogging the RAM bus. 
7. Artifacts from other versions remain in the code for analysis and comparison purposes. done
8. The sieve only used raw values ​​of k per line, and the community requires an interoperable ABC format (k·b^n+c). done
9. Entering residues from composite results for maintaining verification records. (COMPOSITE (a=7 RES64: 0xC044C688349D2382)) All that remains is to save Composite results to the composite.txt file. done
10. Testing Radix-8 supplemented with Prefetching, blocking, and the introduction of _mm512_stream_pd cache write instead of _mm512_store_pd. done


-----------------
## Preparation for next version(v5) and actual work:

1. I am continuing with cache optimization, with the new method huge arrays are now sliced ​​into small tiles, ensuring that the processor always rotates data into the fastest       cache (L1 Cache).  
2. Radix-8 Logic Repair, previously, the engine could get stuck in the slower Radix-4 Bailey mode. 
3. Implementation of the Gerbicz Test (Fault Tolerant Architecture). done
4. Implementation of OpenMP for a test on several threads. done
