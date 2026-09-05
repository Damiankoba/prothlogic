# ProthLogic 
ProthLogic is an experimental, CPU/GPU tool for sieving and testing the primality of Proth numbers.
Sources used to build the program:
Article "What Every Programmer should know about memory",
mersenneforum.org,
Wikipedia,
AI,
To build my program I use the rule: don't copy, but inspire.

This project uses the MPIR / GMP library for multiple precision arithmetic.
The included headers (gmp.h, gmpxx.h) are part of the MPIR Library and are distributed under the terms of the GNU Lesser General Public License (LGPL).


I am an independent the creator of ProthLogic, an open-source mathematical software program designed to test the primality of Proth numbers using both CPU and GPU resources.

What I'm building
Our development focuses heavily on high-performance computing and low-level hardware optimizations. I write code that utilizes CPU cache retention, OpenMP multi-threading, and AVX-512 vector instructions to push modern hardware to its absolute limits and accelerate complex mathematical calculations.


----------------

Version ProthLogic v5
1. Continuing to optimize timne ms/iter, actual for tested N=3322774 time is 1.35 ms/iter and for N=3099097 time is 1.40 ms/iter. both numbers are tested same time with taskset-c 0-5/6-10 and working with 4 threads each test.
2. Implementation in progress: radix for AVX2(very slow for now)
3. Operation Fusion (Eliminate unnecessary RAM reads) new function unpack_karatsuba_overlap_add_vbmi
4. Testing FMA instruction interleaving nn functions like pass_radix5_seq or fft_stockham_radix4_blocked for Bailey2D
5. Testing Factorizer and Mixed-Radix Architecture: Bailey2D and Radix8 working on FFT Len $2^n$, Factorizer allows me to use an array size that is, for example, a combination of factors 2, 3, and 5, for now is much slower than Bailey2D with radix4 and Radix8.
6. Batch mode: I noticed that testing in batch mode runs threads sequentially, and autoconfig for larger numbers allows testing of one number due to a cache conflict, while the AMD Ryzen 9 9900X3D processor architecture allows testing of two numbers simultaneously, assigning them to separate L3 CCD0 and CCD1 chipsets. With the taskset -c 0-4/6-10 command. The next step will be to rebuild the batch mode and autoconfig file for processor analysis and adapt the test to separate L3 memories if possible without data conflicts and clogging the RAM bus.
7. Artifacts from other versions remain in the code for analysis and comparison purposes.
8. The sieve only used raw values ​​of $k$ per line, and the community requires an interoperable ABC format ($k \cdot b^n + c$). done
9. Entering residuals from composite results for maintaining verification records. (COMPOSITE (a=7 RES64: 0xC044C688349D2382)) All that remains is to save Composite results to the results.txt file




-----------------
Preparation for next version(v5) and actual work:

1. I continuing with cache optymize, with the new method huge arrays are now sliced ​​into small tiles, ensuring that the processor always rotates data into the fastest       cache (L1 Cache).  
2. Radix-8 Logic Repair, previously, the engine could get stuck in the slower Radix-4 Bailey mode. fixed
3. Implementation of the Gerbicz Test (Fault Tolerant Architecture). done
4. Implementation of OpenMP for a test on several threads. done
