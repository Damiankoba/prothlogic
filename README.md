# ProthLogic 
ProthLogic is an experimental, CPU/GPU tool for sieving and testing the primality of Proth numbers.


This project uses the MPIR / GMP library for multiple precision arithmetic.
The included headers (gmp.h, gmpxx.h) are part of the MPIR Library and are distributed under the terms of the GNU Lesser General Public License (LGPL).


I am an independent the creator of ProthLogic, an open-source mathematical software program designed to test the primality of Proth numbers using both CPU and GPU resources.

What I building
My development focuses heavily on high-performance computing and low-level hardware optimizations. I write code that utilizes CPU cache retention, OpenMP multi-threading, and AVX-512 vector instructions to push modern hardware to its absolute limits and accelerate complex mathematical calculations.

-----------------
Next version and actual i work:

1. I continuing with cache optymize, with the new method huge arrays are now sliced ​​into small tiles, ensuring that the processor always rotates data into the fastest       cache (L1 Cache).
2. Radix-8 Logic Repair, previously, the engine could get stuck in the slower Radix-4 Bailey mode.
3. Implementation of the Gerbicz Test (Fault Tolerant Architecture).
4. Implementation of OpenMP for a test on several threads.
