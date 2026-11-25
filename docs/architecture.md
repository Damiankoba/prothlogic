# Architecture

This page explains how ProthLogic works inside: how a test runs, how the CPU and GPU engines square
large numbers, and how the sieve works. It is meant for readers who want to understand or change the code.

## Contents

1. [The big picture](#1-the-big-picture)
2. [Source files](#2-source-files)
3. [One Proth test](#3-one-proth-test)
4. [The CPU engine](#4-the-cpu-engine)
5. [The GPU engines](#5-the-gpu-engines)
6. [The GPU sieve](#6-the-gpu-sieve)
7. [Running many tests: workers and the card](#7-running-many-tests-workers-and-the-card)

---

## 1. The big picture

```
command line (cli.cpp)
   │
   ├─ sieve on the GPU (presieve_core.cpp → gpu_engine.cu)      removes candidates with small factors
   │
   └─ list of candidates
         │
         dispatch (dcore.cpp)                                    calibration, queue, workers, pinning
           ├─ CPU workers  → Proth test (proth_logic.cpp) → CPU engine (fft_engine.cpp, fft_pack_vbmi.cpp)
           └─ card workers → Proth test (proth_logic.cpp) → GPU engine (gpu_proth_engine.cu, gpu_hybrid.cuh)
                                     │
                      result lines, primes.txt, composite.txt, checkpoints
```

The same test code runs on the CPU and on the GPU. Only the engine that squares the numbers is different.
This is why checkpoints and results are the same on both.

---

## 2. Source files

| File | Contents |
|---|---|
| `cli.cpp` | command-line options, reading candidate files, choosing the mode, the start of the program |
| `dcore.cpp` | dispatch: CPU calibration, the queue, workers and threads, pinning, `--tune`, the card workers |
| `worker.cpp` | runs one task on one worker |
| `pipeline_proth.cpp` | turns a list of k into tasks |
| `auto_config.cpp` | automatic number of workers |
| `proth_logic.cpp` | the Proth test itself (CPU and GPU), the Gerbicz–Li check, progress lines, results |
| `checkpoint_manager.cpp` | writing and reading checkpoints safely |
| `storage_core.cpp` | appending result lines to files |
| `fft_engine.cpp` | the CPU transforms: Stockham, two-pass Bailey, batch-8 column kernels, IBDWT |
| `fft_pack_vbmi.cpp` | the CPU digit conversion (pack), carries and reduction mod N (unpack) |
| `fft_factorizer.cpp` | splits a transform length into radix steps |
| `fft_types.h` | the CPU engine context (`FFTContext`) |
| `gpu_proth_engine.cu` | the GPU API, the classic GPU engine, choosing the engine and the number of candidates |
| `gpu_hybrid.cuh` | the hybrid GPU engine (included by `gpu_proth_engine.cu`) |
| `gpu_engine.cu` | the GPU sieve |
| `presieve_core.cpp` | the sieve driver on the CPU side |
| `fft_stub.cpp`, `gpu_stub.cpp` | empty stand-ins for builds without the CPU engine or without CUDA |
| `build.sh` | detects the hardware and builds |

---

## 3. One Proth test

### The test

For N = k·2ⁿ + 1, the program picks a base *a* with Jacobi symbol (a/N) = −1 and computes

    a^((N−1)/2) = (a^k)^(2^(n−1))  mod N

as a^k (a short GMP power) followed by n − 1 squarings. N is prime exactly when the result is N − 1.
The squarings take almost all of the time, so they are what the engines do.

### The scaled form

Reducing a product modulo N normally needs a division. ProthLogic avoids it.
Because k·2ⁿ ≡ −1 (mod N), a square S split at bit n as S = L + H·2ⁿ gives

    −k·S ≡ H − k·L   (mod N)

This needs only a multiplication by the small k and a subtraction. The program therefore stores
Y = X·2ⁿ mod N instead of X. Then −k·Y² = X²·2ⁿ, which is again in the same form, so every squaring keeps it.
The conversion to and from Y happens only at the start, when saving a checkpoint, and at the end.
Checkpoints always hold the plain X, so they do not depend on the engine.

### The Gerbicz–Li check

The squarings are done in blocks of L × M squarings (256 × 256 by default). Let v₀ be the value at the start
of a block and vⱼ the value after j·L squarings. During the block the program also keeps the product

    d = v₀ · v₁ · … · v_(M−1)

Squaring each factor L times turns d into v₁ · … · v_M, so at the end

    v₀ · d^(2^L) ≡ d · v_M   (mod N)

must hold. Checking this costs L extra squarings and M + 1 products, about 1% of the block.
A wrong bit anywhere in the block breaks the equation with near certainty. Then the block is repeated
from v₀, which was itself verified. Only verified values are saved.

### Checkpoints

After every verified block the plain X is written to a temporary file, which is then renamed over the
old checkpoint, so a crash never leaves a half-written file. Large checkpoints are also flushed to disk
(`fsync`). The time spent so far goes to a `.time` file next to it.

---

## 4. The CPU engine

The CPU engine multiplies with floating-point FFTs using AVX-512.

### Digits

The number is cut into digits of B bits (16 by default, up to 20). The digits are **balanced**: each lies
between −2^(B−1) and 2^(B−1) instead of 0 and 2^B. This makes the values in the transform much smaller
and lowers the rounding error by about two orders of magnitude.

The balanced digits are computed without a carry chain. With f the B-bit fields of x and h the top bit
of each field,

    d_i = f_i − h_i·2^B + h_(i−1)

The terms telescope, so the sum of d_i·2^(B·i) is exactly x. In the large transforms this step is done
inside the first pass, so it needs no pass over memory of its own.

### The square through a real FFT

A square of a number with 2N digits needs a transform of the product, which has twice as many digits.
ProthLogic stores the 2N real digits as N complex numbers (even digits in the real part, odd in the imaginary part)
and fills the upper half with zeros. One forward FFT, one step that squares the spectrum and untangles the
even and odd parts, and one inverse FFT give the square. That is two transforms per squaring.

### Transform layout

For large lengths the transform is split as a matrix (two-pass Bailey):

1. **pass A:** transforms of the columns;
2. **middle:** twiddle factors, transforms of the rows, the squaring of the spectrum and the inverse row
   transforms, all in one pass while the data is in the cache;
3. **pass B':** the inverse column transforms.

So each squaring reads and writes the data about three times. The column transforms work on 8 columns at once,
one column per AVX-512 lane ("batch-8"). Supported lengths are 2ᵖ, 3·2ᵖ and 7·2ᵖ, so the length can follow the
number size closely.

### The IBDWT

For small k there is a second method, the irrational-base discrete weighted transform. It puts L digits of
varying size (positions e_j = ⌈j·β⌉, β = (n + log₂k)/L) into the transform and multiplies them by weights
a_j = 2^(e_j − j·β). With these weights the cyclic convolution is exactly the product modulo k·2ⁿ + 1 up to
the factor k, and no zero padding is needed: the transform is half as long.

The price is precision: the weights lose about ½·log₂k bits per digit. So the IBDWT wins for small k and
is not used for large k. The calibration decides (see below).

For long runs the IBDWT keeps the digits in its own form between squarings and merges the last pass of
one squaring with the first pass of the next. This saves a full pass over memory each time.

### Reduction modulo N

After the inverse transform the coefficients are rounded to integers, the carries are propagated
(8 independent carry chains in the 8 AVX-512 lanes, then joined) and the scaled reduction H − k·L
from section 3 brings the value back below N. All of these steps run in parallel over the threads.

### Calibration and self-tests

Before a batch, the program tries the possible lengths for the largest n and k of the batch, smallest first:

- each length must pass a **self-test**: the square of a random number is compared with GMP;
- 100 squarings measure the **rounding error**, which must stay well below the limit.

The shortest length that passes is used. During the run the rounding error of every block is checked again
(`Max Err` must stay below 0.45).

---

## 5. The GPU engines

The GPU engines use number-theoretic transforms (NTT): transforms modulo primes below 2³¹.
Integer arithmetic is exact, so there is no rounding error. The result is recovered from several primes
with the Chinese remainder theorem (CRT).

The arithmetic modulo p uses Montgomery multiplication on 32-bit words.

### The classic engine

- primes of the form c·2^e + 1 (15·2²⁷+1, 27·2²⁶+1, 7·2²⁶+1, 45·2²⁴+1), 3 or 4 of them;
- the number is stored as digits of b bits, zero-padded to twice its length;
- forward NTT, pointwise square, inverse NTT;
- CRT (Garner's method) gives each exact coefficient, then the coefficients are split into b-bit parts,
  summed per position, and the scaled reduction is applied;
- a few neighbour passes bring the digits back to nearly balanced form. A digit is never brought to the
  canonical form, so no long carry chain is needed on the card.

### The hybrid engine

The hybrid engine removes the zero padding, like the IBDWT on the CPU, but with exact arithmetic.

- primes of the form p = c·2^e − 1, with p ≡ 3 (mod 4). For such p, −1 is not a square, so
  GF(p²) = F_p[i] with i² = −1 exists: "complex numbers modulo p". One of the primes is 2³¹ − 1;
- the weighted convolution of length L becomes a cyclic transform of length L/2 over GF(p²);
- the weights need an L-th root of k·2ⁿ modulo p, so the primes are chosen for each k;
- the transform is half as long as in the classic engine and uses a complex value per pair of digits,
  which moves much less data.

How the bound works: every output coefficient must stay below half the product P of the primes, so that the
CRT gives it exactly. The plan chooses the digit size and the number of primes so that this holds for the
**worst case** of the input, not just a typical one.

### Wide digits

For large numbers the hybrid engine can use digits of about 80 bits with 6 or 7 primes, instead of 40 bits
with 3 or 4. The transform is then half as long again. With 7 primes the bound holds for the worst case.
With 6 primes it holds for the typical size of the coefficients, and the Gerbicz–Li check guards the rest.

### Structure of one squaring on the card

For the largest sizes (for example 100 million digits, 2²¹ complex values × 7 primes) a squaring is four
passes over memory:

1. **column pass, forward:** the digits are packed into GF(p²) with their weights, and the outer levels of
   the transform are computed on tiles of 8 columns held in shared memory;
2. **block pass:** the lowest 12 levels of the forward transform, the pointwise square and the lowest 12 levels
   of the inverse transform, on blocks of 4096 values;
3. **column pass, inverse:** the outer levels of the inverse transform;
4. **CRT pass:** unweighting, CRT, splitting into digits, the wrap-around correction for k·2ⁿ and the neighbour
   passes, all in one kernel.

A whole batch of squarings is recorded once as a CUDA Graph and replayed, so the CPU does almost nothing.

### Choosing the engine and the plan

- the hybrid engine is used whenever it has a plan for n and k (from 2¹⁴ complex values on);
- for large numbers the normal plan and the 7-prime wide plan are timed against each other, both are checked
  to give the same value, and the faster one is kept;
- for large k without a hybrid plan, the classic engine is timed against the 7-prime wide plan in the same way;
- small candidates share the card: a quick test measures 1, 2, 3, ... candidates at once and keeps the fastest.

---

## 6. The GPU sieve

The sieve tries many primes p at once on the card. For each p it finds which candidates p divides.

**Fixed n, range of k.** p divides k·2ⁿ + 1 exactly when k ≡ −2⁻ⁿ (mod p). One modular power per prime gives
that k, and every k ≡ it in the range is removed. Powers of 2 are computed with Montgomery arithmetic where
"multiply by 2" is only an addition.

**Fixed k, range of n.** p divides k·2ⁿ + 1 exactly when k·2ⁿ ≡ −1 (mod p). This is a discrete logarithm,
solved by **baby-step giant-step**. Write n = n₀ + g·m + j with 0 ≤ j < m:

- baby steps: the m values −2⁻ʲ go into a hash table in shared memory (each step is one halving mod p);
- giant steps: c = k·2^(n₀+g·m) for g = 0, 1, 2, ... (one multiplication each), each looked up in the table.

A match gives n = n₀ + g·m + j. A range of R values of n costs about m + R/m steps instead of R, and m is chosen
near √R / 2. One warp handles one prime. Table entries are short fingerprints; every match is checked exactly,
so a fingerprint clash can never remove a candidate by mistake.

The primes themselves are produced on the card with a segmented sieve of Eratosthenes.

---

## 7. Running many tests: workers and the card

### Workers

The queue holds the candidates. Each worker takes the next one, tests it with its threads, and takes the next.
With `--pin`, each worker and its threads are fixed to their own cores, and a worker stays inside one CCD
(core complex with its own L3 cache) when it fits.

### The card as a worker

With `--gpu` the card takes candidates from the same queue. It can run several engines at once, each with its
own CUDA stream and CPU thread, so several small candidates fill the card.

At the end of a batch the card often finishes its work long before a CPU worker that is still in the middle of a
large candidate. Then the card asks that worker to stop at its next checkpoint, loads the checkpoint and finishes
the test itself. The reported time is the sum of both parts.

### When something fails

| Problem | What happens |
|---|---|
| a block fails the Gerbicz–Li check | the block is repeated from its verified start |
| the rounding error on the CPU is too large | the block is rejected; the IBDWT moves to a longer length |
| the card fails 5 checks in a row, or a CUDA error | the candidate goes back to the CPU, from its checkpoint |
| the card has no plan for a candidate | the candidate goes to the CPU (with `--gpu-only` it is listed as not tested) |
| a checkpoint cannot be read | a loud warning, the file is kept as `.bad`, the test starts again |
