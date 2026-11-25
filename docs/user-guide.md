# ProthLogic User Guide

This guide explains how to build ProthLogic, how to tell it what to test, and how to read what it writes.

## Contents

1. [What ProthLogic does](#1-what-prothlogic-does)
2. [Building](#2-building)
3. [Choosing what to test](#3-choosing-what-to-test)
4. [Sieving](#4-sieving)
5. [Using the CPU and the GPU](#5-using-the-cpu-and-the-gpu)
6. [Reading the output](#6-reading-the-output)
7. [Checkpoints and resuming](#7-checkpoints-and-resuming)
8. [How errors are caught](#8-how-errors-are-caught)
9. [Tips and common mistakes](#9-tips-and-common-mistakes)

---

## 1. What ProthLogic does

A Proth number is N = k·2ⁿ + 1 with k odd and k < 2ⁿ.
Proth's theorem says that N is prime if and only if

    a^((N−1)/2) ≡ −1 (mod N)

for a base *a* that is a quadratic non-residue mod N.
ProthLogic picks such a base automatically (it is shown as `a=3`, `a=5`, ...) and computes this power
with n − 1 modular squarings. The result is a proof: a "prime" answer needs no further test.

The squarings use fast multiplication:

- **on the CPU:** floating-point FFTs with AVX-512, either a classic zero-padded transform or an
  irrational-base weighted transform (IBDWT), whichever is faster and safe for the given k;
- **on the GPU:** exact integer transforms (NTT) modulo several 31-bit primes, combined with the
  Chinese remainder theorem. These have no rounding error at all.

Before testing, a GPU sieve can remove candidates that have a small prime factor. This is much cheaper
than a full test.

---

## 2. Building

### Requirements

| What | Needed for |
|---|---|
| Linux, x86-64 | everything |
| g++ (C++17, OpenMP), GMP (`libgmp-dev`) | everything |
| AVX-512 F, DQ, VBMI, IFMA (Intel Ice Lake+, AMD Zen 4+) | the CPU engine |
| NVIDIA card, driver, CUDA toolkit (`nvcc`) | the GPU sieve and GPU tests |

### Build commands

| Command | Result |
|---|---|
| `./build.sh` | CPU engine, plus the GPU part when `nvcc` and a card are found |
| `./build.sh gpu` | card only: for CPUs without AVX-512 |
| `./build.sh profile` | adds detailed timing lines (`[PROFILE]`) |
| `./build.sh warn` | shows compiler warnings and saves them to `build_warnings.txt` |

Options can be combined, for example `./build.sh gpu warn`.

The script checks the CPU and the cards, compiles for exactly this hardware and writes `prothlogic`
next to the sources. Build on every machine you use; a binary from another machine may be slower or may not start.

A program built with `./build.sh gpu` always works as `--gpu-only` (see section 5), because it has no CPU engine.

---

## 3. Choosing what to test

There are five ways to give the candidates.

| Command | Tests |
|---|---|
| `"K*2^N+1"` | one number, written the usual way, e.g. `./prothlogic "25902068371*2^4000168+1"` |
| `--n N --k K` | the same number, given by its parts |
| `--n N --k-start A --k-end B` | fixed n, every k from A to B (sieved first) |
| `--k K --n-start A --n-end B` | fixed k, every n from A to B (sieved first) |
| `--in-file file.txt` | the candidates listed in a file |

The written form also accepts `2^N+1` (k = 1) and spaces inside quotes (`"3 * 2^100 + 1"`).
Put the number in quotes, otherwise the shell may treat `*` as a file pattern. An even k is moved
into the power of two, as the test needs an odd k: `6*2^10+1` is tested as `3*2^11+1`.
Other options can be added before or after the number, for example `./prothlogic "3*2^5000+1" --gpu`.

### The candidate file

Two formats are read. They can even be mixed in one file.

**ABC format** (also read by LLR2 and other programs). A header line describes the numbers;
`$a`, `$b` are the values given on the following lines:

    ABC $a*2^$b+1 // Sieved to 3e13
    25902068371 4000168
    25902068371 4000312
    25902068371 4000468

| Header | Each line holds |
|---|---|
| `ABC $a*2^$b+1` | k and n (the form the sieve writes) |
| `ABC 25902068371*2^$a+1` | n (fixed k) |
| `ABC $a*2^3321928+1` | k (fixed n) |

ProthLogic reads all three. LLR2 reads only the first one, so the sieve always writes that form.

**Simple format.** One candidate per line, written as four numbers `k 2 n 1`:

    25902068371 2 4000168 1
    25902068371 2 4000312 1

Lines that fit neither format are ignored. The sieve writes the ABC format, with its depth in the header,
so a sieve file can be tested directly, sieved deeper, or given to another program.

ProthLogic looks at the file and chooses the mode by itself:

- all lines have the same n: fixed-n mode;
- all lines have the same k: fixed-k mode;
- otherwise every line is tested on its own, in file order.

When a file holds different values of n, the transform size is chosen for the largest n, so every
candidate fits.

### Limits

- n can be at most 4,294,967,295.
- k must fit in 64 bits.
- The answer "prime" is a proof only when k < 2ⁿ (Proth's theorem). For k ≥ 2ⁿ the program warns at the start
  and reports `is a probable prime!` instead of `is prime!`.

---

## 4. Sieving

The sieve runs on the NVIDIA card. It removes every candidate that has a prime factor up to the
sieve depth `--pmax`. Removed candidates are certainly composite, so they never need a test.

| Option | Meaning |
|---|---|
| `--pmax P` | sieve depth: primes up to P are tried (default 50,000,000,000) |
| `--pstart P` | start at P instead of the beginning: sieves an existing file deeper |
| `--sieve-only` | sieve and stop, test nothing |
| `--out-sieve file.txt` | write the surviving candidates to this file (ABC format, with `// Sieved to P`) |
| `--sparse-sieve` | with `--in-file` and `--n`: sieve only the k values in the file |

`--pmax` and `--pstart` accept a whole number (`10000000000000`) or the short form (`1e13`, `2.5e12`).

**Important:** `--sieve-only` without `--out-sieve` saves nothing.

### Examples

```bash
# Sieve k = 25902068371 for n from 4,000,000 to 5,000,000 up to 10^13 and save the survivors
./prothlogic --k 25902068371 --n-start 4000000 --n-end 5000000 \
             --pmax 1e13 --sieve-only --out-sieve list_p1e13.txt

# Sieve the same list deeper, from 10^13 to 3·10^13
./prothlogic --in-file list_p1e13.txt --pstart 1e13 --pmax 3e13 \
             --sieve-only --out-sieve list_p3e13.txt

# Sieve a range of k for one n
./prothlogic --n 3321928 --k-start 1 --k-end 1000000 --pmax 1e12 \
             --sieve-only --out-sieve n3321928.txt
```

### How deep to sieve

Sieve deeper while the sieve removes candidates faster than the tests would finish them.
A simple rule: run the sieve for a while and look at how many candidates it removes per hour.
If that is more than the number of candidates you can test per hour, keep sieving.

---

## 5. Using the CPU and the GPU

### Workers and threads

A **worker** tests one candidate at a time. Each worker can use several **threads**.

| Option | Meaning |
|---|---|
| `--workers W` | how many candidates are tested at the same time (default: automatic) |
| `--threads T` | threads per candidate (default 1) |
| `--pin` | keep every worker and its threads on fixed CPU cores |

On CPUs with several core complexes (CCDs), such as Ryzen 9 or EPYC, use one worker per CCD with as many
threads as the CCD has cores, and add `--pin`. For example, on a 12-core Ryzen 9 with two CCDs of 6 cores:

```bash
./prothlogic --in-file list.txt --workers 2 --threads 6 --pin
```

### Finding the best split

`--tune` measures several splits of workers × threads on the real candidates and prints the fastest ones.
It tests nothing.

| Option | Meaning |
|---|---|
| `--tune` | measure the splits that use all cores |
| `--tune-all` | also measure splits that leave some cores free |
| `--tune-secs S` | seconds per measured split (default 20) |

### The graphics card

| Option | Meaning |
|---|---|
| `--gpu` | the card works as one more worker, next to the CPU workers |
| `--gpu-only` | only the card tests; the CPU just feeds it (about 2% of one core) |

The card runs several candidates at once when they are small. Before a batch it measures 1, 2, 3, ...
candidates at once and keeps the fastest count; you will see a line `[GPU] quick test ...`.

With `--gpu`, when the queue is empty and a CPU worker is still busy with a long test, the card takes
that candidate over at the next checkpoint and finishes it. The reported time is then the CPU part plus the GPU part.

If the card has no transform plan for a candidate (for example a very small n), the candidate goes to the
CPU with `--gpu`. With `--gpu-only` it is left out and listed at the end as not tested.

### Examples

```bash
# CPU and GPU together
./prothlogic --in-file list.txt --workers 2 --threads 8 --pin --gpu

# the card alone, while the CPU cores do something else
./prothlogic --in-file list.txt --gpu-only
```

---

## 6. Reading the output

### Results on screen

A prime:

    [!] FOUND PRIME: 25902068371*2^4000168+1 is prime! (1204181 decimal digits) Time : 270.897 sec. (a=3)

A composite:

    [+] 843691*2^3000000+1 is not prime. Proth RES64: 0x07875DC38ADAF94D Time : 162.069 sec. (a=3)

- **Time** is the total test time, including earlier runs if the test was resumed.
- **a** is the base used in the test.
- **RES64** is the lowest 64 bits of the final residue. Two programs that test the same number with the
  same base must give the same RES64, so it can be used to double-check a result.

A candidate with a very small factor is reported as `is not prime: divisible by p (trial division)`.
It gets no RES64 and is not written to `composite.txt`.

### Progress

For a single candidate, a progress line shows how far the test is and the time left.

When many candidates are tested at once, one shared line shows the whole batch:

    Batch: 926 / 5894 (15.7%) | primes: 1 | elapsed 12h 26m | ETA 3d 01h

In batch mode composite results are not printed on screen (they still go to `composite.txt`).
Add `--detail` to see them.

| Option | Meaning |
|---|---|
| `--detail` | more information: the chosen transform, calibration, speed estimate and the largest rounding error |
| `--report N` | plain `[STEP]` lines every N % instead of the live line (useful for logs and benchmarks) |

### Result files

| File | Contents |
|---|---|
| `primes.txt` | one line per prime found (another name: `--out-primes file.txt`) |
| `composite.txt` | one line per composite, with its RES64 |

Both files are written in the folder where the program runs, and new lines are always appended.

---

## 7. Checkpoints and resuming

During a test ProthLogic saves its progress to `check_k<k>_n<n>.bin`, together with
`check_k<k>_n<n>.bin.time`, which holds the time spent so far.
A checkpoint is written after every verified block of squarings (65,536 squarings by default).
Both files are deleted when the test ends.

You can stop a run at any time with Ctrl+C. At most the work since the last checkpoint is lost
(for a large number, a few minutes).

To resume, start the same command again in the same folder:

- candidates that already have a line in `primes.txt` or `composite.txt` are skipped;
- a test that was interrupted continues from its checkpoint.

Checkpoints are portable. A test started on the CPU can continue on the GPU and the other way round,
and the files can be copied to another machine.

If a checkpoint cannot be read (for example an empty file after a crash), the program says so loudly,
renames it to `<name>.bad` and starts that test from the beginning.

The CPU calibration of very large numbers can take a long time. When it takes more than 30 seconds,
its result is saved to `fft_calibration.txt` and reused next time. Delete this file to force a new calibration.

---

## 8. How errors are caught

A long test performs millions of squarings, and a single wrong bit would give a wrong answer.
ProthLogic protects the result in three ways.

1. **Gerbicz–Li check.** The squarings are grouped into blocks. At the end of each block a short extra
   computation checks the whole block. If the check fails (a memory or hardware error, an overheated card),
   the block is thrown away and computed again from the last verified state, and you see a `[ROLLBACK]` line.
   Only verified states are saved to checkpoints.
2. **Rounding error check (CPU).** Floating-point transforms round their results. ProthLogic watches the
   largest rounding error (`Max Err`). If it grows too large, the block is rejected; with the IBDWT the test
   then continues on a longer, safer transform.
3. **Self-tests.** Before a transform size is used, it squares a test number and compares the result with
   GMP. A size that fails is not used.

The GPU transforms are exact integer arithmetic, so they have no rounding error; the Gerbicz–Li check still
guards them. If the card fails the check five times in a row, the candidate is handed back to the CPU.

---

## 9. Tips and common mistakes

- **Save your sieve.** `--sieve-only` needs `--out-sieve file.txt`, otherwise the survivors are lost.
- **One process per list.** Two processes on the same file test the same candidates twice.
  Use one process with more workers instead.
- **Run in one folder per list.** Result files and checkpoints are written to the current folder.
- **Check the power limit of rented cards.** A card limited below its normal power runs slower:
  `nvidia-smi -q -d POWER` shows the current and the default limit.
- **A large k is slower.** The fastest GPU transform needs some room in each digit, and a large k takes
  part of it. ProthLogic measures the available transforms and picks the fastest one automatically.
- **Verify important results.** A prime found by any program should be checked by a second, independent
  program (for example LLR2 or PFGW) before it is reported.
