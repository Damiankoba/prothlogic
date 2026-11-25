# Tuning and Diagnostics

ProthLogic chooses its settings by itself: the transform size, the number of candidates on the card,
the CPU engine. You do not need anything from this page for normal work.

This page is for two situations:

- you want to change an automatic choice (section 2);
- you want to measure, check or compare the engines (sections 3 to 5).

## Contents

1. [How to set a variable](#1-how-to-set-a-variable)
2. [Everyday settings](#2-everyday-settings)
3. [The error check](#3-the-error-check)
4. [Measuring and profiling](#4-measuring-and-profiling)
5. [Developer switches](#5-developer-switches)

---

## 1. How to set a variable

All settings on this page are environment variables. Put them in front of the command:

```bash
LOTR_GPU_STREAMS=2 ./prothlogic --in-file list.txt --gpu-only
```

Several at once:

```bash
LOTR_GPU_WIDE=7 LOTR_GPU_SEGTIME=1 ./prothlogic --n 332192799 --k 147 --gpu-only
```

A variable set this way applies to that one run only.

---

## 2. Everyday settings

### CPU

| Variable | Default | What it does |
|---|---|---|
| `LOTR_WORKERS=W` | automatic | number of workers when `--workers` is not given |
| `LOTR_DIGIT_BITS=B` | 19 | largest digit size the CPU may try (16 to 20). Smaller is safer and slower; 16 turns wide digits off |
| `LOTR_NO_IBDWT=1` | off | never use the IBDWT; always the classic transform |
| `LOTR_IB_MAXERR=x` | 0.3 | largest calibration error accepted for an IBDWT size |
| `LOTR_IB_MINGAIN=x` | 1.2 | how much shorter the IBDWT must be than the classic transform to be chosen |
| `LOTR_IB_SAFEERR=x` | 0.05 | below this calibration error the IBDWT is chosen even when it is only a little shorter |
| `LOTR_FORCE_FFT_LEN=L` | off | skip the calibration and use this transform length (a power of two or 3·2ᵖ) |
| `LOTR_IBDWT=1` | off | with `LOTR_FORCE_FFT_LEN`: use the IBDWT at that length |
| `LOTR_KEEP_OMP_BIND=1` | off | keep `OMP_PROC_BIND` / `OMP_PLACES` from the environment (see below) |

**Why `LOTR_KEEP_OMP_BIND` exists.** Some Docker images set OpenMP binding variables. With them, every worker
ends up on the same core and the program runs about 20 times slower. ProthLogic removes these variables and
restarts itself. Set `LOTR_KEEP_OMP_BIND=1` only if you really want them.

**`LOTR_FORCE_FFT_LEN`** is useful for very large numbers, where the calibration takes long, and for tests:
a small number on a large transform must give the same RES64 as on its normal one.

### GPU

| Variable | Default | What it does |
|---|---|---|
| `LOTR_GPU_STREAMS=1..16` | measured | candidates on the card at the same time (skips the quick test) |
| `LOTR_GPU_AUTOTUNE=0` | on | no quick test; use the built-in rule for the number of candidates |
| `LOTR_GPU_HYB=0` / `=1` | automatic | always the classic engine / always the hybrid engine (when it has a plan) |
| `LOTR_GPU_WIDE=0` / `6` / `7` | automatic | wide digits: never / with 6 primes / with 7 primes |
| `LOTR_GPU_WIDEBENCH=0` | on | do not compare the normal plan with the 7-prime wide plan; take the normal one |
| `LOTR_GPU_TAKEOVER=0` | on | with `--gpu`: the card does not take over long tests from the CPU |

**What the GPU engines are.**
The *classic* engine uses a zero-padded integer transform.
The *hybrid* engine needs no zero padding, so its transform is half as long; it is used whenever it has a plan.
*Wide digits* put about twice as many bits in each digit and use 6 or 7 primes instead of 3 or 4.
This moves less data and is faster for large numbers. With 7 primes the bound is proven for the worst case;
with 6 primes it holds for typical values, and the error check stands guard.

**What ProthLogic does by itself.** For a large number it times the normal plan against the 7-prime
wide plan, checks that both give the same value, and keeps the faster one. You see a line like:

    [GPU] WIDE 7 bench: classic NTT 2^19 x 2 (18-bit) 0.0488 ms/iter vs WIDE 7 2^15 x 7 0.0286 ms/iter (-41.3%), values equal -> WIDE 7 (1.0 s)

The choice is remembered for the rest of the run.

---

## 3. The error check

The Gerbicz–Li check verifies the squarings in blocks (see the User Guide, section 8).

| Variable | Default | What it does |
|---|---|---|
| `LOTR_GEC_L=L` | 256 | block size: a block has L × L squarings (256 × 256 = 65,536) |
| `LOTR_GEC_INJECT=i` | off | flips one bit once, after squaring i: the check must find it and redo the block |
| `LOTR_GEC_OFF=1` | off | turns the check off. **For speed comparisons only, never for real tests** |

**Block size.** A larger block costs a little less time per squaring, but a failed check repeats more work
and checkpoints are written less often. The default suits most sizes.

**Testing the check.** `LOTR_GEC_INJECT` shows that the check works on your hardware:

```bash
LOTR_GEC_INJECT=100000 ./prothlogic --n 1110912 --k 25902068371
```

You should see a `[ROLLBACK]` line, and the test must still report `is prime!`.

---

## 4. Measuring and profiling

These settings print extra lines. They are meant for finding out where the time goes.

### Overview

| Setting | Engine | Shows |
|---|---|---|
| `--detail` (option) | all | chosen transform, calibration, speed estimate, rounding error |
| `./build.sh profile` (build) | CPU | `[PROFILE]` lines: time of each stage of a squaring |
| `LOTR_GPU_PROFILE=1` | GPU | `[GPU PROFILE]` lines: time of each kernel of a squaring |
| `LOTR_GPU_SEGTIME=1` | GPU | `[SEG]` lines: squarings vs. check vs. checkpoint in each block |
| `LOTR_GPU_COLBENCH=1` | GPU | a detailed benchmark of every kernel when the engine starts |
| `LOTR_SIEVE_PROFILE=1` | sieve | time of each sieve stage |
| `LOTR_KBENCH=1` | CPU | a microbenchmark of the CPU FFT kernels, then exit |
| `LOTR_VERIFY_UNPACK=1` | CPU | every fast step is recomputed by a slow reference and compared |

### CPU stage profile

Build with `./build.sh profile`. After every block a line shows the milliseconds per squaring for each
stage (pack, forward transform, square, inverse transform, unpack). Use it to compare two settings
on the same number.

### GPU kernel profile

`LOTR_GPU_PROFILE=1` times each kernel of every squaring and prints averages every 10,000 squarings.
It turns CUDA Graphs off and waits after each squaring, so the run itself is slower.
Use the relative sizes, not the total.

### Block timing on the GPU

`LOTR_GPU_SEGTIME=1` prints one `[SEG]` line per block: the time of the squarings, of the error check and
of the checkpoint. It shows how much of a long test goes to the check and to saving.

### Kernel benchmark on the GPU

`LOTR_GPU_COLBENCH=1` runs a benchmark of the engine's kernels when it is created, on garbage data
(timing only). For each kernel it prints microseconds per launch and the memory throughput, and compares
it with a plain memory copy of the same size, which is the speed limit of the card.
For the wide-digit plans of large numbers it also measures energy through NVML: joules per squaring and per kernel.

This matters because a card that runs a long test usually sits at its power limit: there,
**speed = power limit ÷ energy per squaring**. A change that saves energy makes the test faster,
even if a short benchmark does not show it.

### Checking correctness

`LOTR_VERIFY_UNPACK=1` recomputes every fast CPU step with a simple, slow reference version and compares
the results. It prints `MISMATCH` if they differ, and then continues with the reference result.
It is much slower; use it only for tests.

---

## 5. Developer switches

These switches turn single optimizations on or off. They exist to compare an old and a new version of the
same step, and to find the cause if a result is ever wrong. **None of them is needed for normal work**,
and all of them give the same results, only at a different speed.

### CPU transform

| Variable | Default | Switch |
|---|---|---|
| `LOTR_NO_BALANCED=1` | off | plain digits instead of balanced digits (much larger rounding error) |
| `LOTR_DIV_OLD=1` | off | divide by k at the end of each squaring instead of the scaled form |
| `LOTR_CARRY_OLD=1` | off | the old scalar carry step instead of the 8-lane vector version |
| `LOTR_B8_2L_MIN=R` | 256 | from this column length on, column transforms run in two levels |
| `LOTR_B8_NOGAP=1` | off | no spare gap between memory blocks (causes cache conflicts) |
| `LOTR_B8_OUTPLACE=1` | off | column transforms out of place |
| `LOTR_B8_IP_EVEN=1` | off | in place only for an even number of stages |
| `LOTR_B8_IP_MID_EVEN=1` | off | the same, in the middle pass |
| `LOTR_B8_DIT8_MIN=R` | 256 | from this sub-transform length on, the radix-8 split version (0 = never) |
| `LOTR_NAT_PFD=n` | 64 | software prefetch distance in the middle pass (0 = off) |

### CPU IBDWT

| Variable | Default | Switch |
|---|---|---|
| `LOTR_IB_RUN=n` | 256 | squarings kept in the internal digit form before converting back |
| `LOTR_IB_NEWMIN=L` | 49152 | transform length from which the optimizations below are on |
| `LOTR_IB_PACK_FUSE=0/1` | automatic | pack the digits inside the first pass |
| `LOTR_IB_NORM=0/1` | automatic | normalize the digits inside the last pass |
| `LOTR_IB_INPLACE=0/1` | automatic | keep the data in one buffer through all passes |
| `LOTR_IB_OVL=0` | on | keep the digits in a separate array instead of inside the data |
| `LOTR_IB_CHAIN=0` | on | do not merge the last pass of one squaring with the first pass of the next |

### GPU

| Variable | Default | Switch |
|---|---|---|
| `LOTR_GPU_NOGRAPH=1` | off | launch kernels one by one instead of through CUDA Graphs |
| `LOTR_GPU_L2ORD=0..3` | measured | order of two passes over the primes (0 all at once, 1 one by one, 2 overlapped, 3 in column chunks) |
| `LOTR_GPU_CCH=2/4/8/16` | 8 | with `L2ORD=3`: number of column chunks |
| `LOTR_GPU_CPO=0` | 1 | with `L2ORD=3`: the prime as the inner loop of the column pass |
| `LOTR_GPU_COL9=0` | on | the 1024-row column pass instead of the 512-row one (wide digits) |
| `LOTR_GPU_CRTF=1` | off | a different reconstruction method (CRT) for normal digits |
| `LOTR_GPU_L2P=MB` | off | keep part of the data pinned in the card's L2 cache (experiment) |

### Sieve

| Variable | Default | Switch |
|---|---|---|
| `LOTR_NSIEVE_BSGS=0/1/2` | 3 | fixed-k sieve method: 0 simple walk, 1–3 baby-step giant-step versions |
| `LOTR_NSIEVE_MB=8..11` | automatic | baby-step table size 2^MB |
