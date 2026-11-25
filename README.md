# ProthLogic: CPU & GPU primality testing of Proth numbers

ProthLogic tests numbers of the form **N = k·2ⁿ + 1** (Proth numbers) for primality.
It runs on x86-64 CPUs with AVX-512 and on NVIDIA graphics cards, and it includes a GPU sieve
that removes candidates with small factors before testing.

- **Proven results.** For k < 2ⁿ the Proth test is a primality *proof*, not a probable-prime test.
- **Checked while it runs.** Every block of squarings is verified with the Gerbicz–Li check.
  A hardware or rounding error is caught and that block is computed again.
- **CPU and GPU together.** The card can work as one more worker next to the CPU cores, or on its own.
- **Resumable.** Progress is saved to checkpoint files. A stopped run continues where it ended,
  even on another machine or after moving between CPU and GPU.
- **Standard output.** Each composite gets a Proth RES64, so results can be compared with other programs.

The first Top 5000 prime found with ProthLogic is 25902068371·2⁴⁰⁰⁰¹⁶⁸ + 1 (1,204,181 digits),
found in November 2026 and verified with LLR2.

## Requirements

- Linux on x86-64
- g++ with C++17 and OpenMP, and GMP (`libgmp-dev` on Debian/Ubuntu)
- For the CPU engine: a CPU with AVX-512 F, DQ, VBMI and IFMA (Intel Ice Lake or newer, AMD Zen 4 or newer)
- For the GPU: an NVIDIA card, its driver and the CUDA toolkit (`nvcc`)

A machine without AVX-512 can still use its NVIDIA card (see `./build.sh gpu` below).
To see what your CPU has:

```bash
grep -o 'avx512[a-z]*' /proc/cpuinfo | sort -u    # needs avx512f, avx512dq, avx512vbmi, avx512ifma
nvidia-smi                                         # shows the NVIDIA card, if there is one
```

On Debian or Ubuntu the compiler and GMP are installed with:

```bash
sudo apt install build-essential libgmp-dev
```

For the GPU part, install the CUDA toolkit from NVIDIA (it provides `nvcc`).

### Tested hardware

| | Tested on |
|---|---|
| CPU engine (AVX-512) | AMD Ryzen 9 9900X, 9900X3D, 9950X (Zen 5); Ryzen 9 8945HX, Ryzen 7 7700X, EPYC 9334 (Zen 4) |
| NVIDIA cards | RTX 5090, RTX 5080 (ASUS, Zotac, NVIDIA Founders Edition), RTX 5070 Ti, RTX 5060 Ti 16 GB, RTX 5060; RTX 4090, RTX 4070 (ASUS), RTX 4060; RTX 2080 Ti; A100 40 GB |
| GPU-only build (`./build.sh gpu`) | Intel Core i3-8100, Core i5-14600KF, Xeon E5-2680 v4, E5-2698 v3, E5-2690 v2; AMD EPYC 7K62, 7532, 7642; Ryzen 7 5700X |

Other NVIDIA cards with CUDA should work too. The program measures the card at the start and picks its settings itself.

## Building

In the folder with the sources:

```bash
./build.sh          # CPU engine + GPU (GPU part only when nvcc and a card are found)
./build.sh gpu      # card only, for CPUs without AVX-512
```

The script prints what it found (CPU, card) and ends with `[build] done`. The program `prothlogic` is created
in the same folder. It is tuned for the machine it was built on, so build it on each machine you use.

## Quick start

**1. Test one number.** This is a known prime. With 4 threads the test takes a few minutes.

```bash
./prothlogic "25902068371*2^1110912+1" --threads 4
```

At the end you should see `[!] FOUND PRIME: 25902068371*2^1110912+1 is prime!`.
The same number can also be given as `--k 25902068371 --n 1110912`. Keep the quotes around the number:
without them the shell may treat `*` as a file pattern.

**2. Sieve a range** (needs an NVIDIA card). This keeps only the candidates without small factors
and saves them to a file in the ABC format, which LLR2 reads too. Nothing is tested yet.
A deep sieve of a large range can take hours.

```bash
./prothlogic --k 25902068371 --n-start 4000000 --n-end 5000000 --pmax 1e12 --sieve-only --out-sieve list.txt
```

**3. Test the list.** Use the CPU and the card together, or the card alone:

```bash
./prothlogic --in-file list.txt --gpu
./prothlogic --in-file list.txt --gpu-only
```

Primes go to `primes.txt`, composites with their RES64 go to `composite.txt`.

A long run can be stopped at any time with Ctrl+C. Start the same command again in the same folder and it
continues: finished candidates are skipped and the interrupted test resumes from its last checkpoint.
On a remote machine, run long jobs inside `screen` or `tmux` so they survive a lost connection.

## Documentation

- [User Guide](docs/user-guide.md): all options, file formats, output, checkpoints and tips.
- [Tuning and Diagnostics](docs/tuning-and-diagnostics.md): environment variables, measuring and profiling.
- [Architecture](docs/architecture.md): how the test, the CPU and GPU engines and the sieve work inside.
