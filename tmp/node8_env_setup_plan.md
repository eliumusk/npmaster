# Node8 Inference Environment Setup Plan

Investigation date: 2026-04-18. Read-only, nothing installed.

## TL;DR Recommendation

**Option A — `rsync` the entire `/root/miniconda3/envs/bgc` from node7 to node8** is the cheapest, lowest-risk path. The two nodes are clones of the same container image (identical OS / kernel / NVIDIA driver / CUDA toolkit / conda / Python), the env is small (~5.8 GB), node8's `/root` overlay has 47 GB free, and the only build artifact that could go wrong (the locally-compiled `vortex/ops/depr_attn/flash_attn_2_cuda*.so`) was compiled against the same driver/CUDA/glibc that node8 has.

Total estimated time to "ready to launch inference on node8": **15–25 minutes**, with one fallback path (Option C, conda-pack) if the rsync's `.pth` / shebang paths cause trouble.

---

## Key Discoveries

### 1. The two nodes are siblings, not strangers
Side-by-side checks:

| Check | node7 (`Voc_gpu_node7_a800`) | node8 (`Voc_gpu_node8_a800`) |
|---|---|---|
| OS | Ubuntu 22.04.5 LTS | Ubuntu 22.04.5 LTS |
| Kernel | `5.4.250-2-velinux1u1-amd64` | `5.4.250-2-velinux1u1-amd64` |
| NVIDIA driver | `535.129.03` (CUDA 12.2 cap) | `535.129.03` (CUDA 12.9 cap displayed) |
| `nvcc` | CUDA 12.9.41 | CUDA 12.9.41 |
| GPU | 8x A800 | 8x A800 |
| `conda --version` | `24.1.2` | `24.1.2` |
| `/root/miniconda3` Python | `3.12.1` | `3.12.1` |
| Container image hint | `di-20260401151510-qjdn8` | `di-20260401152144-9drh5` (same date/template) |
| `/data` mount | `fs_vepfs-cnbj8e9fa72097f7` GPFS, 10T (8.2T used, 1.9T avail) | **same** GPFS, identical signature |

Both containers were spun up from the same image about 10 minutes apart on 2026-04-01. This is as close to a clone as you can get without actually being one.

### 2. `/data` is shared between the two nodes (verified)
I wrote `/data/muskliu/_share_test_node7.txt` from node7, then read it byte-for-byte from node8 (then deleted it). The mount is `fs_vepfs-cnbj8e9fa72097f7` GPFS, identical inode/usage on both ends. **All datasets, checkpoints, hf_cache, and the repo at `/data/muskliu/npmaster` are already present on node8 with zero copy needed.**

This is why your earlier discovery "`/data/muskliu/npmaster`, `/data/syh/benchmark/...`, `/data/public_models/...`, `/data/muskliu/npmaster/data/phase3/route_b/best_route_b_s0.pt` are already there" is true — they aren't *also* there, they are *literally the same files*. Implication: any output written by inference on node8 to `/data/muskliu/...` is automatically visible from node7.

### 3. `bgc` env contents (node7)
- Size: **5.8 GB**, ~66 pip packages.
- Built by exactly **one conda command**: `conda create -n bgc python=3.11 -y` (history shows nothing else conda-side). Everything else was pip.
- Python: 3.11.15. (Note: this is *inside* `bgc`; base is 3.12.1.)
- Critical packages: `torch 2.6.0+cu124`, `triton 3.2.0`, `evo2 0.5.5` (installed via pip, not editable), `vtx 1.0.8`, `numpy 2.4.4`, `biopython 1.87`.
- **No `flash_attn` package installed.** The flash-attn-2 symbols live as compiled .so files inside the `vortex` package: `vortex/ops/depr_attn/flash_attn_2_cuda.cpython-311-x86_64-linux-gnu.so` (and a 312 variant). The accompanying `build/` dir is 40 MB. This is the manual workaround the user mentioned.
- Only `.pth` file in `site-packages` is `distutils-precedence.pth` (stdlib-style, path-agnostic). **Good news: no editable installs that pin absolute paths.**
- `evo2` `INSTALLER` says `pip` (not `-e`), no `direct_url.json` recording a local path → safe to copy.
- `vtx` has no `direct_url.json` → also installed from a built wheel/sdist, not editable.

### 4. Mirror config (so future installs are fast on node8 too)
- `/root/.pip/pip.conf` on node7 points at `https://mirrors.ivolces.com/pypi/simple/` (volcengine internal mirror — fast).
- `/root/.condarc` uses tsinghua mirrors.
- node8 likely has the same defaults (it's the same image), but if not, copying these two files is trivial.

---

## Options Compared

### Option A — `rsync /root/miniconda3/envs/bgc/` from node7 to node8

**Plan (do NOT execute now, this is what you'd run later):**
```bash
# from a third box, or use SSH ProxyJump if nodes can't ssh each other directly
ssh Voc_gpu_node7_a800 'tar -C /root/miniconda3/envs -cf - bgc' \
  | ssh Voc_gpu_node8_a800 'tar -C /root/miniconda3/envs -xpf -'
# OR if node7 can ssh node8 directly:
# ssh Voc_gpu_node7_a800 'rsync -aHX --numeric-ids /root/miniconda3/envs/bgc/ Voc_gpu_node8_a800:/root/miniconda3/envs/bgc/'
# Then on node8 fix up shebangs (only needed if conda paths differ — they don't here, both are /root/miniconda3):
ssh Voc_gpu_node8_a800 'conda activate bgc && python -c "import torch, evo2, vortex; print(torch.cuda.is_available(), torch.cuda.device_count())"'
```
- **Effort:** ~10 min of one-shot setup, no reasoning required.
- **Time to ready:** 5.8 GB across whatever the inter-node bandwidth is. Same data center, almost certainly 10 GbE+ → **3–8 min transfer**, plus 5 min sanity test. **Total ≤ 25 min.**
- **Risk of breakage:**
  - **Low** for shebangs / `.pth` paths: target prefix is identical (`/root/miniconda3/envs/bgc`). Conda-installed Python scripts hard-code the interpreter path — if it's the same on both sides (it is, both nodes have `/root/miniconda3` and we'd land it under `envs/bgc`), nothing rewrites needed.
  - **Low** for the compiled `flash_attn_2_cuda.so`: built against driver 535.129.03 + CUDA 12.4 runtime (bundled in torch wheel) + glibc from Ubuntu 22.04. node8 has identical driver, identical glibc. The .so should load with no ABI mismatch.
  - **Medium-Low** for "things outside the env": if inference depends on system libs in `/usr/local/cuda/...`, both nodes are on CUDA 12.9 nvcc + driver 535.129.03, so we're fine. No `LD_LIBRARY_PATH` shenanigans were spotted.
  - **The realistic failure mode** is that some pip package wrote an absolute path into a config or cache (e.g. `huggingface_hub` cache dirs are usually env-var driven, but worth verifying). Mitigation: do a `python -c "import torch; import evo2; import vortex; from vortex.ops.depr_attn import flash_attn_2_cuda"` smoke test, then run a tiny single-GPU inference, before launching 8-GPU jobs.
- **Ongoing maintenance:** any future `pip install` in `bgc` on node7 has to be repeated on node8. Same for node8 → node7. Acceptable as long as the env is mostly frozen for the bench runs. Could be scripted with a helper that runs the same `pip install ...` on both via ssh.

### Option B — `conda env export` + `conda env create -f`

**Why it's worse than A:**
- The history shows `bgc` was **not** built with conda installs beyond the bare Python interpreter. Everything else is pip. So `conda env export` would essentially just be a pip requirements list anyway.
- The fragile pieces are: (a) `evo2 0.5.5` (its dependencies pull StripedHyena, which historically has fragile installs), (b) `flash_attn` — which on node7 you sidestepped by building only the `vortex/ops/depr_attn` flash-attn-2 .so. Reinstalling means redoing that workaround on node8 from scratch.
- **Effort:** high — an hour to several hours, depending on whether `vortex` rebuilds the depr_attn .so cleanly.
- **Time to ready:** 1–4 hours best case, can balloon if any pip install hangs on the mirror or if vortex's setup.py for depr_attn needs a particular CUDA toolkit visible.
- **Risk:** **Highest of the four options.** Re-deriving the working env is exactly the painful path you already did once.
- **Maintenance:** Same as A.
- **Use case:** only if rsync produced a corrupt env and we need to start fresh.

### Option C — `conda-pack` bundle and unbundle

**Plan sketch:**
```bash
# on node7
conda install -n base -c conda-forge conda-pack
conda-pack -n bgc -o /data/muskliu/_backup/bgc_node7.tar.gz
# on node8
mkdir -p /root/miniconda3/envs/bgc
tar -xzf /data/muskliu/_backup/bgc_node7.tar.gz -C /root/miniconda3/envs/bgc
/root/miniconda3/envs/bgc/bin/conda-unpack
```
- **Effort:** medium. Requires installing `conda-pack` in `base` on node7 first (one extra step vs. Option A).
- **Time to ready:** similar to A (3–8 min pack, 3–8 min transfer/unpack via shared `/data`, 1 min unpack), maybe 30 min total.
- **Risk:** **Slightly lower than A** because `conda-unpack` explicitly rewrites shebangs, `.pth`, and prefix files. But — see Option A — those rewrites are *not actually needed* here because the prefixes match. So conda-pack mostly buys you insurance you don't need.
- **Bonus:** the resulting tarball lives on shared `/data`, which means it doubles as a checkpoint of the env. Re-using it from a third future node would be one command.
- **Maintenance:** same as A.
- **When to choose C over A:** if A's smoke test fails with a path-rewrite error.

### Option D — Skip node8 entirely, rent more capacity later

- **Effort:** zero now.
- **Time to ready:** infinite for the goal of "doubling local throughput".
- **Risk:** missing a near-free 8-A800 capacity that's literally sitting idle on the same shared filesystem. The marginal cost of trying Option A is ~25 min of work; the upside is **2x parallelism for every benchmark / sweep going forward**, which on Phase 3 / hardneg mining work pays back the first time you launch a sweep.
- **Use case:** only if Options A/B/C all fail for a reason we can't currently anticipate (e.g. node8 turns out to have throttled GPUs or we discover the GPFS gets slow under concurrent heavy IO from both nodes — neither is established).

---

## Recommendation: Option A, with C as the documented fallback

### Why
1. The nodes are sibling containers from the same image — same kernel, driver, CUDA toolkit, conda, Python base, libc. **The riskiest variables for binary copies (driver / glibc / CUDA ABI) are eliminated by construction.**
2. Target conda prefix on both nodes is identical (`/root/miniconda3/envs/bgc`), so the usual rsync footgun (Python shebangs and `.pth` files pointing at the source prefix) doesn't fire. Verified: only `distutils-precedence.pth` is present, and it's path-agnostic.
3. The painful-to-rebuild piece (`vortex/ops/depr_attn/flash_attn_2_cuda*.so`) is just a .so file that was built for the exact driver/CUDA/glibc that node8 also has. Copying it as bytes is the simplest possible "install".
4. `/data` being shared GPFS means **zero data copy is needed** — checkpoints, datasets, repo, hf_cache are already visible from node8.
5. Only ~5.8 GB to move; node8's `/root` overlay has 47 GB free.

### Smoke test sequence (do these in order before launching real inference)
After rsync:
1. `ssh Voc_gpu_node8_a800 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate bgc && python -V'` — Python 3.11.15 expected.
2. `python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"` — expect `2.6.0+cu124 True 8`.
3. `python -c "import evo2, vortex; from vortex.ops.depr_attn import flash_attn_2_cuda; print('ok')"` — this is the critical one; if the .so loads, the ABI is right.
4. Run `scripts/inference_stride.py` on **one** GPU with a tiny shard from the nine_genome bench (the same script that works on node7) and confirm output matches.
5. Only then wire up 8-GPU launches.

### Honest list of what could still go wrong
- **Bandwidth between the two container hosts is unknown.** If they're on the same physical box / same rack switch, the transfer is trivial. If they're across availability zones, 5.8 GB might take longer. Either way, still cheaper than reinstalling.
- **GPFS write contention.** Now you'd potentially have 16 GPUs all hitting the same shared `/data`. If your inference is heavily IO-bound (loading genomes / writing per-shard outputs), you might not get a clean 2x speedup. Worth measuring on the first dual-node run; mitigation is easy (per-node scratch dirs that get rsynced back).
- **HF cache concurrency.** `/data/muskliu/hf_cache` is shared. If both nodes try to hf-download the same model simultaneously on first use it can corrupt the cache. Prevent by making sure the Evo2 weights are already fully downloaded on node7 before node8 starts (they appear to be: `/data/public_models/hub/models--arcinstitute--evo2_7b` exists).
- **The `/root/.pip/pip.conf` and `/root/.condarc` mirror configs may differ on node8.** Not blocking for inference (no installs needed), but copy them over if you want future `pip install`s on node8 to be fast.
- **Driver-side CUDA-version mismatch quirk.** `nvidia-smi` on node8 shows "CUDA Version: 12.9" while node7 shows "12.2", despite identical driver `535.129.03`. This is just `nvidia-smi`'s cosmetic "max CUDA my libraries declare" and not a real driver-ABI difference. Worth noting but not a blocker — torch only cares about the driver version, which matches.
- **No system-level CUDA needed.** torch wheel ships its own CUDA 12.4 runtime; depr_attn .so links against it; nvcc 12.9 is only needed if you ever recompile. So we don't need to align toolkit versions across nodes.

### What we're explicitly **not** doing
- Not modifying anything on either node yet. This document is the plan; the actual rsync + smoke test is a follow-up task that should be run only after you confirm the recommendation.
- Not changing `/root/.condarc` or `/root/.pip/pip.conf` on node8 (not required for inference).
- Not installing `conda-pack` proactively. Only install it if Option A's smoke test fails.
