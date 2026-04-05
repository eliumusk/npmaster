"""
Part B: NLL-based statistics per window, from Evo2 autoregressive logits.
For each window:
  - per-position NLL (softmax cross-entropy of Evo2 next-token predictions)
  - compute: mean, std, autocorr at lag=100/1000, var of diff
Compare BGC windows vs non-BGC windows on these statistics.

Runs on server. Sharded across 8 GPUs.
"""
import os, sys, pickle, time, argparse
import numpy as np
import torch

ROOT = "/data/muskliu/np_master"
OUT_DIR = f"{ROOT}/data/diagnostic_day1/nll"
os.environ.setdefault("HF_HUB_CACHE", "/data/public_models/hub")

GENOMES = ['S_aquilus', 'S_deccanensis', 'S_longhuiensis']

def load_windows():
    """Load all windows from the 3 Strep genomes, tagged with BGC status."""
    all_wins = []
    for g in GENOMES:
        wins = pickle.load(open(f"{ROOT}/data/genome_scan/{g}/windows.pkl", 'rb'))
        for w in wins:
            w['genome'] = g
            all_wins.append(w)
    return all_wins

def compute_nll_stats(model, dna_seq):
    """Forward pass, get per-position NLL, compute stats."""
    clean = ''.join(c for c in dna_seq.upper() if c in 'ATCG')
    if len(clean) < 1000:
        return None
    input_ids = torch.tensor(
        model.tokenizer.tokenize(clean), dtype=torch.int
    ).unsqueeze(0).to('cuda:0')
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        out = model(input_ids)  # nested: ((logits, ...), None)
    logits_full = out[0][0]  # (1, L, V)
    # autoregressive: position i predicts position i+1
    logits = logits_full[0, :-1, :].float()  # (L-1, V)
    targets = input_ids[0, 1:].long()   # (L-1,)
    logprob = torch.log_softmax(logits, dim=-1)
    nll = -logprob.gather(1, targets.unsqueeze(1)).squeeze(1).cpu().numpy()  # (L-1,)
    # stats
    n = len(nll)
    def autocorr(x, lag):
        if len(x) <= lag+1: return 0.0
        x = x - x.mean()
        num = (x[:-lag] * x[lag:]).sum()
        den = (x*x).sum() + 1e-10
        return float(num/den)
    stats = {
        'nll_mean': float(nll.mean()),
        'nll_std': float(nll.std()),
        'nll_median': float(np.median(nll)),
        'nll_q25': float(np.percentile(nll, 25)),
        'nll_q75': float(np.percentile(nll, 75)),
        'autocorr_100': autocorr(nll, 100),
        'autocorr_1000': autocorr(nll, 1000),
        'diff_var': float(np.diff(nll).var()),
        'length': int(n),
    }
    return stats

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--n_shards", type=int, default=8)
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.shard)
    from evo2 import Evo2
    print(f"[shard {args.shard}] loading Evo2 7B ...", flush=True)
    model = Evo2("evo2_7b")
    print(f"[shard {args.shard}] loaded", flush=True)
    windows = load_windows()
    N = len(windows)
    chunk = (N + args.n_shards - 1) // args.n_shards
    lo = args.shard * chunk
    hi = min(N, lo + chunk)
    subset = windows[lo:hi]
    print(f"[shard {args.shard}] processing {lo}..{hi} ({len(subset)}), total pool {N}", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_file = f"{OUT_DIR}/nll_shard_{args.shard}.pkl"
    results = []
    errors = 0
    t0 = time.time()
    for i, w in enumerate(subset):
        try:
            stats = compute_nll_stats(model, w['dna'])
            if stats is None:
                errors += 1; continue
            stats.update({
                'genome': w['genome'], 'contig': w.get('contig',''),
                'start': int(w.get('start',0)), 'end': int(w.get('end',0)),
                'is_bgc': int(w.get('is_known_bgc',0)),
            })
            results.append(stats)
        except Exception as e:
            errors += 1
            print(f"[shard {args.shard}] err {i}: {e}", flush=True)
        if (i+1) % 30 == 0:
            t = time.time()-t0
            r = (i+1)/t
            eta = (len(subset)-(i+1))/r
            print(f"[shard {args.shard}] {i+1}/{len(subset)} rate={r:.2f}/s eta={eta/60:.1f}min errs={errors}", flush=True)
    pickle.dump({'results': results, 'errors': errors}, open(out_file,'wb'))
    print(f"[shard {args.shard}] saved {out_file} ({len(results)} samples, {errors} errors, {time.time()-t0:.1f}s)", flush=True)

if __name__ == '__main__':
    main()
