"""
Day-1 Diagnostic: combine MIBiG + multigenus bg + Strep bg, then extract Evo2 embeddings.

Runs on server. Sharded across 8 GPUs using --shard and --n_shards.
"""
import os, sys, pickle, time, argparse, random
import numpy as np
import torch

ROOT = "/data/muskliu/np_master"
MIBIG_PKL = f"{ROOT}/data/diagnostic_day1/mibig_variable.pkl"
BG_MULTI_PKL = f"{ROOT}/data/diagnostic_day1/bg_multigenus.pkl"
STREP_PKL = f"{ROOT}/data/expanded_strep/w16384/train.pkl"
OUT_DIR = f"{ROOT}/data/diagnostic_day1/embeddings"
MAX_LEN = 16384

os.environ.setdefault("HF_HUB_CACHE", "/data/public_models/hub")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

def build_combined():
    """Return the fully combined sample list (deterministic order)."""
    random.seed(42)
    # 1. MIBiG
    mibig = pickle.load(open(MIBIG_PKL, 'rb'))
    # 2. Multi-genus background
    bg_multi = pickle.load(open(BG_MULTI_PKL, 'rb'))
    # 3. Strep background (sample from expanded_strep w16384 negatives)
    strep_all = pickle.load(open(STREP_PKL, 'rb'))
    strep_negs = [s for s in strep_all if s.get('is_bgc', 0) == 0]
    random.shuffle(strep_negs)
    # Take up to 1500 to roughly balance: 2502 MIBiG vs 2200 (Pseudo+Nostoc+Bacillus) + 1500 Strep = 3700 bg
    strep_take = strep_negs[:1500]
    strep_samples = []
    for s in strep_take:
        strep_samples.append({
            'id': s['id'],
            'dna': s['dna'],
            'full_length': len(s['dna']),
            'used_length': len(s['dna']),
            'categories': [],
            'organism': s.get('organism','strep'),
            'phylum': 'Actinobacteria',
            'is_bgc': 0,
            'source_tag': 'BG_strep',
        })
    combined = mibig + bg_multi + strep_samples
    print(f"Combined: {len(combined)} = MIBiG {len(mibig)} + BG multi {len(bg_multi)} + BG strep {len(strep_samples)}")
    return combined

def extract(model, dna_seq, max_len=MAX_LEN):
    clean = ''.join(c for c in dna_seq.upper() if c in 'ATCG')
    if len(clean) < 10:
        return None
    if len(clean) > max_len:
        start = (len(clean) - max_len) // 2
        clean = clean[start:start+max_len]
    input_ids = torch.tensor(
        model.tokenizer.tokenize(clean), dtype=torch.int
    ).unsqueeze(0).to('cuda:0')
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        _, embeddings = model(
            input_ids, return_embeddings=True,
            layer_names=["blocks.20.mlp.l3"]
        )
    emb = embeddings["blocks.20.mlp.l3"][0].float().mean(dim=0).cpu().numpy()
    return emb

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--n_shards", type=int, default=8)
    args = p.parse_args()
    # shard selects which GPU and which chunk of data
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.shard)
    from evo2 import Evo2
    print(f"[shard {args.shard}] loading Evo2 7B ...", flush=True)
    model = Evo2("evo2_7b")
    print(f"[shard {args.shard}] loaded", flush=True)

    combined = build_combined()
    N = len(combined)
    chunk = (N + args.n_shards - 1) // args.n_shards
    lo = args.shard * chunk
    hi = min(N, lo + chunk)
    subset = combined[lo:hi]
    print(f"[shard {args.shard}] processing {lo}..{hi} ({len(subset)} samples)", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_file = f"{OUT_DIR}/shard_{args.shard}.pkl"

    embeddings, metas = [], []
    errors = 0
    t0 = time.time()
    for i, s in enumerate(subset):
        try:
            emb = extract(model, s['dna'])
            if emb is None:
                errors += 1
                continue
            embeddings.append(emb)
            metas.append({k: v for k, v in s.items() if k != 'dna'})
        except Exception as e:
            errors += 1
            print(f"[shard {args.shard}] err {i} {s['id']}: {e}", flush=True)
        if (i+1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i+1) / elapsed
            eta = (len(subset) - (i+1)) / rate
            print(f"[shard {args.shard}] {i+1}/{len(subset)} rate={rate:.2f}/s eta={eta/60:.1f}min errs={errors}", flush=True)
    # save
    with open(out_file, 'wb') as f:
        pickle.dump({
            'embeddings': np.array(embeddings, dtype=np.float32),
            'metas': metas,
            'shard': args.shard,
            'n_shards': args.n_shards,
            'layer': 'blocks.20.mlp.l3',
            'errors': errors,
        }, f)
    print(f"[shard {args.shard}] saved {out_file} ({len(embeddings)} samples, {errors} errors, {time.time()-t0:.1f}s)", flush=True)

if __name__ == '__main__':
    main()
