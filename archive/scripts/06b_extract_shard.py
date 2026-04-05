"""
分片版embedding提取 - 用于多机并行
用法: python3 06b_extract_shard.py --split train --shard 0 --num_shards 2
"""

import pickle, numpy as np, torch, time, os, sys, argparse

os.environ["HF_HUB_CACHE"] = "/data/public_models/hub"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

DATA_DIR = "/data/muskliu/np_master/data"
LAYER_NAME = "blocks.20.mlp.l3"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='train')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=2)
    args = parser.parse_args()

    print(f"分片: {args.split} shard {args.shard}/{args.num_shards}")

    # 加载数据
    with open(os.path.join(DATA_DIR, f"{args.split}.pkl"), 'rb') as f:
        samples = pickle.load(f)

    # 分片
    total = len(samples)
    shard_size = total // args.num_shards
    start = args.shard * shard_size
    end = total if args.shard == args.num_shards - 1 else start + shard_size
    samples = samples[start:end]
    print(f"  总样本: {total}, 本分片: [{start}:{end}] = {len(samples)}个")

    # 加载模型
    print("加载Evo2...")
    t0 = time.time()
    from evo2 import Evo2
    model = Evo2('evo2_7b')
    print(f"  加载耗时: {time.time()-t0:.0f}s")

    # 提取
    embeddings, labels, categories, ids = [], [], [], []
    errors = 0
    t0 = time.time()

    for i, sample in enumerate(samples):
        seq = ''.join(c for c in sample['dna'].upper() if c in 'ATCG')
        if len(seq) < 10:
            errors += 1
            continue
        if len(seq) > 8192:
            s = (len(seq) - 8192) // 2
            seq = seq[s:s+8192]

        try:
            input_ids = torch.tensor(model.tokenizer.tokenize(seq), dtype=torch.int).unsqueeze(0).to('cuda:0')
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs, embs = model(input_ids, return_embeddings=True, layer_names=[LAYER_NAME])
            emb = embs[LAYER_NAME][0].float().mean(dim=0).cpu().numpy()
            embeddings.append(emb)
            labels.append(sample['is_bgc'])
            categories.append(sample.get('category_vector', [0]*7))
            ids.append(sample.get('id', f'{args.split}_{start+i}'))
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"    错误: {e}")

        if (i+1) % 100 == 0:
            elapsed = time.time() - t0
            speed = (i+1) / elapsed
            eta = (len(samples)-i-1) / speed
            print(f"  {i+1}/{len(samples)} ({speed:.1f}/s, ETA:{eta/60:.0f}min, err:{errors})")

    result = {
        'embeddings': np.array(embeddings), 'labels': np.array(labels),
        'categories': np.array(categories), 'ids': ids,
        'layer': LAYER_NAME, 'embedding_dim': embeddings[0].shape[0] if embeddings else 0,
        'num_samples': len(embeddings), 'num_errors': errors,
    }

    out_path = os.path.join(DATA_DIR, f"{args.split}_emb_shard{args.shard}.pkl")
    with open(out_path, 'wb') as f:
        pickle.dump(result, f)
    print(f"\n完成: {out_path} ({len(embeddings)}个, {errors}错误, {(time.time()-t0)/60:.0f}min)")

if __name__ == '__main__':
    main()
