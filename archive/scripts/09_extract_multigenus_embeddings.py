"""
多属Evo2 embedding提取
- 输入: data/multigenus/{genus}/{train,val,test}.pkl
- 输出: data/multigenus/{genus}/{train,val,test}_emb.pkl
- 用法:
    python3 -u 09_extract_multigenus_embeddings.py              # 全部
    python3 -u 09_extract_multigenus_embeddings.py --shard 0    # node7跑前半
    python3 -u 09_extract_multigenus_embeddings.py --shard 1    # node8跑后半
"""

import pickle, numpy as np, torch, time, os, sys, argparse

os.environ["HF_HUB_CACHE"] = "/data/public_models/hub"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

BASE = "/data/muskliu/np_master/data/multigenus"
LAYER_NAME = "blocks.20.mlp.l3"
DEVICE = "cuda:0"

# 只跑新的3个属 (Streptomyces用已有的embedding, Bacillus去掉因GC bias)
GENERA = ["micromonospora", "amycolatopsis", "pseudomonas"]


def load_model():
    print("加载Evo2 7B...")
    t0 = time.time()
    from evo2 import Evo2
    model = Evo2("evo2_7b")
    print(f"  完成: {time.time()-t0:.0f}s")
    return model


def extract_embedding(model, dna_seq):
    clean_seq = ''.join(c for c in dna_seq.upper() if c in 'ATCG')
    if len(clean_seq) < 10:
        return None
    if len(clean_seq) > 8192:
        start = (len(clean_seq) - 8192) // 2
        clean_seq = clean_seq[start:start + 8192]

    try:
        input_ids = torch.tensor(
            model.tokenizer.tokenize(clean_seq), dtype=torch.int,
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _, embeddings = model(input_ids, return_embeddings=True, layer_names=[LAYER_NAME])

        emb = embeddings[LAYER_NAME][0].float().mean(dim=0).cpu().numpy()
        return emb
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def process_file(model, input_path, output_path):
    if os.path.exists(output_path):
        existing = pickle.load(open(output_path, 'rb'))
        print(f"  [SKIP] {output_path} 已存在 ({existing['num_samples']} samples)")
        return existing

    with open(input_path, 'rb') as f:
        samples = pickle.load(f)

    print(f"  处理 {input_path} ({len(samples)} samples)...")
    embeddings, labels, categories, ids, genera = [], [], [], [], []
    errors = 0
    t0 = time.time()

    for i, s in enumerate(samples):
        emb = extract_embedding(model, s['dna'])
        if emb is not None:
            embeddings.append(emb)
            labels.append(s['is_bgc'])
            categories.append(s.get('category_vector', [0]*7))
            ids.append(s.get('id', f'{i}'))
            genera.append(s.get('genus', 'unknown'))
        else:
            errors += 1

        if (i + 1) % 100 == 0 or (i + 1) == len(samples):
            speed = (i + 1) / (time.time() - t0)
            eta = (len(samples) - i - 1) / max(speed, 0.01)
            print(f"    {i+1}/{len(samples)} ({speed:.1f}/s, ETA:{eta:.0f}s, err:{errors})")

    result = {
        'embeddings': np.array(embeddings),
        'labels': np.array(labels),
        'categories': np.array(categories),
        'ids': ids,
        'genera': genera,
        'layer': LAYER_NAME,
        'embedding_dim': embeddings[0].shape[0] if embeddings else 0,
        'num_samples': len(embeddings),
        'num_errors': errors,
    }

    with open(output_path, 'wb') as f:
        pickle.dump(result, f)

    elapsed = time.time() - t0
    print(f"    完成: {len(embeddings)} emb, {errors} err, {elapsed:.0f}s")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', type=int, default=-1, help='0=前半, 1=后半, -1=全部')
    args = parser.parse_args()

    # 收集所有要处理的文件
    tasks = []
    for genus in GENERA:
        genus_dir = os.path.join(BASE, genus)
        for split in ['test', 'val', 'train']:
            inp = os.path.join(genus_dir, f'{split}.pkl')
            out = os.path.join(genus_dir, f'{split}_emb.pkl')
            if os.path.exists(inp):
                tasks.append((genus, split, inp, out))

    # 加上cross_genus的mixed_train
    cross_dir = os.path.join(BASE, 'cross_genus')
    mixed_in = os.path.join(cross_dir, 'mixed_train.pkl')
    mixed_out = os.path.join(cross_dir, 'mixed_train_emb.pkl')
    if os.path.exists(mixed_in):
        tasks.append(('cross', 'mixed_train', mixed_in, mixed_out))

    # 分片
    if args.shard == 0:
        tasks = tasks[:len(tasks)//2]
        print(f"SHARD 0: 处理前 {len(tasks)} 个任务")
    elif args.shard == 1:
        tasks = tasks[len(tasks)//2:]
        print(f"SHARD 1: 处理后 {len(tasks)} 个任务")

    # 统计总样本数
    total_samples = 0
    for genus, split, inp, out in tasks:
        if not os.path.exists(out):
            with open(inp, 'rb') as f:
                n = len(pickle.load(f))
            total_samples += n
            print(f"  {genus}/{split}: {n} samples")
        else:
            print(f"  {genus}/{split}: SKIP (已完成)")
    print(f"  总计: {total_samples} samples 待处理")

    if total_samples == 0:
        print("全部已完成!")
        return

    # 加载模型
    model = load_model()

    # 测试
    test_emb = extract_embedding(model, "ATGCGATCGATCG" * 200)
    assert test_emb is not None, "模型测试失败!"
    print(f"模型测试OK: dim={test_emb.shape[0]}")

    # 处理
    t_total = time.time()
    for genus, split, inp, out in tasks:
        print(f"\n=== {genus} / {split} ===")
        process_file(model, inp, out)

    print(f"\n全部完成! 总耗时: {(time.time()-t_total)/60:.1f}min")


if __name__ == '__main__':
    main()
