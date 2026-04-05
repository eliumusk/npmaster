"""
扩充版Strep + 窗口消融的Evo2 embedding提取
- 输入: data/expanded_strep/w{4096,8192,16384}/{train,val,test}.pkl
- 输出: 同目录下 {train,val,test}_emb.pkl
- 用法:
    python3 -u 15_extract_expanded_embeddings.py              # 全部
    python3 -u 15_extract_expanded_embeddings.py --shard 0    # node7跑前半
    python3 -u 15_extract_expanded_embeddings.py --shard 1    # node8跑后半
"""

import pickle, numpy as np, torch, time, os, sys, argparse

os.environ["HF_HUB_CACHE"] = "/data/public_models/hub"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

BASE = "/data/muskliu/np_master/data/expanded_strep"
LAYER_NAME = "blocks.20.mlp.l3"
DEVICE = "cuda:0"
WINDOW_SIZES = [4096, 8192, 16384]


def load_model():
    print("加载Evo2 7B...")
    t0 = time.time()
    from evo2 import Evo2
    model = Evo2("evo2_7b")
    print(f"  完成: {time.time()-t0:.0f}s")
    return model


def extract_embedding(model, dna_seq, max_len=None):
    clean_seq = ''.join(c for c in dna_seq.upper() if c in 'ATCG')
    if len(clean_seq) < 10:
        return None
    # 对于16k窗口，Evo2 7b支持长序列但取中间部分以保持一致
    if max_len and len(clean_seq) > max_len:
        start = (len(clean_seq) - max_len) // 2
        clean_seq = clean_seq[start:start + max_len]

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


def process_file(model, input_path, output_path, max_len=None):
    if os.path.exists(output_path):
        existing = pickle.load(open(output_path, 'rb'))
        print(f"  [SKIP] {output_path} ({existing['num_samples']} samples)")
        return existing

    with open(input_path, 'rb') as f:
        samples = pickle.load(f)

    print(f"  处理 {os.path.basename(input_path)} ({len(samples)} samples)...")
    embeddings, labels, categories, ids = [], [], [], []
    errors = 0
    t0 = time.time()

    for i, s in enumerate(samples):
        emb = extract_embedding(model, s['dna'], max_len=max_len)
        if emb is not None:
            embeddings.append(emb)
            labels.append(s['is_bgc'])
            categories.append(s.get('category_vector', [0]*7))
            ids.append(s.get('id', f'{i}'))
        else:
            errors += 1

        if (i + 1) % 200 == 0 or (i + 1) == len(samples):
            speed = (i + 1) / (time.time() - t0)
            eta = (len(samples) - i - 1) / max(speed, 0.01)
            print(f"    {i+1}/{len(samples)} ({speed:.1f}/s, ETA:{eta:.0f}s, err:{errors})")

    result = {
        'embeddings': np.array(embeddings),
        'labels': np.array(labels),
        'categories': np.array(categories),
        'ids': ids,
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
    parser.add_argument('--shard', type=int, default=-1)
    parser.add_argument('--window', type=int, default=0, help='Only process this window size (0=all)')
    args = parser.parse_args()

    # 收集任务
    tasks = []
    window_sizes = [args.window] if args.window else WINDOW_SIZES
    for ws in window_sizes:
        ws_dir = os.path.join(BASE, f"w{ws}")
        if not os.path.exists(ws_dir):
            continue
        for split in ['test', 'val', 'train']:
            inp = os.path.join(ws_dir, f'{split}.pkl')
            out = os.path.join(ws_dir, f'{split}_emb.pkl')
            if os.path.exists(inp):
                tasks.append((ws, split, inp, out))

    # 分片
    if args.shard == 0:
        tasks = tasks[:len(tasks)//2]
        print(f"SHARD 0: {len(tasks)} tasks")
    elif args.shard == 1:
        tasks = tasks[len(tasks)//2:]
        print(f"SHARD 1: {len(tasks)} tasks")

    # 统计
    total = 0
    for ws, split, inp, out in tasks:
        if not os.path.exists(out):
            with open(inp, 'rb') as f:
                n = len(pickle.load(f))
            total += n
            print(f"  w{ws}/{split}: {n} samples")
        else:
            print(f"  w{ws}/{split}: SKIP")
    print(f"总计: {total} samples")

    if total == 0:
        print("全部已完成!")
        return

    model = load_model()

    # 测试
    test_emb = extract_embedding(model, "ATGCGATCGATCG" * 200)
    assert test_emb is not None, "模型测试失败!"
    print(f"模型测试OK: dim={test_emb.shape[0]}")

    t_total = time.time()
    for ws, split, inp, out in tasks:
        # 对于不同窗口大小，限制输入长度
        # 4096和8192直接用，16384截断到16384（Evo2 7b支持）
        max_len = ws if ws <= 16384 else 16384
        print(f"\n=== w{ws} / {split} (max_len={max_len}) ===")
        process_file(model, inp, out, max_len=max_len)

    print(f"\n全部完成! 总耗时: {(time.time()-t_total)/60:.1f}min")


if __name__ == '__main__':
    main()
