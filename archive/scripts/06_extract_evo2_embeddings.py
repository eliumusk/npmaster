"""
在GPU上用Evo2提取DNA序列的embedding
输入: data/train.pkl, data/val.pkl, data/test.pkl
输出: data/train_emb.pkl, data/val_emb.pkl, data/test_emb.pkl
每个样本: 8192bp DNA → Evo2 blocks.20 → mean pooling → 8192维向量
"""

import pickle
import numpy as np
import torch
import time
import os
import sys

import os
os.environ["HF_HUB_CACHE"] = "/data/public_models/hub"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 配置
DATA_DIR = "/data/muskliu/np_master/data"
LAYER_NAME = "blocks.20.mlp.l3"
BATCH_SIZE = 1
DEVICE = "cuda:0"

def load_model():
    """加载Evo2模型"""
    print("加载Evo2 7B模型...")
    t0 = time.time()
    from evo2 import Evo2
    model = Evo2("evo2_7b")
    print(f"  加载完成: {time.time()-t0:.0f}s")
    return model

def extract_embedding(model, dna_seq, layer_name=LAYER_NAME):
    """提取单条DNA序列的embedding"""
    # 清理序列：去掉N，只保留ATCG
    clean_seq = ''.join(c for c in dna_seq.upper() if c in 'ATCG')
    if len(clean_seq) < 10:
        return None

    # 截断到8192（Evo2 7b_base的上限是8k，7b支持更长但先保守）
    if len(clean_seq) > 8192:
        # 取中间部分
        start = (len(clean_seq) - 8192) // 2
        clean_seq = clean_seq[start:start + 8192]

    try:
        input_ids = torch.tensor(
            model.tokenizer.tokenize(clean_seq),
            dtype=torch.int,
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs, embeddings = model(
                input_ids,
                return_embeddings=True,
                layer_names=[layer_name]
            )

        # embeddings[layer_name]: (1, seq_len, hidden_dim)
        emb = embeddings[layer_name][0].float()  # (seq_len, hidden_dim)

        # Mean pooling over sequence dimension
        emb_mean = emb.mean(dim=0).cpu().numpy()  # (hidden_dim,)

        return emb_mean

    except Exception as e:
        print(f"    错误: {e}")
        return None

def process_split(model, split_name):
    """处理一个数据分割"""
    input_path = os.path.join(DATA_DIR, f"{split_name}.pkl")
    output_path = os.path.join(DATA_DIR, f"{split_name}_emb.pkl")

    print(f"\n处理 {split_name}...")
    with open(input_path, 'rb') as f:
        samples = pickle.load(f)

    print(f"  样本数: {len(samples)}")

    embeddings = []
    labels = []
    categories = []
    ids = []
    errors = 0

    t0 = time.time()
    for i, sample in enumerate(samples):
        emb = extract_embedding(model, sample['dna'])

        if emb is not None:
            embeddings.append(emb)
            labels.append(sample['is_bgc'])
            categories.append(sample.get('category_vector', [0]*7))
            ids.append(sample.get('id', f'{split_name}_{i}'))
        else:
            errors += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            speed = (i + 1) / elapsed
            eta = (len(samples) - i - 1) / speed
            print(f"  {i+1}/{len(samples)} ({speed:.1f} samples/s, ETA: {eta/60:.0f}min, 错误: {errors})")

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
    print(f"  完成: {len(embeddings)}个embedding, {errors}个错误, 耗时{elapsed/60:.1f}min")
    print(f"  保存: {output_path} ({os.path.getsize(output_path)/1e6:.1f}MB)")
    print(f"  Embedding维度: {result['embedding_dim']}")

    return result

def main():
    print("=" * 60)
    print("Evo2 Embedding 提取")
    print("=" * 60)

    # 检查GPU
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f}GB")

    # 加载模型
    model = load_model()

    # 先用一个小样本测试
    print("\n--- 测试 ---")
    test_emb = extract_embedding(model, "ATGCGATCGATCGATCGATCGATCGATCGATCGATCG" * 50)
    if test_emb is not None:
        print(f"  测试成功: shape={test_emb.shape}, |mean|={np.abs(test_emb).mean():.4f}")
    else:
        print("  测试失败！退出。")
        sys.exit(1)

    # 处理所有数据
    t_total = time.time()

    # 先处理test（最小），确认没问题
    test_result = process_split(model, "test")
    val_result = process_split(model, "val")
    train_result = process_split(model, "train")

    total_time = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"全部完成! 总耗时: {total_time/60:.0f}min")
    print(f"  train: {train_result['num_samples']}个, dim={train_result['embedding_dim']}")
    print(f"  val:   {val_result['num_samples']}个")
    print(f"  test:  {test_result['num_samples']}个")


if __name__ == '__main__':
    main()
