"""
全基因组滑窗扫描 Pipeline
Step 1: 切窗口 (本地)
Step 2: 提取embedding + 预测 (GPU)
Step 3: 生成概率profile + 标注候选区域 (本地)

用法:
  # Step 1: 本地切窗口
  python3 17_genome_scan.py prepare

  # Step 2: GPU上跑 (需要训练好的模型)
  python3 17_genome_scan.py predict

  # Step 3: 本地分析结果
  python3 17_genome_scan.py analyze
"""

import pickle, json, os, sys, re, argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

# ============ 配置 ============
TARGETS = {
    'S_longhuiensis':  'GCF_020616555.1',
    'S_deccanensis':   'GCF_022385335.1',
    'S_aquilus':        'GCF_003955715.1',
}

WINDOW_SIZE = 16384
STRIDE = 8192  # 50% overlap → 平滑的概率曲线

# 路径 (本地)
LOCAL_BASE = Path("data")
REF_DIR = LOCAL_BASE / "ref_genomes_batch/streptomyces/ncbi_dataset/data"
ANTISMASH_DIR = LOCAL_BASE / "antismash_result"
SCAN_DIR = LOCAL_BASE / "genome_scan"

# 路径 (GPU服务器)
REMOTE_BASE = "/data/muskliu/np_master/data"
REMOTE_SCAN = REMOTE_BASE + "/genome_scan"
REMOTE_EXPANDED = REMOTE_BASE + "/expanded_strep/w16384"


def prepare():
    """Step 1: 从3个基因组切出扫描窗口"""
    from Bio import SeqIO
    import zipfile

    SCAN_DIR.mkdir(parents=True, exist_ok=True)

    for name, gcf_id in TARGETS.items():
        print(f"\n=== {name} ({gcf_id}) ===")

        # 读基因组
        gcf_dir = REF_DIR / gcf_id
        fna = list(gcf_dir.glob("*.fna"))[0]
        contigs = list(SeqIO.parse(fna, 'fasta'))
        total_size = sum(len(c.seq) for c in contigs)
        print(f"  基因组: {total_size/1e6:.1f} Mb, {len(contigs)} contigs")

        # 读antiSMASH标注
        bgc_regions = {}  # contig_id → [(start, end, type), ...]
        as_dir = ANTISMASH_DIR / gcf_id
        as_zip = ANTISMASH_DIR / f"{gcf_id}.zip"

        if as_dir.is_dir():
            for rf in sorted(as_dir.glob("*.region*.gbk")):
                match = re.match(r'(.+)\.region(\d+)\.gbk', rf.name)
                if not match:
                    continue
                contig_id = match.group(1)
                with open(rf) as f:
                    content = f.read()
                    # antiSMASH 8.0: 真实坐标在COMMENT的Orig. start/end
                    orig_start = re.search(r'Orig\. start\s*::\s*(\d+)', content[:2000])
                    orig_end = re.search(r'Orig\. end\s*::\s*(\d+)', content[:2000])
                    if orig_start and orig_end:
                        start = int(orig_start.group(1))
                        end = int(orig_end.group(1))
                    else:
                        # fallback: 旧格式
                        loc = re.search(r'source\s+(\d+)\.\.(\d+)', content[:3000])
                        if loc:
                            start, end = int(loc.group(1)) - 1, int(loc.group(2))
                        else:
                            continue
                    bgc_type = "unknown"
                    type_match = re.search(r'/product="([^"]+)"', content[:5000])
                    if type_match:
                        bgc_type = type_match.group(1)
                    bgc_regions.setdefault(contig_id, []).append((start, end, bgc_type))
        elif as_zip.exists():
            with zipfile.ZipFile(as_zip) as zf:
                for rf in sorted(zf.namelist()):
                    if '.region' not in rf or not rf.endswith('.gbk'):
                        continue
                    basename = os.path.basename(rf)
                    match = re.match(r'(.+)\.region(\d+)\.gbk', basename)
                    if not match:
                        continue
                    contig_id = match.group(1)
                    with zf.open(rf) as f:
                        content = f.read().decode('utf-8', errors='ignore')
                        orig_start = re.search(r'Orig\. start\s*::\s*(\d+)', content[:2000])
                        orig_end = re.search(r'Orig\. end\s*::\s*(\d+)', content[:2000])
                        if orig_start and orig_end:
                            start = int(orig_start.group(1))
                            end = int(orig_end.group(1))
                        else:
                            loc = re.search(r'source\s+(\d+)\.\.(\d+)', content[:3000])
                            if loc:
                                start, end = int(loc.group(1)) - 1, int(loc.group(2))
                            else:
                                continue
                        bgc_type = "unknown"
                        type_match = re.search(r'/product="([^"]+)"', content[:5000])
                        if type_match:
                            bgc_type = type_match.group(1)
                        bgc_regions.setdefault(contig_id, []).append((start, end, bgc_type))

        n_bgc = sum(len(v) for v in bgc_regions.values())
        print(f"  antiSMASH: {n_bgc} BGC regions")

        # 切窗口
        windows = []
        for contig in contigs:
            seq = str(contig.seq).upper()
            contig_id = contig.id

            for start in range(0, len(seq) - WINDOW_SIZE + 1, STRIDE):
                window = seq[start:start + WINDOW_SIZE]
                n_frac = window.count('N') / len(window)
                if n_frac > 0.1:
                    continue  # 跳过N太多的区域

                # 标注这个窗口是否overlaps with known BGC
                is_known_bgc = False
                contig_bgcs = bgc_regions.get(contig_id, [])
                for bgc_start, bgc_end, _ in contig_bgcs:
                    # 窗口中心是否在BGC内
                    win_center = start + WINDOW_SIZE // 2
                    if bgc_start <= win_center < bgc_end:
                        is_known_bgc = True
                        break

                windows.append({
                    'dna': window,
                    'contig': contig_id,
                    'start': start,
                    'end': start + WINDOW_SIZE,
                    'is_known_bgc': is_known_bgc,
                })

        print(f"  切出 {len(windows)} 个窗口 (stride={STRIDE})")
        known = sum(1 for w in windows if w['is_known_bgc'])
        print(f"  已知BGC内: {known}, 非BGC: {len(windows)-known}")

        # 保存
        out_dir = SCAN_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "windows.pkl", 'wb') as f:
            pickle.dump(windows, f)

        # 保存antiSMASH标注供后续分析
        with open(out_dir / "antismash_regions.json", 'w') as f:
            # 转换tuple为list for JSON
            json_regions = {k: [(s, e, t) for s, e, t in v] for k, v in bgc_regions.items()}
            json.dump(json_regions, f, indent=2)

        # 保存元数据
        meta = {
            'gcf_id': gcf_id,
            'name': name,
            'genome_size': total_size,
            'n_contigs': len(contigs),
            'n_windows': len(windows),
            'n_known_bgc_windows': known,
            'n_antismash_bgcs': n_bgc,
            'window_size': WINDOW_SIZE,
            'stride': STRIDE,
            'contigs': {c.id: len(c.seq) for c in contigs},
        }
        with open(out_dir / "metadata.json", 'w') as f:
            json.dump(meta, f, indent=2)

    print("\n准备完成! 下一步: 上传到GPU服务器跑predict")


def predict():
    """Step 2: GPU上提取embedding + RF预测 (在服务器上运行)"""
    import torch, time
    os.environ["HF_HUB_CACHE"] = "/data/public_models/hub"
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    from sklearn.ensemble import RandomForestClassifier

    SCAN_BASE = Path(REMOTE_SCAN)
    LAYER_NAME = "blocks.20.mlp.l3"

    # 1. 训练RF (用16k扩充数据)
    print("加载训练数据...")
    with open(f"{REMOTE_EXPANDED}/train_emb.pkl", 'rb') as f:
        train_data = pickle.load(f)
    X_train = train_data['embeddings']
    y_train = train_data['labels']

    print(f"训练RF (train={len(y_train)} samples)...")
    clf = RandomForestClassifier(100, max_depth=20, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    # 2. 加载Evo2
    print("加载Evo2...")
    t0 = time.time()
    from evo2 import Evo2
    model = Evo2("evo2_7b")
    print(f"  模型加载: {time.time()-t0:.0f}s")

    # 3. 对每个基因组提取embedding并预测
    for name in ['S_longhuiensis', 'S_deccanensis', 'S_aquilus']:
        scan_dir = SCAN_BASE / name
        if not scan_dir.exists():
            print(f"  {name}: 找不到数据, 跳过")
            continue

        with open(scan_dir / "windows.pkl", 'rb') as f:
            windows = pickle.load(f)

        print(f"\n=== {name}: {len(windows)} windows ===")
        embeddings = []
        errors = 0
        t0 = time.time()

        for i, w in enumerate(windows):
            clean_seq = ''.join(c for c in w['dna'] if c in 'ATCG')
            if len(clean_seq) < 100:
                embeddings.append(None)
                errors += 1
                continue

            try:
                input_ids = torch.tensor(
                    model.tokenizer.tokenize(clean_seq), dtype=torch.int,
                ).unsqueeze(0).to("cuda:0")

                with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    _, emb_dict = model(input_ids, return_embeddings=True, layer_names=[LAYER_NAME])

                emb = emb_dict[LAYER_NAME][0].float().mean(dim=0).cpu().numpy()
                embeddings.append(emb)
            except Exception as e:
                embeddings.append(None)
                errors += 1

            if (i + 1) % 100 == 0 or (i + 1) == len(windows):
                speed = (i + 1) / (time.time() - t0)
                eta = (len(windows) - i - 1) / max(speed, 0.01)
                print(f"  {i+1}/{len(windows)} ({speed:.1f}/s, ETA:{eta:.0f}s, err:{errors})")

        # 预测
        valid_idx = [i for i, e in enumerate(embeddings) if e is not None]
        X = np.array([embeddings[i] for i in valid_idx])
        probs = clf.predict_proba(X)[:, 1]

        # 组装结果
        results = []
        prob_idx = 0
        for i, w in enumerate(windows):
            if embeddings[i] is not None:
                results.append({
                    'contig': w['contig'],
                    'start': w['start'],
                    'end': w['end'],
                    'prob': float(probs[prob_idx]),
                    'is_known_bgc': w['is_known_bgc'],
                })
                prob_idx += 1
            else:
                results.append({
                    'contig': w['contig'],
                    'start': w['start'],
                    'end': w['end'],
                    'prob': 0.0,
                    'is_known_bgc': w['is_known_bgc'],
                    'error': True,
                })

        elapsed = time.time() - t0
        print(f"  完成: {len(valid_idx)} ok, {errors} err, {elapsed:.0f}s")

        # 保存
        with open(scan_dir / "predictions.pkl", 'wb') as f:
            pickle.dump(results, f)

        # 也保存为JSON (方便查看)
        with open(scan_dir / "predictions_summary.json", 'w') as f:
            # 只保存候选区域 (prob > 0.5 且不是已知BGC)
            candidates = [r for r in results if r['prob'] > 0.5 and not r['is_known_bgc']]
            known_correct = [r for r in results if r['is_known_bgc'] and r['prob'] > 0.5]
            known_missed = [r for r in results if r['is_known_bgc'] and r['prob'] <= 0.5]

            summary = {
                'total_windows': len(results),
                'known_bgc_detected': len(known_correct),
                'known_bgc_missed': len(known_missed),
                'novel_candidates': len(candidates),
                'novel_candidates_high_conf': len([c for c in candidates if c['prob'] > 0.8]),
                'candidates': sorted(candidates, key=lambda x: -x['prob'])[:50],
            }
            json.dump(summary, f, indent=2)

    print("\n预测完成!")


def analyze():
    """Step 3: 分析预测结果, 找候选区域"""
    print("=" * 70)
    print("全基因组扫描结果分析")
    print("=" * 70)

    for name, gcf_id in TARGETS.items():
        scan_dir = SCAN_DIR / name
        pred_path = scan_dir / "predictions.pkl"
        meta_path = scan_dir / "metadata.json"
        as_path = scan_dir / "antismash_regions.json"

        if not pred_path.exists():
            print(f"\n  {name}: 缺少predictions.pkl, 跳过")
            continue

        with open(pred_path, 'rb') as f:
            predictions = pickle.load(f)
        with open(meta_path) as f:
            meta = json.load(f)
        with open(as_path) as f:
            antismash = json.load(f)

        print(f"\n{'='*60}")
        print(f"  {name} ({gcf_id})")
        print(f"  基因组: {meta['genome_size']/1e6:.1f} Mb, antiSMASH BGC: {meta['n_antismash_bgcs']}")
        print(f"{'='*60}")

        # 统计
        probs = [p['prob'] for p in predictions]
        known = [p for p in predictions if p['is_known_bgc']]
        unknown = [p for p in predictions if not p['is_known_bgc']]

        print(f"\n  窗口统计:")
        print(f"    总计: {len(predictions)}")
        print(f"    已知BGC内: {len(known)} (mean prob={np.mean([p['prob'] for p in known]):.3f})")
        print(f"    非BGC: {len(unknown)} (mean prob={np.mean([p['prob'] for p in unknown]):.3f})")

        # 检测率: 已知BGC是否被正确识别
        if known:
            known_detected = sum(1 for p in known if p['prob'] > 0.5)
            print(f"\n  已知BGC检测率: {known_detected}/{len(known)} ({known_detected/len(known)*100:.0f}%)")

        # 找候选新BGC区域: 连续高概率窗口 + 不在已知BGC内
        print(f"\n  候选新BGC区域 (prob>0.7, 非已知BGC):")
        # 按contig和位置排序
        by_contig = defaultdict(list)
        for p in predictions:
            by_contig[p['contig']].append(p)

        candidate_regions = []
        for contig_id, preds in by_contig.items():
            preds.sort(key=lambda x: x['start'])

            # 合并连续的高概率窗口
            in_region = False
            region_start = 0
            region_probs = []

            for p in preds:
                if p['prob'] > 0.5 and not p['is_known_bgc']:
                    if not in_region:
                        in_region = True
                        region_start = p['start']
                        region_probs = []
                    region_probs.append(p['prob'])
                else:
                    if in_region and len(region_probs) >= 2:  # 至少2个连续窗口
                        region_end = preds[preds.index(p) - 1]['end'] if p != preds[0] else p['end']
                        candidate_regions.append({
                            'contig': contig_id,
                            'start': region_start,
                            'end': region_end,
                            'length': region_end - region_start,
                            'max_prob': max(region_probs),
                            'mean_prob': np.mean(region_probs),
                            'n_windows': len(region_probs),
                        })
                    in_region = False

            # 处理末尾
            if in_region and len(region_probs) >= 2:
                candidate_regions.append({
                    'contig': contig_id,
                    'start': region_start,
                    'end': preds[-1]['end'],
                    'length': preds[-1]['end'] - region_start,
                    'max_prob': max(region_probs),
                    'mean_prob': np.mean(region_probs),
                    'n_windows': len(region_probs),
                })

        # 按max_prob排序
        candidate_regions.sort(key=lambda x: -x['max_prob'])

        if candidate_regions:
            print(f"    找到 {len(candidate_regions)} 个候选区域:")
            for i, c in enumerate(candidate_regions[:10]):
                print(f"    #{i+1}: {c['contig']}:{c['start']}-{c['end']} "
                      f"({c['length']/1000:.1f}kb) "
                      f"max_prob={c['max_prob']:.3f} mean={c['mean_prob']:.3f} "
                      f"({c['n_windows']} windows)")
        else:
            print(f"    未找到候选区域")

        # 保存候选
        with open(scan_dir / "candidate_regions.json", 'w') as f:
            json.dump(candidate_regions, f, indent=2)

        # FP/Mb统计
        genome_mb = meta['genome_size'] / 1e6
        n_fp_windows = sum(1 for p in unknown if p['prob'] > 0.5)
        print(f"\n  假阳性统计:")
        print(f"    非BGC区域高概率窗口: {n_fp_windows} (可能是假阳性或新BGC)")
        print(f"    候选区域数/Mb: {len(candidate_regions)/genome_mb:.1f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('step', choices=['prepare', 'predict', 'analyze'])
    args = parser.parse_args()

    if args.step == 'prepare':
        prepare()
    elif args.step == 'predict':
        predict()
    elif args.step == 'analyze':
        analyze()
