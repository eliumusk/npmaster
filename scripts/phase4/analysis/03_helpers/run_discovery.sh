#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate bgc
cd /data/muskliu/np_master/code/np-master
python -u scripts/eval/ws_discovery_benchmark.py \
  --results /data/muskliu/np_master/data/phase3/route_b/eval/score_swap_s0.json \
  --genomes-dir /data/muskliu/np_master/data/genomes \
  --pfam-db /data/muskliu/np_master/data/stage4_results/pfam_validation/biosyn_pfam.hmm \
  --biosynthetic-list /data/muskliu/np_master/data/stage4_results/pfam_validation/biosyn_names.txt \
  --meta /data/muskliu/np_master/data/stage4_results/p2_t2_hard_neg/windows_metadata_hardneg.csv \
  --out /data/muskliu/np_master/data/phase3/route_b/discovery \
  --cpus 16 --top-n 50
