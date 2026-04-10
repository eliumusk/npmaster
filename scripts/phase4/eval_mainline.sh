#!/bin/bash
# P0: Reproduce Phase 3 Route B 3-seed evaluation.
# Produces test_id + test_ood per-bucket metrics for all 3 seeds.

set -e
GPU=${GPU:-0}

cd /data/muskliu/np_master/code/np-master

CUDA_VISIBLE_DEVICES=$GPU python -u /tmp/eval_3seed.py
# eval_3seed.py is saved at /tmp/eval_3seed.py during original Phase 3 run
# and produces /data/muskliu/np_master/data/phase3/route_b/eval/3seed_results.json
