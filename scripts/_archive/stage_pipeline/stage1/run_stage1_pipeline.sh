#!/bin/bash
# Master pipeline: download all genomes, then run antiSMASH in parallel.
# Designed to be idempotent & resumable.
set -uo pipefail

ROOT=/data/muskliu/np_master/data
SCRIPTS=/data/muskliu/np_master/scripts/stage1
PARALLEL=${PARALLEL:-16}
LOG=$ROOT/stage1_pipeline.log

source /data/conda/ourconda_bashrc
conda activate antismash-env

export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

echo "[$(date)] === Stage 1 pipeline start ===" | tee -a "$LOG"

# ===== Phase 1: download all genomes =====
echo "[$(date)] Phase 1: downloading genomes via aria2..." | tee -a "$LOG"
mkdir -p "$ROOT/genomes"
cd "$ROOT/genomes"
# aria2 auto-skips completed files; retries handles transient fails
aria2c -i urls.txt -j 16 -x 4 -s 4 --continue=true \
    --max-tries=5 --retry-wait=5 --timeout=60 \
    --console-log-level=warn --summary-interval=30 \
    --all-proxy=http://127.0.0.1:7890 2>&1 | tee -a "$LOG"

N_DL=$(ls *.fna.gz 2>/dev/null | wc -l)
echo "[$(date)] Downloaded $N_DL genomes" | tee -a "$LOG"

# ===== Phase 2: antiSMASH in parallel =====
echo "[$(date)] Phase 2: antiSMASH (parallel=$PARALLEL, 8 cpus/job)..." | tee -a "$LOG"
mkdir -p "$ROOT/antismash_results"

# build accession list from manifest (only those with resolved URLs)
awk -F'\t' 'NR>1 && $2!="" {print $1}' "$ROOT/genomes/manifest.tsv" > "$ROOT/antismash_results/accessions.txt"
N_ACC=$(wc -l < "$ROOT/antismash_results/accessions.txt")
echo "[$(date)] Will process $N_ACC accessions" | tee -a "$LOG"

chmod +x "$SCRIPTS/03_run_antismash.sh"
cat "$ROOT/antismash_results/accessions.txt" | \
    xargs -n 1 -P "$PARALLEL" -I {} "$SCRIPTS/03_run_antismash.sh" {} 2>&1 | tee -a "$LOG"

N_OK=$(ls "$ROOT/antismash_results"/*/[!_]*.json 2>/dev/null | wc -l)
echo "[$(date)] === Stage 1 pipeline done. antiSMASH ok: $N_OK / $N_ACC ===" | tee -a "$LOG"
