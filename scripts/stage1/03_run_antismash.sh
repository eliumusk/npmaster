#!/bin/bash
# Worker script: for ONE accession, ensure fasta exists (unzip if needed) then run antiSMASH.
# Idempotent: skips if JSON output already exists.
# Usage: run_antismash.sh <accession>
set -euo pipefail

ACC="$1"
ROOT=/data/muskliu/np_master/data
GENOMES=$ROOT/genomes
OUT=$ROOT/antismash_results/$ACC
LOG_DIR=$ROOT/antismash_results/_logs
mkdir -p "$LOG_DIR"

# Already done?
if [[ -s "$OUT/${ACC}.json" ]]; then
    echo "[$(date +%H:%M:%S)] SKIP $ACC (already done)"
    exit 0
fi

FNA="$GENOMES/${ACC}.fna"
GZ="$GENOMES/${ACC}.fna.gz"

if [[ ! -s "$FNA" ]]; then
    if [[ -s "$GZ" ]]; then
        gunzip -k "$GZ"
    else
        echo "[$(date +%H:%M:%S)] ERROR $ACC: fasta not found" >&2
        exit 2
    fi
fi

# clean partial output
rm -rf "$OUT"

START=$(date +%s)
antismash --minimal --no-enable-html --no-zip-output --no-summary-gbk --no-region-gbks \
    --genefinding-tool prodigal --cpus 8 \
    --output-dir "$OUT" "$FNA" > "$LOG_DIR/${ACC}.log" 2>&1
RC=$?
DT=$(( $(date +%s) - START ))

if [[ $RC -eq 0 && -s "$OUT/${ACC}.json" ]]; then
    echo "[$(date +%H:%M:%S)] OK   $ACC (${DT}s)"
    # free decompressed fasta to save disk
    rm -f "$FNA"
else
    echo "[$(date +%H:%M:%S)] FAIL $ACC rc=$RC (${DT}s)"
    exit 1
fi
