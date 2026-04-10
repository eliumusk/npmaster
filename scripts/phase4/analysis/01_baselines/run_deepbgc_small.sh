#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate deepbgc
export PATH=/root/miniconda3/envs/deepbgc/bin:/data/conda/miniconda/envs/antismash-env/bin:$PATH

OUT_DIR=/data/muskliu/np_master/data/phase4/baselines/deepbgc
mkdir -p $OUT_DIR /tmp/deepbgc_in2

# Use only the 46 candidate genomes
N_WORKERS=8
echo "=== DeepBGC batch (46 top-candidate genomes) ==="
date

process_one() {
  local gid=$1
  local out=$OUT_DIR/$gid
  if [ -f "$out/out.bgc.tsv" ]; then return; fi  # already complete
  rm -rf $out
  mkdir -p $out
  local fna=/data/muskliu/np_master/data/genomes/${gid}.fna.gz
  local tmpfna=/tmp/deepbgc_in2/${gid}.fna
  zcat $fna > $tmpfna
  deepbgc pipeline --classifier product_activity -o $out $tmpfna > $out/deepbgc.log 2>&1 && touch $out/.done || echo "FAILED $gid" >> $OUT_DIR/failed.log
  rm -f $tmpfna
  rm -f $out/out.full.gbk $out/tmp/*.pfam.tsv 2>/dev/null
}
export -f process_one
export OUT_DIR

cat /tmp/cand_genomes.txt | xargs -P $N_WORKERS -I {} bash -c 'process_one "$@"' _ {}
date
echo DONE
