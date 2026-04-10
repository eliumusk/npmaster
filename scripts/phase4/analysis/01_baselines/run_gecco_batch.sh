#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate gecco

OUT_DIR=/data/muskliu/np_master/data/phase4/baselines/gecco
mkdir -p $OUT_DIR /tmp/gecco_in

N_WORKERS=16
N_GENOMES=$(wc -l < /tmp/test_genomes.txt)
echo "=== GECCO batch: $N_GENOMES genomes x $N_WORKERS workers ==="
date

process_one() {
  local gid=$1
  local out=$OUT_DIR/$gid
  if [ -f "$out/.done" ]; then
    return
  fi
  mkdir -p $out
  local fna=/data/muskliu/np_master/data/genomes/${gid}.fna.gz
  if [ ! -f "$fna" ]; then
    echo "MISSING $gid" >> $OUT_DIR/missing.log
    return
  fi
  local tmpfna=/tmp/gecco_in/${gid}.fna
  zcat $fna > $tmpfna
  gecco run -o $out -g $tmpfna > $out/gecco.log 2>&1 && touch $out/.done || echo "FAILED $gid" >> $OUT_DIR/failed.log
  rm -f $tmpfna
}
export -f process_one
export OUT_DIR

cat /tmp/test_genomes.txt | xargs -P $N_WORKERS -I {} bash -c 'process_one "$@"' _ {}
date
echo DONE
