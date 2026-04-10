#!/bin/bash
# Run Prodigal + hmmscan (biosyn_pfam) on all null region FASTAs
set -e
source /data/conda/miniconda/etc/profile.d/conda.sh
conda activate antismash-env  # has prodigal + hmmscan

OUT=/data/muskliu/np_master/data/phase4/d2_matched_null
FASTA_DIR=$OUT/null_fasta
PFAM_HMM=/data/muskliu/np_master/data/stage4_results/pfam_validation/biosyn_pfam.hmm
mkdir -p $OUT/null_hits

process_one() {
  local fna=$1
  local base=$(basename $fna .fna)
  local out_dir=$OUT/null_hits/$base
  mkdir -p $out_dir
  
  if [ -f "$out_dir/.done" ]; then return; fi
  
  # Prodigal
  prodigal -i $fna -a $out_dir/proteins.faa -p meta -q > /dev/null 2>&1
  # hmmscan on biosyn Pfam
  hmmscan --cut_ga --tblout $out_dir/pfam.tbl --cpu 1 $PFAM_HMM $out_dir/proteins.faa > /dev/null 2>&1 || true
  touch $out_dir/.done
}
export -f process_one
export OUT PFAM_HMM

ls $FASTA_DIR/*.fna | xargs -P 16 -I {} bash -c 'process_one "$@"' _ {}
echo DONE
