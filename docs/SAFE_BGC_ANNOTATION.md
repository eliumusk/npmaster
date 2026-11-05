# Safe BGC Annotation Layer

This layer keeps NP-Master as a DNA-model candidate detector, but prevents broad
embedding labels from being reported as definitive BGC mechanism classes.

Design principles, aligned with antiSMASH/GECCO/DeepBGC-style evidence reporting:

- Detection and annotation are separate. A DNA-model region is a candidate until
  protein/domain/context evidence is attached.
- antiSMASH rule hits are evidence, not a veto. Novel BGC discovery must not
  require exact antiSMASH product-rule satisfaction.
- Type labels are conservative. `Polyketide` is split into mechanism-aware labels
  such as `typeI_PKS_supported`, `typeII_PKS_like`, `typeII_or_FAS_like_risk`,
  `partial_PKS_like`, and `polyketide_embedding_only_ambiguous`.
- Short, contig-edge, single-gene, low-margin, and embedding-only calls are kept
  auditable but are not `safe_pass` candidates.
- Hybrid/ambiguous signals are represented by evidence tiers instead of forcing a
  single final product class.

Main scripts:

- `scripts/annotate/safe_bgc_annotation.py`
- `scripts/evaluate/compare_antismash_npmaster_safe.py`
- `scripts/workflows/monitor_sequenced_benchmark.sh`

The current sequenced-genome benchmark writes to `/data/syh/bench_34`.
