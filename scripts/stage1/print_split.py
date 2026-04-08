#!/usr/bin/env python3
import json
d = json.load(open('data/windows/split_summary.json'))
print('=== split sizes (genomes) ===')
print(json.dumps(d['split_genomes_final'], indent=2))
print()
print('=== windows per split ===')
for spl, c in d['split_counts'].items():
    pct = round(100*c['positive']/(c['positive']+c['negative']), 2)
    print(f"  {spl:10s}: {c['windows']:>8,} total  {c['positive']:>6,} pos  {c['negative']:>7,} neg  {c['ambiguous']:>5,} amb  pos%={pct}")
print()
print('=== type distribution per split ===')
for spl, cls in d['class_distribution_by_split'].items():
    parts = '  '.join(f'{k}={v}' for k,v in sorted(cls.items()))
    print(f'  {spl:10s}: {parts}')
print()
print('=== GCF leak ===')
print(json.dumps(d['gcf_leakage_check'], indent=2))
