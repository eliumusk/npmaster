#!/usr/bin/env python3
"""Analyze GCF leak at old (class|genus) vs new (class|genus|species) granularity."""
import csv, random
from collections import defaultdict, Counter

genome_info = {}
with open('data/selected_genomes.tsv') as f:
    for row in csv.DictReader(f, delimiter='\t'):
        genome_info[row['ncbi_accession']] = row

def primary_class(types_str):
    for c in ['Polyketide','NRP','Terpene','RiPP','Saccharide','Alkaloid','Other']:
        if c in types_str.split(';'): return c
    return 'Unknown'

gcf_old = defaultdict(set)
gcf_new = defaultdict(set)
with open('data/bgc_annotations.csv') as f:
    for row in csv.DictReader(f):
        g = row['genome_id']
        gi = genome_info.get(g, {})
        genus = gi.get('genus', '?')
        species = gi.get('species', '?')
        pc = primary_class(row['bgc_types'])
        gcf_old[pc + '|' + genus].add(g)
        gcf_new[pc + '|' + genus + '|' + species].add(g)

# reconstruct original split
random.seed(20260405)
OOD = {'p__Bacteroidota', 'p__Cyanobacteriota'}
split_orig = {}
remaining = []
for g in sorted(genome_info.keys()):
    if genome_info[g]['phylum'] in OOD:
        split_orig[g] = 'test_ood'
    else:
        remaining.append(g)
random.shuffle(remaining)
n = len(remaining)
n_test = int(round(0.20 * n))
n_val = int(round(0.125 * (n - n_test)))
for g in remaining[:n_test]: split_orig[g] = 'test_id'
for g in remaining[n_test:n_test + n_val]: split_orig[g] = 'val'
for g in remaining[n_test + n_val:]: split_orig[g] = 'train'

def find_leaks(gcf_dict, smap):
    leaks = []
    moved = 0
    genus_ct = Counter()
    for gcf_id, gs in gcf_dict.items():
        sps = set(smap.get(g, '?') for g in gs)
        if 'train' in sps and (sps & {'val', 'test_id', 'test_ood'}):
            victims = [(g, smap[g]) for g in gs if smap.get(g) in ('val', 'test_id', 'test_ood')]
            leaks.append({
                'gcf_id': gcf_id, 'total': len(gs), 'victims': len(victims),
                'by_split': Counter(s for _, s in victims)
            })
            moved += len(victims)
            parts = gcf_id.split('|')
            genus_ct[parts[1] if len(parts) > 1 else gcf_id] += len(victims)
    leaks.sort(key=lambda x: -x['victims'])
    return leaks, moved, genus_ct

old_l, old_m, old_g = find_leaks(gcf_old, split_orig)
new_l, new_m, new_g = find_leaks(gcf_new, split_orig)

print('=== OLD: primary_class|genus ===')
print(f'leaked GCFs: {len(old_l)}, moved: {old_m}')
print('Top 15 leaked GCFs:')
for x in old_l[:15]:
    bs = dict(x['by_split'])
    print(f'  {x["gcf_id"]:50s}  genomes={x["total"]:4d}  moved={x["victims"]:3d}  {bs}')
print('Top 10 genera by damage:')
for g, n in old_g.most_common(10):
    print(f'  {g:40s}  {n}')

print()
print('=== NEW: primary_class|genus|species ===')
print(f'leaked GCFs: {len(new_l)}, moved: {new_m}')
print('Top 15:')
for x in new_l[:15]:
    bs = dict(x['by_split'])
    print(f'  {x["gcf_id"]:70s}  total={x["total"]:3d}  moved={x["victims"]:2d}  {bs}')
print('Top 10 genera by damage:')
for g, n in new_g.most_common(10):
    print(f'  {g:40s}  {n}')

print()
print('=== COMPARISON ===')
print(f'OLD: {len(old_l)} leaked, {old_m} moved')
print(f'NEW: {len(new_l)} leaked, {new_m} moved')
print(f'Recovered: {old_m - new_m} genomes back to val/test')

# simulate new split
new_split = dict(split_orig)
for x in new_l:
    for g in gcf_new[x['gcf_id']]:
        if new_split.get(g) in ('val', 'test_id', 'test_ood'):
            new_split[g] = 'train'
final = Counter(new_split.values())
orig_ct = Counter(split_orig.values())
print(f'Original:  {dict(orig_ct)}')
print(f'After NEW fix: {dict(final)}')
