import json
with open("/data/muskliu/np_master/data/phase3/route_b/discovery/discovery_benchmark.json") as f:
    d = json.load(f)
genomes = set()
for c in d["candidate_results"]:
    if not c.get("skip", False):
        genomes.add(c["genome_id"])
print(f"Unique genomes in top-50 candidates: {len(genomes)}")
with open("/tmp/cand_genomes.txt", "w") as f:
    for g in sorted(genomes):
        f.write(g + "\n")
# Also get all Route B unmatched genomes (all clean_missed, not just top 50)  
with open("/data/muskliu/np_master/data/phase3/route_b/eval/score_swap_s0.json") as f:
    sw = json.load(f)
all_unmatched_genomes = set()
for split in ["test_id", "test_ood"]:
    for u in sw["results"][split]["score_maxtype"].get("unmatched_details", []):
        all_unmatched_genomes.add(u["genome_id"])
print(f"Unique genomes with any Route B unmatched call: {len(all_unmatched_genomes)}")
with open("/tmp/all_unmatched_genomes.txt", "w") as f:
    for g in sorted(all_unmatched_genomes):
        f.write(g + "\n")
