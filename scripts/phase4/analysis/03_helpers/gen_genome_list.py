import pandas as pd
df = pd.read_csv("/data/muskliu/np_master/data/stage4_results/p2_t2_hard_neg/windows_metadata_hardneg.csv")
test_genomes = df[df["split"].isin(["test_id", "test_ood"])]["genome_id"].unique()
with open("/tmp/test_genomes.txt", "w") as f:
    for g in sorted(test_genomes):
        f.write(g + "\n")
print(f"Test genomes: {len(test_genomes)}")
# Also split into test_id and test_ood
df_id = df[df["split"]=="test_id"]["genome_id"].unique()
df_ood = df[df["split"]=="test_ood"]["genome_id"].unique()
with open("/tmp/test_id_genomes.txt", "w") as f:
    for g in sorted(df_id): f.write(g + "\n")
with open("/tmp/test_ood_genomes.txt", "w") as f:
    for g in sorted(df_ood): f.write(g + "\n")
print(f"test_id: {len(df_id)}, test_ood: {len(df_ood)}")
