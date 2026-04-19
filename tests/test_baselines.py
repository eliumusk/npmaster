import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(rel_path: str, module_name: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class AntismashParserTests(unittest.TestCase):
    def test_reads_areas_and_converts_to_zero_based(self):
        mod = load_module("scripts/baselines/parse_antismash.py", "parse_antismash")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gid = "GCA_test.1"
            (root / gid).mkdir()
            (root / gid / f"{gid}.json").write_text(json.dumps({
                "records": [
                    {"id": "contigA", "areas": [
                        {"start": 1001, "end": 30000, "products": ["PKS", "NRPS"]},
                        {"start": 50001, "end": 60000, "products": []},
                    ]},
                    {"id": "contigB", "areas": []},
                ]
            }))
            preds = mod.parse_antismash_dir(root)
            self.assertEqual(len(preds[gid]), 2)
            p0 = preds[gid][0]
            self.assertEqual(p0["contig_id"], "contigA")
            self.assertEqual(p0["start_bp"], 1000)
            self.assertEqual(p0["end_bp"], 30000)
            self.assertEqual(p0["predicted_type"], "PKS,NRPS")

    def test_skips_malformed_area(self):
        mod = load_module("scripts/baselines/parse_antismash.py", "parse_antismash_skip")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gid = "GCA_x.1"
            (root / gid).mkdir()
            (root / gid / f"{gid}.json").write_text(json.dumps({
                "records": [
                    {"id": "c1", "areas": [
                        {"start": 10, "end": 5},   # inverted -> dropped
                        {"start": None, "end": 100},  # missing start -> dropped
                        {"start": 1, "end": 100},
                    ]},
                ]
            }))
            preds = mod.parse_antismash_dir(root)
            self.assertEqual(len(preds[gid]), 1)


class BgcProphetParserTests(unittest.TestCase):
    def test_parses_per_genome_lists(self):
        mod = load_module("scripts/baselines/parse_bgc_prophet.py", "parse_bgcp")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bgc_prophet_regions.json"
            path.write_text(json.dumps({
                "GCA_1.1": [["c1", 100, 2000], ["c2", 5000, 8000]],
                "GCA_2.1": [["c1", 0, 0]],  # zero-length -> dropped
            }))
            preds = mod.parse_bgc_prophet_file(path)
            self.assertEqual(len(preds["GCA_1.1"]), 2)
            self.assertEqual(preds["GCA_1.1"][0]["contig_id"], "c1")
            self.assertEqual(preds["GCA_2.1"], [])

    def test_genome_id_subset(self):
        mod = load_module("scripts/baselines/parse_bgc_prophet.py", "parse_bgcp_subset")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.json"
            path.write_text(json.dumps({
                "A": [["c", 0, 10]],
                "B": [["c", 0, 10]],
            }))
            preds = mod.parse_bgc_prophet_file(path, genome_ids=["B"])
            self.assertEqual(list(preds.keys()), ["B"])


class DeepbgcParserTests(unittest.TestCase):
    def test_reads_bgc_tsv_and_subtracts_one_from_start(self):
        mod = load_module("scripts/baselines/parse_deepbgc.py", "parse_deepbgc_mod")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gid = "GCA_y.1"
            (root / gid).mkdir()
            (root / gid / f"{gid}.bgc.tsv").write_text(
                "sequence_id\tnucl_start\tnucl_end\tproduct_activity\tdeepbgc_score\tbgc_candidate_id\n"
                "c1\t101\t2000\tsome\t0.9\tbgc_1\n"
                "c1\tbad\t2\t\t\t\n"
            )
            preds = mod.parse_deepbgc_dir(root)
            self.assertEqual(len(preds[gid]), 1)
            self.assertEqual(preds[gid][0]["start_bp"], 100)
            self.assertEqual(preds[gid][0]["end_bp"], 2000)


class GeccoParserTests(unittest.TestCase):
    def test_requires_features_tsv_and_handles_empty_clusters(self):
        mod = load_module("scripts/baselines/parse_gecco.py", "parse_gecco_mod")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            empty = root / "GCA_empty.1"
            empty.mkdir()
            (empty / "GCA_empty.1.features.tsv").write_text("x\n")
            full = root / "GCA_full.1"
            full.mkdir()
            (full / "GCA_full.1.features.tsv").write_text("x\n")
            (full / "GCA_full.1.clusters.tsv").write_text(
                "sequence_id\tstart\tend\ttype\tmax_p\tcluster_id\n"
                "c1\t1\t5000\tPolyketide\t0.8\tclust_1\n"
            )
            no_features = root / "GCA_nofeat.1"
            no_features.mkdir()
            (no_features / "GCA_nofeat.1.clusters.tsv").write_text(
                "sequence_id\tstart\tend\ttype\tmax_p\tcluster_id\n"
                "c1\t1\t100\tx\t0.5\tx\n"
            )
            preds = mod.parse_gecco_dir(root)
            self.assertEqual(preds["GCA_empty.1"], [])
            self.assertEqual(len(preds["GCA_full.1"]), 1)
            self.assertEqual(preds["GCA_full.1"][0]["start_bp"], 0)
            self.assertNotIn("GCA_nofeat.1", preds)


class NineGenomeBaselineAggregatorTests(unittest.TestCase):
    def test_bench_aggregates_tp_counts_across_genomes(self):
        mod = load_module(
            "scripts/evaluate_nine_genome_baselines.py",
            "evaluate_nine_genome_baselines",
        )
        gt = {
            "G1": [("c1", 0, 1000)],
            "G2": [("c1", 0, 1000), ("c1", 2000, 3000)],
        }
        pred = {
            "G1": [("c1", 0, 1500)],    # rec tp (ov/gt=1.0), prec tp (ov/pred=0.67)
            "G2": [("c1", 100, 400)],   # rec no (ov/gt=0.3), prec tp (ov/pred=1.0)
        }
        result = mod.aggregate_over_genomes(gt, pred, ["G1", "G2"], modes=("bench",))
        b = result["summary"]["bench"]
        self.assertEqual(b["tp_recall"], 1)
        self.assertEqual(b["tp_precision"], 2)
        self.assertEqual(b["n_gt"], 3)
        self.assertEqual(b["n_pred"], 2)
        self.assertAlmostEqual(b["recall"], 1 / 3)
        self.assertAlmostEqual(b["precision"], 1.0)

    def test_aggregates_fragmentation_metrics_across_genomes(self):
        mod = load_module(
            "scripts/evaluate_nine_genome_baselines.py",
            "evaluate_nine_genome_baselines_frag",
        )
        gt = {
            "G1": [("c1", 0, 1000)],
            "G2": [("c1", 0, 1000)],
        }
        pred = {
            "G1": [("c1", 0, 600), ("c1", 100, 700)],  # duplicate precision hit
            "G2": [("c1", 0, 600), ("c1", 2000, 2600)],
        }

        result = mod.aggregate_over_genomes(gt, pred, ["G1", "G2"], modes=("bench",))
        frag = result["summary"]["bench"]["fragmentation"]

        self.assertEqual(frag["unique_gt_precision_tp"], 2)
        self.assertEqual(frag["duplicate_precision_tp"], 1)
        self.assertEqual(frag["max_preds_per_gt"], 2)
        self.assertAlmostEqual(frag["dedup_precision"], 0.5)

    def test_region_length_stats_empty(self):
        mod = load_module(
            "scripts/evaluate_nine_genome_baselines.py",
            "evaluate_nine_genome_baselines_empty",
        )
        stats = mod.region_length_stats({"G1": []})
        self.assertEqual(stats, {"n": 0})

    def test_markdown_table_has_bench_columns(self):
        mod = load_module(
            "scripts/evaluate_nine_genome_baselines.py",
            "evaluate_nine_genome_baselines_md",
        )
        rows = [{
            "label": "Foo",
            "summary": {"bench": {"recall": 0.5, "precision": 0.3, "f1": 0.375,
                                  "n_pred": 42, "tp_recall": 10, "tp_precision": 10, "n_gt": 20,
                                  "fragmentation": {"dedup_precision": 0.2, "dedup_f1": 0.2857}}},
            "length_stats": {"n": 42, "median": 12345, "min": 10, "max": 99999, "mean": 5000},
        }]
        md = mod.format_markdown_table(rows)
        self.assertIn("| Foo |", md)
        self.assertIn("0.3750", md)
        self.assertIn("0.2857", md)
        self.assertIn("12,345 bp", md)


if __name__ == "__main__":
    unittest.main()
