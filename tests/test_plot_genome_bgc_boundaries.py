import csv
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
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class PlotGenomeBcgBoundariesTests(unittest.TestCase):
    def setUp(self):
        self.plotter = load_module(
            "scripts/plot/genome_bgc_boundaries.py",
            "plot_genome_bgc_boundaries",
        )

    def test_parse_npmaster_track_spec(self):
        spec = (
            "Best F1|tmp/preds|max_type|0.995|0.99|4|30000"
        )

        track = self.plotter.parse_npmaster_track_spec(spec)

        self.assertEqual(track.label, "Best F1")
        self.assertEqual(track.results_dir, Path("tmp/preds"))
        self.assertEqual(track.score_family, "max_type")
        self.assertEqual(track.start_threshold, 0.995)
        self.assertEqual(track.extend_threshold, 0.99)
        self.assertEqual(track.min_support_windows, 4)
        self.assertEqual(track.length_cap, 30000)

    def test_plot_genome_writes_png_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            gt_path = tmp / "ground_truth.json"
            pred_dir = tmp / "preds"
            pred_dir.mkdir()
            out_png = tmp / "plot.png"
            out_json = tmp / "plot.json"
            genome_id = "GCA_TEST.1"

            gt_path.write_text(
                json.dumps(
                    {
                        "ground_truth": {
                            genome_id: [
                                ["contigA", 100, 220],
                                ["contigA", 500, 650],
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            with (pred_dir / f"{genome_id}_predictions.csv").open(
                "w",
                newline="",
                encoding="utf-8",
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "contig",
                        "start",
                        "end",
                        "bgc_score",
                        "non_bgc_score",
                    ],
                )
                writer.writeheader()
                for start in range(80, 700, 20):
                    writer.writerow(
                        {
                            "contig": "contigA",
                            "start": start,
                            "end": start + 80,
                            "bgc_score": "0.999",
                            "non_bgc_score": "0.001",
                        }
                    )

            self.plotter.plot_genome_boundaries(
                genome_id=genome_id,
                gt_path=gt_path,
                out_png=out_png,
                out_json=out_json,
                baseline_tracks=[],
                npmaster_tracks=[
                    self.plotter.NPMasterTrack(
                        label="NP test",
                        results_dir=pred_dir,
                        score_family="max_type",
                        start_threshold=0.99,
                        extend_threshold=0.99,
                        min_support_windows=1,
                        length_cap=500,
                    )
                ],
                max_contigs=None,
            )

            self.assertTrue(out_png.exists())
            self.assertGreater(out_png.stat().st_size, 0)
            manifest = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(manifest["genome_id"], genome_id)
            self.assertIn("GT", manifest["tracks"])
            self.assertIn("NP test", manifest["tracks"])
            self.assertEqual(manifest["region_counts"]["GT"], 2)

    def test_plot_zoom_writes_png_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            gt_path = tmp / "ground_truth.json"
            pred_dir = tmp / "preds"
            pred_dir.mkdir()
            out_png = tmp / "zoom.png"
            out_json = tmp / "zoom.json"
            genome_id = "GCA_TEST.1"

            gt_path.write_text(
                json.dumps(
                    {
                        "ground_truth": {
                            genome_id: [
                                ["contigA", 100, 220],
                                ["contigA", 500, 650],
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            with (pred_dir / f"{genome_id}_predictions.csv").open(
                "w",
                newline="",
                encoding="utf-8",
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["contig", "start", "end", "bgc_score", "non_bgc_score"],
                )
                writer.writeheader()
                for start in range(80, 700, 20):
                    writer.writerow(
                        {
                            "contig": "contigA",
                            "start": start,
                            "end": start + 80,
                            "bgc_score": "0.999",
                            "non_bgc_score": "0.001",
                        }
                    )

            self.plotter.plot_zoom_boundaries(
                genome_id=genome_id,
                gt_path=gt_path,
                out_png=out_png,
                out_json=out_json,
                baseline_tracks=[],
                npmaster_tracks=[
                    self.plotter.NPMasterTrack(
                        label="NP test",
                        results_dir=pred_dir,
                        score_family="max_type",
                        start_threshold=0.99,
                        extend_threshold=0.99,
                        min_support_windows=1,
                        length_cap=500,
                    )
                ],
                contig="contigA",
                start_bp=0,
                end_bp=1000,
            )

            self.assertTrue(out_png.exists())
            self.assertGreater(out_png.stat().st_size, 0)
            manifest = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(manifest["mode"], "zoom")
            self.assertEqual(manifest["contig"], "contigA")
            self.assertEqual(manifest["window"], {"start_bp": 0, "end_bp": 1000})


if __name__ == "__main__":
    unittest.main()
