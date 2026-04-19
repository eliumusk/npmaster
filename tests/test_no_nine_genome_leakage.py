"""Tests for the Stage 5 nine-genome embargo leakage checker.

The checker scans text-like files for occurrences of embargoed 9-genome
assembly IDs. It is used as a pre-flight guard before launching any
Stage 5 training / mining / selection job, to ensure no 9-genome data
leaks into the selection pipeline.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module(rel_path: str, module_name: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_checker = _load_module(
    "scripts/check_no_nine_genome_leakage.py",
    "check_no_nine_genome_leakage",
)
load_embargo_ids = _checker.load_embargo_ids
scan_paths_for_embargo_ids = _checker.scan_paths_for_embargo_ids


EMBARGO_FIXTURE = {
    "version": "test_embargo",
    "embargoed_genome_ids": [
        "GCA_000154945.1",
        "GCA_000158915.1",
    ],
}


class LoadEmbargoIdsTests(unittest.TestCase):
    def test_loads_ids_from_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "embargo.json"
            path.write_text(json.dumps(EMBARGO_FIXTURE), encoding="utf-8")
            ids = load_embargo_ids(path)
            self.assertEqual(
                ids, {"GCA_000154945.1", "GCA_000158915.1"}
            )

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_embargo_ids(Path("/nonexistent/embargo.json"))

    def test_malformed_json_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "embargo.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_embargo_ids(path)


class ScanPathsForEmbargoIdsTests(unittest.TestCase):
    def test_detects_embargo_id_in_text_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text("genome: GCA_000158915.1\n", encoding="utf-8")
            hits = scan_paths_for_embargo_ids(
                [path], {"GCA_000158915.1", "GCA_000154945.1"}
            )
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0]["genome_id"], "GCA_000158915.1")
            self.assertEqual(hits[0]["line"], 1)

    def test_ignores_clean_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text("genome: GCA_999999999.1\n", encoding="utf-8")
            hits = scan_paths_for_embargo_ids([path], {"GCA_000158915.1"})
            self.assertEqual(hits, [])

    def test_recursively_scans_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "sub").mkdir()
            clean = root / "sub" / "clean.txt"
            clean.write_text("hello\n", encoding="utf-8")
            dirty = root / "sub" / "dirty.csv"
            dirty.write_text(
                "id,path\n1,GCA_000154945.1/chrom.fna\n", encoding="utf-8"
            )
            hits = scan_paths_for_embargo_ids(
                [root], {"GCA_000154945.1"}
            )
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0]["genome_id"], "GCA_000154945.1")
            self.assertEqual(hits[0]["line"], 2)
            self.assertTrue(hits[0]["path"].endswith("dirty.csv"))

    def test_multiple_hits_on_same_line(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mixed.txt"
            path.write_text(
                "both GCA_000154945.1 and GCA_000158915.1 here\n",
                encoding="utf-8",
            )
            hits = scan_paths_for_embargo_ids(
                [path],
                {"GCA_000154945.1", "GCA_000158915.1"},
            )
            self.assertEqual(len(hits), 2)
            found = {hit["genome_id"] for hit in hits}
            self.assertEqual(
                found, {"GCA_000154945.1", "GCA_000158915.1"}
            )

    def test_skips_binary_files_without_raising(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "blob.bin"
            path.write_bytes(b"\x00\x01\x02\x03GCA_000154945.1\xff\xfe")
            # Should not crash; binary files are skipped.
            hits = scan_paths_for_embargo_ids([path], {"GCA_000154945.1"})
            self.assertEqual(hits, [])

    def test_skips_common_noise_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            noise = root / ".git"
            noise.mkdir()
            (noise / "HEAD").write_text(
                "ref: refs/heads/GCA_000154945.1\n", encoding="utf-8"
            )
            hits = scan_paths_for_embargo_ids([root], {"GCA_000154945.1"})
            self.assertEqual(hits, [])

    def test_allowlist_skips_explicitly_allowed_files(self):
        """Final 9-genome report files legitimately contain embargo IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            allowed = root / "final_9genome.json"
            allowed.write_text('{"GCA_000154945.1": 0.5}', encoding="utf-8")
            hits = scan_paths_for_embargo_ids(
                [root],
                {"GCA_000154945.1"},
                allowlist_globs=["final_9genome.json"],
            )
            self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
