import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


from experiments.manifest import load_split, validate_manifest  # noqa: E402


SPLIT_PATH = ROOT / "configs" / "splits" / "nine_genome_dev_lockbox_v1.json"
SCHEMA_PATH = ROOT / "configs" / "experiments" / "stage0_manifest_schema.json"

EXPECTED_GENOME_IDS = [
    "GCA_000154945.1",
    "GCA_000156475.1",
    "GCA_000158875.1",
    "GCA_000158915.1",
    "GCA_000568915.1",
    "GCA_000156435.1",
    "GCA_000158815.1",
    "GCA_000158895.1",
    "GCA_000568255.1",
]


class ExperimentSplitTests(unittest.TestCase):
    def test_split_file_contains_all_nine_genomes_exactly_once(self):
        split = load_split(SPLIT_PATH)

        combined = split["dev"] + split["lockbox"]
        self.assertEqual(combined, EXPECTED_GENOME_IDS)
        self.assertEqual(len(combined), len(set(combined)))

    def test_split_file_has_stratified_5_4_partition(self):
        split = load_split(SPLIT_PATH)

        self.assertEqual(split["version"], "nine_genome_dev_lockbox_v1")
        self.assertEqual(len(split["dev"]), 5)
        self.assertEqual(len(split["lockbox"]), 4)

    def test_manifest_schema_lists_required_stage0_fields(self):
        with SCHEMA_PATH.open(encoding="utf-8") as handle:
            schema = json.load(handle)

        self.assertEqual(
            schema["required"],
            ["experiment_id", "git_commit", "server", "command"],
        )


class ExperimentManifestValidationTests(unittest.TestCase):
    def test_validate_manifest_accepts_minimal_required_fields(self):
        manifest = {
            "experiment_id": "exp-001",
            "git_commit": "abc1234",
            "server": "server-a",
            "command": "python scripts/run.py --dry-run",
        }

        validated = validate_manifest(manifest)
        self.assertEqual(validated, manifest)

    def test_validate_manifest_preserves_extra_metadata_fields(self):
        manifest = {
            "experiment_id": "exp-001",
            "git_commit": "abc1234",
            "server": "server-a",
            "command": "python scripts/run.py --dry-run",
            "split_version": "nine_genome_dev_lockbox_v1",
            "window": 16384,
        }

        validated = validate_manifest(manifest)
        self.assertEqual(validated["split_version"], "nine_genome_dev_lockbox_v1")
        self.assertEqual(validated["window"], 16384)

    def test_validate_manifest_rejects_missing_required_fields(self):
        for field in ("experiment_id", "git_commit", "server", "command"):
            manifest = {
                "experiment_id": "exp-001",
                "git_commit": "abc1234",
                "server": "server-a",
                "command": "python scripts/run.py --dry-run",
            }
            manifest.pop(field)

            with self.assertRaisesRegex(ValueError, field):
                validate_manifest(manifest)

    def test_load_split_rejects_duplicate_genomes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "split.json"
            path.write_text(
                json.dumps(
                    {
                        "version": "nine_genome_dev_lockbox_v1",
                        "dev": EXPECTED_GENOME_IDS[:5],
                        "lockbox": EXPECTED_GENOME_IDS[4:8],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "exactly once"):
                load_split(path)


if __name__ == "__main__":
    unittest.main()
