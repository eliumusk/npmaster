import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


from experiments.lockbox import (  # noqa: E402
    LockboxGateError,
    assert_candidate_frozen,
    gate_lockbox_run,
    is_lockbox_split,
    record_lockbox_open,
)


def _frozen_manifest() -> dict:
    return {
        "experiment_id": "exp-007",
        "git_commit": "abc1234",
        "git_dirty": False,
        "frozen": True,
        "server": "node7",
        "command": "python scripts/run.py",
    }


class IsLockboxSplitTests(unittest.TestCase):
    def test_lockbox_returns_true(self):
        self.assertTrue(is_lockbox_split("lockbox"))

    def test_dev_returns_false(self):
        self.assertFalse(is_lockbox_split("dev"))

    def test_other_returns_false(self):
        self.assertFalse(is_lockbox_split("LOCKBOX"))
        self.assertFalse(is_lockbox_split(""))


class GateLockboxRunTests(unittest.TestCase):
    def test_dev_split_passes_through_without_writing_audit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            audit_log = tmp / "audit.jsonl"
            manifest_path = tmp / "missing.json"  # not even required for dev

            result = gate_lockbox_run(
                split_part="dev",
                candidate_manifest_path=manifest_path,
                audit_log_path=audit_log,
                reason="dev sweep",
            )

            self.assertIsNone(result)
            self.assertFalse(audit_log.exists())

    def test_lockbox_split_with_non_frozen_manifest_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest_path = tmp / "candidate.json"
            manifest = _frozen_manifest()
            manifest["frozen"] = False
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            audit_log = tmp / "audit.jsonl"

            with self.assertRaisesRegex(LockboxGateError, "frozen"):
                gate_lockbox_run(
                    split_part="lockbox",
                    candidate_manifest_path=manifest_path,
                    audit_log_path=audit_log,
                    reason="final eval",
                )
            self.assertFalse(audit_log.exists())

    def test_lockbox_split_with_dirty_git_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest_path = tmp / "candidate.json"
            manifest = _frozen_manifest()
            manifest["git_dirty"] = True
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            audit_log = tmp / "audit.jsonl"

            with self.assertRaisesRegex(LockboxGateError, "git_dirty"):
                gate_lockbox_run(
                    split_part="lockbox",
                    candidate_manifest_path=manifest_path,
                    audit_log_path=audit_log,
                    reason="final eval",
                )
            self.assertFalse(audit_log.exists())

    def test_lockbox_split_with_frozen_manifest_writes_audit_entry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest_path = tmp / "candidate.json"
            manifest = _frozen_manifest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            audit_log = tmp / "audit_subdir" / "audit.jsonl"

            entry = gate_lockbox_run(
                split_part="lockbox",
                candidate_manifest_path=manifest_path,
                audit_log_path=audit_log,
                reason="final lockbox eval",
            )

            self.assertTrue(audit_log.is_file())
            lines = audit_log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            written = json.loads(lines[0])

            self.assertEqual(written, entry)
            self.assertEqual(written["experiment_id"], "exp-007")
            self.assertEqual(written["git_commit"], "abc1234")
            self.assertEqual(written["reason"], "final lockbox eval")
            self.assertIn("timestamp", written)
            self.assertTrue(written["timestamp"])

    def test_lockbox_split_appends_subsequent_audit_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest_path = tmp / "candidate.json"
            manifest_path.write_text(json.dumps(_frozen_manifest()), encoding="utf-8")
            audit_log = tmp / "audit.jsonl"

            gate_lockbox_run(
                split_part="lockbox",
                candidate_manifest_path=manifest_path,
                audit_log_path=audit_log,
                reason="first open",
            )
            gate_lockbox_run(
                split_part="lockbox",
                candidate_manifest_path=manifest_path,
                audit_log_path=audit_log,
                reason="second open",
            )

            lines = audit_log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)

    def test_empty_reason_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest_path = tmp / "candidate.json"
            manifest_path.write_text(json.dumps(_frozen_manifest()), encoding="utf-8")
            audit_log = tmp / "audit.jsonl"

            for bad_reason in ("", "   "):
                with self.assertRaisesRegex(ValueError, "reason"):
                    gate_lockbox_run(
                        split_part="lockbox",
                        candidate_manifest_path=manifest_path,
                        audit_log_path=audit_log,
                        reason=bad_reason,
                    )

    def test_missing_manifest_file_raises_lockbox_gate_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest_path = tmp / "does_not_exist.json"
            audit_log = tmp / "audit.jsonl"

            with self.assertRaisesRegex(LockboxGateError, "not found"):
                gate_lockbox_run(
                    split_part="lockbox",
                    candidate_manifest_path=manifest_path,
                    audit_log_path=audit_log,
                    reason="final eval",
                )
            self.assertFalse(audit_log.exists())


class AssertCandidateFrozenTests(unittest.TestCase):
    def test_returns_validated_manifest_when_frozen(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "candidate.json"
            manifest_path.write_text(json.dumps(_frozen_manifest()), encoding="utf-8")
            result = assert_candidate_frozen(manifest_path)
            self.assertEqual(result["git_commit"], "abc1234")

    def test_missing_git_commit_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "candidate.json"
            manifest = _frozen_manifest()
            manifest["git_commit"] = ""
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(LockboxGateError, "git_commit"):
                assert_candidate_frozen(manifest_path)


class RecordLockboxOpenTests(unittest.TestCase):
    def test_record_requires_non_empty_reason(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_log = Path(tmpdir) / "audit.jsonl"
            with self.assertRaisesRegex(ValueError, "reason"):
                record_lockbox_open(
                    audit_log,
                    experiment_id="exp-007",
                    git_commit="abc1234",
                    candidate_manifest_path=Path("/tmp/foo.json"),
                    reason="",
                )


if __name__ == "__main__":
    unittest.main()
