"""The honesty gate needs its own guard, for the same reason the DDP test does.

A checker that cannot fail proves nothing, so every test here is paired with
its inverse.
"""

from __future__ import annotations

import json
from pathlib import Path

from dvit.verify_claims import extract, main, measured, rounds_to


class TestRoundsTo:
    def test_a_claim_passes_at_the_precision_it_was_written_with(self) -> None:
        assert rounds_to(1.963, "1.96")
        assert rounds_to(98.14, "98")
        assert rounds_to(1853.0, "1,853")

    def test_a_wrong_claim_fails(self) -> None:
        # The prediction that was actually wrong in this project.
        assert not rounds_to(1.963, "1.8")
        assert not rounds_to(98.14, "57")

    def test_extra_precision_is_not_granted(self) -> None:
        # Writing 1.96x from 1.963 is rounding. Writing 1.963x from 1.96 is not.
        assert rounds_to(1.963, "1.96")
        assert not rounds_to(1.96, "1.963")


class TestMeasured:
    def test_ratios_are_only_formed_within_one_hardware_group(self) -> None:
        runs = [
            {"hardware": "A", "median_images_per_sec": 100.0, "world_size": 1,
             "best_test_acc": 0.5, "peak_memory_mb": 0.0, "model_params": 10},
            {"hardware": "B", "median_images_per_sec": 200.0, "world_size": 1,
             "best_test_acc": 0.5, "peak_memory_mb": 0.0, "model_params": 10},
        ]
        # 200/100 = 2.0 would be a cross hardware ratio and must not appear.
        assert not any(rounds_to(v, "2.00") for v in measured(runs)["ratio"])

    def test_gradient_payload_is_derived_from_recorded_parameters(self) -> None:
        runs = [{"hardware": "A", "median_images_per_sec": 1.0, "world_size": 1,
                 "best_test_acc": 0.0, "peak_memory_mb": 0.0,
                 "model_params": 1_806_538}]
        assert any(rounds_to(v, "7") for v in measured(runs)["memory"])


class TestExtraction:
    def test_percent_sign_is_caught_not_just_the_word(self, tmp_path: Path) -> None:
        """Regression: a trailing word boundary after the alternation meant
        `98%` never matched, because a percent sign followed by a space is
        two non word characters. Only `98 percent` was ever checked, so the
        gate passed a document while silently skipping most of its figures."""
        f = tmp_path / "doc.md"
        f.write_text("Efficiency was 98% in fp32 and 57 percent in fp16.")
        found = sorted(c.text for c in extract(f) if c.kind == "percent")
        assert found == ["57", "98"]

    def test_units_are_required(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.md"
        f.write_text("Trained for 30 epochs at batch 128, reaching 4,706 img/s.")
        kinds = {c.kind for c in extract(f)}
        # 30 and 128 carry no unit and are configuration, not measurements.
        assert kinds == {"throughput"}
        assert [c.text for c in extract(f)] == ["4,706"]


class TestGateEndToEnd:
    def _artifact(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        (d / "r.json").write_text(json.dumps({
            "finished": True, "hardware": "2x Tesla T4", "world_size": 1,
            "median_images_per_sec": 1853.0, "best_test_acc": 0.807,
            "peak_memory_mb": 553.0, "model_params": 1_806_538,
        }))

    def test_a_true_claim_passes(self, tmp_path: Path) -> None:
        self._artifact(tmp_path / "runs")
        doc = tmp_path / "ok.md"
        doc.write_text("Reached 1,853 img/s at 553 MB.")
        assert main(["--runs", str(tmp_path / "runs"), "--sources", str(doc)]) == 0

    def test_a_fabricated_claim_fails_the_gate(self, tmp_path: Path) -> None:
        self._artifact(tmp_path / "runs")
        doc = tmp_path / "bad.md"
        doc.write_text("Reached 9,999 img/s.")
        assert main(["--runs", str(tmp_path / "runs"), "--sources", str(doc)]) == 1
