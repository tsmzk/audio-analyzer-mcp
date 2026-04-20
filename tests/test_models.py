"""Input-validation tests for the Pydantic models.

Focuses on the new DetectHighlightsLocalInput (required-field semantics +
range/cross-field validation). Also re-verifies the existing DetectHighlightsInput
behavior so accidental regressions surface quickly.
"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from audio_analyzer_mcp.models import (
    DetectHighlightsInput,
    DetectHighlightsLocalInput,
)


class DetectHighlightsLocalInputTests(unittest.TestCase):
    """`detect_highlights_local` の入力バリデーション。"""

    def _base_kwargs(self) -> dict:
        return {
            "file_path": "/tmp/any.mp4",
            "top_n": 20,
            "min_gap_sec": 10,
        }

    def test_valid_minimal_input(self) -> None:
        m = DetectHighlightsLocalInput(**self._base_kwargs())
        self.assertEqual(m.file_path, "/tmp/any.mp4")
        self.assertEqual(m.top_n, 20)
        self.assertEqual(m.min_gap_sec, 10)
        # optional fields default to None / DEFAULTs.
        self.assertIsNone(m.start_sec)
        self.assertIsNone(m.end_sec)

    def test_file_path_is_required(self) -> None:
        kwargs = self._base_kwargs()
        del kwargs["file_path"]
        with self.assertRaises(ValidationError) as cm:
            DetectHighlightsLocalInput(**kwargs)
        self.assertIn("file_path", str(cm.exception))

    def test_top_n_is_required(self) -> None:
        kwargs = self._base_kwargs()
        del kwargs["top_n"]
        with self.assertRaises(ValidationError) as cm:
            DetectHighlightsLocalInput(**kwargs)
        self.assertIn("top_n", str(cm.exception))

    def test_min_gap_sec_is_required(self) -> None:
        kwargs = self._base_kwargs()
        del kwargs["min_gap_sec"]
        with self.assertRaises(ValidationError) as cm:
            DetectHighlightsLocalInput(**kwargs)
        self.assertIn("min_gap_sec", str(cm.exception))

    def test_top_n_out_of_range(self) -> None:
        for bad in (0, 101):
            kwargs = self._base_kwargs()
            kwargs["top_n"] = bad
            with self.assertRaises(ValidationError):
                DetectHighlightsLocalInput(**kwargs)

    def test_min_gap_sec_out_of_range(self) -> None:
        for bad in (0, 301):
            kwargs = self._base_kwargs()
            kwargs["min_gap_sec"] = bad
            with self.assertRaises(ValidationError):
                DetectHighlightsLocalInput(**kwargs)

    def test_file_path_length_bounds(self) -> None:
        # Empty string is rejected by min_length=1.
        kwargs = self._base_kwargs()
        kwargs["file_path"] = ""
        with self.assertRaises(ValidationError):
            DetectHighlightsLocalInput(**kwargs)

        # Just over max_length is rejected.
        kwargs["file_path"] = "/" + ("a" * 1000)
        with self.assertRaises(ValidationError):
            DetectHighlightsLocalInput(**kwargs)

    def test_frame_length_must_be_ge_hop_length(self) -> None:
        kwargs = self._base_kwargs()
        kwargs["hop_length"] = 2048
        kwargs["frame_length"] = 1024  # < hop_length
        with self.assertRaises(ValidationError) as cm:
            DetectHighlightsLocalInput(**kwargs)
        self.assertIn("frame_length", str(cm.exception))

    def test_end_sec_must_exceed_start_sec(self) -> None:
        kwargs = self._base_kwargs()
        kwargs["start_sec"] = 60.0
        kwargs["end_sec"] = 60.0  # not strictly greater
        with self.assertRaises(ValidationError):
            DetectHighlightsLocalInput(**kwargs)

    def test_extra_fields_forbidden(self) -> None:
        kwargs = self._base_kwargs()
        kwargs["unknown_field"] = "oops"
        with self.assertRaises(ValidationError):
            DetectHighlightsLocalInput(**kwargs)

    def test_optional_range_accepted(self) -> None:
        kwargs = self._base_kwargs()
        kwargs["start_sec"] = 0.0
        kwargs["end_sec"] = 120.5
        m = DetectHighlightsLocalInput(**kwargs)
        self.assertEqual(m.start_sec, 0.0)
        self.assertEqual(m.end_sec, 120.5)


class DetectHighlightsInputRegressionTests(unittest.TestCase):
    """既存 YouTube 版の挙動が壊れていないことの確認。"""

    def test_defaults_still_apply(self) -> None:
        m = DetectHighlightsInput(youtube_url="https://youtu.be/abcdefghijk")
        self.assertEqual(m.top_n, 20)
        self.assertEqual(m.min_gap_sec, 10)


if __name__ == "__main__":
    unittest.main()
