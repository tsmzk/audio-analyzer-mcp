"""Unit tests for the highlight ranking + formatting helpers.

These are pure-Python tests that avoid librosa / network / file I/O, so they
run fast (<1s) and don't depend on external tools.
"""

from __future__ import annotations

import unittest

from audio_analyzer_mcp.constants import AudioFrame
from audio_analyzer_mcp.highlights import (
    _detect_highlight_moments,
    _format_hms,
    build_highlight_summary,
)


def _make_frame(
    time_sec: int,
    *,
    rms_db: float = -30.0,
    rms_norm: float = 50.0,
    pitch_hz: float = 150.0,
    spectral_centroid: float = 1500.0,
    is_speech: bool = True,
    volume_spike: bool = False,
) -> AudioFrame:
    """小さなヘルパー。テストしたい属性だけ指定できるようにしてある。"""
    return AudioFrame(
        time_sec=time_sec,
        rms_db=rms_db,
        rms_norm=rms_norm,
        pitch_hz=pitch_hz,
        spectral_centroid=spectral_centroid,
        is_speech=is_speech,
        volume_spike=volume_spike,
    )


class FormatHmsTests(unittest.TestCase):
    """`_format_hms` の基本境界テスト。"""

    def test_under_one_minute(self) -> None:
        self.assertEqual(_format_hms(0), "0:00")
        self.assertEqual(_format_hms(5), "0:05")
        self.assertEqual(_format_hms(59), "0:59")

    def test_under_one_hour(self) -> None:
        self.assertEqual(_format_hms(60), "1:00")
        self.assertEqual(_format_hms(306), "5:06")
        self.assertEqual(_format_hms(3599), "59:59")

    def test_with_hours(self) -> None:
        self.assertEqual(_format_hms(3600), "1:00:00")
        self.assertEqual(_format_hms(5636), "1:33:56")
        self.assertEqual(_format_hms(10 * 3600 + 2 * 60 + 3), "10:02:03")

    def test_negative_is_clamped_to_zero(self) -> None:
        # 負値は想定外だが、落ちずに "0:00" を返すのが望ましい。
        self.assertEqual(_format_hms(-10), "0:00")


class DetectHighlightMomentsTests(unittest.TestCase):
    """スコアリング / 間引き / 出力キーのテスト。"""

    def test_returns_empty_when_no_speech(self) -> None:
        frames = [_make_frame(sec, is_speech=False) for sec in range(30)]
        self.assertEqual(_detect_highlight_moments(frames, top_n=5, min_gap_sec=1), [])

    def test_output_shape_includes_rank_time_hms_spectral_centroid(self) -> None:
        # 音量が徐々に上がって最後だけ叫ぶような構成。
        frames = [_make_frame(sec, rms_db=-40.0) for sec in range(20)]
        frames.append(
            _make_frame(
                20,
                rms_db=-5.0,
                pitch_hz=400.0,
                spectral_centroid=3500.0,
                volume_spike=True,
            )
        )

        result = _detect_highlight_moments(frames, top_n=3, min_gap_sec=1)
        self.assertTrue(result, "expected at least one highlight")

        expected_keys = {
            "rank",
            "timestamp",
            "time_sec",
            "time_hms",
            "score",
            "rms_db",
            "rms_norm",
            "pitch_hz",
            "spectral_centroid",
            "reasons",
        }
        for item in result:
            self.assertTrue(
                expected_keys.issubset(item.keys()),
                f"missing keys: {expected_keys - item.keys()}",
            )

    def test_rank_is_1_indexed_and_contiguous(self) -> None:
        frames = [
            _make_frame(sec, rms_db=-40.0 + sec, volume_spike=(sec % 5 == 0))
            for sec in range(40)
        ]
        result = _detect_highlight_moments(frames, top_n=5, min_gap_sec=1)
        ranks = [item["rank"] for item in result]
        self.assertEqual(ranks, list(range(1, len(result) + 1)))

    def test_min_gap_sec_drops_adjacent_candidates(self) -> None:
        # 3連続フレーム全てが強力な highlight 候補。min_gap_sec=5 なら
        # 1つしか残らないはず。
        frames = []
        for sec in range(10):
            frames.append(
                _make_frame(
                    sec,
                    rms_db=-40.0,
                    pitch_hz=120.0,
                    spectral_centroid=1500.0,
                )
            )
        for sec in (10, 11, 12):
            frames.append(
                _make_frame(
                    sec,
                    rms_db=-5.0,
                    pitch_hz=400.0,
                    spectral_centroid=3500.0,
                    volume_spike=True,
                )
            )

        result = _detect_highlight_moments(frames, top_n=10, min_gap_sec=5)
        times = sorted(item["time_sec"] for item in result)
        # どのペアも min_gap_sec 以上離れている必要がある。
        for a, b in zip(times, times[1:]):
            self.assertGreaterEqual(
                b - a, 5, f"gap violation between {a} and {b}"
            )

    def test_top_n_caps_result_length(self) -> None:
        # min_gap_sec=1 で、全フレーム highlight になるような構成。
        frames = [
            _make_frame(
                sec,
                rms_db=-5.0,
                pitch_hz=400.0,
                spectral_centroid=3500.0,
                volume_spike=True,
            )
            for sec in range(0, 100, 3)
        ]
        result = _detect_highlight_moments(frames, top_n=4, min_gap_sec=1)
        self.assertEqual(len(result), 4)

    def test_sorted_by_score_descending(self) -> None:
        frames = [_make_frame(sec, rms_db=-40.0) for sec in range(30)]
        # 異なるスコアが出るように多様な profile を散りばめる。
        frames.append(_make_frame(30, rms_db=-5.0, volume_spike=True, pitch_hz=400.0,
                                   spectral_centroid=3500.0))
        frames.append(_make_frame(50, rms_db=-20.0))  # 弱めの候補
        frames.append(_make_frame(70, rms_db=-10.0, pitch_hz=380.0))

        result = _detect_highlight_moments(frames, top_n=5, min_gap_sec=1)
        scores = [item["score"] for item in result]
        self.assertEqual(scores, sorted(scores, reverse=True))


class BuildHighlightSummaryTests(unittest.TestCase):
    """共通サマリー helper が期待通りの辞書を返すか。"""

    def test_summary_keys_and_counts(self) -> None:
        frames = [
            _make_frame(sec, is_speech=(sec % 2 == 0), volume_spike=(sec == 10))
            for sec in range(20)
        ]
        summary = build_highlight_summary(frames, top_n=3, min_gap_sec=1)
        self.assertEqual(summary["total_duration_sec"], 20)
        self.assertEqual(summary["speech_seconds"], 10)
        self.assertEqual(summary["silence_seconds"], 10)
        self.assertEqual(summary["total_volume_spikes"], 1)
        self.assertIn("highlights", summary)
        self.assertIsInstance(summary["highlights"], list)


if __name__ == "__main__":
    unittest.main()
