"""並列チャンク処理(`_resolve_parallel_workers` と並列パス) のテスト。

実音声を使った full e2e は /Volumes/Extreme SSD/ 配下の本番ファイルでのみ
検証可能なので CI には乗せない。代わりに以下を検証する:

  1) `_resolve_parallel_workers` のワーカー数決定ロジック
  2) `_load_and_analyze_chunk` がトップレベル関数として import 可能
     (= ProcessPoolExecutor の spawn 起動から呼べる picklable 形になっている)
  3) 短い合成 WAV を `analyze_audio_file` に通したとき、逐次/並列の両方で
     同じ frame 数・同じ time_sec シーケンスが返ること
"""

from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

from audio_analyzer_mcp import analyzer as analyzer_module
from audio_analyzer_mcp.analyzer import (
    _load_and_analyze_chunk,
    _resolve_parallel_workers,
    analyze_audio_file,
)


def _make_test_wav(path: Path, *, duration_sec: int, sample_rate: int = 8000) -> None:
    """正弦波 + ノイズを混ぜた合成 WAV を duration_sec 秒ぶん書き出す。

    librosa が解析を成立させるため、完全な無音ではなく最低限の信号を入れる。
    """
    n = duration_sec * sample_rate
    t = np.arange(n) / sample_rate
    # 220Hz の正弦波 + 振幅ゆらぎ。pyin がピッチを取れる程度の信号強度にする。
    signal = (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    pcm = (signal * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


class ResolveParallelWorkersTests(unittest.TestCase):
    """`_resolve_parallel_workers` の分岐検証。"""

    def test_explicit_one_forces_serial(self) -> None:
        with patch.object(analyzer_module, "PARALLEL_WORKERS", 1):
            self.assertEqual(_resolve_parallel_workers(10), 1)

    def test_explicit_value_capped_by_chunks(self) -> None:
        # 4 ワーカー指定でもチャンクが 2 個しかなければ 2 に切り詰める。
        with patch.object(analyzer_module, "PARALLEL_WORKERS", 4):
            self.assertEqual(_resolve_parallel_workers(2), 2)

    def test_auto_uses_cpu_count_capped(self) -> None:
        with patch.object(analyzer_module, "PARALLEL_WORKERS", 0), \
             patch.object(analyzer_module, "PARALLEL_WORKERS_CAP", 4), \
             patch.object(analyzer_module.os, "cpu_count", return_value=16):
            # cpu_count=16 でも CAP=4 で頭打ち。チャンク 10 個なので 4。
            self.assertEqual(_resolve_parallel_workers(10), 4)

    def test_auto_handles_unknown_cpu_count(self) -> None:
        # os.cpu_count() が None を返すレアケース。1 を採用してクラッシュしない。
        with patch.object(analyzer_module, "PARALLEL_WORKERS", 0), \
             patch.object(analyzer_module.os, "cpu_count", return_value=None):
            self.assertEqual(_resolve_parallel_workers(10), 1)


class LoadAndAnalyzeChunkPicklableTests(unittest.TestCase):
    """worker 関数が ProcessPoolExecutor から呼べる形になっていることの確認。"""

    def test_function_is_module_level_importable(self) -> None:
        # spawn 起動の worker は __main__ ではなく定義モジュールから関数を import する。
        # トップレベル + __qualname__ にネストがないことを確認する。
        self.assertEqual(_load_and_analyze_chunk.__module__, "audio_analyzer_mcp.analyzer")
        self.assertEqual(_load_and_analyze_chunk.__qualname__, "_load_and_analyze_chunk")


class ParallelEquivalenceTests(unittest.TestCase):
    """逐次/並列モードで同じ入力に対して同じ shape の frames を返すこと。

    実値のbit一致までは保証しない(チャンク境界のオーバーラップ統計値が
    並列化前後で完全一致するのは設計上の保証ではあるが、テストとしては
    frame 数 + time_sec 列の一致まで見れば回帰検出としては十分)。
    """

    def setUp(self) -> None:
        # `AUTO_CHUNK_THRESHOLD_SEC` をテスト用に小さくして、短い WAV でも
        # チャンク分割パスを踏むようにする(本番値 1200s だと 20分の WAV が必要になる)。
        # 同様に CHUNK_DURATION_SEC / CHUNK_OVERLAP_SEC も縮める。
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.wav_path = Path(self.tmp.name) / "synthetic.wav"
        _make_test_wav(self.wav_path, duration_sec=30)

        self._patches = [
            patch.object(analyzer_module, "AUTO_CHUNK_THRESHOLD_SEC", 10),
            patch.object(analyzer_module, "CHUNK_DURATION_SEC", 8),
            patch.object(analyzer_module, "CHUNK_OVERLAP_SEC", 2),
        ]
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)

    def _run_with_workers(self, workers: int) -> list:
        with patch.object(analyzer_module, "PARALLEL_WORKERS", workers):
            return analyze_audio_file(str(self.wav_path))

    def test_serial_and_parallel_produce_same_frames(self) -> None:
        serial = self._run_with_workers(1)
        parallel = self._run_with_workers(3)

        self.assertGreater(len(serial), 0)
        self.assertEqual(len(serial), len(parallel))
        self.assertEqual(
            [f.time_sec for f in serial],
            [f.time_sec for f in parallel],
        )
        # rms_db / pitch_hz は完全一致を期待してよい (同じ波形 → 同じ結果)。
        for s, p in zip(serial, parallel):
            self.assertAlmostEqual(s.rms_db, p.rms_db, places=3)
            self.assertAlmostEqual(s.pitch_hz, p.pitch_hz, places=3)
            self.assertEqual(s.is_speech, p.is_speech)


if __name__ == "__main__":
    unittest.main()
