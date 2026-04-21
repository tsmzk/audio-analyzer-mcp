"""`analyze_local` の分岐ロジックに対する単体テスト。

実 ffmpeg / librosa を呼ぶと遅く環境依存になるので、
- `transcode_to_wav`  → モック
- `analyze_audio_file` → モック
して「どちらがどの引数で呼ばれるか」だけを検証する。

ここで検証したいのは以下 3 点:
  1) 存在しない file_path で AnalysisError が上がる
  2) .wav 入力では transcode_to_wav を呼ばずに analyze_audio_file に直接渡す
  3) .mp4 入力では transcode_to_wav が呼ばれ、その戻り値が analyze_audio_file に渡る
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from audio_analyzer_mcp import analyzer as analyzer_module
from audio_analyzer_mcp.constants import (
    AnalysisError,
    FRAME_LENGTH,
    HOP_LENGTH,
    SAMPLE_RATE,
)


class AnalyzeLocalBranchTests(unittest.TestCase):
    """analyze_local() の前処理分岐(拡張子判定・存在チェック)のテスト。"""

    def test_missing_file_raises_analysis_error(self) -> None:
        # 実在しないパスを渡せば、ffmpeg / librosa を呼ぶ前に AnalysisError が出るはず。
        missing = "/tmp/does-not-exist-ever/__nope__.mp4"
        with self.assertRaises(AnalysisError) as cm:
            analyzer_module.analyze_local(missing)
        self.assertIn("File not found", str(cm.exception))

    def test_wav_input_skips_transcode(self) -> None:
        """.wav を渡した場合 transcode_to_wav は呼ばれず analyze_audio_file に素通し。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "sample.wav"
            wav_path.touch()  # 空ファイルでよい(実際の読み込みはモックしている)

            fake_frames: list = []
            with patch.object(
                analyzer_module,
                "analyze_audio_file",
                return_value=fake_frames,
            ) as mock_analyze, patch.object(
                analyzer_module,
                "transcode_to_wav",
            ) as mock_transcode:
                result = analyzer_module.analyze_local(str(wav_path))

            self.assertIs(result, fake_frames)
            mock_transcode.assert_not_called()
            mock_analyze.assert_called_once_with(
                str(wav_path),
                sample_rate=SAMPLE_RATE,
                hop_length=HOP_LENGTH,
                frame_length=FRAME_LENGTH,
                start_sec=None,
                end_sec=None,
                progress_cb=None,
            )

    def test_mp4_input_transcodes_then_analyzes(self) -> None:
        """.mp4 を渡すと transcode_to_wav → その戻り値で analyze_audio_file。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mp4_path = Path(tmpdir) / "movie.mp4"
            mp4_path.touch()

            fake_wav = "/some/tmpdir/movie.wav"
            fake_frames: list = []
            # progress_cb が素通しされることも確認するためにダミーを 1 つ用意。
            dummy_cb = lambda msg, frac: None  # noqa: E731

            # transcode_to_wav が呼ばれた時点で tmpdir がまだ存在していることを
            # side_effect の中で検証する(analyze_local が return した後は
            # TemporaryDirectory の __exit__ で削除されるため、call_args を
            # 事後に見るだけだと確認できない)。
            captured_tmpdir: list[str] = []

            def _fake_transcode(input_path, output_dir, **kwargs):
                captured_tmpdir.append(output_dir)
                # 呼び出された瞬間は with ブロック内なので存在するはず。
                assert Path(output_dir).exists(), (
                    f"TemporaryDirectory should exist during transcode call: {output_dir}"
                )
                return fake_wav

            with patch.object(
                analyzer_module,
                "transcode_to_wav",
                side_effect=_fake_transcode,
            ) as mock_transcode, patch.object(
                analyzer_module,
                "analyze_audio_file",
                return_value=fake_frames,
            ) as mock_analyze:
                result = analyzer_module.analyze_local(
                    str(mp4_path),
                    sample_rate=16000,
                    hop_length=1024,
                    frame_length=2048,
                    start_sec=10.0,
                    end_sec=30.0,
                    progress_cb=dummy_cb,
                )

            self.assertIs(result, fake_frames)

            # transcode_to_wav は (mp4_path, <tmpdir>, sample_rate=..., progress_cb=...) で
            # 1 回だけ呼ばれる。
            mock_transcode.assert_called_once()
            transcode_args, transcode_kwargs = mock_transcode.call_args
            self.assertEqual(transcode_args[0], str(mp4_path))
            # 第 2 引数(tmpdir)は side_effect 側で存在確認済み。
            self.assertEqual(transcode_args[1], captured_tmpdir[0])
            self.assertEqual(transcode_kwargs["sample_rate"], 16000)
            self.assertIs(transcode_kwargs["progress_cb"], dummy_cb)

            # analyze_audio_file には transcode の戻り値がそのまま渡る。
            mock_analyze.assert_called_once_with(
                fake_wav,
                sample_rate=16000,
                hop_length=1024,
                frame_length=2048,
                start_sec=10.0,
                end_sec=30.0,
                progress_cb=dummy_cb,
            )

    def test_non_wav_suffix_case_insensitive(self) -> None:
        """拡張子判定が大文字小文字を区別しないこと(".MP4" でも transcode される)。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mp4_path = Path(tmpdir) / "MOVIE.MP4"
            mp4_path.touch()

            with patch.object(
                analyzer_module,
                "transcode_to_wav",
                return_value="/fake.wav",
            ) as mock_transcode, patch.object(
                analyzer_module,
                "analyze_audio_file",
                return_value=[],
            ):
                analyzer_module.analyze_local(str(mp4_path))

            mock_transcode.assert_called_once()

    def test_wav_suffix_case_insensitive(self) -> None:
        """"".WAV" でも transcode されずに直接 analyze_audio_file に渡る。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "SAMPLE.WAV"
            wav_path.touch()

            with patch.object(
                analyzer_module,
                "analyze_audio_file",
                return_value=[],
            ) as mock_analyze, patch.object(
                analyzer_module,
                "transcode_to_wav",
            ) as mock_transcode:
                analyzer_module.analyze_local(str(wav_path))

            mock_transcode.assert_not_called()
            # 第 1 引数は元のパスがそのまま渡る(変換スキップ)。
            self.assertEqual(mock_analyze.call_args.args[0], str(wav_path))


if __name__ == "__main__":
    unittest.main()
