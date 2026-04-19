# audio-analyzer-mcp

YouTube動画やローカルファイルの音声を分析し、1秒ごとの声量(dB)・ピッチ(Hz)・発話検出をCSV/JSONで返すMCPサーバー。

ゲーム実況のショート動画素材選定など、「どこで声が跳ねたか」を数値で特定する用途に最適。

## ツール一覧

| ツール名 | 説明 |
|---------|------|
| `analyze_youtube_audio` | YouTube URLから音声をDL→1秒ごとの分析データ（CSV/JSON） |
| `analyze_local_audio` | ローカルファイル（mp4, wav等）を分析 |
| `detect_highlights` | YouTube URLから音声のハイライト（声量スパイク、興奮ポイント）を自動検出 |

## 出力データの列

| 列名 | 説明 |
|------|------|
| `timestamp` | MM:SS形式のタイムスタンプ |
| `time_sec` | 秒数（字幕データとの突き合わせ用） |
| `rms_db` | 声量（dB）。-60以下≒無音、-20以上≒叫び |
| `rms_norm` | 声量を0-100に正規化 |
| `pitch_hz` | 声の高さ（Hz）。男性通常100-150Hz、興奮時200Hz超 |
| `spectral_centroid` | 声の鋭さ（Hz）。ツッコミ等で上昇 |
| `is_speech` | 発話中か（true/false） |
| `volume_spike` | 直前3秒と比べて急激に声量上昇（true/false） |

## セットアップ

### 前提条件

- Python 3.10以上
- ffmpeg（`brew install ffmpeg` or `apt install ffmpeg`）
- uv（推奨）または pip

### インストール

```bash
cd audio-analyzer-mcp
uv sync
```

### Claude Desktop設定

`claude_desktop_config.json` に以下を追加：

```json
{
  "mcpServers": {
    "audio-analyzer": {
      "command": "uv",
      "args": [
        "--directory", "/path/to/audio-analyzer-mcp",
        "run", "python", "-m", "audio_analyzer_mcp"
      ]
    }
  }
}
```

## 使い方の例

### ショート動画素材の選定フロー

1. `detect_highlights` でハイライト候補を自動検出
2. `youtube-transcript` MCPで字幕を取得
3. `time_sec` で突き合わせ → 「何を言った瞬間に声が跳ねたか」が分かる

### Claude への指示例

```
このYouTube動画のハイライトを検出して、字幕と突き合わせてショート候補を出して。
URL: https://www.youtube.com/watch?v=XXXXX
```

## 技術詳細

- **音声DL**: yt-dlp（音声のみ抽出、WAV変換）
- **分析**: librosa
  - RMS: `librosa.feature.rms` → `amplitude_to_db`
  - ピッチ: `librosa.pyin`（人声特化のF0推定アルゴリズム）
  - スペクトル重心: `librosa.feature.spectral_centroid`
- **ハイライト検出**: 声量スパイク + 上位パーセンタイル + ピッチ上昇の複合スコアリング

## 注意事項

- 1時間の動画の分析には数分かかります（主にピッチ推定が重い）
- yt-dlpとffmpegがPATHに必要です
- YouTube動画は公開動画のみ対応
