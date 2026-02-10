# MioTTS-Inference

[![Hugging Face Collection](https://img.shields.io/badge/Collection-HuggingFace-yellow)](https://huggingface.co/collections/Aratako/miotts)
[![Demo](https://img.shields.io/badge/Demo-HuggingFace%20Space-blue)](https://huggingface.co/spaces/Aratako/MioTTS-0.1B-Demo)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](LICENSE)

## 概要

軽量・高速なTTSモデル [MioTTS](https://huggingface.co/collections/Aratako/miotts) の推論用コードです。

主な特徴:
- 一般的なLLM推論フレームワーク（llama.cpp、Ollama、vLLMなど）を使用可能
- REST API経由での音声合成
- 参照音声のプリセット登録機能
- Best-of-N による高品質な音声選択

## モデル一覧

| モデル名 | パラメータ | ライセンス |
|---|---|---|
| [MioTTS-0.1B](https://huggingface.co/Aratako/MioTTS-0.1B) | 0.1B | [Falcon-LLM License](https://falconllm.tii.ae/falcon-terms-and-conditions.html) |
| [MioTTS-0.4B](https://huggingface.co/Aratako/MioTTS-0.4B) | 0.4B | [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2-350M/blob/main/LICENSE) |
| [MioTTS-0.6B](https://huggingface.co/Aratako/MioTTS-0.6B) | 0.6B | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/) |
| [MioTTS-1.2B](https://huggingface.co/Aratako/MioTTS-1.2B) | 1.2B | [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Base/blob/main/LICENSE) |
| [MioTTS-1.7B](https://huggingface.co/Aratako/MioTTS-1.7B) | 1.7B | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/) |
| [MioTTS-2.6B](https://huggingface.co/Aratako/MioTTS-2.6B) | 2.6B | [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2-2.6B/blob/main/LICENSE) |

量子化配布モデル:

| モデル | 用途 |
|---|---|
| [MioTTS-GGUF](https://huggingface.co/Aratako/MioTTS-GGUF) | llama.cpp / Ollama 向け量子化モデル |

## 環境構築

```bash
git clone https://github.com/Aratako/MioTTS-Inference.git
cd MioTTS-Inference
uv sync
# flash-attentionのインストール（推奨）
# MAX_JOBSの値はCPUスペックによって変更してください
MAX_JOBS=8 uv pip install --no-build-isolation -v flash-attn
```

## 使い方

### 1. TTSモデル推論サーバの起動

TTSモデルの推論サーバを起動します。モデルのアーキテクチャは通常のLLMと完全に同一であるため、一般的なLLM推論フレームワークで起動できます。OpenAI Compatible APIが立ち上がるようにしてください。

#### llama.cpp の場合

公式の [Quick Start](https://github.com/ggml-org/llama.cpp?tab=readme-ov-file#quick-start) に従い、llama.cppをインストールしてください。その後、以下のように推論サーバを起動します。使いたいモデルに応じて `-hff` の指定を適宜変更してください。

```bash
llama-server -hf Aratako/MioTTS-GGUF -hff MioTTS-1.2B-BF16.gguf -c 8192 --cont-batching --batch_size 8 --port 8000
```

#### Ollama の場合

公式の [Download](https://ollama.com/download) に従い、Ollamaをインストールしてください。その後、以下のように推論サーバを起動します。使いたいモデルに応じてモデル名を適宜変更してください。

```bash
# CLIを使う場合
OLLAMA_HOST=localhost:8000 ollama serve
# 別ウィンドウで
OLLAMA_HOST=localhost:8000 ollama run hf.co/Aratako/MioTTS-GGUF:MioTTS-1.2B-BF16.gguf
```

#### vLLM の場合

公式の [Installation](https://docs.vllm.ai/en/latest/getting_started/installation/) に従い、vLLMをインストールしてください。その後、以下のように推論サーバを起動します。使いたいモデルに応じてモデル名を適宜変更してください。また、GPUスペックに応じて `--gpu-memory-utilization` を調整してください。

```bash
vllm serve Aratako/MioTTS-1.2B --max-model-len 1024 --gpu-memory-utilization 0.2
```

LMStudioやSGLangなどその他の推論フレームワークでも、OpenAI Compatible APIを立てることができれば動作します。

### 2. 音声合成APIの起動

本リポジトリで提供している音声合成のAPIサーバを起動します。ポート番号は手順1で立てたサーバに合わせてください（例えば、Ollamaはデフォルトではポート11434を使用します）。

```bash
python run_server.py --llm-base-url http://localhost:8000/v1
```

`--best-of-n-enabled` を付けて起動すると、Best-of-N による音声合成が可能になります。この設定では、1つの入力テキストに対して指定したN個の候補を同時に生成し、ASR（Whisper）によるエラー率などのヒューリスティックな評価で最も良いと判断された音声を返します。

```bash
python run_server.py --llm-base-url http://localhost:8000/v1 --best-of-n-enabled
```

### 3. WebUIの起動

上記の手順で起動した音声合成APIを使った簡易的なWebUIのデモを用意しています。

```bash
python run_gradio.py
```

実行後、`http://localhost:7860` でWebUIにアクセスできます。

## 環境変数 / CLI引数

### run_server.py（音声合成APIサーバ）

環境変数やCLI引数で以下の設定を変更できます。CLI引数が優先されます。

#### サーバ設定

| 引数 | 環境変数 | デフォルト | 説明 |
|------|---------|-----------|------|
| `--host` | `MIOTTS_HOST` | `0.0.0.0` | サーバホスト |
| `--port` | `MIOTTS_PORT` | `8001` | サーバポート |
| `--reload` | `MIOTTS_RELOAD` | `false` | ホットリロードの有効化 |
| `--log-level` | `MIOTTS_LOG_LEVEL` | `info` | ログレベル |

#### LLM設定

| 引数 | 環境変数 | デフォルト | 説明 |
|------|---------|-----------|------|
| `--llm-base-url` | `MIOTTS_LLM_BASE_URL` | `http://localhost:8000/v1` | LLM APIのベースURL |
| `--llm-api-key` | `MIOTTS_LLM_API_KEY` | なし | LLM APIキー（必要な場合） |
| `--llm-model` | `MIOTTS_LLM_MODEL` | 自動取得 | LLMのモデル名 |
| `--llm-timeout` | `MIOTTS_LLM_TIMEOUT` | `120.0` | LLMリクエストのタイムアウト（秒） |

#### サンプリングパラメータ

| 環境変数 | デフォルト | 説明 |
|---------|-----------|------|
| `MIOTTS_LLM_TEMPERATURE` | `0.8` | Temperature |
| `MIOTTS_LLM_TOP_P` | `1.0` | Top-P |
| `MIOTTS_LLM_MAX_TOKENS` | `700` | 最大生成トークン数 |
| `MIOTTS_LLM_REPETITION_PENALTY` | `1.0` | Repetition Penalty（1.0-1.5） |
| `MIOTTS_LLM_PRESENCE_PENALTY` | `0.0` | Presence Penalty（0.0-1.0） |
| `MIOTTS_LLM_FREQUENCY_PENALTY` | `0.0` | Frequency Penalty（0.0-1.0） |

#### コーデック設定

| 引数 | 環境変数 | デフォルト | 説明 |
|------|---------|-----------|------|
| `--codec-model` | `MIOTTS_CODEC_MODEL` | `Aratako/MioCodec-25Hz-24kHz` | MioCodecのモデル名 |
| `--device` | `MIOTTS_DEVICE` | `cuda`（なければ`cpu`） | コーデックの推論デバイス |

#### プリセット設定

| 引数 | 環境変数 | デフォルト | 説明 |
|------|---------|-----------|------|
| `--presets-dir` | `MIOTTS_PRESETS_DIR` | `presets` | プリセットのディレクトリ |

#### Best-of-N設定

| 引数 | 環境変数 | デフォルト | 説明 |
|------|---------|-----------|------|
| `--best-of-n-enabled` | `MIOTTS_BEST_OF_N_ENABLED` | `false` | Best-of-Nの有効化 |
| `--best-of-n-default` | `MIOTTS_BEST_OF_N_DEFAULT` | `1` | デフォルトのN（1=通常の生成） |
| `--best-of-n-max` | `MIOTTS_BEST_OF_N_MAX` | `8` | Nの最大値 |
| `--best-of-n-language` | `MIOTTS_BEST_OF_N_LANGUAGE` | `auto` | Best-of-N用の言語設定（`auto`/`ja`/`en`） |

#### ASR設定（Best-of-N用）

| 引数 | 環境変数 | デフォルト | 説明 |
|------|---------|-----------|------|
| `--asr-model` | `MIOTTS_ASR_MODEL` | `openai/whisper-large-v3-turbo` | ASRモデル |
| `--asr-device` | `MIOTTS_ASR_DEVICE` | `MIOTTS_DEVICE`と同じ | ASRの推論デバイス |
| `--asr-compute-type` | `MIOTTS_ASR_COMPUTE_TYPE` | `float16`（cuda）/ `int8`（cpu） | ASRの計算精度 |
| `--asr-batch-size` | `MIOTTS_ASR_BATCH_SIZE` | `0`（全並列） | ASRのバッチサイズ |
| `--asr-language` | `MIOTTS_ASR_LANGUAGE` | `auto` | ASR言語 |

#### その他設定

| 引数 | 環境変数 | デフォルト | 説明 |
|------|---------|-----------|------|
| `--max-text-length` | `MIOTTS_MAX_TEXT_LENGTH` | `300` | 入力テキストの最大文字数 |
| `--max-reference-mb` | `MIOTTS_MAX_REFERENCE_MB` | `20` | 参照音声の最大サイズ（MB） |
| `--allowed-audio-exts` | `MIOTTS_ALLOWED_AUDIO_EXTS` | `.wav,.flac,.ogg` | 許可する音声拡張子 |

参照音声の最大長は20秒に固定されています。

### run_gradio.py（WebUI）

| 環境変数 | デフォルト | 説明 |
|---------|-----------|------|
| `MIOTTS_API_BASE` | `http://localhost:8001` | 音声合成APIサーバのベースURL |

WebUI上の「Advanced Settings」からもAPI Base URLを変更できます。

## 参照音声のプリセットについて

参照音声を毎回入力する代わりに、事前にコーデックでエンコードした結果をプリセットとして登録して使いまわすことができます。

```bash
python scripts/generate_preset.py --audio /path/to/audio.wav --preset-id preset_name
```

### generate_preset.py の引数

| 引数 | 必須 | デフォルト | 説明 |
|------|------|-----------|------|
| `--audio` | はい | - | 参照音声ファイルのパス |
| `--preset-id` | はい | - | プリセットID（ファイル名になります） |
| `--output-dir` | いいえ | `presets` | 出力ディレクトリ |
| `--model-id` | いいえ | `Aratako/MioCodec-25Hz-24kHz` | MioCodecのモデル名 |
| `--device` | いいえ | `cuda` | 推論デバイス |

### デフォルトプリセット

以下のプリセットが同梱されています:

- `jp_female` - 日本語・女性音声
- `jp_male` - 日本語・男性音声
- `en_female` - 英語・女性音声
- `en_male` - 英語・男性音声

## API仕様

### ヘルスチェック

```
GET /health
```

**レスポンス:**
```json
{"status": "ok"}
```

### プリセット一覧取得

```
GET /v1/presets
```

**レスポンス:**
```json
{"presets": ["en_female", "en_male", "jp_female", "jp_male"]}
```

### 音声合成（JSONリクエスト）

```
POST /v1/tts
Content-Type: application/json
```

**リクエストボディ:**
```json
{
  "text": "合成するテキスト",
  "reference": {
    "type": "preset",
    "preset_id": "jp_female"
  },
  "llm": {
    "temperature": 0.8,
    "top_p": 1.0,
    "max_tokens": 700,
    "repetition_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0
  },
  "output": {
    "format": "base64"
  },
  "best_of_n": {
    "enabled": false,
    "n": 1,
    "language": "auto"
  }
}
```

`reference` は必須です。
テキスト前処理は、入力が日本語なら正規化、それ以外は `strip()` のみを適用します。

| フィールド | 型 | 必須 | 説明 |
|-----------|---|------|------|
| `text` | string | はい | 合成するテキスト |
| `reference.type` | string | はい | `preset` または `base64` |
| `reference.preset_id` | string | 条件付き | `type=preset` の場合に必須 |
| `reference.data` | string | 条件付き | `type=base64` の場合に必須 |
| `llm.*` | - | いいえ | LLMパラメータ |
| `output.format` | string | いいえ | `wav` または `base64`（デフォルト: base64） |
| `best_of_n.*` | - | いいえ | Best-of-N設定 |

**レスポンス:**
```json
{
  "audio": "Base64エンコードされたWAVデータ",
  "format": "base64",
  "sample_rate": 24000,
  "token_count": 123,
  "timings": {
    "llm_sec": 0.5,
    "parse_sec": 0.01,
    "codec_sec": 0.2,
    "total_sec": 0.71,
    "best_of_n_sec": null,
    "asr_sec": null
  },
  "normalized_text": "前処理後のテキスト"
}
```

### 音声合成（ファイルアップロード）

```
POST /v1/tts/file
Content-Type: multipart/form-data
```

**フォームフィールド:**

`reference_audio` または `reference_preset_id` のどちらかが必須です。

| フィールド | 型 | 必須 | 説明 |
|-----------|---|------|------|
| `text` | string | はい | 合成するテキスト |
| `reference_audio` | file | 条件付き | `reference_preset_id` 未指定時は必須 |
| `reference_preset_id` | string | 条件付き | `reference_audio` 未指定時は必須 |
| `model` | string | いいえ | LMのモデル名 |
| `temperature` | float | いいえ | Temperature |
| `top_p` | float | いいえ | Top-Pサ |
| `max_tokens` | int | いいえ | 最大生成トークン数 |
| `repetition_penalty` | float | いいえ | Repetition Penalty |
| `presence_penalty` | float | いいえ | Presence Penalty |
| `frequency_penalty` | float | いいえ | Frequency Penalty |
| `output_format` | string | いいえ | `wav` または `base64` |
| `best_of_n_enabled` | boolean | いいえ | Best-of-Nの有効化 |
| `best_of_n_n` | int | いいえ | Nの値 |
| `best_of_n_language` | string | いいえ | Best-of-N用の言語設定 |

**レスポンス:**
- `output_format=wav`: WAVファイル（`audio/wav`）
- `output_format=base64`: JSONレスポンス

## ライセンス・クレジット

- **コード**: MIT License
- **デフォルトプリセット**: `presets`以下のデフォルトプリセットは [T5Gemma-TTS](https://huggingface.co/Aratako/T5Gemma-TTS-2b-2b) と [gemini-2.5-pro-tts](https://cloud.google.com/text-to-speech/docs/gemini-tts) の生成音声を利用しているため、これを用いて合成した音声は商用利用出来ません。
- **モデル**: 各モデルのライセンスに従ってください。
