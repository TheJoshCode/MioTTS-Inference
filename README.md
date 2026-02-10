# MioTTS-Inference

[![Hugging Face Collection](https://img.shields.io/badge/Collection-HuggingFace-yellow)](https://huggingface.co/collections/Aratako/miotts)
[![Demo](https://img.shields.io/badge/Demo-HuggingFace%20Space-blue)](https://huggingface.co/spaces/Aratako/MioTTS-0.1B-Demo)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](LICENSE)

**[日本語版 README はこちら](README_ja.md)**

## Overview

Inference code for [MioTTS](https://huggingface.co/collections/Aratako/miotts), a lightweight and fast TTS model.

Key features:
- Compatible with common LLM inference frameworks (llama.cpp, Ollama, vLLM, etc.)
- Speech synthesis via REST API
- Reference audio preset registration
- Best-of-N for high-quality audio selection

## Models

| Model Name | Parameters | License |
|---|---|---|
| [MioTTS-0.1B](https://huggingface.co/Aratako/MioTTS-0.1B) | 0.1B | [Falcon-LLM License](https://falconllm.tii.ae/falcon-terms-and-conditions.html) |
| [MioTTS-0.4B](https://huggingface.co/Aratako/MioTTS-0.4B) | 0.4B | [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2-350M/blob/main/LICENSE) |
| [MioTTS-0.6B](https://huggingface.co/Aratako/MioTTS-0.6B) | 0.6B | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/) |
| [MioTTS-1.2B](https://huggingface.co/Aratako/MioTTS-1.2B) | 1.2B | [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Base/blob/main/LICENSE) |
| [MioTTS-1.7B](https://huggingface.co/Aratako/MioTTS-1.7B) | 1.7B | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/) |
| [MioTTS-2.6B](https://huggingface.co/Aratako/MioTTS-2.6B) | 2.6B | [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2-2.6B/blob/main/LICENSE) |

Quantized models:

| Model | Purpose |
|---|---|
| [MioTTS-GGUF](https://huggingface.co/Aratako/MioTTS-GGUF) | Quantized models for llama.cpp / Ollama |

## Setup

```bash
git clone https://github.com/Aratako/MioTTS-Inference.git
cd MioTTS-Inference
uv sync
# Install flash-attention (recommended)
# Adjust MAX_JOBS based on your CPU specs
MAX_JOBS=8 uv pip install --no-build-isolation -v flash-attn
```

## Usage

### 1. Starting the TTS Model Inference Server

Start the inference server for the TTS model. Since the model architecture is identical to standard LLMs, you can use common LLM inference frameworks. Make sure to set up an OpenAI Compatible API.

#### llama.cpp

Follow the official [Quick Start](https://github.com/ggml-org/llama.cpp?tab=readme-ov-file#quick-start) to install llama.cpp, then start the inference server as follows. Adjust the `-hff` parameter according to the model you want to use.

```bash
llama-server -hf Aratako/MioTTS-GGUF -hff MioTTS-1.2B-BF16.gguf -c 8192 --cont-batching --batch_size 8 --port 8000
```

#### Ollama

Follow the official [Download](https://ollama.com/download) to install Ollama, then start the inference server as follows. Adjust the model name according to your preference.

```bash
# Using CLI
OLLAMA_HOST=localhost:8000 ollama serve
# In a separate window
OLLAMA_HOST=localhost:8000 ollama run hf.co/Aratako/MioTTS-GGUF:MioTTS-1.2B-BF16.gguf
```

#### vLLM

Follow the official [Installation](https://docs.vllm.ai/en/latest/getting_started/installation/) to install vLLM, then start the inference server as follows. Adjust the model name according to your preference. Also adjust `--gpu-memory-utilization` based on your GPU specs.

```bash
vllm serve Aratako/MioTTS-1.2B --max-model-len 1024 --gpu-memory-utilization 0.2
```

Other inference frameworks such as LMStudio or SGLang will also work as long as they can provide an OpenAI Compatible API.

### 2. Starting the Speech Synthesis API

Start the speech synthesis API server provided in this repository. Make sure the port matches the server started in step 1 (for example, Ollama uses port 11434 by default).

```bash
python run_server.py --llm-base-url http://localhost:8000/v1
```

Add `--best-of-n-enabled` to enable Best-of-N speech synthesis. This setting generates N candidates simultaneously for a single input text and returns the best audio based on heuristic evaluation such as ASR (Whisper) error rate.

```bash
python run_server.py --llm-base-url http://localhost:8000/v1 --best-of-n-enabled
```

### 3. Starting the WebUI

A simple WebUI demo is available that uses the speech synthesis API started in the steps above.

```bash
python run_gradio.py
```

After running, access the WebUI at `http://localhost:7860`.

## Environment Variables / CLI Arguments

### run_server.py (Speech Synthesis API Server)

Settings can be changed via environment variables or CLI arguments. CLI arguments take precedence.

#### Server Settings

| Argument | Environment Variable | Default | Description |
|----------|---------------------|---------|-------------|
| `--host` | `MIOTTS_HOST` | `0.0.0.0` | Server host |
| `--port` | `MIOTTS_PORT` | `8001` | Server port |
| `--reload` | `MIOTTS_RELOAD` | `false` | Enable hot reload |
| `--log-level` | `MIOTTS_LOG_LEVEL` | `info` | Log level |

#### LLM Settings

| Argument | Environment Variable | Default | Description |
|----------|---------------------|---------|-------------|
| `--llm-base-url` | `MIOTTS_LLM_BASE_URL` | `http://localhost:8000/v1` | LLM API base URL |
| `--llm-api-key` | `MIOTTS_LLM_API_KEY` | None | LLM API key (if required) |
| `--llm-model` | `MIOTTS_LLM_MODEL` | Auto-detected | LLM model name |
| `--llm-timeout` | `MIOTTS_LLM_TIMEOUT` | `120.0` | LLM request timeout (seconds) |

#### Sampling Parameters

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `MIOTTS_LLM_TEMPERATURE` | `0.8` | Temperature |
| `MIOTTS_LLM_TOP_P` | `1.0` | Top-P |
| `MIOTTS_LLM_MAX_TOKENS` | `700` | Maximum generation tokens |
| `MIOTTS_LLM_REPETITION_PENALTY` | `1.0` | Repetition Penalty (1.0-1.5) |
| `MIOTTS_LLM_PRESENCE_PENALTY` | `0.0` | Presence Penalty (0.0-1.0) |
| `MIOTTS_LLM_FREQUENCY_PENALTY` | `0.0` | Frequency Penalty (0.0-1.0) |

#### Codec Settings

| Argument | Environment Variable | Default | Description |
|----------|---------------------|---------|-------------|
| `--codec-model` | `MIOTTS_CODEC_MODEL` | `Aratako/MioCodec-25Hz-24kHz` | MioCodec model name |
| `--device` | `MIOTTS_DEVICE` | `cuda` (or `cpu` if unavailable) | Codec inference device |

#### Preset Settings

| Argument | Environment Variable | Default | Description |
|----------|---------------------|---------|-------------|
| `--presets-dir` | `MIOTTS_PRESETS_DIR` | `presets` | Presets directory |

#### Best-of-N Settings

| Argument | Environment Variable | Default | Description |
|----------|---------------------|---------|-------------|
| `--best-of-n-enabled` | `MIOTTS_BEST_OF_N_ENABLED` | `false` | Enable Best-of-N |
| `--best-of-n-default` | `MIOTTS_BEST_OF_N_DEFAULT` | `1` | Default N (1 = normal generation) |
| `--best-of-n-max` | `MIOTTS_BEST_OF_N_MAX` | `8` | Maximum value of N |
| `--best-of-n-language` | `MIOTTS_BEST_OF_N_LANGUAGE` | `auto` | Language setting for Best-of-N (`auto`/`ja`/`en`) |

#### ASR Settings (for Best-of-N)

| Argument | Environment Variable | Default | Description |
|----------|---------------------|---------|-------------|
| `--asr-model` | `MIOTTS_ASR_MODEL` | `openai/whisper-large-v3-turbo` | ASR model |
| `--asr-device` | `MIOTTS_ASR_DEVICE` | Same as `MIOTTS_DEVICE` | ASR inference device |
| `--asr-compute-type` | `MIOTTS_ASR_COMPUTE_TYPE` | `float16` (cuda) / `int8` (cpu) | ASR compute precision |
| `--asr-batch-size` | `MIOTTS_ASR_BATCH_SIZE` | `0` (all parallel) | ASR batch size |
| `--asr-language` | `MIOTTS_ASR_LANGUAGE` | `auto` | ASR language |

#### Other Settings

| Argument | Environment Variable | Default | Description |
|----------|---------------------|---------|-------------|
| `--max-text-length` | `MIOTTS_MAX_TEXT_LENGTH` | `300` | Maximum input text length |
| `--max-reference-mb` | `MIOTTS_MAX_REFERENCE_MB` | `20` | Maximum reference audio size (MB) |
| `--allowed-audio-exts` | `MIOTTS_ALLOWED_AUDIO_EXTS` | `.wav,.flac,.ogg` | Allowed audio extensions |

The maximum reference audio length is fixed at 20 seconds.

### run_gradio.py (WebUI)

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `MIOTTS_API_BASE` | `http://localhost:8001` | Speech synthesis API server base URL |

You can also change the API Base URL from "Advanced Settings" in the WebUI.

## Reference Audio Presets

Instead of providing reference audio each time, you can pre-encode audio with the codec and register it as a reusable preset.

```bash
python scripts/generate_preset.py --audio /path/to/audio.wav --preset-id preset_name
```

### generate_preset.py Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--audio` | Yes | - | Path to reference audio file |
| `--preset-id` | Yes | - | Preset ID (becomes the filename) |
| `--output-dir` | No | `presets` | Output directory |
| `--model-id` | No | `Aratako/MioCodec-25Hz-24kHz` | MioCodec model name |
| `--device` | No | `cuda` | Inference device |

### Default Presets

The following presets are included:

- `jp_female` - Japanese female voice
- `jp_male` - Japanese male voice
- `en_female` - English female voice
- `en_male` - English male voice

## API Specification

### Health Check

```
GET /health
```

**Response:**
```json
{"status": "ok"}
```

### List Presets

```
GET /v1/presets
```

**Response:**
```json
{"presets": ["en_female", "en_male", "jp_female", "jp_male"]}
```

### Speech Synthesis (JSON Request)

```
POST /v1/tts
Content-Type: application/json
```

**Request Body:**
```json
{
  "text": "Text to synthesize",
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

`reference` is required.
Text preprocessing applies normalization for Japanese input, and only `strip()` for other languages.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | Yes | Text to synthesize |
| `reference.type` | string | Yes | `preset` or `base64` |
| `reference.preset_id` | string | Conditional | Required when `type=preset` |
| `reference.data` | string | Conditional | Required when `type=base64` |
| `llm.*` | - | No | LLM parameters |
| `output.format` | string | No | `wav` or `base64` (default: base64) |
| `best_of_n.*` | - | No | Best-of-N settings |

**Response:**
```json
{
  "audio": "Base64-encoded WAV data",
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
  "normalized_text": "Preprocessed text"
}
```

### Speech Synthesis (File Upload)

```
POST /v1/tts/file
Content-Type: multipart/form-data
```

**Form Fields:**

Either `reference_audio` or `reference_preset_id` is required.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | Yes | Text to synthesize |
| `reference_audio` | file | Conditional | Required when `reference_preset_id` is not specified |
| `reference_preset_id` | string | Conditional | Required when `reference_audio` is not specified |
| `model` | string | No | LLM model name |
| `temperature` | float | No | Temperature |
| `top_p` | float | No | Top-P |
| `max_tokens` | int | No | Maximum generation tokens |
| `repetition_penalty` | float | No | Repetition Penalty |
| `presence_penalty` | float | No | Presence Penalty |
| `frequency_penalty` | float | No | Frequency Penalty |
| `output_format` | string | No | `wav` or `base64` |
| `best_of_n_enabled` | boolean | No | Enable Best-of-N |
| `best_of_n_n` | int | No | Value of N |
| `best_of_n_language` | string | No | Language setting for Best-of-N |

**Response:**
- `output_format=wav`: WAV file (`audio/wav`)
- `output_format=base64`: JSON response

## License & Credits

- **Code**: MIT License
- **Default presets**: The default presets under `presets` use audio generated by [T5Gemma-TTS](https://huggingface.co/Aratako/T5Gemma-TTS-2b-2b) and [gemini-2.5-pro-tts](https://cloud.google.com/text-to-speech/docs/gemini-tts), so audio synthesized using these presets cannot be used commercially.
- **Models**: Please follow the license of each model.
