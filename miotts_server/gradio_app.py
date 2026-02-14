from __future__ import annotations

import base64
import io
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

import gradio as gr
import httpx
import numpy as np
import soundfile as sf
from pypdf import PdfReader
from pydub import AudioSegment

DEFAULT_API_BASE = os.getenv("MIOTTS_API_BASE", "http://localhost:8001")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using simple regex."""
    # Split on period, exclamation, question mark followed by space or newline
    sentences = re.split(r'([.!?]+(?:\s+|$))', text)
    # Rejoin punctuation with sentences
    result = []
    for i in range(0, len(sentences) - 1, 2):
        sentence = (sentences[i] + sentences[i + 1]).strip()
        if sentence:
            result.append(sentence)
    # Handle last item if odd number of splits
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        result.append(sentences[-1].strip())
    return result


def _extract_text_from_file(file_path: str) -> str:
    """Extract text from txt or pdf files."""
    path = Path(file_path)
    
    if path.suffix.lower() == '.pdf':
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    elif path.suffix.lower() == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


def _fetch_presets(api_base: str) -> list[str]:
    try:
        res = httpx.get(f"{api_base}/v1/presets", timeout=5.0)
        res.raise_for_status()
        data = res.json()
        presets = data.get("presets", [])
        if isinstance(presets, list):
            return presets
    except Exception:
        pass
    return []


def _refresh_presets(api_base: str) -> gr.Dropdown:
    presets = _fetch_presets(api_base)
    value = presets[0] if presets else None
    return gr.update(choices=presets, value=value)


def _decode_wav_bytes(data: bytes) -> tuple[int, np.ndarray]:
    with io.BytesIO(data) as buff:
        audio, sr = sf.read(buff, dtype="float32")
    return sr, audio


def _call_tts(
    api_base: str,
    text: str,
    reference_mode: str,
    reference_audio: tuple[int, np.ndarray] | None,
    preset_id: str,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    presence_penalty: float,
    frequency_penalty: float,
    best_of_n_enabled: bool,
    best_of_n_n: int,
    best_of_n_language: str,
) -> tuple[tuple[int, np.ndarray] | None, str]:
    if not text:
        return None, "No text provided"
    api_base = api_base.rstrip("/")

    payload: dict[str, Any] = {
        "text": text,
        "llm": {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": 700,
            "repetition_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
        },
    }
    if reference_mode == "upload" and reference_audio is not None:
        sr, audio = reference_audio
        buffer = io.BytesIO()
        sf.write(buffer, audio, sr, format="WAV")
        audio_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        payload["reference"] = {"type": "base64", "data": audio_b64}
    elif preset_id:
        payload["reference"] = {"type": "preset", "preset_id": preset_id}
    if best_of_n_enabled:
        payload["best_of_n"] = {
            "enabled": True,
            "n": best_of_n_n,
            "language": best_of_n_language,
        }
    
    try:
        response = httpx.post(f"{api_base}/v1/tts", json=payload, timeout=120.0)
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP {e.response.status_code}"
        try:
            error_detail = e.response.json()
            if isinstance(error_detail, dict) and "detail" in error_detail:
                error_msg += f": {error_detail['detail']}"
        except Exception:
            pass
        return None, f"Server error: {error_msg}"
    except httpx.TimeoutException:
        return None, "Request timeout (120s). Try shorter text or check server."
    except httpx.ConnectError:
        return None, f"Connection error: Cannot reach server at {api_base}"
    except Exception as e:
        return None, f"Error: {str(e)}"

    content_type = response.headers.get("content-type", "")
    if content_type.startswith("audio/"):
        return _decode_wav_bytes(response.content), ""
    
    try:
        data = response.json()
    except Exception:
        return None, "Invalid JSON response from server"
        
    audio_b64 = data.get("audio")
    if not audio_b64:
        return None, "No audio in response."
    
    try:
        audio_bytes = base64.b64decode(audio_b64)
        sr, audio = _decode_wav_bytes(audio_bytes)
    except Exception as e:
        return None, f"Failed to decode audio: {str(e)}"
        
    audio_samples = audio.shape[0] if hasattr(audio, "shape") else len(audio)
    audio_sec = float(audio_samples) / float(sr) if sr else 0.0
    timings = data.get("timings") or {}
    total_sec = timings.get("total_sec") or 0.0
    rtf = (float(total_sec) / audio_sec) if audio_sec > 0 else 0.0

    def _fmt(label: str, value: Any) -> str:
        if value is None:
            return f"- {label}: n/a"
        try:
            return f"- {label}: {float(value):.3f}s"
        except Exception:
            return f"- {label}: {value}"

    total_sec = timings.get("total_sec")
    llm_sec = timings.get("llm_sec")
    parse_sec = timings.get("parse_sec")
    codec_sec = timings.get("codec_sec")
    best_of_n_sec = timings.get("best_of_n_sec")
    asr_sec = timings.get("asr_sec")
    rtf_line = f"- RTF: {rtf:.3f}" if rtf else "- RTF: n/a"

    info_text = "\n".join(
        [
            "Timings",
            _fmt("Total", total_sec),
            _fmt("LLM", llm_sec),
            _fmt("Parse", parse_sec),
            _fmt("Codec", codec_sec),
            _fmt("Best-of-N", best_of_n_sec),
            _fmt("ASR", asr_sec),
            rtf_line,
        ]
    )
    return (sr, audio), info_text


def _call_tts_batch(
    api_base: str,
    sentences: list[str],
    reference_mode: str,
    reference_audio: tuple[int, np.ndarray] | None,
    preset_id: str,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    presence_penalty: float,
    frequency_penalty: float,
    best_of_n_enabled: bool,
    best_of_n_n: int,
    best_of_n_language: str,
    progress=gr.Progress(),
) -> tuple[tuple[int, np.ndarray] | None, str]:
    """Process multiple sentences and concatenate audio."""
    if not sentences:
        return None, "No sentences to process"
    
    # Create UUID subfolder for chunks
    session_id = str(uuid.uuid4())
    chunks_dir = Path("outs") / session_id
    chunks_dir.mkdir(parents=True, exist_ok=True)
    
    all_audio = []
    sample_rate = None
    failed = []
    total_sentences = len(sentences)
    start_time = time.time()
    
    for i, sentence in enumerate(sentences):
        # Calculate ETA
        if i > 0:
            elapsed = time.time() - start_time
            avg_time_per_sentence = elapsed / i
            remaining_sentences = total_sentences - i
            eta_seconds = avg_time_per_sentence * remaining_sentences
            eta_mins = int(eta_seconds // 60)
            eta_secs = int(eta_seconds % 60)
            eta_str = f"{eta_mins}m {eta_secs}s" if eta_mins > 0 else f"{eta_secs}s"
            desc = f"Sentence {i+1}/{total_sentences} | ETA: {eta_str}"
        else:
            desc = f"Sentence {i+1}/{total_sentences}"
        
        # Update progress
        if progress is not None:
            progress((i + 1) / total_sentences, desc=desc)
        
        # Skip empty sentences
        if not sentence.strip():
            continue
            
        # Retry logic for failed requests
        max_retries = 3
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            try:
                result, _ = _call_tts(
                    api_base=api_base,
                    text=sentence,
                    reference_mode=reference_mode,
                    reference_audio=reference_audio,
                    preset_id=preset_id,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    best_of_n_enabled=best_of_n_enabled,
                    best_of_n_n=best_of_n_n,
                    best_of_n_language=best_of_n_language,
                )
                if result:
                    sr, audio = result
                    if sample_rate is None:
                        sample_rate = sr
                    elif sr != sample_rate:
                        # Resample if sample rates don't match (shouldn't happen but just in case)
                        failed.append(f"Sentence {i+1}: Sample rate mismatch")
                        break
                    
                    # Ensure audio is 1D array
                    if audio.ndim > 1:
                        audio = audio.flatten()
                    
                    # Save chunk as MP3
                    chunk_path = chunks_dir / f"chunk_{i+1:03d}.mp3"
                    try:
                        # Convert to AudioSegment and export as MP3
                        audio_int16 = (audio * 32767).astype(np.int16)
                        audio_segment = AudioSegment(
                            audio_int16.tobytes(),
                            frame_rate=sr,
                            sample_width=2,
                            channels=1
                        )
                        audio_segment.export(str(chunk_path), format="mp3", bitrate="192k")
                    except Exception as e:
                        failed.append(f"Sentence {i+1}: Failed to save MP3 - {str(e)[:50]}")
                    
                    all_audio.append(audio)
                    success = True
                else:
                    failed.append(f"Sentence {i+1}: No audio returned")
                    break
                    
            except httpx.HTTPStatusError as e:
                retry_count += 1
                if retry_count >= max_retries:
                    error_msg = f"Sentence {i+1}: HTTP {e.response.status_code}"
                    try:
                        error_detail = e.response.json()
                        if isinstance(error_detail, dict) and "detail" in error_detail:
                            error_msg += f" - {error_detail['detail']}"
                    except Exception:
                        pass
                    failed.append(error_msg)
                else:
                    # Wait a bit before retrying
                    time.sleep(1.0 * retry_count)
                    
            except httpx.TimeoutException:
                retry_count += 1
                if retry_count >= max_retries:
                    failed.append(f"Sentence {i+1}: Request timeout")
                else:
                    time.sleep(2.0 * retry_count)
                    
            except Exception as e:
                failed.append(f"Sentence {i+1}: {str(e)[:100]}")
                break
    
    if not all_audio:
        error_msg = "All sentences failed to process"
        if failed:
            error_msg += ":\n" + "\n".join(failed[:10])  # Show first 10 errors
            if len(failed) > 10:
                error_msg += f"\n... and {len(failed) - 10} more errors"
        return None, error_msg
    
    # Concatenate all audio with silence between sentences (0.3 seconds)
    silence_duration = 0.3
    silence_samples = int(sample_rate * silence_duration)
    silence = np.zeros(silence_samples, dtype=all_audio[0].dtype)
    
    audio_with_pauses = []
    for i, audio in enumerate(all_audio):
        audio_with_pauses.append(audio)
        if i < len(all_audio) - 1:  # Don't add silence after last sentence
            audio_with_pauses.append(silence)
    
    combined = np.concatenate(audio_with_pauses, axis=0)
    
    total_time = time.time() - start_time
    total_mins = int(total_time // 60)
    total_secs = int(total_time % 60)
    time_str = f"{total_mins}m {total_secs}s" if total_mins > 0 else f"{total_secs}s"
    
    info = f"✓ Successfully processed {len(all_audio)}/{total_sentences} sentences in {time_str}\n"
    info += f"📁 Chunks saved to: outs/{session_id}/"
    if failed:
        info += f"\n\n⚠ Failed: {len(failed)} sentences"
        if len(failed) <= 5:
            info += "\n" + "\n".join(failed)
        else:
            info += "\n" + "\n".join(failed[:5])
            info += f"\n... and {len(failed) - 5} more errors"
    
    return (sample_rate, combined), info


def build_app() -> gr.Blocks:
    presets = _fetch_presets(DEFAULT_API_BASE)

    with gr.Blocks(title="MioTTS Demo") as demo:
        gr.Markdown("# MioTTS Demo")

        with gr.Accordion("Advanced Settings", open=False):
            api_base = gr.Textbox(
                label="API Base URL",
                value=DEFAULT_API_BASE,
                placeholder="http://localhost:8001",
            )

        with gr.Tabs() as tabs:
            with gr.Tab("Single Text"):
                text = gr.Textbox(label="Text", lines=6, placeholder="Type text to synthesize...")
                
            with gr.Tab("Batch Processing"):
                file_upload = gr.File(
                    label="Upload File (TXT or PDF)",
                    file_types=[".txt", ".pdf"],
                )
                batch_text_preview = gr.Textbox(
                    label="Extracted Text Preview",
                    lines=6,
                    interactive=False,
                )
                batch_sentence_count = gr.Markdown("No file loaded")

        with gr.Row():
            reference_mode = gr.Radio(
                choices=["preset", "upload"],
                value="preset",
                label="Reference Mode",
            )
            preset_id = gr.Dropdown(
                choices=presets,
                value=presets[0] if presets else None,
                label="Preset ID",
                allow_custom_value=True,
                visible=True,
            )
            with gr.Column(scale=0, min_width=72):
                refresh_presets = gr.Button("↻", size="md")

        reference_audio = gr.Audio(
            label="Reference Audio",
            sources=["upload"],
            type="numpy",
            visible=False,
        )

        def _update_reference_visibility(mode):
            if mode == "preset":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)

        reference_mode.change(
            fn=_update_reference_visibility,
            inputs=[reference_mode],
            outputs=[preset_id, reference_audio],
        )

        with gr.Row():
            temperature = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Top-p")
            repetition_penalty = gr.Slider(
                1.0, 1.5, value=1.0, step=0.05, label="Repetition Penalty"
            )
            presence_penalty = gr.Slider(0.0, 0.5, value=0.0, step=0.05, label="Presence Penalty")
            frequency_penalty = gr.Slider(0.0, 0.5, value=0.0, step=0.05, label="Frequency Penalty")

        with gr.Row():
            best_of_n_enabled = gr.Checkbox(value=False, label="Best-of-N")
            best_of_n_n = gr.Slider(1, 8, value=2, step=1, label="N")
            best_of_n_language = gr.Dropdown(
                choices=["auto", "ja", "en"],
                value="auto",
                label="Language",
            )
        
        # Buttons and outputs
        with gr.Row():
            synth_btn = gr.Button("Synthesize Single Text", variant="primary")
            batch_synth_btn = gr.Button("Synthesize Batch", variant="primary")
        
        output_audio = gr.Audio(label="Output", type="numpy")
        output_info = gr.Markdown(label="Timings")

        # File processing handler
        def _process_file(file):
            if file is None:
                return "", "No file loaded"
            try:
                text = _extract_text_from_file(file.name)
                sentences = _split_sentences(text)
                preview = text[:500] + ("..." if len(text) > 500 else "")
                count_info = f"**{len(sentences)} sentences detected**"
                return preview, count_info
            except Exception as e:
                return "", f"Error: {str(e)}"

        file_upload.change(
            _process_file,
            inputs=[file_upload],
            outputs=[batch_text_preview, batch_sentence_count],
        )

        refresh_presets.click(
            _refresh_presets,
            inputs=api_base,
            outputs=preset_id,
        )

        synth_btn.click(
            _call_tts,
            inputs=[
                api_base,
                text,
                reference_mode,
                reference_audio,
                preset_id,
                temperature,
                top_p,
                repetition_penalty,
                presence_penalty,
                frequency_penalty,
                best_of_n_enabled,
                best_of_n_n,
                best_of_n_language,
            ],
            outputs=[output_audio, output_info],
        )

        # Batch synthesis handler
        def _batch_synthesis(
            file,
            api_base_val,
            ref_mode,
            ref_audio,
            preset,
            temp,
            topp,
            rep_pen,
            pres_pen,
            freq_pen,
            bon_enabled,
            bon_n,
            bon_lang,
            progress=gr.Progress(),
        ):
            if file is None:
                return None, "No file uploaded"
            try:
                text = _extract_text_from_file(file.name)
                sentences = _split_sentences(text)
                return _call_tts_batch(
                    api_base=api_base_val,
                    sentences=sentences,
                    reference_mode=ref_mode,
                    reference_audio=ref_audio,
                    preset_id=preset,
                    temperature=temp,
                    top_p=topp,
                    repetition_penalty=rep_pen,
                    presence_penalty=pres_pen,
                    frequency_penalty=freq_pen,
                    best_of_n_enabled=bon_enabled,
                    best_of_n_n=bon_n,
                    best_of_n_language=bon_lang,
                    progress=progress,
                )
            except Exception as e:
                return None, f"Error: {str(e)}"

        batch_synth_btn.click(
            _batch_synthesis,
            inputs=[
                file_upload,
                api_base,
                reference_mode,
                reference_audio,
                preset_id,
                temperature,
                top_p,
                repetition_penalty,
                presence_penalty,
                frequency_penalty,
                best_of_n_enabled,
                best_of_n_n,
                best_of_n_language,
            ],
            outputs=[output_audio, output_info],
        )

    return demo


def main() -> None:
    app = build_app()
    app.launch()


if __name__ == "__main__":
    main()