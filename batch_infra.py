#!/usr/bin/env python3
"""
MioTTS Batch Processor CLI

A robust command-line tool for batch processing text through MioTTS.
Processes chunks sequentially while maintaining order for final concatenation.
"""

import argparse
import base64
import json
import os
import sys
import time
import wave
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
import shutil

import requests
from tqdm import tqdm


class PathEncoder(json.JSONEncoder):
    """JSON Encoder that handles Path objects."""
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def convert_paths_to_strings(obj):
    """Recursively convert Path objects to strings."""
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: convert_paths_to_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_paths_to_strings(item) for item in obj]
    return obj


@dataclass
class TTSConfig:
    """Configuration for TTS processing."""
    api_url: str = "http://localhost:8001"
    reference_preset: Optional[str] = "en_female"
    reference_audio: Optional[Path] = None
    temperature: float = 0.8
    top_p: float = 1.0
    max_tokens: int = 700
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    best_of_n_enabled: bool = False
    best_of_n_n: int = 1
    best_of_n_language: str = "auto"
    use_file_upload: bool = False
    
    def validate(self):
        if self.reference_audio and self.reference_preset:
            raise ValueError("Cannot specify both reference_audio and reference_preset")
        if not self.reference_audio and not self.reference_preset:
            raise ValueError("Must specify either reference_audio or reference_preset")
    
    def to_dict(self):
        d = asdict(self)
        return convert_paths_to_strings(d)


class MioTTSBatchProcessor:
    """Main processor for batch TTS processing."""
    
    def __init__(self, config: TTSConfig):
        self.config = config
        self.config.validate()
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json, audio/wav"})
        
        # State tracking
        self.state = {
            "processed_indices": [],
            "failed_indices": [],
            "output_files": {},
            "config": config.to_dict()
        }
        self.state_file = None
        
        self._check_health()
        
    def _check_health(self):
        """Verify API is reachable."""
        try:
            resp = requests.get(f"{self.config.api_url}/health", timeout=10)
            resp.raise_for_status()
            print(f"✅ API healthy at {self.config.api_url}")
        except Exception as e:
            print(f"⚠️  API health check failed: {e}")
            raise
    
    def _load_state(self, state_path: Path):
        """Load processing state with error handling."""
        if not state_path.exists():
            return
            
        try:
            with open(state_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if not content.strip():
                    return
                loaded = json.loads(content)
            
            current_config = self.config.to_dict()
            if loaded.get("config") == current_config:
                self.state = loaded
                # Convert output_files dict keys back to int
                if "output_files" in self.state:
                    self.state["output_files"] = {
                        int(k): Path(v) if v else None 
                        for k, v in self.state["output_files"].items()
                    }
                processed = len(self.state.get("processed_indices", []))
                failed = len(self.state.get("failed_indices", []))
                print(f"📂 Resuming: {processed} done, {failed} failed")
            else:
                print("⚠️  Config changed, starting fresh")
                backup = state_path.with_suffix('.json.backup')
                shutil.copy(state_path, backup)
                
        except json.JSONDecodeError as e:
            print(f"⚠️  Corrupted state file: {e}")
            backup = state_path.with_suffix('.json.corrupted')
            shutil.move(state_path, backup)
            print("   Starting fresh")
        except Exception as e:
            print(f"⚠️  Error loading state: {e}")
    
    def _save_state(self):
        """Save processing state safely."""
        if self.state_file:
            # Convert to serializable format
            state_to_save = {
                "processed_indices": self.state["processed_indices"],
                "failed_indices": self.state["failed_indices"],
                "output_files": {
                    str(k): str(v) if v else None 
                    for k, v in self.state["output_files"].items()
                },
                "config": self.state["config"]
            }
            try:
                with open(self.state_file, 'w', encoding='utf-8') as f:
                    json.dump(state_to_save, f, indent=2, cls=PathEncoder)
            except Exception as e:
                print(f"⚠️  Warning: Could not save state: {e}")
    
    def _get_reference_params(self):
        """Build reference parameters."""
        if self.config.reference_preset:
            return {"type": "preset", "preset_id": self.config.reference_preset}
        elif self.config.reference_audio:
            if self.config.use_file_upload:
                return {"reference_audio": open(self.config.reference_audio, 'rb')}
            else:
                with open(self.config.reference_audio, 'rb') as f:
                    data = base64.b64encode(f.read()).decode('utf-8')
                return {"type": "base64", "data": data}
        return {}
    
    def _make_tts_request_json(self, text: str, retries: int = 3) -> bytes:
        """Make TTS request using JSON API."""
        payload = {
            "text": text,
            "reference": self._get_reference_params(),
            "llm": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_tokens": self.config.max_tokens,
                "repetition_penalty": self.config.repetition_penalty,
                "presence_penalty": self.config.presence_penalty,
                "frequency_penalty": self.config.frequency_penalty
            },
            "output": {"format": "base64"}
        }
        
        if self.config.best_of_n_enabled:
            payload["best_of_n"] = {
                "enabled": True,
                "n": self.config.best_of_n_n,
                "language": self.config.best_of_n_language
            }
        
        last_error = None
        for attempt in range(retries):
            try:
                resp = self.session.post(
                    f"{self.config.api_url}/v1/tts",
                    json=payload,
                    timeout=120
                )
                resp.raise_for_status()
                data = resp.json()
                return base64.b64decode(data["audio"])
                
            except Exception as e:
                last_error = e
                time.sleep((attempt + 1) * 2)
        
        raise Exception(f"Failed after {retries} attempts: {last_error}")
    
    def _make_tts_request_file(self, text: str, retries: int = 3) -> bytes:
        """Make TTS request using file upload API."""
        files = {}
        data = {
            "text": text,
            "temperature": str(self.config.temperature),
            "top_p": str(self.config.top_p),
            "max_tokens": str(self.config.max_tokens),
            "repetition_penalty": str(self.config.repetition_penalty),
            "presence_penalty": str(self.config.presence_penalty),
            "frequency_penalty": str(self.config.frequency_penalty),
            "output_format": "wav"
        }
        
        if self.config.reference_preset:
            data["reference_preset_id"] = self.config.reference_preset
        elif self.config.reference_audio:
            files["reference_audio"] = open(self.config.reference_audio, 'rb')
        
        if self.config.best_of_n_enabled:
            data["best_of_n_enabled"] = "true"
            data["best_of_n_n"] = str(self.config.best_of_n_n)
        
        last_error = None
        for attempt in range(retries):
            try:
                resp = self.session.post(
                    f"{self.config.api_url}/v1/tts/file",
                    data=data,
                    files=files,
                    timeout=120
                )
                resp.raise_for_status()
                return resp.content
                
            except Exception as e:
                last_error = e
                time.sleep((attempt + 1) * 2)
            finally:
                for f in files.values():
                    if hasattr(f, 'close'):
                        f.close()
        
        raise Exception(f"Failed after {retries} attempts: {last_error}")
    
    def _process_chunk(self, chunk_idx: int, chunk_text: str, output_path: Path) -> Tuple[int, bool, Optional[Path]]:
        """
        Process a single chunk.
        
        Returns: (chunk_idx, success, output_path_or_none)
        """
        try:
            if self.config.use_file_upload:
                audio = self._make_tts_request_file(chunk_text)
            else:
                audio = self._make_tts_request_json(chunk_text)
            
            with open(output_path, 'wb') as f:
                f.write(audio)
            
            return (chunk_idx, True, output_path)
            
        except Exception as e:
            print(f"\n❌ Failed on chunk {chunk_idx}: {e}")
            return (chunk_idx, False, None)
    
    def _split_text(self, text: str, mode: str = "line", max_chars: Optional[int] = None) -> List[str]:
        """Split text into processing chunks."""
        if mode == "line":
            lines = [line.strip() for line in text.split('\n')]
            chunks = [line for line in lines if line]
        elif mode == "paragraph":
            paras = text.split('\n\n')
            chunks = [p.strip() for p in paras if p.strip()]
        elif mode == "sentence":
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text)
            chunks = [s.strip() for s in sentences if s.strip()]
        elif mode == "chunk":
            if not max_chars:
                max_chars = 300
            chunks = []
            words = text.split()
            current = ""
            for word in words:
                if len(current) + len(word) + 1 > max_chars:
                    if current:
                        chunks.append(current.strip())
                    current = word
                else:
                    current += " " + word if current else word
            if current:
                chunks.append(current.strip())
        else:
            raise ValueError(f"Unknown split mode: {mode}")
        
        # Filter by max length
        if max_chars:
            valid_chunks = []
            for chunk in chunks:
                if len(chunk) <= max_chars:
                    valid_chunks.append(chunk)
                else:
                    sub_chunks = self._split_text(chunk, "chunk", max_chars)
                    valid_chunks.extend(sub_chunks)
            chunks = valid_chunks
        
        return chunks
    
    def _concatenate_wavs(self, wav_files: Dict[int, Path], output_path: Path, 
                          add_silence_ms: int = 0):
        """Concatenate WAV files in index order."""
        if not wav_files:
            return
        
        # Sort by index
        sorted_indices = sorted(wav_files.keys())
        sorted_files = [wav_files[i] for i in sorted_indices if wav_files[i] and wav_files[i].exists()]
        
        if not sorted_files:
            print("⚠️  No valid files to concatenate")
            return
        
        print(f"\n🔧 Concatenating {len(sorted_files)} segments...")
        
        with wave.open(str(sorted_files[0]), 'rb') as w:
            params = w.getparams()
            frames = [w.readframes(w.getnframes())]
        
        for wav_file in sorted_files[1:]:
            with wave.open(str(wav_file), 'rb') as w:
                frames.append(w.readframes(w.getnframes()))
                if add_silence_ms > 0:
                    silence_frames = b'\x00' * int(params.framerate * params.sampwidth * add_silence_ms / 1000)
                    frames.append(silence_frames)
        
        with wave.open(str(output_path), 'wb') as w:
            w.setparams(params)
            for frame in frames:
                w.writeframes(frame)
        
        print(f"✅ Final: {output_path}")
    
    def process_file(
        self,
        input_path: Path,
        output_dir: Path,
        split_mode: str = "line",
        max_line_length: Optional[int] = None,
        concatenate: bool = True,
        add_silence_ms: int = 0,
        delete_chunks: bool = False,
        on_error: str = "prompt"
    ) -> Optional[Path]:
        """Process file sequentially."""
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        job_name = input_path.stem
        self.state_file = output_dir / f".{job_name}.state.json"
        chunks_dir = output_dir / f"{job_name}_chunks"
        chunks_dir.mkdir(exist_ok=True)
        
        # Load previous state
        self._load_state(self.state_file)
        
        # Read and split text
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        chunks = self._split_text(text, split_mode, max_line_length)
        total = len(chunks)
        
        if total == 0:
            print("⚠️  No text to process")
            return None
        
        # Determine which chunks remain
        processed_set = set(self.state["processed_indices"])
        remaining = [(i, chunks[i]) for i in range(total) if i not in processed_set]
        
        print(f"\n📚 {input_path.name}")
        print(f"   Mode: {split_mode} | Total: {total} | Remaining: {len(remaining)}")
        print(f"   Output: {output_dir}")
        print(f"   Ref: {self.config.reference_preset or self.config.reference_audio}\n")
        
        if not remaining:
            print("✅ All chunks already processed!")
        
        # Process with progress bar
        pbar = tqdm(remaining, desc="Processing", unit="chunk")
        
        for chunk_idx, chunk_text in pbar:
            # Skip empty
            if not chunk_text.strip():
                self.state["processed_indices"].append(chunk_idx)
                continue
            
            output_path = chunks_dir / f"{job_name}_{chunk_idx:05d}.wav"
            
            idx, success, path = self._process_chunk(chunk_idx, chunk_text, output_path)
            
            if success:
                self.state["processed_indices"].append(idx)
                self.state["output_files"][idx] = path
            else:
                self.state["failed_indices"].append(idx)
                
                if on_error == "prompt":
                    choice = input(f"\nChunk {idx} failed. [c]ontinue, [s]top? ").lower()
                    if choice == 's':
                        raise KeyboardInterrupt
            
            # Save state periodically
            if len(self.state["processed_indices"]) % 5 == 0:
                self._save_state()
        
        pbar.close()
        self._save_state()
        
        # Concatenate results
        final_output = None
        if concatenate and self.state["output_files"]:
            final_output = output_dir / f"{job_name}_complete.wav"
            self._concatenate_wavs(self.state["output_files"], final_output, add_silence_ms)
            
            if delete_chunks:
                print("🗑️  Cleaning up chunks...")
                for path in self.state["output_files"].values():
                    if path and path.exists():
                        path.unlink()
                chunks_dir.rmdir()
                self.state_file.unlink(missing_ok=True)
        
        # Summary
        success = len(self.state["processed_indices"])
        failed = len(self.state["failed_indices"])
        print(f"\n📊 Done: {success} success, {failed} failed, {total} total")
        
        return final_output or chunks_dir


def main():
    parser = argparse.ArgumentParser(
        description="MioTTS Batch Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python miotts_batch.py book.txt -o ./out/
  
  # Paragraph splitting
  python miotts_batch.py book.txt -o ./out/ --split paragraph
  
  # Custom reference
  python miotts_batch.py book.txt --reference-audio ./voice.wav -o ./out/
        """
    )
    
    parser.add_argument("input", help="Input text file")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    
    # Voice selection
    voice_group = parser.add_mutually_exclusive_group()
    voice_group.add_argument("--preset", default="en_female", help="Voice preset")
    voice_group.add_argument("--reference-audio", type=Path, help="Custom reference audio")
    
    # API configuration
    parser.add_argument("--api-url", default="http://localhost:8001", help="API URL")
    parser.add_argument("--use-file-upload", action="store_true", help="Use file upload API")
    
    # Text processing
    parser.add_argument("--split", choices=["line", "paragraph", "sentence", "chunk"],
                       default="line", help="Text splitting mode")
    parser.add_argument("--max-length", type=int, help="Max chars per chunk")
    
    # LLM parameters
    llm_group = parser.add_argument_group("LLM Parameters")
    llm_group.add_argument("--temperature", type=float, default=0.8)
    llm_group.add_argument("--top-p", type=float, default=1.0)
    llm_group.add_argument("--max-tokens", type=int, default=700)
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument("--silence", type=int, default=0, 
                             help="Silence between chunks (ms)")
    output_group.add_argument("--delete-chunks", action="store_true",
                             help="Delete chunks after concatenation")
    
    args = parser.parse_args()
    
    config = TTSConfig(
        api_url=args.api_url,
        reference_preset=args.preset if not args.reference_audio else None,
        reference_audio=args.reference_audio,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        use_file_upload=args.use_file_upload
    )
    
    processor = MioTTSBatchProcessor(config)
    
    try:
        processor.process_file(
            Path(args.input),
            args.output,
            split_mode=args.split,
            max_line_length=args.max_length,
            add_silence_ms=args.silence,
            delete_chunks=args.delete_chunks
        )
    except KeyboardInterrupt:
        print("\n\n⏸️  Interrupted. Run again to resume.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()