#!/usr/bin/env python3
"""
MioTTS Batch Processor CLI - Enhanced Edition

A robust command-line tool for batch processing text through MioTTS.
Supports both JSON and file upload APIs, custom reference audio, presets,
and advanced batching strategies.
"""

import argparse
import base64
import json
import os
import sys
import time
import wave
import io
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
from dataclasses import dataclass, asdict
import tempfile
import shutil

import requests
from tqdm import tqdm


class PathEncoder(json.JSONEncoder):
    """JSON Encoder that handles Path objects."""
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


@dataclass
class TTSConfig:
    """Configuration for TTS processing."""
    api_url: str = "http://localhost:8001"
    llm_base_url: Optional[str] = None
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
    output_format: str = "wav"
    use_file_upload: bool = False  # Use multipart/form-data instead of JSON
    
    def validate(self):
        """Validate configuration."""
        if self.reference_audio and self.reference_preset:
            raise ValueError("Cannot specify both reference_audio and reference_preset")
        if not self.reference_audio and not self.reference_preset:
            raise ValueError("Must specify either reference_audio or reference_preset")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with Path objects as strings."""
        d = asdict(self)
        if d.get('reference_audio'):
            d['reference_audio'] = str(d['reference_audio'])
        return d


class MioTTSBatchProcessor:
    def __init__(self, config: TTSConfig):
        self.config = config
        self.config.validate()
        
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json, audio/wav"
        })
        
        # State tracking
        self.state: Dict[str, Any] = {
            "processed_indices": [],
            "failed_indices": [],
            "output_files": [],
            "config": config.to_dict()
        }
        self.state_file: Optional[Path] = None
        
        # Check API health on init
        self._check_health()
        
    def _check_health(self):
        """Verify API is reachable."""
        try:
            resp = self.session.get(f"{self.config.api_url}/health", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != "ok":
                raise Exception(f"API unhealthy: {data}")
            print(f"✅ API healthy at {self.config.api_url}")
        except Exception as e:
            print(f"⚠️  API health check failed: {e}")
            print(f"   Make sure MioTTS server is running: python run_server.py --llm-base-url ...")
            raise
    
    def _get_reference_params(self) -> Dict[str, Any]:
        """Build reference parameters for API request."""
        if self.config.reference_preset:
            return {"type": "preset", "preset_id": self.config.reference_preset}
        elif self.config.reference_audio:
            # For file upload, this is handled separately
            if self.config.use_file_upload:
                return {"reference_audio": open(self.config.reference_audio, 'rb')}
            else:
                # Encode as base64 for JSON
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
            "output": {
                "format": "base64"
            }
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
                
                # Decode base64 audio
                audio_data = base64.b64decode(data["audio"])
                
                # Log timing info if available
                if "timings" in data:
                    timings = data["timings"]
                    total = timings.get("total_sec", 0)
                    print(f"  ⏱️  Generated in {total:.2f}s", end="\r")
                
                return audio_data
                
            except requests.exceptions.RequestException as e:
                last_error = e
                wait = (attempt + 1) * 2
                print(f"  ⚠️  Retry {attempt + 1}/{retries} in {wait}s...")
                time.sleep(wait)
        
        raise Exception(f"Failed after {retries} attempts: {last_error}")
    
    def _make_tts_request_file(self, text: str, retries: int = 3) -> bytes:
        """Make TTS request using file upload API (multipart/form-data)."""
        files = {}
        data = {
            "text": text,
            "temperature": str(self.config.temperature),
            "top_p": str(self.config.top_p),
            "max_tokens": str(self.config.max_tokens),
            "repetition_penalty": str(self.config.repetition_penalty),
            "presence_penalty": str(self.config.presence_penalty),
            "frequency_penalty": str(self.config.frequency_penalty),
            "output_format": self.config.output_format
        }
        
        # Handle reference
        if self.config.reference_preset:
            data["reference_preset_id"] = self.config.reference_preset
        elif self.config.reference_audio:
            files["reference_audio"] = open(self.config.reference_audio, 'rb')
        
        # Best-of-N
        if self.config.best_of_n_enabled:
            data["best_of_n_enabled"] = "true"
            data["best_of_n_n"] = str(self.config.best_of_n_n)
            data["best_of_n_language"] = self.config.best_of_n_language
        
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
                
                # Return raw content (WAV file)
                return resp.content
                
            except requests.exceptions.RequestException as e:
                last_error = e
                wait = (attempt + 1) * 2
                print(f"  ⚠️  Retry {attempt + 1}/{retries} in {wait}s...")
                time.sleep(wait)
            finally:
                # Close file handles
                for f in files.values():
                    if hasattr(f, 'close'):
                        f.close()
        
        raise Exception(f"Failed after {retries} attempts: {last_error}")
    
    def synthesize(self, text: str, retries: int = 3) -> bytes:
        """Synthesize text to audio."""
        if self.config.use_file_upload:
            return self._make_tts_request_file(text, retries)
        else:
            return self._make_tts_request_json(text, retries)
    
    def _load_state(self, state_path: Path):
        """Load processing state."""
        if state_path.exists():
            with open(state_path, 'r') as f:
                loaded = json.load(f)
                # Verify config matches
                if loaded.get("config") == self.config.to_dict():
                    self.state = loaded
                    # Convert string paths back to Path objects
                    self.state["output_files"] = [Path(p) for p in self.state.get("output_files", [])]
                    print(f"📂 Resuming: {len(self.state['processed_indices'])} done, "
                          f"{len(self.state['failed_indices'])} failed")
                else:
                    print("⚠️  Config mismatch, starting fresh")
    
    def _save_state(self):
        """Save processing state."""
        if self.state_file:
            # Convert Path objects to strings for JSON
            state_to_save = self.state.copy()
            state_to_save["output_files"] = [str(p) for p in state_to_save["output_files"]]
            with open(self.state_file, 'w') as f:
                json.dump(state_to_save, f, indent=2, cls=PathEncoder)
    
    def _split_text(self, text: str, mode: str = "line", max_chars: Optional[int] = None) -> List[str]:
        """
        Split text into processing chunks.
        
        Modes:
        - line: Split by newlines
        - paragraph: Split by double newlines  
        - sentence: Split by sentence boundaries
        - chunk: Fixed character chunks (respects word boundaries)
        """
        if mode == "line":
            lines = [line.strip() for line in text.split('\n')]
            chunks = [line for line in lines if line]
            
        elif mode == "paragraph":
            paras = text.split('\n\n')
            chunks = [p.strip() for p in paras if p.strip()]
            
        elif mode == "sentence":
            import re
            # Simple sentence splitting
            sentences = re.split(r'(?<=[.!?])\s+', text)
            chunks = [s.strip() for s in sentences if s.strip()]
            
        elif mode == "chunk":
            if not max_chars:
                max_chars = 300  # MioTTS default max
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
        
        # Filter by max length if specified
        if max_chars:
            valid_chunks = []
            for chunk in chunks:
                if len(chunk) <= max_chars:
                    valid_chunks.append(chunk)
                else:
                    # Further split long chunks
                    sub_chunks = self._split_text(chunk, "chunk", max_chars)
                    valid_chunks.extend(sub_chunks)
            chunks = valid_chunks
        
        return chunks
    
    def _concatenate_wavs(self, wav_files: List[Path], output_path: Path, 
                          add_silence_ms: int = 0):
        """Concatenate WAV files with optional silence between them."""
        if not wav_files:
            return
        
        # Read first file for parameters
        with wave.open(str(wav_files[0]), 'rb') as w:
            params = w.getparams()
            frames = [w.readframes(w.getnframes())]
        
        # Read remaining files
        for wav_file in wav_files[1:]:
            with wave.open(str(wav_file), 'rb') as w:
                frames.append(w.readframes(w.getnframes()))
                # Add silence if requested
                if add_silence_ms > 0:
                    silence_frames = b'\x00' * int(params.framerate * params.sampwidth * add_silence_ms / 1000)
                    frames.append(silence_frames)
        
        # Write output
        with wave.open(str(output_path), 'wb') as w:
            w.setparams(params)
            for frame in frames:
                w.writeframes(frame)
    
    def process_file(
        self,
        input_path: Path,
        output_dir: Path,
        split_mode: str = "line",
        max_line_length: Optional[int] = None,
        concatenate: bool = True,
        add_silence_ms: int = 0,
        delete_chunks: bool = False,
        on_error: str = "prompt"  # "prompt", "skip", "stop"
    ) -> Optional[Path]:
        """
        Process a text file through TTS.
        
        Args:
            input_path: Input text file
            output_dir: Output directory
            split_mode: How to split text
            max_line_length: Maximum characters per chunk
            concatenate: Join output into single file
            add_silence_ms: Milliseconds of silence between chunks
            delete_chunks: Remove individual files after concatenation
            on_error: How to handle errors ("prompt", "skip", "stop")
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        job_name = input_path.stem
        self.state_file = output_dir / f".{job_name}.state.json"
        chunks_dir = output_dir / f"{job_name}_chunks"
        chunks_dir.mkdir(exist_ok=True)
        
        # Load previous state
        self._load_state(self.state_file)
        
        # Read input
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split into chunks
        chunks = self._split_text(text, split_mode, max_line_length)
        total = len(chunks)
        
        if total == 0:
            print("⚠️  No text to process")
            return None
        
        # Filter already processed
        processed_set = set(self.state["processed_indices"])
        failed_set = set(self.state["failed_indices"])
        remaining = [i for i in range(total) if i not in processed_set]
        
        print(f"\n📚 {input_path.name}")
        print(f"   Mode: {split_mode} | Chunks: {total} | Remaining: {len(remaining)}")
        print(f"   Output: {output_dir}")
        print(f"   Ref: {self.config.reference_preset or self.config.reference_audio}")
        print()
        
        # Process chunks
        pbar = tqdm(remaining, desc="Synthesizing", initial=len(processed_set), total=total)
        
        try:
            for idx in pbar:
                chunk_text = chunks[idx]
                
                # Skip empty
                if not chunk_text.strip():
                    self.state["processed_indices"].append(idx)
                    continue
                
                # Update progress bar description
                preview = chunk_text[:50].replace('\n', ' ')
                pbar.set_description(f"Chunk {idx}: {preview}...")
                
                try:
                    audio = self.synthesize(chunk_text)
                    
                    # Save
                    chunk_path = chunks_dir / f"{job_name}_{idx:05d}.wav"
                    with open(chunk_path, 'wb') as f:
                        f.write(audio)
                    
                    self.state["processed_indices"].append(idx)
                    self.state["output_files"].append(chunk_path)
                    
                    # Save state periodically
                    if len(self.state["processed_indices"]) % 5 == 0:
                        self._save_state()
                    
                except Exception as e:
                    print(f"\n❌ Error on chunk {idx}: {e}")
                    print(f"   Text: {chunk_text[:100]}...")
                    
                    self.state["failed_indices"].append(idx)
                    self._save_state()
                    
                    if on_error == "stop":
                        raise
                    elif on_error == "prompt":
                        choice = input("  [c]ontinue, [s]kip, [q]uit? ").lower()
                        if choice == 'q':
                            raise KeyboardInterrupt
                        elif choice == 's':
                            continue
                    # else: continue automatically
        
        except KeyboardInterrupt:
            print("\n\n⏸️  Paused. Run again to resume.")
            self._save_state()
            return None
        
        finally:
            pbar.close()
            self._save_state()
        
        # Concatenate results
        final_output = None
        if concatenate and self.state["output_files"]:
            final_output = output_dir / f"{job_name}_complete.wav"
            
            # Sort by index to maintain order
            sorted_files = sorted(
                self.state["output_files"],
                key=lambda p: int(p.stem.split('_')[-1])
            )
            
            print(f"\n🔧 Concatenating {len(sorted_files)} segments...")
            self._concatenate_wavs(sorted_files, final_output, add_silence_ms)
            print(f"✅ Final: {final_output}")
            
            # Cleanup
            if delete_chunks:
                print("🗑️  Cleaning up chunks...")
                shutil.rmtree(chunks_dir)
                self.state_file.unlink(missing_ok=True)
        
        # Summary
        success = len(self.state["processed_indices"])
        failed = len(self.state["failed_indices"])
        print(f"\n📊 Done: {success} success, {failed} failed, {total} total")
        
        return final_output or chunks_dir
    
    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        pattern: str = "*.txt",
        **kwargs
    ) -> List[Path]:
        """Process multiple files in a directory."""
        files = sorted(Path(input_dir).glob(pattern))
        
        if not files:
            print(f"No files matching '{pattern}' in {input_dir}")
            return []
        
        print(f"Found {len(files)} files")
        outputs = []
        
        for file_path in files:
            # Reset state for each file
            self.state = {
                "processed_indices": [],
                "failed_indices": [],
                "output_files": [],
                "config": self.config.to_dict()
            }
            self.state_file = None
            
            result = self.process_file(file_path, output_dir, **kwargs)
            if result:
                outputs.append(result)
            print("\n" + "="*60 + "\n")
        
        return outputs


def list_presets(api_url: str):
    """List available voice presets."""
    try:
        resp = requests.get(f"{api_url}/v1/presets", timeout=10)
        resp.raise_for_status()
        presets = resp.json().get("presets", [])
        print("Available presets:")
        for p in presets:
            print(f"  - {p}")
    except Exception as e:
        print(f"Error fetching presets: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="MioTTS Batch Processor - Convert books to audiobooks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with preset voice
  python miotts_batch.py book.txt -o ./audiobook/
  
  # Use custom reference audio
  python miotts_batch.py book.txt --reference-audio ./my_voice.wav -o ./out/
  
  # Process with paragraph splitting and silence between paragraphs
  python miotts_batch.py book.txt --split paragraph --silence 500 -o ./out/
  
  # Resume interrupted job
  python miotts_batch.py book.txt -o ./out/
  
  # Batch process directory with specific voice
  python miotts_batch.py ./chapters/ --preset jp_female -o ./audiobook/
  
  # Enable Best-of-4 for higher quality
  python miotts_batch.py book.txt --best-of-n 4 -o ./out/
  
  # Use file upload API instead of JSON (better for large reference audio)
  python miotts_batch.py book.txt --use-file-upload --reference-audio ./large.wav -o ./out/
        """
    )
    
    # Input/output
    parser.add_argument("input", help="Input text file or directory")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    
    # Voice selection (mutually exclusive)
    voice_group = parser.add_mutually_exclusive_group()
    voice_group.add_argument("--preset", default="en_female", 
                            help="Voice preset (default: en_female)")
    voice_group.add_argument("--reference-audio", type=Path,
                            help="Path to custom reference audio file")
    
    # API configuration
    parser.add_argument("--api-url", default="http://localhost:8001",
                       help="MioTTS API URL")
    parser.add_argument("--use-file-upload", action="store_true",
                       help="Use multipart/form-data API (better for large files)")
    parser.add_argument("--list-presets", action="store_true",
                       help="List available presets and exit")
    
    # Text processing
    parser.add_argument("--split", choices=["line", "paragraph", "sentence", "chunk"],
                       default="line", help="Text splitting mode")
    parser.add_argument("--max-length", type=int,
                       help="Maximum characters per chunk (default: 300)")
    
    # LLM parameters
    llm_group = parser.add_argument_group("LLM Parameters")
    llm_group.add_argument("--temperature", type=float, default=0.8)
    llm_group.add_argument("--top-p", type=float, default=1.0)
    llm_group.add_argument("--max-tokens", type=int, default=700)
    llm_group.add_argument("--repetition-penalty", type=float, default=1.0)
    llm_group.add_argument("--presence-penalty", type=float, default=0.0)
    llm_group.add_argument("--frequency-penalty", type=float, default=0.0)
    
    # Best-of-N
    bon_group = parser.add_argument_group("Best-of-N Quality")
    bon_group.add_argument("--best-of-n", type=int,
                          help="Enable best-of-N with specified N value (1-8)")
    bon_group.add_argument("--best-of-n-lang", default="auto",
                          help="Language for best-of-n evaluation")
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument("--no-concatenate", action="store_true",
                             help="Keep individual chunk files")
    output_group.add_argument("--silence", type=int, default=0,
                             help="Milliseconds of silence between chunks")
    output_group.add_argument("--delete-chunks", action="store_true",
                             help="Delete chunk files after concatenation")
    output_group.add_argument("--on-error", choices=["prompt", "skip", "stop"],
                             default="prompt", help="Error handling mode")
    
    # Control
    control_group = parser.add_argument_group("Control")
    control_group.add_argument("--no-resume", action="store_true",
                              help="Start fresh (ignore previous state)")
    control_group.add_argument("--pattern", default="*.txt",
                              help="File pattern for directory processing")
    
    args = parser.parse_args()
    
    # List presets mode
    if args.list_presets:
        list_presets(args.api_url)
        return
    
    # Build config
    config = TTSConfig(
        api_url=args.api_url,
        reference_preset=args.preset if not args.reference_audio else None,
        reference_audio=args.reference_audio,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        best_of_n_enabled=args.best_of_n is not None,
        best_of_n_n=args.best_of_n or 1,
        best_of_n_language=args.best_of_n_lang,
        use_file_upload=args.use_file_upload
    )
    
    # Initialize processor
    processor = MioTTSBatchProcessor(config)
    
    # Process
    input_path = Path(args.input)
    
    try:
        if input_path.is_file():
            processor.process_file(
                input_path,
                args.output,
                split_mode=args.split,
                max_line_length=args.max_length,
                concatenate=not args.no_concatenate,
                add_silence_ms=args.silence,
                delete_chunks=args.delete_chunks,
                on_error=args.on_error
            )
        elif input_path.is_dir():
            processor.process_directory(
                input_path,
                args.output,
                pattern=args.pattern,
                split_mode=args.split,
                max_line_length=args.max_length,
                concatenate=not args.no_concatenate,
                add_silence_ms=args.silence,
                delete_chunks=args.delete_chunks,
                on_error=args.on_error
            )
        else:
            print(f"Error: Input not found: {input_path}")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()