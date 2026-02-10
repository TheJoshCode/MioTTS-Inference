import argparse
from pathlib import Path

import torch
from miocodec import MioCodecModel
from miocodec.util import load_audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MioTTS preset embedding")
    parser.add_argument("--audio", required=True, help="Reference audio path")
    parser.add_argument(
        "--preset-id", required=True, help="Preset id (file name without extension)"
    )
    parser.add_argument("--output-dir", default="presets", help="Output directory")
    parser.add_argument(
        "--model-id", default="Aratako/MioCodec-25Hz-24kHz", help="MioCodec model id"
    )
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.preset_id}.pt"

    codec = MioCodecModel.from_pretrained(args.model_id)
    codec = codec.eval().to(args.device)
    waveform = load_audio(args.audio, sample_rate=codec.config.sample_rate)
    waveform = waveform.to(args.device)

    features = codec.encode(waveform, return_content=False, return_global=True)
    torch.save(features.global_embedding.cpu(), output_path)
    print(f"Saved preset embedding to {output_path}")


if __name__ == "__main__":
    main()
