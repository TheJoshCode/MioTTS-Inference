import argparse
import logging
import os

import uvicorn


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    return value


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _set_env_if(name: str, value: str | None) -> None:
    if value is None:
        return
    os.environ[name] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MioTTS API server")

    parser.add_argument("--host", default=_env("MIOTTS_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=_env_int("MIOTTS_PORT", 8001))
    parser.add_argument("--reload", action="store_true", default=_env_bool("MIOTTS_RELOAD", False))
    parser.add_argument("--log-level", default=_env("MIOTTS_LOG_LEVEL", "info"))

    parser.add_argument(
        "--llm-base-url", default=_env("MIOTTS_LLM_BASE_URL", "http://localhost:8000/v1")
    )
    parser.add_argument("--llm-api-key", default=_env("MIOTTS_LLM_API_KEY"))
    parser.add_argument("--llm-model", default=_env("MIOTTS_LLM_MODEL"))
    parser.add_argument(
        "--llm-timeout", type=float, default=_env_float("MIOTTS_LLM_TIMEOUT", 120.0)
    )

    parser.add_argument(
        "--codec-model", default=_env("MIOTTS_CODEC_MODEL", "Aratako/MioCodec-25Hz-24kHz")
    )
    parser.add_argument("--device", default=_env("MIOTTS_DEVICE"))

    parser.add_argument("--presets-dir", default=_env("MIOTTS_PRESETS_DIR", "presets"))

    parser.add_argument(
        "--max-text-length", type=int, default=_env_int("MIOTTS_MAX_TEXT_LENGTH", 300)
    )
    parser.add_argument(
        "--max-reference-mb", type=int, default=_env_int("MIOTTS_MAX_REFERENCE_MB", 20)
    )

    parser.add_argument("--allowed-audio-exts", default=_env("MIOTTS_ALLOWED_AUDIO_EXTS"))

    parser.add_argument(
        "--best-of-n-enabled",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("MIOTTS_BEST_OF_N_ENABLED", False),
    )
    parser.add_argument(
        "--best-of-n-default", type=int, default=_env_int("MIOTTS_BEST_OF_N_DEFAULT", 1)
    )
    parser.add_argument("--best-of-n-max", type=int, default=_env_int("MIOTTS_BEST_OF_N_MAX", 8))
    parser.add_argument("--best-of-n-language", default=_env("MIOTTS_BEST_OF_N_LANGUAGE", "auto"))

    parser.add_argument(
        "--asr-model",
        default=_env("MIOTTS_ASR_MODEL", "openai/whisper-large-v3-turbo"),
    )
    parser.add_argument("--asr-device", default=_env("MIOTTS_ASR_DEVICE"))
    parser.add_argument("--asr-compute-type", default=_env("MIOTTS_ASR_COMPUTE_TYPE"))
    parser.add_argument("--asr-batch-size", type=int, default=_env_int("MIOTTS_ASR_BATCH_SIZE", 0))
    parser.add_argument("--asr-language", default=_env("MIOTTS_ASR_LANGUAGE", "auto"))

    return parser.parse_args()


def _configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("miotts_server").setLevel(level)
    logging.getLogger("miocodec").setLevel(level)


def main() -> None:
    args = parse_args()
    _configure_logging(args.log_level)

    _set_env_if("MIOTTS_LLM_BASE_URL", args.llm_base_url)
    _set_env_if("MIOTTS_LLM_API_KEY", args.llm_api_key)
    _set_env_if("MIOTTS_LLM_MODEL", args.llm_model)
    _set_env_if("MIOTTS_LLM_TIMEOUT", str(args.llm_timeout))
    _set_env_if("MIOTTS_CODEC_MODEL", args.codec_model)
    if args.device:
        _set_env_if("MIOTTS_DEVICE", args.device)
    _set_env_if("MIOTTS_PRESETS_DIR", args.presets_dir)
    _set_env_if("MIOTTS_MAX_TEXT_LENGTH", str(args.max_text_length))
    _set_env_if("MIOTTS_MAX_REFERENCE_MB", str(args.max_reference_mb))
    if args.allowed_audio_exts:
        _set_env_if("MIOTTS_ALLOWED_AUDIO_EXTS", args.allowed_audio_exts)

    _set_env_if("MIOTTS_BEST_OF_N_ENABLED", "true" if args.best_of_n_enabled else "false")
    _set_env_if("MIOTTS_BEST_OF_N_DEFAULT", str(args.best_of_n_default))
    _set_env_if("MIOTTS_BEST_OF_N_MAX", str(args.best_of_n_max))
    _set_env_if("MIOTTS_BEST_OF_N_LANGUAGE", args.best_of_n_language)

    _set_env_if("MIOTTS_ASR_MODEL", args.asr_model)
    if args.asr_device:
        _set_env_if("MIOTTS_ASR_DEVICE", args.asr_device)
    if args.asr_compute_type:
        _set_env_if("MIOTTS_ASR_COMPUTE_TYPE", args.asr_compute_type)
    _set_env_if("MIOTTS_ASR_BATCH_SIZE", str(args.asr_batch_size))
    _set_env_if("MIOTTS_ASR_LANGUAGE", args.asr_language)

    uvicorn.run(
        "miotts_server.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
