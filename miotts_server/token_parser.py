from __future__ import annotations

import re
from collections.abc import Iterable

TOKEN_PATTERN = re.compile(r"<\|s_(\d+)\|>")


def parse_speech_tokens(text: str) -> list[int]:
    tokens = [int(value) for value in TOKEN_PATTERN.findall(text)]
    if not tokens:
        raise ValueError("No speech tokens found in LLM output.")
    return tokens


def tokens_to_str(tokens: Iterable[int]) -> str:
    return "".join(f"<|s_{token}|>" for token in tokens)
