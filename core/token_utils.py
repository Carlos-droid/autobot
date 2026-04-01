"""
core/token_utils.py — Estimación de tokens sin dependencias externas.
Inspirado en services/tokenEstimation.ts y microCompact.ts de Claude Code.

Razón: tiktoken/sentencepiece son pesados; para decisiones de umbral,
una estimación rápida con padding conservador (4/3) es suficiente.
"""

import json
from typing import Any, Dict, List, Union


# ~4 caracteres por token es la heurística estándar para modelos modernos
CHARS_PER_TOKEN = 4
# Padding conservador (4/3 ≈ 1.33x) para no subestimar — replicado de
# microCompact.ts:estimateMessageTokens línea 204
CONSERVATIVE_PADDING = 4 / 3


def estimate_tokens(text: str) -> int:
    """Estimación rápida de tokens a partir de longitud de texto."""
    if not text:
        return 0
    return int(len(text) / CHARS_PER_TOKEN * CONSERVATIVE_PADDING)


def estimate_message_tokens(messages: List[Dict[str, Any]]) -> int:
    """Suma conservadora de tokens de un historial completo de mensajes."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            # Formato multi-bloque (tool_result arrays, etc.)
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        total += estimate_tokens(block.get("text", ""))
                    elif block.get("type") == "tool_result":
                        total += estimate_tokens(block.get("content", ""))
                    elif block.get("type") == "tool_use":
                        total += estimate_tokens(
                            block.get("name", "") + json.dumps(block.get("input", {}))
                        )
                    else:
                        total += estimate_tokens(json.dumps(block))
                elif isinstance(block, str):
                    total += estimate_tokens(block)

        # Contabilizar tool_calls embebidos (formato OpenAI)
        tool_calls = msg.get("tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                fn = tc if isinstance(tc, dict) else {}
                func = fn.get("function", {})
                total += estimate_tokens(
                    func.get("name", "") + func.get("arguments", "")
                )

    return total


def is_above_threshold(messages: List[Dict[str, Any]], threshold: int) -> bool:
    """Chequeo rápido de si el historial excede el umbral configurado."""
    return estimate_message_tokens(messages) >= threshold
