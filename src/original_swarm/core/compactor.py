"""
core/compactor.py — Sistema de autocompactación de 3 niveles.
Inspirado en microCompact.ts, autoCompact.ts y compact/prompt.ts de Claude Code.

Nivel 1 (Pruning): Trunca tool outputs masivos — gratis, sin LLM.
Nivel 2 (Summarize): Resume turnos antiguos con modelo económico.
Nivel 3 (KAIROS): Delegado a memory_manager.py (consolidación periódica).

Razón: sin compactación, sesiones largas explotan el presupuesto de tokens
y degradan la calidad del modelo al diluir el contexto reciente.
"""

import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.token_utils import estimate_message_tokens, estimate_tokens

# ── Configuración ──────────────────────────────────────────────────────────

# Umbral de contexto antes de activar compactación (configurable)
CONTEXT_THRESHOLD_TOKENS = 80_000

# Caracteres máximos por tool result antes de truncar (Nivel 1)
# Basado en microCompact.ts: los resultados de grep/ls/cat masivos son los
# principales consumidores de tokens innecesarios.
MAX_TOOL_OUTPUT_CHARS = 2000

# Mensajes protegidos (Sticky Latches): los últimos N mensajes nunca se
# compactan porque contienen el Chain-of-Thought activo del modelo.
# Patrón de compact.ts: preservar la conversación reciente para no romper
# la coherencia del reasoning actual.
STICKY_LATCH_COUNT = 5

# Máximo de intentos de compactación antes de abortar
MAX_COMPACT_FAILURES = 3

# Placeholder inyectado en tool results truncados — mantiene la semántica
# del flujo (el agente sabe que ejecutó la herramienta) sin el costo.
TRUNCATED_MARKER = "[TOOL OUTPUT TRUNCATED: {n} chars omitted to save context]"

# Marker para tool results completamente limpiados (Nivel 2 / time-based)
CLEARED_MARKER = "[Old tool result content cleared]"

# ── Prompt de compactación (Nivel 2) ───────────────────────────────────────
# Adaptado de compact/prompt.ts — simplificado para nuestro caso de uso.

SUMMARIZATION_PROMPT = """Tu tarea es crear un resumen técnico conciso de los siguientes mensajes de una sesión de agente autónomo. Este resumen reemplazará los mensajes originales.

REGLAS:
1. Preserva TODOS los hechos técnicos: nombres de funciones, decisiones de diseño, errores encontrados.
2. Preserva las preferencias del usuario y reglas explícitas.
3. Elimina logs crudos, outputs de herramientas repetitivos y charla sin valor.
4. Sé extremadamente conciso — cada frase debe aportar información irrecuperable.

Mensajes a resumir:
{messages_text}

Responde ÚNICAMENTE con el resumen técnico en texto plano, sin markup adicional."""


class Compactor:
    """Motor de compactación de 3 niveles para gestión de ventana de contexto."""

    def __init__(
        self,
        threshold: int = CONTEXT_THRESHOLD_TOKENS,
        max_tool_chars: int = MAX_TOOL_OUTPUT_CHARS,
        sticky_count: int = STICKY_LATCH_COUNT,
        summarize_fn: Optional[Callable[[str], str]] = None,
    ):
        self.threshold = threshold
        self.max_tool_chars = max_tool_chars
        self.sticky_count = sticky_count
        # Función de summarización inyectada (para testabilidad)
        # Firma: summarize_fn(prompt: str) -> str
        self._summarize_fn = summarize_fn
        self._consecutive_failures = 0

    # ── Nivel 1: Pruning Quirúrgico ────────────────────────────────────────

    def prune_tool_results(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Trunca tool outputs masivos conservando inicio+final.
        Retorna (mensajes compactados, tokens ahorrados).

        Inspirado en microCompact.ts:maybeTimeBasedMicrocompact (líneas 446-529):
        itera tool results, calcula tokens, reemplaza con marker si exceden umbral.
        """
        compacted = []
        tokens_saved = 0

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Solo compactamos resultados de herramientas con contenido largo
            if role in ("tool", "function") and isinstance(content, str):
                if len(content) > self.max_tool_chars:
                    original_tokens = estimate_tokens(content)
                    half = self.max_tool_chars // 2
                    truncated = (
                        f"{content[:half]}\n\n"
                        f"... {TRUNCATED_MARKER.format(n=len(content) - self.max_tool_chars)} ...\n\n"
                        f"{content[-half:]}"
                    )
                    new_msg = {**msg, "content": truncated}
                    compacted.append(new_msg)
                    tokens_saved += original_tokens - estimate_tokens(truncated)
                    continue

            # Formato multi-bloque (arrays de tool_result)
            if role == "user" and isinstance(content, list):
                new_content = []
                touched = False
                for block in content:
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "tool_result"
                        and isinstance(block.get("content", ""), str)
                        and len(block["content"]) > self.max_tool_chars
                    ):
                        original_tokens = estimate_tokens(block["content"])
                        half = self.max_tool_chars // 2
                        truncated = (
                            f"{block['content'][:half]}\n\n"
                            f"... {TRUNCATED_MARKER.format(n=len(block['content']) - self.max_tool_chars)} ...\n\n"
                            f"{block['content'][-half:]}"
                        )
                        new_content.append({**block, "content": truncated})
                        tokens_saved += original_tokens - estimate_tokens(truncated)
                        touched = True
                    else:
                        new_content.append(block)
                if touched:
                    compacted.append({**msg, "content": new_content})
                    continue

            compacted.append(msg)

        if tokens_saved > 0:
            print(f"🧹 [Compactor L1] Pruning completado. Tokens ahorrados: ~{tokens_saved}")

        return compacted, tokens_saved

    # ── Nivel 2: Colapso de Turnos (Summarization) ──────────────────────────

    def summarize_old_turns(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Resume los turnos más antiguos usando un modelo económico.
        Protege los últimos `sticky_count` mensajes (Sticky Latches).

        Patrón de compact/prompt.ts: genera un resumen estructurado y
        reemplaza los mensajes comprimibles por un solo mensaje de sistema.
        """
        if not self._summarize_fn:
            print("⚠️  [Compactor L2] Sin función de summarización configurada. Saltando.")
            return messages

        if len(messages) <= self.sticky_count + 2:
            # No hay suficientes mensajes para comprimir
            return messages

        # Separar mensajes protegidos de comprimibles
        protected = messages[-self.sticky_count :]
        compressible = messages[: -self.sticky_count]

        # Condensar a texto para el prompt de summarización
        lines = []
        for msg in compressible:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            if isinstance(content, list):
                content = json.dumps(content, ensure_ascii=False)[:500]
            elif isinstance(content, str) and len(content) > 500:
                content = content[:500] + "..."
            lines.append(f"[{role}]: {content}")

        messages_text = "\n".join(lines)
        prompt = SUMMARIZATION_PROMPT.format(messages_text=messages_text)

        try:
            summary = self._summarize_fn(prompt)
            self._consecutive_failures = 0
        except Exception as e:
            self._consecutive_failures += 1
            print(f"❌ [Compactor L2] Summarización falló ({self._consecutive_failures}/{MAX_COMPACT_FAILURES}): {e}")
            if self._consecutive_failures >= MAX_COMPACT_FAILURES:
                print("🛑 [Compactor L2] Circuit breaker activado — no más reintentos esta sesión.")
            return messages

        # Construir resumen como mensaje de sistema
        summary_msg = {
            "role": "system",
            "content": f"[CONTEXT SUMMARY — {len(compressible)} mensajes compactados]\n\n{summary}",
        }

        result = [summary_msg] + protected
        old_tokens = estimate_message_tokens(messages)
        new_tokens = estimate_message_tokens(result)
        print(
            f"📉 [Compactor L2] Colapso completado. "
            f"{old_tokens} → {new_tokens} tokens ({len(compressible)} msgs → 1 resumen)"
        )
        return result

    # ── Orquestador de Compactación ────────────────────────────────────────

    def compact_if_needed(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Punto de entrada principal. Aplica niveles progresivamente.
        Replicado de autoCompact.ts:autoCompactIfNeeded (líneas 241-351):
        chequear threshold → pruning → si insuficiente → summarize.
        """
        current_tokens = estimate_message_tokens(messages)

        if current_tokens < self.threshold:
            return messages

        print(
            f"⚠️  [Compactor] Contexto en {current_tokens} tokens "
            f"(threshold: {self.threshold}). Iniciando compactación..."
        )

        # Nivel 1: Pruning
        messages, saved = self.prune_tool_results(messages)
        current_tokens -= saved

        if current_tokens < self.threshold:
            return messages

        # Nivel 2: Summarization (si circuit breaker no está activo)
        if self._consecutive_failures < MAX_COMPACT_FAILURES:
            messages = self.summarize_old_turns(messages)

        return messages
