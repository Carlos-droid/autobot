"""
core/query_engine.py — Motor de ejecución ReAct con autocompactación integrada.
Inspirado en query.ts:queryLoop (línea 241) y QueryEngine.ts:submitMessage (línea 209).

El ReAct pattern: Reasoning → Acting → Observing → repeat.
La IA no "termina" su turno hasta que responde sin tool_calls.

Razón: un agente single-shot pierde la capacidad de auto-corregirse.
Con ReAct, puede leer el resultado de su acción y decidir si iterar.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.compactor import Compactor
from core.memory_manager import MemoryManager
from core.token_utils import estimate_message_tokens

# ── Configuración ──────────────────────────────────────────────────────────

# Auto-retry para truncamiento de output (query.ts línea 164)
MAX_OUTPUT_RECOVERY_LIMIT = 3

# Límite de turnos en un solo ciclo ReAct para evitar loops infinitos
MAX_REACT_TURNS = 15

# Umbral de contexto por defecto (tokens)
DEFAULT_CONTEXT_THRESHOLD = 80_000


@dataclass
class AgentState:
    """Estado mutable del agente durante un ciclo ReAct."""
    session_id: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    total_tokens: int = 0
    turn_count: int = 0
    tools_executed: List[str] = field(default_factory=list)


class QueryEngine:
    """
    Motor de ejecución ReAct que integra compactación y memoria.
    Un QueryEngine por agente; cada run_cycle() es un turno completo.
    """

    def __init__(
        self,
        call_model_fn: Callable[..., Any],
        tools: List[Dict[str, Any]],
        system_prompt: str,
        compactor: Optional[Compactor] = None,
        memory_manager: Optional[MemoryManager] = None,
        context_threshold: int = DEFAULT_CONTEXT_THRESHOLD,
        max_react_turns: int = MAX_REACT_TURNS,
        model_name: str = "claude-sonnet-4-20250514",
        use_tool_calling: bool = True,
    ):
        # Función genérica para llamar al modelo (Anthropic/Ollama/DeepSeek)
        # Firma: call_model_fn(messages, system_prompt, tools) -> response
        self._call_model = call_model_fn
        self.tools = tools
        self.system_prompt = system_prompt
        self.compactor = compactor or Compactor()
        self.memory = memory_manager
        self.context_threshold = context_threshold
        self.max_react_turns = max_react_turns
        self.model_name = model_name
        self.use_tool_calling = use_tool_calling

        # Handlers de herramientas registrados
        self._tool_handlers: Dict[str, Callable[..., str]] = {}

    # ── Registro de Herramientas ───────────────────────────────────────────

    def register_tool(self, name: str, handler: Callable[..., str]) -> None:
        """
        Registra un handler para una herramienta específica.
        Patrón de Tool.ts:findToolByName — lookup por nombre.
        """
        self._tool_handlers[name] = handler

    # ── Ensamblaje de Contexto ─────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        """
        Ensambla el system prompt completo con memoria inyectada.
        Patrón de context.ts:getUserContext + QueryEngine.ts líneas 321-325:
        systemPrompt = base + memory + append.

        El orden importa para prompt caching: secciones estáticas al inicio.
        """
        parts = [self.system_prompt]

        if self.memory:
            parts.append(self.memory.build_system_prompt_section())

        return "\n".join(parts)

    # ── Bucle ReAct Principal ──────────────────────────────────────────────

    def run_cycle(self, task: str) -> Tuple[str, AgentState]:
        """
        Ejecuta un ciclo ReAct completo para una tarea.
        Retorna (respuesta_final, estado).

        Basado en query.ts:queryLoop (líneas 307-end):
        while(true) → compactación → llamada API → tool_calls? → continue/break
        """
        state = AgentState(session_id=f"cycle_{time.time():.0f}")
        state.messages.append({"role": "user", "content": task})
        state.total_tokens = estimate_message_tokens(state.messages)

        full_system_prompt = self._build_system_prompt()
        output_recovery_count = 0

        while state.turn_count < self.max_react_turns:
            state.turn_count += 1

            # ── 1. Compactación Preventiva ─────────────────────────────────
            # Patrón de query.ts líneas 412-468: microcompact → autocompact
            state.messages = self.compactor.compact_if_needed(state.messages)
            state.total_tokens = estimate_message_tokens(state.messages)

            # ── 2. Llamar al Modelo ────────────────────────────────────────
            try:
                response = self._call_model(
                    messages=state.messages,
                    system_prompt=full_system_prompt,
                    tools=self.tools if self.use_tool_calling else None,
                    model=self.model_name,
                )
            except Exception as e:
                error_msg = str(e)
                print(f"❌ [QueryEngine] Error en llamada API: {error_msg}")

                # Auto-retry para max_tokens (query.ts línea 164)
                if "max_tokens" in error_msg.lower() or "max_output" in error_msg.lower():
                    output_recovery_count += 1
                    if output_recovery_count <= MAX_OUTPUT_RECOVERY_LIMIT:
                        print(
                            f"🔄 [QueryEngine] Recovery {output_recovery_count}/{MAX_OUTPUT_RECOVERY_LIMIT}"
                        )
                        state.messages.append({
                            "role": "user",
                            "content": "Continúa exactamente donde lo dejaste. No repitas lo anterior.",
                        })
                        continue
                    else:
                        print("🛑 [QueryEngine] Límite de recovery alcanzado.")

                return f"Error: {error_msg}", state

            # ── 3. Procesar Respuesta ──────────────────────────────────────
            content, tool_calls, stop_reason = self._parse_response(response)

            # Guardar respuesta del asistente en el estado
            assistant_msg: Dict[str, Any] = {"role": "assistant", "content": content or ""}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            state.messages.append(assistant_msg)
            state.total_tokens = estimate_message_tokens(state.messages)

            # ── 4. Señal de Continuación: Acting ───────────────────────────
            # Patrón de query.ts línea 557: if toolUseBlocks → execute → continue
            if tool_calls:
                for tc in tool_calls:
                    result = self._execute_tool(tc)
                    state.tools_executed.append(
                        tc.get("function", {}).get("name", "unknown")
                        if isinstance(tc, dict) else "unknown"
                    )
                    state.messages.append(result)
                state.total_tokens = estimate_message_tokens(state.messages)
                continue

            # ── 5. Auto-retry para stop_reason=max_tokens ──────────────────
            # Patrón de query.ts línea 164: MAX_OUTPUT_TOKENS_RECOVERY_LIMIT
            if stop_reason == "max_tokens":
                output_recovery_count += 1
                if output_recovery_count <= MAX_OUTPUT_RECOVERY_LIMIT:
                    print(
                        f"🔄 [QueryEngine] max_tokens recovery "
                        f"{output_recovery_count}/{MAX_OUTPUT_RECOVERY_LIMIT}"
                    )
                    state.messages.append({
                        "role": "user",
                        "content": "Continúa exactamente donde lo dejaste.",
                    })
                    continue

            # ── 6. Señal de Terminación: Finalizing ────────────────────────
            # Patrón de query.ts: si no hay tool_calls, el agente ha terminado
            print(f"✅ [QueryEngine] Ciclo completado en {state.turn_count} turnos.")
            return content or "", state

        # Safety: límite de turnos alcanzado
        print(f"⚠️  [QueryEngine] Límite de {self.max_react_turns} turnos alcanzado.")
        last_content = ""
        for msg in reversed(state.messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                last_content = msg["content"]
                break
        return last_content, state

    # ── Parsing de Respuesta (Agnóstico al Backend) ────────────────────────

    def _parse_response(
        self, response: Any
    ) -> Tuple[Optional[str], Optional[List[Dict]], Optional[str]]:
        """
        Extrae contenido, tool_calls y stop_reason de la respuesta.
        Soporta formato Anthropic y formato OpenAI-compatible (DeepSeek/Ollama).
        """
        # Formato Anthropic (Messages API)
        if hasattr(response, "content") and isinstance(response.content, list):
            text_parts = []
            tool_calls = []
            for block in response.content:
                if hasattr(block, "type"):
                    if block.type == "text":
                        text_parts.append(block.text)
                    elif block.type == "tool_use":
                        tool_calls.append({
                            "id": block.id,
                            "function": {
                                "name": block.name,
                                "arguments": json.dumps(block.input),
                            },
                        })
            stop_reason = getattr(response, "stop_reason", None)
            return (
                "\n".join(text_parts) if text_parts else None,
                tool_calls if tool_calls else None,
                stop_reason,
            )

        # Formato OpenAI-compatible (DeepSeek, Ollama con tool calling)
        if hasattr(response, "choices") and response.choices:
            msg = response.choices[0].message
            content = getattr(msg, "content", None)
            tool_calls_raw = getattr(msg, "tool_calls", None)
            stop_reason = getattr(response.choices[0], "finish_reason", None)

            tool_calls = None
            if tool_calls_raw:
                tool_calls = []
                for tc in tool_calls_raw:
                    tool_calls.append({
                        "id": getattr(tc, "id", ""),
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    })

            return content, tool_calls, stop_reason

        # Formato dict genérico (respuestas ya parseadas)
        if isinstance(response, dict):
            return (
                response.get("content"),
                response.get("tool_calls"),
                response.get("stop_reason"),
            )

        # Fallback: tratar como texto plano
        return str(response), None, None

    # ── Ejecución de Herramientas ──────────────────────────────────────────

    def _execute_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta una herramienta y devuelve el resultado como mensaje.
        Patrón de toolOrchestration.ts:runTools — lookup handler → execute → format.
        """
        func_info = tool_call.get("function", {})
        name = func_info.get("name", "unknown")
        args_str = func_info.get("arguments", "{}")
        tool_id = tool_call.get("id", "")

        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except json.JSONDecodeError:
            args = {}

        print(f"🔧 [Tool] Ejecutando: {name}")

        handler = self._tool_handlers.get(name)
        if handler:
            try:
                result = handler(**args) if isinstance(args, dict) else handler(args)
            except Exception as e:
                result = f"Error ejecutando {name}: {e}"
                print(f"❌ [Tool] {name} falló: {e}")
        else:
            result = f"Herramienta '{name}' no registrada."
            print(f"⚠️  [Tool] Handler no encontrado para: {name}")

        return {
            "role": "tool",
            "tool_call_id": tool_id,
            "content": str(result),
        }
