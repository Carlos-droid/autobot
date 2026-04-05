"""
Fix Protocol module para el AgentFixer del Framework Inmune.

Implementa el ciclo de 3 pasos que convierte un fallo detectado
en una intervención inyectable en el historial del agente:

  1. ANCLAJE    → Identifica el checkpoint fallido exacto.
  2. RCA        → Explica por qué el razonamiento fue defectuoso.
  3. DIRECTIVA  → Instrucción imperativa para retomar desde estado válido.
"""

import logging
from dataclasses import dataclass
from string import Template
from typing import Any, List, Tuple, Dict

from .checkpoints import CheckpointDefinition, CheckpointPriority, FailureType
from ..core.config import AgentConfig
from ..llm.backend import LLMMessage, LLMBackend

logger = logging.getLogger(__name__)

_FIX_PROMPT_CRITICAL = """\
Eres el QA_Fixer_Agent. Detectaste un fallo CRÍTICO que interrumpe el flujo.
Tu intervención tiene 3 partes exactas. Sé conciso, imperativo y técnico.

## 1. ANCLAJE
Checkpoint fallido: {checkpoint_name}
Tipo de fallo: {failure_type}

## 2. ANÁLISIS DE CAUSA RAÍZ (RCA)
Explica en 1-2 oraciones por qué el razonamiento del agente fue defectuoso.
Contexto del fallo: {failure_context}

## 3. REPAIR_DIRECTIVE
Genera una instrucción imperativa en segunda persona que:
- Empiece con "REPAIR_DIRECTIVE:"
- Especifique EXACTAMENTE qué cambiar (no generalidades).
- Indique desde qué estado válido retomar.
- Evite que el agente repita el mismo error.

Responde ÚNICAMENTE con el texto de la REPAIR_DIRECTIVE. Sin explicaciones adicionales.\
"""

_FIX_PROMPT_MEDIUM = """\
Eres el QA_Fixer_Agent. Detectaste un fallo MEDIO en la salida del Worker.
El Worker debe corregirlo antes de entregar al Router.
Tu intervención tiene 3 partes. Usa un tono reflexivo, no alarmista.

## 1. ANCLAJE
Checkpoint fallido: {checkpoint_name}
Tipo de fallo: {failure_type}

## 2. ANÁLISIS DE CAUSA RAÍZ (RCA)
Explica en 1-2 oraciones el error específico en el output.
Contexto del fallo: {failure_context}

## 3. REFLEXIÓN INTERNA
Genera una instrucción que:
- Empiece con "REFLEXIÓN:"
- Señale exactamente qué está mal en el output (con referencia a líneas si aplica).
- Indique el estándar que debe cumplirse.
- Pida al Worker que genere una versión corregida.

Responde ÚNICAMENTE con el texto de la REFLEXIÓN. Sin explicaciones adicionales.\
"""

_FIX_PROMPT_LOW = """\
Eres el QA_Fixer_Agent operando en modo Auto-Dreaming (batch nocturno).
Detectaste un patrón de bajo impacto a optimizar en el genoma de prompts.
Tu intervención modifica permanentemente las instrucciones para mañana.

## 1. ANCLAJE
Checkpoint: {checkpoint_name}
Patrón detectado: {failure_type}

## 2. ANÁLISIS DE CAUSA RAÍZ (RCA)
Identifica el patrón sistemático que causa esta degradación.
Contexto: {failure_context}

## 3. GENOME_UPDATE
Genera una modificación al system prompt que:
- Empiece con "GENOME_UPDATE:"
- Añada UNA instrucción clara que prevenga este patrón.
- Use un ejemplo negativo (lo que NO hacer) y positivo (lo que SÍ hacer).
- Sea lo suficientemente general para aplicar en futuras tareas.

Responde ÚNICAMENTE con el GENOME_UPDATE. Sin explicaciones adicionales.\
"""

_PRIORITY_PROMPT_MAP: Dict[CheckpointPriority, str] = {
    CheckpointPriority.CRITICAL: _FIX_PROMPT_CRITICAL,
    CheckpointPriority.MEDIUM: _FIX_PROMPT_MEDIUM,
    CheckpointPriority.LOW: _FIX_PROMPT_LOW,
}

@dataclass
class FixDirective:
    failure_type: FailureType
    priority: CheckpointPriority
    checkpoint_name: str
    anchor: str
    rca: str
    directive: str
    directive_type: str
    context: Dict[str, Any]

    def to_injection_message(self) -> LLMMessage:
        content = (
            f"[QA_FIXER_AGENT — {self.priority.value}]\n\n"
            f"**Checkpoint fallido:** {self.checkpoint_name}\n"
            f"**Tipo:** {self.failure_type.value}\n\n"
            f"**Causa Raíz:** {self.rca}\n\n"
            f"{self.directive}"
        )
        return LLMMessage(role="user", content=content)

class FixProtocol:
    def __init__(self, config: AgentConfig) -> None:
        self._config = config
        self._llm = LLMBackend(config)

    def generate(self, checkpoint: CheckpointDefinition, failure_context: Dict[str, Any]) -> FixDirective:
        anchor = self._build_anchor(checkpoint, failure_context)
        context_str = self._format_context(failure_context)
        prompt_template = _PRIORITY_PROMPT_MAP[checkpoint.priority]
        prompt = prompt_template.format(
            checkpoint_name=checkpoint.name,
            failure_type=checkpoint.failure_type.value,
            failure_context=context_str,
        )

        try:
            response = self._llm.complete([LLMMessage(role="user", content=prompt)])
            full_directive = response.content.strip()
        except Exception as exc:
            logger.warning(f"FixProtocol LLM falló: {exc}. Usando template estático.")
            full_directive = self._static_directive(checkpoint, failure_context)

        rca, directive, directive_type = self._parse_directive(full_directive, checkpoint.priority)
        return FixDirective(
            failure_type=checkpoint.failure_type,
            priority=checkpoint.priority,
            checkpoint_name=checkpoint.name,
            anchor=anchor,
            rca=rca,
            directive=directive,
            directive_type=directive_type,
            context=failure_context,
        )

    def generate_batch(self, failures: List[Tuple[CheckpointDefinition, Dict[str, Any]]]) -> List[FixDirective]:
        low_failures = [(cp, ctx) for cp, ctx in failures if cp.priority == CheckpointPriority.LOW]
        directives: List[FixDirective] = []
        for checkpoint, context in low_failures:
            try:
                directive = self.generate(checkpoint, context)
                directives.append(directive)
            except Exception as exc:
                logger.error(f"[FixProtocol batch] Fallo al procesar {checkpoint.name}: {exc}")
        return directives

    @staticmethod
    def _build_anchor(checkpoint: CheckpointDefinition, context: Dict[str, Any]) -> str:
        base = f"Checkpoint '{checkpoint.name}' activado ({checkpoint.phase.value})."
        extras = [f"{k}={v}" for k, v in context.items() if k in ("tool_name", "step", "file", "line", "error")]
        if extras:
            base += f" Contexto: {', '.join(extras)}."
        return base

    @staticmethod
    def _format_context(context: Dict[str, Any]) -> str:
        return "\n".join([f"  {k}: {str(v)[:200]}" for k, v in context.items()]) or "Sin contexto adicional."

    @staticmethod
    def _parse_directive(full_output: str, priority: CheckpointPriority) -> Tuple[str, str, str]:
        type_map = {CheckpointPriority.CRITICAL: "REPAIR_DIRECTIVE", CheckpointPriority.MEDIUM: "REFLEXIÓN", CheckpointPriority.LOW: "GENOME_UPDATE"}
        directive_type = type_map[priority]
        if directive_type + ":" in full_output:
            idx = full_output.index(directive_type + ":")
            directive = full_output[idx:].strip()
            rca = full_output[:idx].strip() or "Ver directiva para detalles del fallo."
        else:
            directive = f"{directive_type}: {full_output.strip()}"
            rca = "Análisis integrado en la directiva."
        return rca, directive, directive_type

    @staticmethod
    def _static_directive(checkpoint: CheckpointDefinition, context: Dict[str, Any]) -> str:
        try:
            tmpl = Template(checkpoint.intervention.replace("{", "${"))
            return tmpl.safe_substitute(context)
        except Exception:
            return f"{checkpoint.intervention.split(':')[0]}: Fallo en {checkpoint.name}. Revisa el contexto: {list(context.keys())}."
