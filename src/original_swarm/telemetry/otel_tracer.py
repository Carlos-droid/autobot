"""
OpenTelemetry tracer module para el stack de agentes autónomos.

Implementa la capa de telemetría descrita en Mulian et al. 2026
(AgentFixer, IBM Research) y referenciada en el Related Work como
el estándar OpenTelemetry con spans LLM-específicos.

Arquitectura de spans jerárquicos:
  agent_run
    └── agent_step
          ├── llm_call   (provider, model, tokens, latency)
          ├── tool_call  (tool_name, exit_code, duration_ms)
          └── checkpoint (failure_type, severity, directive_type)

Los spans se exportan a:
  1. JSON log estructurado en disco (siempre activo).
  2. Console OTLP (opcional, para debugging en tiempo real).

El diseño es agnóstico de framework — se integra con AutoAgent,
AG2-Coder v6, y el Framework Inmune sin acoplamientos directos.
"""

import json
import logging
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Generator
from typing import Iterator

logger = logging.getLogger(__name__)


class SpanKind(str, Enum):
    """Tipo de span según la jerarquía del agente."""

    AGENT_RUN = "agent_run"
    AGENT_STEP = "agent_step"
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    CHECKPOINT = "checkpoint"
    VALIDATION = "validation"
    DREAMING = "dreaming"


class SpanStatus(str, Enum):
    """Estado final del span."""

    OK = "OK"
    ERROR = "ERROR"
    UNSET = "UNSET"


class IssueSeverity(str, Enum):
    """
    Niveles de severidad según el paper AgentFixer (Mulian et al. 2026).

    CRITICAL  → syntax errors, schema violations (bloquean el pipeline)
    MODERATE  → reasoning mismatches, format violations, inconsistencies
    MINOR     → instruction adherence, coverage gaps, token anomalies
    """

    CRITICAL = "CRITICAL"
    MODERATE = "MODERATE"
    MINOR = "MINOR"


@dataclass
class SpanAttributes:
    """Atributos estructurados de un span de telemetría."""

    # Identidad
    span_id: str = ""
    trace_id: str = ""
    parent_span_id: str = ""
    span_kind: SpanKind = SpanKind.AGENT_RUN
    name: str = ""

    # Timing
    start_time_iso: str = ""
    end_time_iso: str = ""
    duration_ms: float = 0.0

    # Estado
    status: SpanStatus = SpanStatus.UNSET
    error_message: str = ""

    # Atributos LLM (para llm_call spans)
    llm_provider: str = ""
    llm_model: str = ""
    llm_tokens_input: int = 0
    llm_tokens_output: int = 0
    llm_temperature: float = 0.0
    llm_finish_reason: str = ""

    # Atributos de herramienta (para tool_call spans)
    tool_name: str = ""
    tool_exit_code: int = -1
    tool_truncated: bool = False

    # Atributos de checkpoint / validación (paper §3)
    failure_type: str = ""
    issue_severity: str = ""
    checkpoint_phase: str = ""
    directive_type: str = ""
    issues_detected: list[str] = field(default_factory=list)

    # Atributos de agente
    agent_task_id: str = ""
    agent_step_index: int = -1
    agent_objective: str = ""
    agent_iteration: int = -1

    # Metadatos extra (extensible)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """
    Un span de telemetría completo con lifecycle management.

    Los spans son inmutables una vez finalizados (end() llamado).
    """

    attributes: SpanAttributes
    _start_ns: int = field(default=0, repr=False)
    _ended: bool = field(default=False, repr=False)

    def set_attribute(self, key: str, value: Any) -> None:
        """
        Añadir o actualizar un atributo del span.

        Args:
            key: Nombre del atributo (campo de SpanAttributes o extra key).
            value: Valor del atributo.
        """
        if hasattr(self.attributes, key):
            object.__setattr__(self.attributes, key, value)
        else:
            self.attributes.extra[key] = value

    def set_status(self, status: SpanStatus, message: str = "") -> None:
        """
        Establecer el estado final del span.

        Args:
            status: SpanStatus (OK / ERROR / UNSET).
            message: Mensaje de error si status es ERROR.
        """
        self.attributes.status = status
        if message:
            self.attributes.error_message = message

    def record_exception(self, exc: Exception) -> None:
        """
        Registrar una excepción en el span.

        Args:
            exc: La excepción capturada.
        """
        self.attributes.status = SpanStatus.ERROR
        self.attributes.error_message = (
            f"{type(exc).__name__}: {exc}\n"
            f"{traceback.format_exc()[-500:]}"
        )

    def end(self) -> None:
        """Finalizar el span y calcular la duración."""
        if self._ended:
            return
        end_ns = time.monotonic_ns()
        now_iso = datetime.now(timezone.utc).isoformat()
        self.attributes.end_time_iso = now_iso
        self.attributes.duration_ms = (end_ns - self._start_ns) / 1_000_000
        if self.attributes.status == SpanStatus.UNSET:
            self.attributes.status = SpanStatus.OK
        self._ended = True

    def to_dict(self) -> dict[str, Any]:
        """Serializar el span a un dict JSON-serializable."""
        d = asdict(self.attributes)
        d.pop("_start_ns", None)
        d.pop("_ended", None)
        # Convertir Enums a strings
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
        return d


class AgentTracer:
    """
    Tracer principal para el stack de agentes autónomos.

    Crea y gestiona spans jerárquicos siguiendo el modelo del paper
    AgentFixer. Los spans se acumulan en memoria y se exportan al
    JSON log estructurado en cada flush() o al finalizar el run.

    Thread-safety: no garantizada para spans concurrentes —
    diseñado para el modelo single-thread de AutoAgent.

    Example:
        >>> tracer = AgentTracer(log_dir=Path("./logs"))
        >>> with tracer.start_agent_run("task-001", "Refactorizar auth") as run_span:
        ...     with tracer.start_llm_call("ollama", "llama3.1:8b") as llm_span:
        ...         response = llm.complete(messages)
        ...         tracer.record_llm_result(llm_span, response)
        ...     with tracer.start_tool_call("run_pytest") as tool_span:
        ...         result = executor.execute(call)
        ...         tracer.record_tool_result(tool_span, result)
        >>> tracer.flush()
    """

    def __init__(
        self,
        log_dir: Path | None = None,
        service_name: str = "auto-agent",
        enable_console: bool = False,
    ) -> None:
        """
        Inicializar el tracer con configuración de exportación.

        Args:
            log_dir: Directorio donde se escriben los JSON logs.
                     Default: ./logs/telemetry/
            service_name: Nombre del servicio para los spans.
            enable_console: Si True, imprime spans a stdout también.
        """
        self._service_name = service_name
        self._enable_console = enable_console
        self._spans: list[Span] = []
        self._active_trace_id: str = ""
        self._active_run_span_id: str = ""
        self._span_stack: list[str] = []  # Parent ID stack

        self._log_dir = log_dir or Path("./logs/telemetry")
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._session_log = (
            self._log_dir
            / f"trace_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jsonl"
        )

    # ── Span factories ─────────────────────────────────────────────────

    @contextmanager
    def start_agent_run(
        self,
        task_id: str,
        objective: str,
    ) -> Generator[Span, None, None]:
        """
        Context manager para un run completo del agente.

        Establece el trace_id raíz para todos los spans hijos.

        Args:
            task_id: ID único de la tarea.
            objective: Objetivo en lenguaje natural.

        Yields:
            Span del agent_run.
        """
        self._active_trace_id = self._new_id()
        span = self._new_span(SpanKind.AGENT_RUN, f"agent_run:{task_id}")
        span.attributes.agent_task_id = task_id
        span.attributes.agent_objective = objective[:200]
        self._active_run_span_id = span.attributes.span_id
        self._span_stack.append(span.attributes.span_id)

        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            raise
        finally:
            self._span_stack.pop()
            span.end()
            self._record(span)

    @contextmanager
    def start_agent_step(
        self,
        step_index: int,
        description: str,
        iteration: int = 0,
    ) -> Generator[Span, None, None]:
        """
        Context manager para un paso individual del loop del agente.

        Args:
            step_index: Índice 0-based del paso en el plan.
            description: Descripción del paso.
            iteration: Número de iteración del loop.

        Yields:
            Span del agent_step.
        """
        span = self._new_span(
            SpanKind.AGENT_STEP,
            f"agent_step:{step_index}",
        )
        span.attributes.agent_step_index = step_index
        span.attributes.agent_objective = description[:200]
        span.attributes.agent_iteration = iteration
        self._span_stack.append(span.attributes.span_id)

        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            raise
        finally:
            self._span_stack.pop()
            span.end()
            self._record(span)

    @contextmanager
    def start_llm_call(
        self,
        provider: str,
        model: str,
        temperature: float = 0.1,
    ) -> Generator[Span, None, None]:
        """
        Context manager para una llamada al LLM backend.

        Captura provider, model, tokens (input+output), latency, y
        finish_reason — los atributos que el paper usa para análisis
        de cost/latency anomalies (LOW checkpoint).

        Args:
            provider: Nombre del provider (nim / huggingface / ollama).
            model: Nombre del modelo usado.
            temperature: Temperatura de sampling.

        Yields:
            Span del llm_call.
        """
        span = self._new_span(SpanKind.LLM_CALL, f"llm_call:{provider}/{model}")
        span.attributes.llm_provider = provider
        span.attributes.llm_model = model
        span.attributes.llm_temperature = temperature
        self._span_stack.append(span.attributes.span_id)

        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            raise
        finally:
            self._span_stack.pop()
            span.end()
            self._record(span)

    @contextmanager
    def start_tool_call(self, tool_name: str) -> Generator[Span, None, None]:
        """
        Context manager para la ejecución de una herramienta CLI.

        Args:
            tool_name: Nombre de la herramienta (run_pytest, run_mypy, etc.).

        Yields:
            Span del tool_call.
        """
        span = self._new_span(SpanKind.TOOL_CALL, f"tool_call:{tool_name}")
        span.attributes.tool_name = tool_name
        self._span_stack.append(span.attributes.span_id)

        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            raise
        finally:
            self._span_stack.pop()
            span.end()
            self._record(span)

    @contextmanager
    def start_checkpoint(
        self,
        failure_type: str,
        severity: IssueSeverity,
        phase: str,
    ) -> Generator[Span, None, None]:
        """
        Context manager para la detección de un checkpoint AgentFixer.

        Registra el failure_type, severity (CRITICAL/MODERATE/MINOR),
        y la fase del pipeline (PRE_FLIGHT/EXECUTION/OUTPUT_VALIDATION).

        Args:
            failure_type: Nombre del FailureType detectado.
            severity: Nivel de severidad según el paper.
            phase: Fase del pipeline donde se detectó.

        Yields:
            Span del checkpoint.
        """
        span = self._new_span(
            SpanKind.CHECKPOINT, f"checkpoint:{failure_type}"
        )
        span.attributes.failure_type = failure_type
        span.attributes.issue_severity = severity.value
        span.attributes.checkpoint_phase = phase
        self._span_stack.append(span.attributes.span_id)

        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            raise
        finally:
            self._span_stack.pop()
            span.end()
            self._record(span)

    @contextmanager
    def start_validation(self, validator_name: str) -> Generator[Span, None, None]:
        """
        Context manager para el Validator de AG2-Coder.

        Args:
            validator_name: Nombre del validador (pytest/mypy/ruff/bandit).

        Yields:
            Span del validation.
        """
        span = self._new_span(SpanKind.VALIDATION, f"validation:{validator_name}")
        span.set_attribute("tool_name", validator_name)
        self._span_stack.append(span.attributes.span_id)

        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            raise
        finally:
            self._span_stack.pop()
            span.end()
            self._record(span)

    # ── Result recorders ───────────────────────────────────────────────

    def record_llm_result(self, span: Span, response: Any) -> None:
        """
        Registrar el resultado de una LLM call en el span.

        Compatible con LLMResponse de AutoAgent.

        Args:
            span: El span del llm_call activo.
            response: LLMResponse con tokens y finish_reason.
        """
        span.attributes.llm_tokens_input = getattr(response, "input_tokens", 0)
        span.attributes.llm_tokens_output = getattr(response, "output_tokens", 0)
        span.attributes.llm_finish_reason = getattr(response, "finish_reason", "stop")
        span.set_status(SpanStatus.OK)

    def record_tool_result(self, span: Span, result: Any) -> None:
        """
        Registrar el resultado de una tool_call en el span.

        Compatible con ToolResult de AutoAgent.

        Args:
            span: El span del tool_call activo.
            result: ToolResult con exit_code y truncated.
        """
        span.attributes.tool_exit_code = getattr(result, "exit_code", -1)
        span.attributes.tool_truncated = getattr(result, "truncated", False)
        span.attributes.duration_ms = getattr(result, "duration_ms", 0)
        succeeded = getattr(result, "succeeded", result.exit_code == 0)
        span.set_status(SpanStatus.OK if succeeded else SpanStatus.ERROR)

    def record_checkpoint_result(
        self,
        span: Span,
        failures: list[str],
        directive: str = "",
    ) -> None:
        """
        Registrar el resultado de un checkpoint en el span.

        Args:
            span: El span del checkpoint activo.
            failures: Lista de failure_type strings detectados.
            directive: Tipo de directiva generada (REPAIR_DIRECTIVE, etc.).
        """
        span.attributes.issues_detected = failures
        span.attributes.directive_type = directive
        status = SpanStatus.ERROR if failures else SpanStatus.OK
        span.set_status(status)

    # ── Flush y export ─────────────────────────────────────────────────

    def flush(self) -> Path:
        """
        Escribir todos los spans acumulados al JSON log.

        Cada span se escribe como una línea JSONL (JSON Lines format)
        para facilitar el análisis con herramientas como jq o pandas.

        Returns:
            Path al archivo de log generado.
        """
        with self._session_log.open("a", encoding="utf-8") as fh:
            for span in self._spans:
                fh.write(json.dumps(span.to_dict(), ensure_ascii=False) + "\n")
        count = len(self._spans)
        self._spans.clear()
        logger.info(f"[Telemetry] Flushed {count} spans → {self._session_log}")
        return self._session_log

    def get_session_log_path(self) -> Path:
        """Retornar la ruta del log de sesión actual."""
        return self._session_log

    def get_buffered_spans(self) -> list[Span]:
        """Retornar los spans en buffer sin flusear (para tests)."""
        return list(self._spans)

    # ── Private helpers ────────────────────────────────────────────────

    def _new_span(self, kind: SpanKind, name: str) -> Span:
        """
        Crear un nuevo span con IDs y timestamp.

        Args:
            kind: Tipo de span.
            name: Nombre descriptivo del span.

        Returns:
            Span inicializado con trace_id heredado y parent_id del stack.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        span_id = self._new_id()
        parent_id = self._span_stack[-1] if self._span_stack else ""

        attrs = SpanAttributes(
            span_id=span_id,
            trace_id=self._active_trace_id or self._new_id(),
            parent_span_id=parent_id,
            span_kind=kind,
            name=f"{self._service_name}/{name}",
            start_time_iso=now_iso,
        )
        span = Span(attributes=attrs, _start_ns=time.monotonic_ns())

        if self._enable_console:
            logger.debug(f"[SPAN START] {kind.value}: {name}")

        return span

    def _record(self, span: Span) -> None:
        """
        Acumular un span terminado en el buffer y opcionalmente al console.

        Args:
            span: El span finalizado a registrar.
        """
        self._spans.append(span)
        if self._enable_console:
            logger.debug(
                f"[SPAN END] {span.attributes.span_kind.value}: "
                f"{span.attributes.name} | "
                f"status={span.attributes.status.value} | "
                f"duration={span.attributes.duration_ms:.1f}ms"
            )

    @staticmethod
    def _new_id() -> str:
        """Generar un ID hexadecimal de 16 caracteres."""
        import secrets
        return secrets.token_hex(8)


# ── Singleton global para uso cross-módulo ─────────────────────────────────
# Los módulos de AutoAgent, AG2-Coder, e Immune Framework importan esto.
# Se puede reemplazar con una instancia configurada antes del inicio del run.

_default_tracer: AgentTracer | None = None


def get_tracer(
    log_dir: Path | None = None,
    service_name: str = "auto-agent",
) -> AgentTracer:
    """
    Obtener o crear el tracer singleton global.

    Llamar con parámetros solo la primera vez para configurar.
    Llamadas subsecuentes sin parámetros retornan la instancia existente.

    Args:
        log_dir: Directorio de logs (solo en la primera llamada).
        service_name: Nombre del servicio (solo en la primera llamada).

    Returns:
        AgentTracer singleton configurado.
    """
    global _default_tracer
    if _default_tracer is None:
        _default_tracer = AgentTracer(
            log_dir=log_dir,
            service_name=service_name,
        )
    return _default_tracer


def reset_tracer() -> None:
    """
    Resetear el singleton global (uso exclusivo en tests).

    Permite crear tracers frescos en cada test sin estado compartido.
    """
    global _default_tracer
    _default_tracer = None
