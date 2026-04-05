"""
Instrumentation module — telemetría para AutoAgent, AG2-Coder e Immune Framework.

Provee decoradores y wrappers que añaden spans OpenTelemetry a los
métodos existentes sin modificar su lógica de negocio.

Patrón: wrap-and-delegate (sin herencia, sin monkey-patching global).
Cada wrapper crea el span, delega al original, registra el resultado.

Según el paper AgentFixer (§2 Related Work):
  "OpenTelemetry gained LLM- and agent-specific spans and attributes
  to standardize telemetry across heterogeneous providers."
"""

import functools
import logging
from pathlib import Path
from typing import Any
from typing import Callable
from typing import TypeVar

from telemetry.otel_tracer import AgentTracer
from telemetry.otel_tracer import IssueSeverity
from telemetry.otel_tracer import SpanStatus
from telemetry.otel_tracer import get_tracer

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# ── Decoradores reutilizables ──────────────────────────────────────────────


def trace_llm_call(provider: str, model_attr: str = "model") -> Callable[[F], F]:
    """
    Decorador para instrumentar métodos que hacen LLM calls.

    Crea un span llm_call con provider y model extraído del resultado.

    Args:
        provider: Nombre del provider LLM (nim / huggingface / ollama).
        model_attr: Nombre del atributo del resultado que contiene el model.

    Returns:
        Decorador que envuelve el método con telemetría.

    Example:
        >>> @trace_llm_call(provider="ollama", model_attr="model")
        ... def complete(self, messages):
        ...     return ollama_client.chat(...)
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            model = "unknown"

            # Intentar extraer el modelo de la instancia (self)
            if args:
                instance = args[0]
                config = getattr(instance, "_config", None)
                if config:
                    model_field = f"{provider}_model"
                    model = getattr(config, model_field, model)

            with tracer.start_llm_call(provider=provider, model=model) as span:
                try:
                    result = func(*args, **kwargs)
                    tracer.record_llm_result(span, result)
                    return result
                except Exception as exc:
                    span.record_exception(exc)
                    raise

        return wrapper  # type: ignore[return-value]
    return decorator


def trace_tool_call(func: F) -> F:
    """
    Decorador para instrumentar ToolExecutor.execute() y execute_raw().

    Crea un span tool_call con el nombre de la herramienta y el resultado.

    Args:
        func: El método a instrumentar (debe recibir ToolCall o command list).

    Returns:
        Método wrapped con telemetría.
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        tracer = get_tracer()
        # Extraer tool_name del primer argumento posicional no-self
        tool_name = "unknown"
        if len(args) >= 2:
            call_arg = args[1]
            if hasattr(call_arg, "tool_name"):
                tool_name = call_arg.tool_name
            elif isinstance(call_arg, list) and call_arg:
                tool_name = str(call_arg[0])

        with tracer.start_tool_call(tool_name=tool_name) as span:
            try:
                result = func(*args, **kwargs)
                tracer.record_tool_result(span, result)
                return result
            except Exception as exc:
                span.record_exception(exc)
                raise

    return wrapper  # type: ignore[return-value]


def trace_checkpoint(severity: IssueSeverity, phase: str) -> Callable[[F], F]:
    """
    Decorador para instrumentar métodos de detección de checkpoints AgentFixer.

    Crea un span checkpoint con failure_type, severity, y directivas generadas.

    Args:
        severity: Nivel de severidad del checkpoint (CRITICAL/MODERATE/MINOR).
        phase: Fase del pipeline (PRE_FLIGHT/OUTPUT_VALIDATION/AUTO_DREAMING).

    Returns:
        Decorador que envuelve el método con telemetría de checkpoint.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            failure_type = func.__name__.replace("_", " ").upper()

            with tracer.start_checkpoint(
                failure_type=failure_type,
                severity=severity,
                phase=phase,
            ) as span:
                try:
                    result = func(*args, **kwargs)
                    # Extraer failures del InspectionResult si aplica
                    failures = []
                    if hasattr(result, "failures_detected"):
                        failures = [f.value for f in result.failures_detected]
                    directive = ""
                    if hasattr(result, "directives") and result.directives:
                        directive = result.directives[0].directive_type

                    tracer.record_checkpoint_result(span, failures, directive)
                    return result
                except Exception as exc:
                    span.record_exception(exc)
                    raise

        return wrapper  # type: ignore[return-value]
    return decorator


# ── Wrappers de clase para instrumentación completa ────────────────────────


class InstrumentedLLMBackend:
    """
    Wrapper del LLMBackend de AutoAgent con telemetría completa.

    Envuelve cada llamada al backend con un span llm_call que captura:
    provider, model, tokens_input, tokens_output, latency, finish_reason.

    Example:
        >>> config = AgentConfig()
        >>> from llm.backend import LLMBackend
        >>> backend = InstrumentedLLMBackend(LLMBackend(config))
        >>> response = backend.complete(messages)  # → span generado
    """

    def __init__(self, backend: Any, tracer: AgentTracer | None = None) -> None:
        """
        Inicializar con el LLMBackend original.

        Args:
            backend: Instancia real de LLMBackend.
            tracer: AgentTracer a usar. Default: singleton global.
        """
        self._backend = backend
        self._tracer = tracer or get_tracer()

    def complete(self, messages: list) -> Any:
        """
        Ejecutar una LLM call con telemetría.

        Args:
            messages: Lista de LLMMessage a enviar al backend.

        Returns:
            LLMResponse del backend original.
        """
        # Determinar qué provider se usará (intentando el primero disponible)
        provider = "unknown"
        model = "unknown"
        config = getattr(self._backend, "_config", None)
        if config and hasattr(config, "llm_provider_priority"):
            first_provider = config.llm_provider_priority[0]
            provider = first_provider.value if hasattr(first_provider, "value") else str(first_provider)
            model_field = f"{provider}_model"
            model = getattr(config, model_field, "unknown")

        with self._tracer.start_llm_call(provider=provider, model=model) as span:
            try:
                response = self._backend.complete(messages)
                self._tracer.record_llm_result(span, response)
                # Actualizar con el provider real del response
                actual_provider = getattr(response, "provider", None)
                if actual_provider:
                    pv = actual_provider.value if hasattr(actual_provider, "value") else str(actual_provider)
                    span.attributes.llm_provider = pv
                return response
            except Exception as exc:
                span.record_exception(exc)
                # Marcar como parsing error si aplica
                if "json" in str(exc).lower() or "parse" in str(exc).lower():
                    span.set_attribute("parsing_error", True)
                raise


class InstrumentedToolExecutor:
    """
    Wrapper del ToolExecutor de AutoAgent con telemetría completa.

    Envuelve cada tool_call con un span que captura:
    tool_name, exit_code, duration_ms, truncated.

    Example:
        >>> config = AgentConfig()
        >>> from tools.executor import ToolExecutor
        >>> executor = InstrumentedToolExecutor(ToolExecutor(config))
        >>> result = executor.execute(call)  # → span generado
    """

    def __init__(self, executor: Any, tracer: AgentTracer | None = None) -> None:
        """
        Inicializar con el ToolExecutor original.

        Args:
            executor: Instancia real de ToolExecutor.
            tracer: AgentTracer a usar. Default: singleton global.
        """
        self._executor = executor
        self._tracer = tracer or get_tracer()

    def execute(self, call: Any) -> Any:
        """Ejecutar un ToolCall con telemetría."""
        tool_name = getattr(call, "tool_name", "unknown")
        with self._tracer.start_tool_call(tool_name) as span:
            try:
                result = self._executor.execute(call)
                self._tracer.record_tool_result(span, result)
                return result
            except Exception as exc:
                span.record_exception(exc)
                raise

    def execute_raw(self, command: list, cwd: Any = None) -> Any:
        """Ejecutar un comando raw con telemetría."""
        tool_name = command[0] if command else "unknown"
        with self._tracer.start_tool_call(tool_name) as span:
            try:
                result = self._executor.execute_raw(command, cwd)
                self._tracer.record_tool_result(span, result)
                return result
            except Exception as exc:
                span.record_exception(exc)
                raise

    def list_tools(self) -> list:
        """Delegar list_tools sin instrumentación (no es una ejecución)."""
        return self._executor.list_tools()


class InstrumentedAgentFixer:
    """
    Wrapper del AgentFixer con telemetría para todos los checkpoints.

    Envuelve inspect_pre_flight, inspect_output, y run_auto_dreaming
    con spans que capturan failure_type, severity, y directivas.

    Example:
        >>> fixer = InstrumentedAgentFixer(AgentFixer(config))
        >>> result = fixer.inspect_pre_flight(call, tools)  # → span
    """

    def __init__(self, fixer: Any, tracer: AgentTracer | None = None) -> None:
        """
        Inicializar con el AgentFixer original.

        Args:
            fixer: Instancia real de AgentFixer.
            tracer: AgentTracer a usar. Default: singleton global.
        """
        self._fixer = fixer
        self._tracer = tracer or get_tracer()

    def inspect_pre_flight(
        self,
        tool_call: Any,
        available_tools: list,
        generated_code: str = "",
    ) -> Any:
        """Inspeccionar pre-vuelo con telemetría CRITICAL."""
        with self._tracer.start_checkpoint(
            failure_type="PRE_FLIGHT_INSPECTION",
            severity=IssueSeverity.CRITICAL,
            phase="PRE_FLIGHT",
        ) as span:
            result = self._fixer.inspect_pre_flight(
                tool_call, available_tools, generated_code
            )
            failures = [f.value for f in result.failures_detected]
            directive = (
                result.directives[0].directive_type if result.directives else ""
            )
            self._tracer.record_checkpoint_result(span, failures, directive)
            if result.should_interrupt:
                span.set_status(SpanStatus.ERROR, f"CRITICAL: {failures}")
            return result

    def inspect_output(
        self,
        tool_result: Any,
        generated_code: str = "",
        step_objective: str = "",
        estimated_tokens: int = 0,
    ) -> Any:
        """Inspeccionar output con telemetría MODERATE."""
        with self._tracer.start_checkpoint(
            failure_type="OUTPUT_INSPECTION",
            severity=IssueSeverity.MODERATE,
            phase="OUTPUT_VALIDATION",
        ) as span:
            result = self._fixer.inspect_output(
                tool_result, generated_code, step_objective, estimated_tokens
            )
            failures = [f.value for f in result.failures_detected]
            self._tracer.record_checkpoint_result(span, failures)
            return result

    def run_auto_dreaming(self, findings_path: Path | None = None) -> Any:
        """Ejecutar Auto-Dreaming con telemetría LOW."""
        with self._tracer.start_checkpoint(
            failure_type="AUTO_DREAMING",
            severity=IssueSeverity.MINOR,
            phase="AUTO_DREAMING",
        ) as span:
            result = self._fixer.run_auto_dreaming(findings_path)
            span.set_attribute("genome_updates", len(result))
            span.set_status(SpanStatus.OK)
            return result

    def reset_session(self) -> None:
        """Delegar sin instrumentación."""
        self._fixer.reset_session()
