"""
Metrics collector module para el stack de agentes autónomos.

Implementa las métricas cuantitativas descritas en el paper AgentFixer
(Mulian et al. 2026, §5): issue_frequency, detection_rate por tool,
token cost tracking, latency distributions, y parsing error rate.

El paper reportó:
  - 1,940 LLM calls totales analizados
  - 64-88% detection rates por herramienta
  - 38% de fallos causados por parsing errors
  - Alta frecuencia de issues en Input y Output validation

Esta clase captura exactamente esas métricas en tiempo real para
que los logs permitan identificar los mismos patrones en nuestro stack.
"""

import json
import statistics
from collections import defaultdict
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

from telemetry.otel_tracer import IssueSeverity
from telemetry.otel_tracer import SpanKind
from telemetry.otel_tracer import SpanStatus


@dataclass
class ToolMetrics:
    """Métricas acumuladas para una herramienta o checkpoint específico."""

    name: str
    calls: int = 0
    successes: int = 0
    failures: int = 0
    issues_detected: int = 0
    total_duration_ms: float = 0.0
    durations_ms: list[float] = field(default_factory=list)

    @property
    def failure_rate(self) -> float:
        """Tasa de fallos como porcentaje [0-100]."""
        return (self.failures / self.calls * 100) if self.calls > 0 else 0.0

    @property
    def detection_rate(self) -> float:
        """
        Tasa de detección de issues como porcentaje [0-100].

        Equivale al porcentaje reportado por herramienta en el paper
        (Figura 1: 64-88% para la mayoría de herramientas).
        """
        return (self.issues_detected / self.calls * 100) if self.calls > 0 else 0.0

    @property
    def avg_duration_ms(self) -> float:
        """Latencia media en milisegundos."""
        return statistics.mean(self.durations_ms) if self.durations_ms else 0.0

    @property
    def p95_duration_ms(self) -> float:
        """Latencia percentil 95 en milisegundos."""
        if not self.durations_ms:
            return 0.0
        sorted_d = sorted(self.durations_ms)
        idx = int(len(sorted_d) * 0.95)
        return sorted_d[min(idx, len(sorted_d) - 1)]


@dataclass
class LLMMetrics:
    """Métricas acumuladas para un provider/modelo LLM."""

    provider: str
    model: str
    calls: int = 0
    failures: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_duration_ms: float = 0.0
    parsing_errors: int = 0  # JSON decode failures — el 38% del paper

    @property
    def avg_input_tokens(self) -> float:
        """Media de tokens de entrada por llamada."""
        return self.total_input_tokens / self.calls if self.calls > 0 else 0.0

    @property
    def avg_output_tokens(self) -> float:
        """Media de tokens de salida por llamada."""
        return self.total_output_tokens / self.calls if self.calls > 0 else 0.0

    @property
    def parsing_error_rate(self) -> float:
        """
        Tasa de errores de parsing como porcentaje.

        El paper (§1) reportó que parsing errors = 38% de todos los fallos.
        Esta métrica permite verificar si nuestro stack tiene el mismo patrón.
        """
        return (self.parsing_errors / self.failures * 100) if self.failures > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        """Latencia media por llamada en ms."""
        return self.total_duration_ms / self.calls if self.calls > 0 else 0.0


@dataclass
class SessionSummary:
    """
    Resumen completo de una sesión del agente.

    Equivale al análisis por run que el paper hace sobre los 1,940 LLM calls.
    """

    task_id: str
    objective: str
    start_time: str
    end_time: str = ""
    total_steps: int = 0
    completed_steps: int = 0
    total_iterations: int = 0

    # Issue counts por severidad (paper §5 tabla 1)
    critical_issues: int = 0
    moderate_issues: int = 0
    minor_issues: int = 0

    # Top failure types (como Figura 1 del paper)
    top_failure_types: dict[str, int] = field(default_factory=dict)

    # LLM metrics
    total_llm_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_parsing_errors: int = 0

    # Tool metrics
    total_tool_calls: int = 0
    tool_failure_counts: dict[str, int] = field(default_factory=dict)

    # Outcome
    success: bool = False
    correction_rounds: int = 0


class MetricsCollector:
    """
    Recopila y agrega métricas del stack de agentes en tiempo real.

    Lee los spans del AgentTracer y calcula:
    - Issue frequency por herramienta (Figura 1 del paper)
    - Detection rate por checkpoint (64-88% en el paper)
    - Parsing error rate (38% en el paper)
    - Token cost y latency distributions
    - Top failure types por sesión

    Los reportes se escriben en disco como JSON y como texto legible
    para que los logs permitan identificar dónde mejorar el sistema.

    Example:
        >>> tracer = AgentTracer(log_dir=Path("./logs"))
        >>> collector = MetricsCollector(tracer, report_dir=Path("./logs/metrics"))
        >>> # ... ejecución del agente ...
        >>> collector.ingest_session_log(tracer.get_session_log_path())
        >>> report = collector.generate_report("task-001")
        >>> print(report)
    """

    def __init__(
        self,
        report_dir: Path | None = None,
    ) -> None:
        """
        Inicializar el collector con directorio de reportes.

        Args:
            report_dir: Directorio para escribir reportes JSON.
        """
        self._report_dir = report_dir or Path("./logs/metrics")
        self._report_dir.mkdir(parents=True, exist_ok=True)

        # Métricas acumuladas por nombre de herramienta/checkpoint
        self._tool_metrics: dict[str, ToolMetrics] = defaultdict(
            lambda: ToolMetrics(name="unknown")
        )
        self._llm_metrics: dict[str, LLMMetrics] = {}
        self._sessions: dict[str, SessionSummary] = {}

        # Issue tracking
        self._issue_counts: dict[str, int] = defaultdict(int)
        self._severity_counts: dict[str, int] = defaultdict(int)

    def ingest_span(self, span_dict: dict[str, Any]) -> None:
        """
        Procesar un span individual y actualizar las métricas.

        Llamar por cada span del log JSONL para construir el perfil
        de la sesión. Compatible con el formato de AgentTracer.to_dict().

        Args:
            span_dict: Diccionario del span serializado por AgentTracer.
        """
        kind = span_dict.get("span_kind", "")
        duration = float(span_dict.get("duration_ms", 0))
        status = span_dict.get("status", "UNSET")
        succeeded = status == "OK"

        if kind == SpanKind.LLM_CALL.value:
            self._process_llm_span(span_dict, duration, succeeded)

        elif kind == SpanKind.TOOL_CALL.value:
            self._process_tool_span(span_dict, duration, succeeded)

        elif kind == SpanKind.CHECKPOINT.value:
            self._process_checkpoint_span(span_dict, succeeded)

        elif kind == SpanKind.AGENT_RUN.value:
            self._process_run_span(span_dict)

    def ingest_session_log(self, log_path: Path) -> int:
        """
        Leer y procesar todos los spans de un archivo JSONL de sesión.

        Args:
            log_path: Ruta al archivo .jsonl generado por AgentTracer.flush().

        Returns:
            Número de spans procesados.
        """
        if not log_path.exists():
            return 0
        count = 0
        with log_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    span = json.loads(line)
                    self.ingest_span(span)
                    count += 1
                except json.JSONDecodeError:
                    continue
        return count

    def get_tool_report(self) -> list[dict[str, Any]]:
        """
        Generar reporte de métricas por herramienta (equivale Figura 1 del paper).

        Retorna una lista ordenada por detection_rate descendente,
        replicando el análisis de issue frequency del paper.

        Returns:
            Lista de dicts con métricas por herramienta/checkpoint.
        """
        rows: list[dict[str, Any]] = []
        for name, tm in self._tool_metrics.items():
            rows.append({
                "tool_name": name,
                "calls": tm.calls,
                "successes": tm.successes,
                "failures": tm.failures,
                "failure_rate_pct": round(tm.failure_rate, 1),
                "issues_detected": tm.issues_detected,
                "detection_rate_pct": round(tm.detection_rate, 1),
                "avg_duration_ms": round(tm.avg_duration_ms, 1),
                "p95_duration_ms": round(tm.p95_duration_ms, 1),
            })
        return sorted(rows, key=lambda r: r["detection_rate_pct"], reverse=True)

    def get_llm_report(self) -> list[dict[str, Any]]:
        """
        Generar reporte de métricas LLM incluyendo parsing error rate.

        El parsing_error_rate es la métrica clave del paper (38% en producción).

        Returns:
            Lista de dicts con métricas por provider/modelo.
        """
        rows: list[dict[str, Any]] = []
        for key, lm in self._llm_metrics.items():
            rows.append({
                "provider": lm.provider,
                "model": lm.model,
                "total_calls": lm.calls,
                "failures": lm.failures,
                "total_input_tokens": lm.total_input_tokens,
                "total_output_tokens": lm.total_output_tokens,
                "avg_input_tokens": round(lm.avg_input_tokens, 1),
                "avg_output_tokens": round(lm.avg_output_tokens, 1),
                "avg_latency_ms": round(lm.avg_latency_ms, 1),
                "parsing_errors": lm.parsing_errors,
                "parsing_error_rate_pct": round(lm.parsing_error_rate, 1),
            })
        return sorted(rows, key=lambda r: r["total_calls"], reverse=True)

    def get_issue_frequency_report(self) -> dict[str, Any]:
        """
        Generar reporte de frecuencia de issues por tipo y severidad.

        Replica el análisis de Tabla 1 y Figura 1 del paper.

        Returns:
            Dict con counts por failure_type y por severidad.
        """
        total = sum(self._issue_counts.values())
        top_types = sorted(
            self._issue_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return {
            "total_issues": total,
            "by_severity": dict(self._severity_counts),
            "top_failure_types": [
                {
                    "failure_type": ft,
                    "count": count,
                    "pct_of_total": round(count / total * 100, 1) if total > 0 else 0,
                }
                for ft, count in top_types[:10]
            ],
            "parsing_error_concern": (
                "⚠️ Parsing errors >30% — consistent with Mulian et al. 2026 §1"
                if self._issue_counts.get("SCHEMA_VIOLATION", 0) > total * 0.3
                else "✅ Parsing error rate within acceptable bounds"
            ),
        }

    def generate_report(self, task_id: str = "") -> str:
        """
        Generar reporte de texto legible para el log de la sesión.

        Produce el mismo tipo de insight que el paper describe en §5
        para IBM CUGA: recurrent patterns, bottleneck agents, top issues.

        Args:
            task_id: ID de la tarea para personalizar el reporte.

        Returns:
            String multilínea con el reporte completo.
        """
        lines: list[str] = [
            "=" * 60,
            f"TELEMETRY REPORT — {task_id or 'session'}",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "=" * 60,
            "",
            "── LLM CALLS ─────────────────────────────────────",
        ]

        llm_report = self.get_llm_report()
        if llm_report:
            for row in llm_report:
                lines.append(
                    f"  {row['provider']}/{row['model']}: "
                    f"{row['total_calls']} calls | "
                    f"avg {row['avg_latency_ms']:.0f}ms | "
                    f"parsing_errors={row['parsing_errors']} "
                    f"({row['parsing_error_rate_pct']}%)"
                )
        else:
            lines.append("  No LLM calls recorded.")

        lines += [
            "",
            "── TOOL EXECUTION ────────────────────────────────",
        ]
        tool_report = self.get_tool_report()
        if tool_report:
            for row in tool_report[:8]:
                lines.append(
                    f"  {row['tool_name']}: "
                    f"{row['calls']} calls | "
                    f"fail={row['failure_rate_pct']}% | "
                    f"detect={row['detection_rate_pct']}% | "
                    f"p95={row['p95_duration_ms']:.0f}ms"
                )
        else:
            lines.append("  No tool calls recorded.")

        issue_report = self.get_issue_frequency_report()
        lines += [
            "",
            "── ISSUE FREQUENCY (Mulian et al. 2026 §5) ──────",
            f"  Total issues: {issue_report['total_issues']}",
            f"  By severity:  {issue_report['by_severity']}",
            "",
            "  Top failure types:",
        ]
        for entry in issue_report["top_failure_types"][:5]:
            lines.append(
                f"    {entry['failure_type']}: "
                f"{entry['count']} ({entry['pct_of_total']}%)"
            )
        lines += [
            "",
            f"  {issue_report['parsing_error_concern']}",
            "",
            "=" * 60,
        ]

        return "\n".join(lines)

    def save_report(self, task_id: str = "") -> Path:
        """
        Guardar el reporte completo en JSON y texto.

        Args:
            task_id: ID de tarea para el nombre del archivo.

        Returns:
            Path al archivo JSON del reporte.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_path = self._report_dir / f"metrics_{task_id}_{timestamp}.json"

        payload = {
            "task_id": task_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "llm_metrics": self.get_llm_report(),
            "tool_metrics": self.get_tool_report(),
            "issue_frequency": self.get_issue_frequency_report(),
        }
        report_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # También guardar el texto legible
        text_path = report_path.with_suffix(".txt")
        text_path.write_text(self.generate_report(task_id), encoding="utf-8")

        return report_path

    # ── Private processors ─────────────────────────────────────────────

    def _process_llm_span(
        self,
        span: dict[str, Any],
        duration: float,
        succeeded: bool,
    ) -> None:
        """Procesar un span de tipo llm_call."""
        provider = span.get("llm_provider", "unknown")
        model = span.get("llm_model", "unknown")
        key = f"{provider}/{model}"

        if key not in self._llm_metrics:
            self._llm_metrics[key] = LLMMetrics(provider=provider, model=model)

        lm = self._llm_metrics[key]
        lm.calls += 1
        lm.total_input_tokens += span.get("llm_tokens_input", 0)
        lm.total_output_tokens += span.get("llm_tokens_output", 0)
        lm.total_duration_ms += duration

        if not succeeded:
            lm.failures += 1
            # Detectar si fue un parsing error (schema_violation en el mensaje)
            error_msg = span.get("error_message", "").lower()
            if any(p in error_msg for p in ("json", "parse", "schema", "decode")):
                lm.parsing_errors += 1

    def _process_tool_span(
        self,
        span: dict[str, Any],
        duration: float,
        succeeded: bool,
    ) -> None:
        """Procesar un span de tipo tool_call."""
        tool_name = span.get("tool_name", "unknown")
        if tool_name not in self._tool_metrics:
            self._tool_metrics[tool_name] = ToolMetrics(name=tool_name)

        tm = self._tool_metrics[tool_name]
        tm.calls += 1
        tm.total_duration_ms += duration
        tm.durations_ms.append(duration)

        if succeeded:
            tm.successes += 1
        else:
            tm.failures += 1
            tm.issues_detected += 1

    def _process_checkpoint_span(
        self,
        span: dict[str, Any],
        succeeded: bool,
    ) -> None:
        """Procesar un span de tipo checkpoint."""
        failure_type = span.get("failure_type", "unknown")
        severity = span.get("issue_severity", IssueSeverity.MINOR.value)
        issues = span.get("issues_detected", [])
        duration = float(span.get("duration_ms", 0))

        if failure_type not in self._tool_metrics:
            self._tool_metrics[failure_type] = ToolMetrics(name=failure_type)

        tm = self._tool_metrics[failure_type]
        tm.calls += 1
        tm.durations_ms.append(duration)

        if issues:
            tm.issues_detected += len(issues)
            tm.failures += 1
            for issue in issues:
                self._issue_counts[issue] += 1
            self._severity_counts[severity] = (
                self._severity_counts.get(severity, 0) + 1
            )
        else:
            tm.successes += 1

    def _process_run_span(self, span: dict[str, Any]) -> None:
        """Procesar un span de tipo agent_run para el resumen de sesión."""
        task_id = span.get("agent_task_id", "unknown")
        self._sessions[task_id] = SessionSummary(
            task_id=task_id,
            objective=span.get("agent_objective", ""),
            start_time=span.get("start_time_iso", ""),
            end_time=span.get("end_time_iso", ""),
            success=span.get("status") == SpanStatus.OK.value,
        )
