"""
Critic module para AG2-Coder v6.

Reemplaza el placeholder `return "Fix errors"` con análisis real
del ValidationReport usando el LLMBackend de AutoAgent.

El Critic lee la salida real de pytest/mypy/ruff/bandit, identifica
los errores concretos, y genera instrucciones específicas para que
el Coder pueda corregirlos en la siguiente iteración.
"""

import logging
from dataclasses import dataclass

from core.config import AgentConfig
from llm.backend import LLMBackend
from llm.backend import LLMMessage
from validation.validator import ValidationReport

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CriticFeedback:
    """
    Feedback estructurado generado por el Critic.

    Attributes:
        has_errors: True si hay errores que corregir.
        error_summary: Resumen legible para el Planner.
        corrective_instructions: Instrucciones específicas para el Coder.
        severity: 'low' | 'medium' | 'high' basado en tipo de error.
        failed_checks: Lista de checks que fallaron.
    """

    has_errors: bool
    error_summary: str
    corrective_instructions: str
    severity: str
    failed_checks: list[str]

    def to_planner_message(self) -> str:
        """Formato para update_plan() del Planner."""
        return f"[{self.severity.upper()}] {self.error_summary}"

    def to_coder_context(self) -> str:
        """Formato para generate_fix() del Coder."""
        return self.corrective_instructions


class Critic:
    """
    Analiza ValidationReport real y genera feedback correctivo con LLM.

    A diferencia del placeholder que retornaba "Fix errors",
    esta implementación:
    - Lee los outputs reales de cada herramienta de validación.
    - Categoriza errores por severidad (lint < tipos < tests < seguridad).
    - Usa el LLM para generar instrucciones de corrección específicas.
    - Prioriza qué corregir primero según impacto.

    Example:
        >>> config = AgentConfig()
        >>> critic = Critic(config)
        >>> report = validator.run_all()
        >>> feedback = critic.analyze(report)
        >>> if feedback.has_errors:
        ...     coder.generate_fix(step, feedback.to_coder_context())
    """

    _MAX_ITERATIONS: int = 5  # Límite anti-loop infinito

    def __init__(self, config: AgentConfig) -> None:
        """
        Inicializar Critic con LLMBackend de AutoAgent.

        Args:
            config: AgentConfig con settings de LLM.
        """
        self._config = config
        self._llm = LLMBackend(config)
        self._iteration_count: int = 0

    def analyze(self, report: ValidationReport) -> CriticFeedback:
        """
        Analizar un ValidationReport y producir feedback accionable.

        Args:
            report: ValidationReport con resultados reales de cada check.

        Returns:
            CriticFeedback con instrucciones específicas de corrección.
        """
        self._iteration_count += 1

        if report.all_passed:
            return CriticFeedback(
                has_errors=False,
                error_summary="Todos los checks pasaron correctamente.",
                corrective_instructions="",
                severity="none",
                failed_checks=[],
            )

        if self._iteration_count >= self._MAX_ITERATIONS:
            logger.warning(
                f"Critic: límite de {self._MAX_ITERATIONS} iteraciones alcanzado."
            )
            return CriticFeedback(
                has_errors=True,
                error_summary=(
                    f"Límite de corrección ({self._MAX_ITERATIONS} iteraciones) "
                    "alcanzado. Requiere intervención manual."
                ),
                corrective_instructions=(
                    "El agente no pudo corregir todos los errores automáticamente. "
                    f"Checks fallidos: {report.failed_checks}. "
                    "Revisar manualmente los archivos en el workspace."
                ),
                severity="high",
                failed_checks=report.failed_checks,
            )

        severity = self._classify_severity(report)
        raw_summary = report.to_summary()
        instructions = self._generate_instructions(raw_summary, report.failed_checks)

        return CriticFeedback(
            has_errors=True,
            error_summary=raw_summary,
            corrective_instructions=instructions,
            severity=severity,
            failed_checks=report.failed_checks,
        )

    def reset_iteration_count(self) -> None:
        """Resetear contador para una nueva tarea."""
        self._iteration_count = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _classify_severity(self, report: ValidationReport) -> str:
        """
        Clasificar severidad basada en qué checks fallaron.

        Prioridad (mayor a menor): security > tests > types > coverage > lint.

        Args:
            report: ValidationReport con resultados reales.

        Returns:
            'low', 'medium', o 'high' como string.
        """
        if report.security and not report.security.succeeded:
            return "high"
        if report.tests and not report.tests.succeeded:
            return "high"
        if report.types and not report.types.succeeded:
            return "medium"
        if report.coverage and not report.coverage.succeeded:
            return "medium"
        return "low"  # Solo lint

    def _generate_instructions(
        self,
        validation_summary: str,
        failed_checks: list[str],
    ) -> str:
        """
        Usar el LLM para generar instrucciones correctivas específicas.

        Args:
            validation_summary: Salida real de las herramientas.
            failed_checks: Lista de checks que fallaron.

        Returns:
            Instrucciones concretas para el Coder, con ejemplos de corrección.
        """
        messages = [
            LLMMessage(
                role="system",
                content=(
                    "Eres un revisor de código senior. Analiza estos errores "
                    "de herramientas de validación y genera instrucciones "
                    "ESPECÍFICAS y CONCRETAS para corregirlos. "
                    "Para cada error: indica el archivo, línea si está disponible, "
                    "y el cambio exacto a hacer. "
                    "Responde en español, en formato de lista numerada."
                ),
            ),
            LLMMessage(
                role="user",
                content=(
                    f"Checks fallidos: {', '.join(failed_checks)}\n\n"
                    f"Salida de las herramientas:\n{validation_summary}"
                ),
            ),
        ]

        try:
            response = self._llm.complete(messages)
            return response.content.strip()
        except Exception as exc:
            logger.warning(f"Critic LLM falló: {exc}. Usando análisis básico.")
            return self._basic_instructions(validation_summary, failed_checks)

    @staticmethod
    def _basic_instructions(summary: str, failed_checks: list[str]) -> str:
        """
        Generar instrucciones sin LLM cuando el backend no está disponible.

        Args:
            summary: Resumen del reporte de validación.
            failed_checks: Checks que fallaron.

        Returns:
            Instrucciones básicas basadas en reglas heurísticas.
        """
        instructions: list[str] = []

        if "tests" in failed_checks:
            instructions.append(
                "1. Ejecuta pytest -v y lee cada FAILED traceback. "
                "Corrige las aserciones o la lógica de las funciones fallidas."
            )
        if "types" in failed_checks:
            instructions.append(
                "2. Ejecuta mypy --strict y añade type hints faltantes. "
                "Corrige tipos incompatibles en las líneas indicadas."
            )
        if "lint" in failed_checks:
            instructions.append(
                "3. Ejecuta ruff check --fix para auto-corregir. "
                "Corrige manualmente las violaciones que ruff no puede auto-fijar."
            )
        if "security" in failed_checks:
            instructions.append(
                "4. Revisa las alertas de bandit. Elimina uso de funciones "
                "inseguras (eval, exec, subprocess sin validación)."
            )
        if "coverage" in failed_checks:
            instructions.append(
                "5. Añade tests unitarios para las líneas no cubiertas "
                "indicadas en el reporte de cobertura."
            )

        if not instructions:
            instructions.append(f"Corregir errores en: {', '.join(failed_checks)}")

        return "\n".join(instructions)
