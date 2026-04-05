"""
Validator module para AG2-Coder v6.

Reemplaza el placeholder que retornaba `{"tests": True, "coverage": True}`
hardcodeado con ejecución real de herramientas CLI usando el
ToolExecutor de AutoAgent.

Ejecuta: pytest, pytest --cov, ruff, mypy, bandit.
Nunca asume que algo pasa — lee el exit code y stdout reales.
"""

import logging
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

from core.config import AgentConfig
from tools.executor import ToolExecutor
from tools.tool_schema import ToolCall
from tools.tool_schema import ToolResult

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """
    Resultado completo de todas las validaciones ejecutadas.

    Cada campo es un ToolResult real, no un bool hardcodeado.
    """

    tests: ToolResult | None = None
    coverage: ToolResult | None = None
    lint: ToolResult | None = None
    types: ToolResult | None = None
    security: ToolResult | None = None
    extra_results: list[ToolResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        """True sólo si TODOS los checks terminaron con exit code 0."""
        checks = [self.tests, self.coverage, self.lint, self.types, self.security]
        active = [c for c in checks if c is not None]
        return all(c.succeeded for c in active)

    @property
    def failed_checks(self) -> list[str]:
        """Lista de nombres de checks que fallaron."""
        mapping = {
            "tests": self.tests,
            "coverage": self.coverage,
            "lint": self.lint,
            "types": self.types,
            "security": self.security,
        }
        return [name for name, result in mapping.items() if result and not result.succeeded]

    def to_dict(self) -> dict[str, bool]:
        """
        Interfaz compatible con el contrato original del Adapter.

        Returns:
            Diccionario {check_name: passed} con valores reales.
        """
        mapping = {
            "tests": self.tests,
            "coverage": self.coverage,
            "lint": self.lint,
            "types": self.types,
            "security": self.security,
        }
        return {
            name: (result.succeeded if result is not None else True)
            for name, result in mapping.items()
        }

    def to_summary(self) -> str:
        """
        Generar resumen legible para el Critic.

        Returns:
            Texto estructurado con resultados y salidas de error.
        """
        lines: list[str] = ["=== Validation Report ==="]
        mapping = {
            "tests": self.tests,
            "coverage": self.coverage,
            "lint": self.lint,
            "types": self.types,
            "security": self.security,
        }
        for name, result in mapping.items():
            if result is None:
                lines.append(f"  {name}: SKIPPED")
                continue
            icon = "✅" if result.succeeded else "❌"
            lines.append(
                f"  {icon} {name}: exit={result.exit_code} ({result.duration_ms}ms)"
            )
            if not result.succeeded:
                output = result.stderr or result.stdout
                if output:
                    lines.append(f"    └─ {output[:300].strip()}")

        status = "PASS" if self.all_passed else f"FAIL ({len(self.failed_checks)} check(s))"
        lines.append(f"\nOverall: {status}")
        return "\n".join(lines)


class Validator:
    """
    Ejecuta validaciones reales sobre el código generado.

    A diferencia del placeholder, esta clase usa el ToolExecutor de
    AutoAgent para correr subprocesos reales y capturar su salida.
    El Adapter recibe resultados reales, no booleans hardcodeados.

    Example:
        >>> config = AgentConfig()
        >>> validator = Validator(config, target_path=Path("src/"))
        >>> report = validator.run_all()
        >>> print(report.all_passed)   # False si hay errores reales
        >>> print(report.to_summary()) # Muestra qué falló exactamente
    """

    def __init__(
        self,
        config: AgentConfig,
        target_path: Path | None = None,
        tests_path: Path | None = None,
        min_coverage: int = 80,
    ) -> None:
        """
        Inicializar Validator con rutas y configuración real.

        Args:
            config: AgentConfig para ToolExecutor.
            target_path: Directorio fuente a validar (default: src/).
            tests_path: Directorio de tests (default: tests/).
            min_coverage: Porcentaje mínimo de cobertura requerido.
        """
        self._config = config
        self._executor = ToolExecutor(config)
        self._target = str(target_path or config.workspace_dir.parent / "src")
        self._tests = str(tests_path or config.workspace_dir.parent / "tests")
        self._min_coverage = min_coverage

    def run_all(self) -> ValidationReport:
        """
        Ejecutar todos los checks de validación en secuencia.

        Orden: tests → coverage → lint → types → security.
        Si tests falla, los demás igualmente se ejecutan para dar
        feedback completo al Critic.

        Returns:
            ValidationReport con resultados reales de cada herramienta.
        """
        report = ValidationReport()

        logger.info("Validator: ejecutando pytest...")
        report.tests = self._run_tests()

        logger.info("Validator: ejecutando pytest --cov...")
        report.coverage = self._run_coverage()

        logger.info("Validator: ejecutando ruff...")
        report.lint = self._run_lint()

        logger.info("Validator: ejecutando mypy...")
        report.types = self._run_types()

        logger.info("Validator: ejecutando bandit...")
        report.security = self._run_security()

        status = "PASS" if report.all_passed else f"FAIL: {report.failed_checks}"
        logger.info(f"Validator completo: {status}")
        return report

    def run_fast(self) -> ValidationReport:
        """
        Ejecutar sólo lint y types (sin pytest) para feedback rápido.

        Útil durante el loop de corrección del Critic para evitar
        ejecutar la suite completa en cada iteración menor.

        Returns:
            ValidationReport parcial con sólo lint y types.
        """
        report = ValidationReport()
        report.lint = self._run_lint()
        report.types = self._run_types()
        return report

    # ------------------------------------------------------------------
    # Individual check methods
    # ------------------------------------------------------------------

    def _run_tests(self) -> ToolResult:
        """Ejecutar pytest con output detallado y tb=short."""
        return self._executor.execute(
            ToolCall(
                tool_name="run_pytest",
                arguments={"path": self._tests},
                call_id="validator-tests",
            )
        )

    def _run_coverage(self) -> ToolResult:
        """Ejecutar pytest con cobertura y fallo si < min_coverage."""
        # Extender el comando base con --cov flags usando execute_raw
        return self._executor.execute_raw(
            [
                "python", "-m", "pytest", self._tests,
                f"--cov={self._target}",
                "--cov-report=term-missing",
                f"--cov-fail-under={self._min_coverage}",
                "-q", "--no-header",
            ]
        )

    def _run_lint(self) -> ToolResult:
        """Ejecutar ruff check sobre el código fuente."""
        return self._executor.execute(
            ToolCall(
                tool_name="run_ruff",
                arguments={"path": self._target},
                call_id="validator-lint",
            )
        )

    def _run_types(self) -> ToolResult:
        """Ejecutar mypy --strict sobre el código fuente."""
        return self._executor.execute(
            ToolCall(
                tool_name="run_mypy",
                arguments={"path": self._target},
                call_id="validator-types",
            )
        )

    def _run_security(self) -> ToolResult:
        """Ejecutar bandit para análisis de seguridad estático."""
        return self._executor.execute_raw(
            [
                "python", "-m", "bandit",
                "-r", self._target,
                "-ll",           # Solo severidad MEDIUM o HIGH
                "--quiet",
            ]
        )
