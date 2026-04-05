"""
AgentFixer — el Sistema Inmune del Framework Inmune (Pilar IV).

Orquesta la detección de los 15 tipos de fallo, la generación de
FixDirective via Fix Protocol, y la inyección en el historial del
agente para auto-curación en tiempo real.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Tuple, Dict, Optional, Set

from .checkpoints import (
    CHECKPOINT_REGISTRY,
    CheckpointDefinition,
    CheckpointPriority,
    FailureType,
    get_checkpoint,
)
from .fix_protocol import FixDirective, FixProtocol
from ..core.config import AgentConfig
from ..llm.backend import LLMMessage
from ..schemas.schemas import ToolCall, ToolResult

logger = logging.getLogger(__name__)

@dataclass
class InspectionResult:
    passed: bool
    failures_detected: List[FailureType] = field(default_factory=list)
    directives: List[FixDirective] = field(default_factory=list)
    low_priority_log: List[Tuple[CheckpointDefinition, Dict[str, Any]]] = field(default_factory=list)
    should_interrupt: bool = False

    @property
    def injection_messages(self) -> List[LLMMessage]:
        return [d.to_injection_message() for d in self.directives]

class AgentFixer:
    _LOOP_THRESHOLD: int = 3
    _CTX_THRESHOLD: float = 0.85
    _INCOMPLETENESS_PATTERNS: List[str] = [r"^\s*pass\s*$", r"^\s*\.\.\.\s*$", r"#\s*(TODO|FIXME|HACK|XXX)", r"raise NotImplementedError"]

    def __init__(self, config: AgentConfig) -> None:
        self._config = config
        self._fix_protocol = FixProtocol(config)
        self._call_history: List[ToolCall] = []
        self._low_priority_accumulator: List[Tuple[CheckpointDefinition, Dict[str, Any]]] = []
        self._prior_test_failures: Set[str] = set()

    def inspect_pre_flight(self, tool_call: ToolCall, available_tools: List[str], generated_code: str = "") -> InspectionResult:
        failures: List[Tuple[CheckpointDefinition, Dict[str, Any]]] = []
        if not tool_call.tool_name or not isinstance(tool_call.arguments, dict):
            cp = get_checkpoint(FailureType.SCHEMA_VIOLATION)
            if cp: failures.append((cp, {"raw_call": str(tool_call), "issue": "tool_name vacío o arguments no es dict"}))
        if tool_call.tool_name not in available_tools + ["task_complete"]:
            cp = get_checkpoint(FailureType.TOOL_HALLUCINATION)
            if cp: failures.append((cp, {"tool_name": tool_call.tool_name, "available_tools": ", ".join(available_tools[:8])}))
        if generated_code:
            security_issue = self._detect_security(generated_code)
            if security_issue:
                cp = get_checkpoint(FailureType.SECURITY_VIOLATION)
                if cp: failures.append((cp, {"pattern": security_issue, "safe_alternative": self._safe_alternative(security_issue)}))
        self._call_history.append(tool_call)
        if self._detect_loop():
            cp = get_checkpoint(FailureType.INFINITE_LOOP)
            if cp: failures.append((cp, {"tool_name": tool_call.tool_name, "count": self._LOOP_THRESHOLD, "last_result": "sin cambio de estado", "alternative_approach": "cambiar herramienta o parámetros"}))
        return self._build_result(failures)

    def inspect_output(self, tool_result: ToolResult, generated_code: str = "", step_objective: str = "", estimated_tokens: int = 0) -> InspectionResult:
        failures: List[Tuple[CheckpointDefinition, Dict[str, Any]]] = []
        if generated_code:
            placeholder_lines = self._detect_placeholders(generated_code)
            if placeholder_lines:
                cp = get_checkpoint(FailureType.INCOMPLETENESS)
                if cp: failures.append((cp, {"lines": ", ".join(str(l) for l in placeholder_lines), "count": len(placeholder_lines)}))
        if tool_result.tool_name == "run_ruff" and not tool_result.succeeded:
            cp = get_checkpoint(FailureType.FORMAT_INTEGRITY)
            if cp: failures.append((cp, {"count": tool_result.stdout.count("\n"), "violations": tool_result.stdout[:300] or tool_result.stderr[:300]}))
        if tool_result.tool_name == "run_mypy" and not tool_result.succeeded:
            cp = get_checkpoint(FailureType.TYPE_SAFETY_VIOLATION)
            if cp: failures.append((cp, {"count": (tool_result.stdout[:400] or tool_result.stderr[:400]).count(": error:"), "errors": tool_result.stdout[:400] or tool_result.stderr[:400]}))
        if tool_result.tool_name == "run_pytest" and not tool_result.succeeded:
            new_failures = self._detect_regression(tool_result.stdout)
            if new_failures:
                cp = get_checkpoint(FailureType.TEST_REGRESSION)
                if cp: failures.append((cp, {"test_ids": ", ".join(new_failures), "expected": "pass", "actual": "FAILED"}))
        max_tokens = 32000 # Default fallback
        if estimated_tokens > 0 and estimated_tokens > max_tokens * self._CTX_THRESHOLD:
            cp = get_checkpoint(FailureType.CONTEXT_OVERFLOW)
            if cp: failures.append((cp, {"usage": f"{int(estimated_tokens / max_tokens * 100)}%"}))
        return self._build_result(failures)

    def run_auto_dreaming(self, findings_path: Optional[Path] = None) -> List[FixDirective]:
        if not self._low_priority_accumulator: return []
        if findings_path and findings_path.exists(): self._enrich_low_priority_from_findings(findings_path)
        directives = self._fix_protocol.generate_batch(self._low_priority_accumulator)
        self._low_priority_accumulator.clear()
        return directives

    def reset_session(self) -> None:
        self._call_history.clear()
        self._prior_test_failures.clear()

    def _build_result(self, failures: List[Tuple[CheckpointDefinition, Dict[str, Any]]]) -> InspectionResult:
        if not failures: return InspectionResult(passed=True)
        directives: List[FixDirective] = []
        failure_types: List[FailureType] = []
        should_interrupt = False
        for checkpoint, context in failures:
            failure_types.append(checkpoint.failure_type)
            if checkpoint.priority == CheckpointPriority.LOW:
                self._low_priority_accumulator.append((checkpoint, context))
                continue
            directives.append(self._fix_protocol.generate(checkpoint, context))
            if checkpoint.priority == CheckpointPriority.CRITICAL: should_interrupt = True
        return InspectionResult(passed=not directives, failures_detected=failure_types, directives=directives, low_priority_log=list(self._low_priority_accumulator), should_interrupt=should_interrupt)

    def _detect_loop(self) -> bool:
        if len(self._call_history) < self._LOOP_THRESHOLD: return False
        last_n = self._call_history[-self._LOOP_THRESHOLD:]
        return all(c.tool_name == last_n[0].tool_name and c.arguments == last_n[0].arguments for c in last_n)

    def _detect_placeholders(self, code: str) -> List[int]:
        hp: List[int] = []
        for i, line in enumerate(code.splitlines(), start=1):
            if any(re.search(p, line) for p in self._INCOMPLETENESS_PATTERNS): hp.append(i)
        return hp

    @staticmethod
    def _detect_security(code: str) -> str:
        danger = {r"\beval\s*\(": "eval()", r"\bexec\s*\(": "exec()", r"(password|secret)\s*=\s*['\"][^'\"]+['\"]": "hardcoded secret", r"__import__\s*\(": "__import__()"}
        for p, l in danger.items():
            if re.search(p, code, re.IGNORECASE): return l
        return ""

    @staticmethod
    def _safe_alternative(pattern: str) -> str:
        alt = {"eval()": "ast.literal_eval()", "exec()": "subprocess", "hardcoded secret": "os.environ"}
        return alt.get(pattern, "una alternativa segura")

    def _detect_regression(self, pytest_output: str) -> List[str]:
        cur = set(re.findall(r"FAILED\s+([\w/.:]+)", pytest_output))
        reg = cur - self._prior_test_failures
        self._prior_test_failures = cur
        return list(reg)

    def _enrich_low_priority_from_findings(self, findings_path: Path) -> None:
        try:
            content = findings_path.read_text(encoding="utf-8")
            paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 50]
            for i in range(len(paragraphs) - 2):
                if self._similarity(paragraphs[i], paragraphs[i + 2]) > 0.85:
                    cp = get_checkpoint(FailureType.CIRCULAR_REASONING)
                    if cp: self._low_priority_accumulator.append((cp, {"pattern": paragraphs[i][:100], "count": 3}))
                    break
        except Exception: pass

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        wa, wb = set(a.lower().split()), set(b.lower().split())
        return len(wa & wb) / len(wa | wb) if wa | wb else 0.0
