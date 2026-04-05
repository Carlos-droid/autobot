"""
Checkpoints module para el AgentFixer del Framework Inmune.

Define los 15 tipos de fallo detectables, su prioridad, y la
estrategia de intervención correspondiente según el documento
del Framework Inmune (Pilar IV).
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

class CheckpointPriority(str, Enum):
    """Nivel de prioridad que determina el tipo de intervención."""
    CRITICAL = "CRITICAL"   # Intervención Directa — interrumpe el flujo
    MEDIUM = "MEDIUM"       # Reflexión Interna — pide corrección antes de entregar
    LOW = "LOW"             # Optimización Batch — modifica prompts para el día siguiente

class CheckpointPhase(str, Enum):
    """Fase del pipeline donde opera el checkpoint."""
    PRE_FLIGHT = "PRE_FLIGHT"           # Antes de ejecutar el paso
    EXECUTION = "EXECUTION"             # Durante la ejecución del paso
    OUTPUT_VALIDATION = "OUTPUT_VALIDATION"  # Validación de salida del Worker
    AUTO_DREAMING = "AUTO_DREAMING"     # Batch nocturno

class FailureType(str, Enum):
    """Los 15 tipos de fallo detectables por el AgentFixer."""
    SECURITY_VIOLATION = "SECURITY_VIOLATION"
    SCHEMA_VIOLATION = "SCHEMA_VIOLATION"
    TOOL_HALLUCINATION = "TOOL_HALLUCINATION"
    INFINITE_LOOP = "INFINITE_LOOP"
    CONTEXT_OVERFLOW = "CONTEXT_OVERFLOW"
    PLANNER_MISALIGNMENT = "PLANNER_MISALIGNMENT"
    INCOMPLETENESS = "INCOMPLETENESS"
    FORMAT_INTEGRITY = "FORMAT_INTEGRITY"
    TEST_REGRESSION = "TEST_REGRESSION"
    TYPE_SAFETY_VIOLATION = "TYPE_SAFETY_VIOLATION"
    AMBIGUITY_SCORE = "AMBIGUITY_SCORE"
    LATENCY_COST_ANOMALY = "LATENCY_COST_ANOMALY"
    CIRCULAR_REASONING = "CIRCULAR_REASONING"
    COVERAGE_DEGRADATION = "COVERAGE_DEGRADATION"
    PROMPT_DRIFT = "PROMPT_DRIFT"

@dataclass(frozen=True)
class CheckpointDefinition:
    """Definición completa de un checkpoint del AgentFixer."""
    failure_type: FailureType
    priority: CheckpointPriority
    phase: CheckpointPhase
    name: str
    description: str
    intervention: str
    detector_hint: str

CHECKPOINT_REGISTRY: List[CheckpointDefinition] = [
    # ── CRITICAL ──────────────────────────────────────────────────────
    CheckpointDefinition(
        failure_type=FailureType.SECURITY_VIOLATION,
        priority=CheckpointPriority.CRITICAL,
        phase=CheckpointPhase.PRE_FLIGHT,
        name="Security Gate",
        description="Detecta código con eval(), exec(), hardcoded secrets o bandit HIGH.",
        intervention="REPAIR_DIRECTIVE: Elimina el uso de {pattern}. Usa la alternativa segura: {safe_alternative}.",
        detector_hint="bandit exit_code=1 OR keywords: eval|exec|secret|password=",
    ),
    CheckpointDefinition(
        failure_type=FailureType.SCHEMA_VIOLATION,
        priority=CheckpointPriority.CRITICAL,
        phase=CheckpointPhase.PRE_FLIGHT,
        name="Schema Validator",
        description="Verifica que el tool call sea JSON válido con tool_name y arguments.",
        intervention="REPAIR_DIRECTIVE: Tu respuesta anterior no es JSON válido. Responde ÚNICAMENTE con: {\"tool_name\": \"...\", \"arguments\": {...}}",
        detector_hint="json.JSONDecodeError OR missing 'tool_name' key",
    ),
    CheckpointDefinition(
        failure_type=FailureType.TOOL_HALLUCINATION,
        priority=CheckpointPriority.CRITICAL,
        phase=CheckpointPhase.PRE_FLIGHT,
        name="Tool Registry Guard",
        description="El agente invocó una herramienta que no está en el SkillRegistry.",
        intervention="REPAIR_DIRECTIVE: La herramienta '{tool_name}' no existe. Herramientas disponibles: {available_tools}. Elige una de estas.",
        detector_hint="tool_name NOT IN ToolExecutor._BUILTIN_COMMANDS",
    ),
    CheckpointDefinition(
        failure_type=FailureType.INFINITE_LOOP,
        priority=CheckpointPriority.CRITICAL,
        phase=CheckpointPhase.EXECUTION,
        name="Loop Detector",
        description="El mismo tool call se repitió ≥3 veces consecutivas sin cambio.",
        intervention="REPAIR_DIRECTIVE: Has ejecutado '{tool_name}' {count} veces sin progreso. El resultado fue: {last_result}. Cambia de estrategia: intenta {alternative_approach}.",
        detector_hint="last 3 tool_calls identical AND step status unchanged",
    ),
    CheckpointDefinition(
        failure_type=FailureType.CONTEXT_OVERFLOW,
        priority=CheckpointPriority.CRITICAL,
        phase=CheckpointPhase.EXECUTION,
        name="Context Window Guard",
        description="El contexto acumulado supera el 85% del límite de tokens.",
        intervention="REPAIR_DIRECTIVE: Contexto crítico ({usage}% usado). Resume el estado en máximo 3 bullets y continúa desde ahí.",
        detector_hint="estimated_tokens > 0.85 * max_context_tokens",
    ),

    # ── MEDIUM ────────────────────────────────────────────────────────
    CheckpointDefinition(
        failure_type=FailureType.PLANNER_MISALIGNMENT,
        priority=CheckpointPriority.MEDIUM,
        phase=CheckpointPhase.OUTPUT_VALIDATION,
        name="Objective Alignment Check",
        description="La salida del Worker no corresponde al objetivo del paso del Planner.",
        intervention="REFLEXIÓN: Tu output para '{step}' no corresponde al objetivo. El objetivo era: {objective}. Tu output fue: {output_summary}. Vuelve a generar enfocándote en el objetivo.",
        detector_hint="LLM-as-Judge similarity_score < 0.6",
    ),
    CheckpointDefinition(
        failure_type=FailureType.INCOMPLETENESS,
        priority=CheckpointPriority.MEDIUM,
        phase=CheckpointPhase.OUTPUT_VALIDATION,
        name="Completeness Checker",
        description="El código contiene pass, ..., TODO, FIXME o NotImplementedError.",
        intervention="REFLEXIÓN: Tu código tiene placeholders en las líneas {lines}. Reemplaza cada placeholder con la implementación real completa.",
        detector_hint="grep: pass|\\.\\.\\.|TODO|FIXME|NotImplementedError in generated code",
    ),
    CheckpointDefinition(
        failure_type=FailureType.FORMAT_INTEGRITY,
        priority=CheckpointPriority.MEDIUM,
        phase=CheckpointPhase.OUTPUT_VALIDATION,
        name="Format Enforcer",
        description="ruff o black --check detectan violaciones de formato.",
        intervention="REFLEXIÓN: Tu código tiene {count} violaciones de formato: {violations}. Corrígelas aplicando Black y ruff antes de entregar.",
        detector_hint="ruff exit_code=1 OR black --check exit_code=1",
    ),
    CheckpointDefinition(
        failure_type=FailureType.TEST_REGRESSION,
        priority=CheckpointPriority.MEDIUM,
        phase=CheckpointPhase.OUTPUT_VALIDATION,
        name="Regression Guard",
        description="Tests que pasaban en la iteración anterior ahora fallan.",
        intervention="REFLEXIÓN: Tu cambio introdujo una regresión en {test_ids}. El test esperaba {expected}, obtuvo {actual}. Revierte o corrige sin romper el contrato existente.",
        detector_hint="pytest failed_tests NOT IN prior_failed_tests",
    ),
    CheckpointDefinition(
        failure_type=FailureType.TYPE_SAFETY_VIOLATION,
        priority=CheckpointPriority.MEDIUM,
        phase=CheckpointPhase.OUTPUT_VALIDATION,
        name="Type Safety Gate",
        description="mypy --strict detecta errores de tipos en el código generado.",
        intervention="REFLEXIÓN: mypy encontró {count} errores de tipos: {errors}. Añade los type hints correctos. No uses Any a menos que sea necesario.",
        detector_hint="mypy exit_code=1",
    ),

    # ── LOW (Auto-Dreaming) ───────────────────────────────────────────
    CheckpointDefinition(
        failure_type=FailureType.AMBIGUITY_SCORE,
        priority=CheckpointPriority.LOW,
        phase=CheckpointPhase.AUTO_DREAMING,
        name="Ambiguity Scorer",
        description="El paso del Planner generó múltiples interpretaciones distintas.",
        intervention="GENOME_UPDATE: Reescribir paso '{step}' con verbo imperativo y artefacto de salida explícito. Nuevo prompt: {improved_step}",
        detector_hint="std_dev(coder_outputs_for_same_step) > threshold",
    ),
    CheckpointDefinition(
        failure_type=FailureType.LATENCY_COST_ANOMALY,
        priority=CheckpointPriority.LOW,
        phase=CheckpointPhase.AUTO_DREAMING,
        name="Cost Optimizer",
        description="Un paso consumió >2x el promedio de tokens/tiempo esperado.",
        intervention="GENOME_UPDATE: Paso '{step}' costó {actual_tokens} tokens (baseline: {baseline_tokens}). Simplificar contexto inyectado.",
        detector_hint="step_tokens > 2 * rolling_average_tokens",
    ),
    CheckpointDefinition(
        failure_type=FailureType.CIRCULAR_REASONING,
        priority=CheckpointPriority.LOW,
        phase=CheckpointPhase.AUTO_DREAMING,
        name="Circular Reasoning Detector",
        description="findings.md muestra el mismo razonamiento en ≥3 iteraciones.",
        intervention="GENOME_UPDATE: El agente repitió el razonamiento '{pattern}' {count} veces. Añadir instrucción explícita: 'Si ya intentaste X sin éxito, prueba Y en su lugar.'",
        detector_hint="similarity(finding[n], finding[n-2]) > 0.85",
    ),
    CheckpointDefinition(
        failure_type=FailureType.COVERAGE_DEGRADATION,
        priority=CheckpointPriority.LOW,
        phase=CheckpointPhase.AUTO_DREAMING,
        name="Coverage Watchdog",
        description="La cobertura de tests bajó >5% respecto al baseline.",
        intervention="GENOME_UPDATE: Cobertura bajó de {baseline_pct}% a {current_pct}%. Añadir al Planner: paso explícito 'Escribir tests para líneas no cubiertas antes de marcar el paso como completado.'",
        detector_hint="coverage_pct < prior_coverage_pct - 5",
    ),
    CheckpointDefinition(
        failure_type=FailureType.PROMPT_DRIFT,
        priority=CheckpointPriority.LOW,
        phase=CheckpointPhase.AUTO_DREAMING,
        name="Prompt Drift Monitor",
        description="El agente ignora instrucciones del system prompt en iteraciones ≥5.",
        intervention="GENOME_UPDATE: El agente omitió la regla '{rule}' en la iteración {iteration}. Reforzar la regla con ejemplo negativo en el system prompt.",
        detector_hint="LLM-as-Judge: output violates system_prompt rule X at iteration N",
    ),
]

def get_checkpoints_by_priority(priority: CheckpointPriority) -> List[CheckpointDefinition]:
    return [cp for cp in CHECKPOINT_REGISTRY if cp.priority == priority]

def get_checkpoint(failure_type: FailureType) -> Optional[CheckpointDefinition]:
    return next((cp for cp in CHECKPOINT_REGISTRY if cp.failure_type == failure_type), None)
