from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

class ToolStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    ERROR = "error"

@dataclass
class ToolCall:
    tool_name: str
    arguments: dict
    call_id: str = ""

@dataclass
class ToolResult:
    tool_name: str
    status: ToolStatus
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int = 0
    succeeded: bool = False

@dataclass
class EvalScore:
    score: float
    rationale: str

@dataclass
class FixDirective:
    failure_type: str
    priority: str
    anchor: str
    rca: str
    directive: str

@dataclass
class TraceProfile:
    task_id: str = ""
    success: bool = False
    llm_provider: str = ""
    total_steps: int = 0
    completed_steps: int = 0
    total_llm_calls: int = 0
    total_tokens: int = 0
    parsing_errors: int = 0
    tool_failures: Dict[str, int] = field(default_factory=dict)
    checkpoints_triggered: List[str] = field(default_factory=list)
    critical_failures: List[str] = field(default_factory=list)
    divergence_step: int = -1
    spans: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class ComparisonResult:
    objective: str
    successful_profile: TraceProfile
    failed_profile: TraceProfile
    divergence_point: str = ""
    root_cause: str = ""
    recommendations: List[str] = field(default_factory=list)
    severity: str = "MODERATE"
