import os
from typing import Optional
from pathlib import Path

class AgentConfig:
    def __init__(
        self,
        llm_model: str = "qwen2.5:7b",
        ollama_host: str = "http://localhost:11434",
        workspace_dir: Optional[Path] = None,
        enable_coverage: bool = True
    ) -> None:
        self.llm_model = llm_model
        # Use http://host.docker.internal:11434 if in Docker, else localhost
        self.ollama_host = os.environ.get("OLLAMA_HOST", ollama_host)
        self.workspace_dir = workspace_dir or Path.cwd()
        self.enable_coverage = enable_coverage
