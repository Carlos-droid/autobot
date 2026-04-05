import ollama
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class LLMMessage:
    role: str
    content: str

class LLMBackend:
    def __init__(self, config: Any) -> None:
        self._client = ollama.Client(host=config.ollama_host)
        self._model = config.llm_model

    def complete(self, messages: List[LLMMessage]) -> Any:
        response = self._client.chat(
            model=self._model,
            messages=[{"role": m.role, "content": m.content} for m in messages]
        )
        # Mocking the response internal structure for use
        class Response:
            def __init__(self, content: str):
                self.content = content
        return Response(content=response['message']['content'])
