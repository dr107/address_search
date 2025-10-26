import logging
from typing import Optional, Dict, Any

import requests

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Minimal Ollama REST client. Talks to /api/generate.
    """

    def __init__(self, base_url: str):
        # e.g. "http://localhost:11434"
        self.base_url = base_url.rstrip("/")

    def generate(
        self,
        model: str,
        prompt: str,
        options: Optional[Dict[str, Any]] = None,
        timeout: int = 300,
    ) -> str:
        """
        Call /api/generate on Ollama.

        model:    name of the model registered in Ollama
        prompt:   text prompt to send
        options:  inference options, e.g. {"temperature":0.2,"num_ctx":4096}
        timeout:  request timeout (seconds)

        Returns the 'response' string from Ollama.
        """
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if options:
            payload["options"] = options

        url = f"{self.base_url}/api/generate"
        logger.debug(
            "Calling Ollama generate: url=%s model=%s options=%s", url, model, options
        )
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()

        data = resp.json()
        logger.debug(
            "Ollama response metadata: model=%s done=%s", data.get("model"), data.get("done")
        )
        # Expected shape: {"model":"...","created_at":"...","response":"...","done":true,...}
        return data.get("response", "")
