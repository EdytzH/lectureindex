"""Keyless Ollama provider adapter, used by the existing validated LLM planner.

The frontend Puter.js provider is deliberately browser-only and does not expose
Puter tokens to Streamlit Python. All model JSON is parsed and validated locally.
"""
from __future__ import annotations
import json
import urllib.error
import urllib.request
from types import SimpleNamespace


class OllamaResponses:
    def __init__(self, model_base='http://127.0.0.1:11434', timeout=240):
        # This application only sends lecture data to the user's *local* Ollama.
        if model_base.rstrip('/') not in ('http://127.0.0.1:11434', 'http://localhost:11434'):
            raise ValueError('For privacy, only the local Ollama server on port 11434 is supported.')
        self.base = model_base.rstrip('/')
        self.timeout = timeout
        self.responses = self

    def parse(self, *, model, input, text_format):
        body = json.dumps({
            'model': model, 'messages': input, 'stream': False,
            'format': text_format.model_json_schema(),
            'options': {'temperature': 0.1},
        }).encode('utf-8')
        request=urllib.request.Request(self.base + '/api/chat', body,
                                       headers={'Content-Type': 'application/json'}, method='POST')
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                obj = json.loads(response.read().decode('utf-8'))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError('Cannot reach local Ollama. Install Ollama, run `ollama pull qwen3:8b`, '
                               'and ensure its server is running on localhost:11434.') from exc
        content = obj.get('message',{}).get('content','')
        if not content:
            raise ValueError('Ollama returned an empty answer.')
        parsed = text_format.model_validate_json(content)
        return SimpleNamespace(output_parsed=parsed)
