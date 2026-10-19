"""Concrete providers: Ollama (local), any OpenAI-compatible API (Groq, Gemini, OpenRouter, OpenAI),
and a scripted mock for tests. Selected by name from config/llm.yaml."""
from __future__ import annotations

import json
import os
import time
from typing import Callable, Optional, Union

from auv_mission_planner.llm.base import LLMError, LLMProvider, LLMResponse, http_post_json, with_retries


class OllamaProvider(LLMProvider):
    """Local model via Ollama /api/chat. Uses `format=<json schema>` for constrained decoding."""

    def __init__(self, name, cfg):
        super().__init__(name, cfg)
        self.host = os.environ.get(cfg.get('host_env', 'OLLAMA_HOST')) or cfg.get('host', 'http://127.0.0.1:11434')
        self.num_ctx = cfg.get('num_ctx', 8192)
        self.use_schema = cfg.get('constrained', True)

    def complete(self, messages, json_schema=None):
        body = {'model': self.model, 'messages': messages, 'stream': False,
                'options': {'temperature': self.temperature, 'num_ctx': self.num_ctx,
                            'seed': self.cfg.get('seed', 0)},
                'keep_alive': self.cfg.get('keep_alive', '10m')}
        if json_schema is not None:
            body['format'] = json_schema if self.use_schema else 'json'
        t0 = time.time()
        r, retries = with_retries(lambda: http_post_json(f'{self.host}/api/chat', body, {}, self.timeout_s),
                                  attempts=2)
        if 'message' not in r:
            raise LLMError(f'unexpected Ollama response: {str(r)[:300]}')
        return LLMResponse(r['message'].get('content', ''),
                           self._usage(r.get('prompt_eval_count', 0), r.get('eval_count', 0),
                                       time.time() - t0, retries), r)


class OpenAICompatProvider(LLMProvider):
    """Any /chat/completions endpoint. json_mode: json_schema | json_object | none."""

    def __init__(self, name, cfg):
        super().__init__(name, cfg)
        self.base_url = cfg['base_url'].rstrip('/')
        self.key_env = cfg.get('api_key_env', 'OPENAI_API_KEY')
        self.json_mode = cfg.get('json_mode', 'json_object')

    def api_key(self) -> str:
        k = os.environ.get(self.key_env, '')
        if not k:
            raise LLMError(f'{self.key_env} is not set (put it in .env)')
        return k

    def complete(self, messages, json_schema=None):
        body = {'model': self.model, 'messages': messages, 'temperature': self.temperature,
                **self.cfg.get('extra_body', {})}
        if json_schema is not None and self.json_mode == 'json_schema':
            body['response_format'] = {'type': 'json_schema',
                                       'json_schema': {'name': 'mission_plan', 'schema': json_schema}}
        elif json_schema is not None and self.json_mode == 'json_object':
            body['response_format'] = {'type': 'json_object'}
        headers = {'Authorization': f'Bearer {self.api_key()}'}
        t0 = time.time()
        r, retries = with_retries(lambda: http_post_json(f'{self.base_url}/chat/completions', body, headers,
                                                         self.timeout_s), attempts=self.cfg.get('retries', 4))
        try:
            text = r['choices'][0]['message']['content'] or ''
        except (KeyError, IndexError, TypeError):
            raise LLMError(f'unexpected response: {str(r)[:300]}')
        u = r.get('usage') or {}
        return LLMResponse(text, self._usage(u.get('prompt_tokens', 0), u.get('completion_tokens', 0),
                                             time.time() - t0, retries), r)


class MockProvider(LLMProvider):
    """Scripted responses for tests: a list of strings/dicts (consumed in order) or a callable."""

    def __init__(self, name='mock', cfg=None, responses: Union[list, Callable, None] = None):
        super().__init__(name, cfg or {'model': 'mock'})
        rfile = (cfg or {}).get('responses_file') or os.environ.get('AUV_MOCK_RESPONSES')
        if responses is None and rfile:
            with open(rfile) as f:                       # JSON list of outputs, used in order
                responses = json.load(f)
        self.responses = responses if responses is not None else []
        self.calls: list[list[dict]] = []

    def complete(self, messages, json_schema=None):
        self.calls.append([dict(m) for m in messages])   # snapshot; caller keeps appending
        if callable(self.responses):
            out = self.responses(messages)
        elif self.responses:
            out = self.responses.pop(0)
        else:
            raise LLMError('mock has no scripted responses left')
        text = out if isinstance(out, str) else json.dumps(out)
        tin = sum(len(m['content']) for m in messages) // 4
        return LLMResponse(text, self._usage(tin, len(text) // 4, 0.0))


TYPES = {'ollama': OllamaProvider, 'openai_compat': OpenAICompatProvider, 'mock': MockProvider}


def make_provider(llm_cfg: dict, name: Optional[str] = None, **kw) -> LLMProvider:
    name = name or llm_cfg.get('default_provider', 'ollama')
    providers = llm_cfg.get('providers', {})
    if name not in providers:
        raise ValueError(f'unknown provider {name!r}; configured: {sorted(providers)}')
    cfg = providers[name]
    cls = TYPES[cfg.get('type', name)]
    return cls(name, cfg, **kw)
