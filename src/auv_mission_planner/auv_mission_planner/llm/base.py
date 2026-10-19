"""LLM provider abstraction. Providers only turn messages into text; parsing lives in the planner."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class Usage:
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0          # what this call actually cost (free tiers -> 0)
    list_cost_usd: float = 0.0     # what it would cost at paid list price (for comparison)
    retries: int = 0

    def as_dict(self):
        return asdict(self)


@dataclass
class LLMResponse:
    text: str
    usage: Usage
    raw: dict = field(default_factory=dict)


class LLMError(RuntimeError):
    def __init__(self, msg, retryable=False):
        super().__init__(msg)
        self.retryable = retryable


class LLMProvider(ABC):
    def __init__(self, name: str, cfg: dict):
        self.name = name
        self.cfg = cfg
        self.model = cfg.get('model', name)
        self.temperature = cfg.get('temperature', 0.1)
        self.timeout_s = cfg.get('timeout_s', 120)
        p = cfg.get('pricing', {})
        self.price_in = p.get('input_per_mtok', 0.0)
        self.price_out = p.get('output_per_mtok', 0.0)
        self.list_in = p.get('list_input_per_mtok', self.price_in)
        self.list_out = p.get('list_output_per_mtok', self.price_out)

    @abstractmethod
    def complete(self, messages: list[dict], json_schema: Optional[dict] = None) -> LLMResponse: ...

    def _usage(self, tin, tout, latency, retries=0) -> Usage:
        return Usage(self.name, self.model, int(tin), int(tout), round(latency, 3),
                     round((tin * self.price_in + tout * self.price_out) / 1e6, 6),
                     round((tin * self.list_in + tout * self.list_out) / 1e6, 6), retries)

    def describe(self) -> dict:
        return {'provider': self.name, 'model': self.model, 'type': self.cfg.get('type')}


def http_post_json(url: str, body: dict, headers: dict, timeout_s: float) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method='POST',
                                 headers={'Content-Type': 'application/json', **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors='replace')[:500]
        retry_after = e.headers.get('Retry-After') if e.headers else None
        err = LLMError(f'HTTP {e.code} from {url}: {detail}', retryable=e.code in (429, 500, 502, 503, 504))
        err.retry_after = float(retry_after) if retry_after and retry_after.replace('.', '').isdigit() else None
        raise err
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise LLMError(f'cannot reach {url}: {e}', retryable=True)


def with_retries(fn, attempts=4, base_delay=2.0, sleep=time.sleep):
    """Call fn(); retry retryable LLMErrors with exponential backoff (honours Retry-After)."""
    for i in range(attempts):
        try:
            return fn(), i
        except LLMError as e:
            if not e.retryable or i == attempts - 1:
                raise
            delay = getattr(e, 'retry_after', None) or base_delay * (2 ** i)
            sleep(min(delay, 30.0))
