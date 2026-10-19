"""OpenTelemetry tracing for the multi-agent graph (exported over OTLP to a self-hosted Jaeger).

Span tree per mission:
  mission  (prompt, outcome, referee results)
    graph.<node>           one per LangGraph node execution (supervisor, validate, execute_step, ...)
      llm.<kind>           every LLM call: gen_ai.* attributes (system, model, tokens, latency)
      tool.<name>          every vehicle tool call: ok, error_code, sim time, duration
The JSON traces of v1 are kept; OTel adds a live, queryable view (Jaeger UI on :16686).

Enable with AUV_OTEL=1 (endpoint OTEL_EXPORTER_OTLP_ENDPOINT, default http://127.0.0.1:4318).
"""
from __future__ import annotations

import functools
import os
from contextlib import contextmanager

_tracer = None


def init(service: str = 'auv-agents', exporter=None):
    """Set up the tracer. `exporter` overrides OTLP (tests pass an in-memory exporter)."""
    global _tracer
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
    provider = TracerProvider(resource=Resource.create({'service.name': service}))
    if exporter is None:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        ep = os.environ.get('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://127.0.0.1:4318').rstrip('/')
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=f'{ep}/v1/traces')))
    else:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    _tracer = provider.get_tracer('auv_agents')
    return provider


def enabled() -> bool:
    return _tracer is not None


def maybe_init_from_env():
    if os.environ.get('AUV_OTEL') == '1' and _tracer is None:
        return init()
    return None


@contextmanager
def span(name: str, **attrs):
    if _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name) as s:
        for k, v in attrs.items():
            _set(s, k, v)
        yield s


def _set(s, k, v):
    if s is None or v is None:
        return
    if isinstance(v, (str, bool, int, float)):
        s.set_attribute(k, v)
    else:
        s.set_attribute(k, str(v)[:1000])


def node(name: str, fn):
    """Wrap a LangGraph node so each execution is a span."""
    @functools.wraps(fn)
    def wrapped(state):
        with span(f'graph.{name}', **{'auv.phase_in': state.get('phase')}) as s:
            out = fn(state)
            if s is not None and isinstance(out, dict):
                _set(s, 'auv.phase_out', out.get('phase'))
                for ev in out.get('events', [])[:5]:
                    s.add_event(ev.get('agent', '?'), {'msg': str(ev.get('msg', ''))[:300]})
            return out
    return wrapped


class TracedProvider:
    """LLM provider proxy: one span per completion with GenAI semantic-convention attributes."""

    def __init__(self, inner, kind: str = 'chat'):
        self._inner = inner
        self._kind = kind

    def __getattr__(self, k):
        return getattr(self._inner, k)

    def complete(self, messages, json_schema=None):
        with span(f'llm.{self._kind}', **{'gen_ai.system': self._inner.name,
                                          'gen_ai.request.model': self._inner.model,
                                          'gen_ai.operation.name': 'chat'}) as s:
            r = self._inner.complete(messages, json_schema)
            if s is not None:
                u = r.usage
                _set(s, 'gen_ai.usage.input_tokens', u.input_tokens)
                _set(s, 'gen_ai.usage.output_tokens', u.output_tokens)
                _set(s, 'auv.llm.latency_s', u.latency_s)
                _set(s, 'auv.llm.cost_usd', u.cost_usd)
            return r


class TracedRegistry:
    """Tool registry proxy: one span per tool call."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, k):
        return getattr(self._inner, k)

    def call(self, name, params=None, timeout_s=60.0):
        with span(f'tool.{name}', **{'auv.tool.params': params or {}, 'auv.tool.timeout_s': timeout_s}) as s:
            obs = self._inner.call(name, params, timeout_s)
            if s is not None:
                _set(s, 'auv.tool.ok', obs.ok)
                _set(s, 'auv.tool.error_code', obs.error_code)
                _set(s, 'auv.sim_time', obs.sim_time)
                _set(s, 'auv.tool.duration_s', obs.duration_s)
            return obs
