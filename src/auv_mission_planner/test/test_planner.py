"""Planner + provider abstraction with a mocked LLM (no network)."""
import io
import json
import os
import urllib.error

import pytest

from auv_mission_planner.config import load_config, load_llm_config
from auv_mission_planner.llm import base as llm_base
from auv_mission_planner.llm.base import LLMError, with_retries
from auv_mission_planner.llm.providers import (MockProvider, OllamaProvider, OpenAICompatProvider,
                                                  make_provider)
from auv_mission_planner.planner import Planner, extract_json, parse_output
from auv_mission_planner.prompts import EXAMPLE_PLAN
from auv_mission_planner.tools.fake_backend import FakeBackend
from auv_mission_planner.tools.registry import ToolRegistry

CTX = {'pose': {'x': 0, 'y': 0, 'depth': 0.1, 'yaw_deg': 0}, 'battery_percent': 90.0, 'remembered_objects': {}}


@pytest.fixture
def cfg():
    return load_config()


def planner_with(cfg, responses, retries=2):
    reg = ToolRegistry(FakeBackend(), cfg)
    mock = MockProvider(responses=responses)
    return Planner(mock, reg.schemas(), cfg['limits'], max_format_retries=retries), mock


def test_valid_plan_parsed(cfg):
    p, mock = planner_with(cfg, [EXAMPLE_PLAN])
    r = p.plan('pass the gate', CTX)
    assert r.status == 'plan' and len(r.plan.steps) == 5
    assert r.plan.steps[1].params == {'target': 'gate', 'pass_through': True}
    assert r.plan.abort_conditions[0].percent == 30
    assert len(r.calls) == 1 and r.calls[0]['kind'] == 'plan'
    assert r.tokens[0] > 0 and r.tokens[1] > 0


def test_prompt_contains_tools_limits_objects_and_state(cfg):
    p, mock = planner_with(cfg, [EXAMPLE_PLAN])
    p.plan('touch the red buoy', CTX)
    system, user = mock.calls[0][0]['content'], mock.calls[0][1]['content']
    for tool in cfg['limits']['allowed_tools']:
        assert tool in system
    for obj in cfg['limits']['known_objects']:
        assert obj in system
    assert f"0.0..{cfg['limits']['max_depth_m']} m" in system
    assert 'forward_m' in system                                   # relative moves are advertised
    assert 'touch the red buoy' in user and '"battery_percent": 90.0' in user


def test_format_errors_are_fed_back_and_retried(cfg):
    p, mock = planner_with(cfg, ['I think you should go to the gate.',
                                 '{"status": "plan", "plan": {"steps": []}}',
                                 '```json\n' + json.dumps(EXAMPLE_PLAN) + '\n```'])
    r = p.plan('x', CTX)
    assert r.status == 'plan'
    assert [c['kind'] for c in r.calls] == ['plan', 'plan_format_retry', 'plan_format_retry']
    assert 'no JSON object' in r.calls[0]['parse_error']
    assert 'at least one step' in r.calls[1]['parse_error']
    assert 'not a valid plan' in mock.calls[1][-1]['content']


def test_gives_up_after_max_format_retries(cfg):
    p, _ = planner_with(cfg, ['nope'] * 5, retries=1)
    r = p.plan('x', CTX)
    assert r.status == 'error' and len(r.calls) == 2


def test_refusal(cfg):
    p, _ = planner_with(cfg, [{'status': 'refuse', 'reason': 'no thruster tool', 'plan': None}])
    r = p.plan('thrusters 100%', CTX)
    assert r.status == 'refuse' and 'thruster' in r.reason


def test_refusal_needs_reason_and_bad_conditions_rejected():
    with pytest.raises(Exception):
        parse_output('{"status": "refuse", "plan": null}')
    bad = json.loads(json.dumps(EXAMPLE_PLAN))
    bad['plan']['steps'][2]['success'] = {'type': 'detected'}           # missing target
    with pytest.raises(Exception):
        parse_output(json.dumps(bad))
    bad = json.loads(json.dumps(EXAMPLE_PLAN))
    bad['plan']['steps'][0]['bogus'] = 1                                  # extra field
    with pytest.raises(Exception):
        parse_output(json.dumps(bad))


def test_lenient_wrappers():
    assert parse_output(json.dumps(EXAMPLE_PLAN['plan'])).status == 'plan'          # bare plan
    assert extract_json('Sure! {"a": {"b": "}"}} trailing') == {'a': {'b': '}'}}


def test_unknown_tools_and_unsafe_values_still_parse(cfg):
    """Structure-valid but unsafe plans must reach the validator (to be caught and counted)."""
    plan = {'status': 'plan', 'plan': {'steps': [
        {'id': 1, 'tool': 'set_thrusters', 'params': {'all': 100}},
        {'id': 2, 'tool': 'hold_depth', 'params': {'depth': 30}}]}}
    p, _ = planner_with(cfg, [plan])
    r = p.plan('x', CTX)
    assert r.status == 'plan' and r.plan.steps[0].tool == 'set_thrusters'


def test_repair_continues_conversation(cfg):
    p, mock = planner_with(cfg, [EXAMPLE_PLAN, EXAMPLE_PLAN])
    first = p.plan('x', CTX)
    r = p.repair(first, ['depth 30 m exceeds the maximum 4.0 m'])
    assert r.status == 'plan' and r.calls[0]['kind'] == 'validation_repair'
    convo = mock.calls[1]
    assert convo[2]['role'] == 'assistant' and 'REJECTED' in convo[3]['content']
    assert 'fixable' in convo[3]['content']
    assert 'depth 30 m' in convo[3]['content']


def test_replan_includes_failure_and_state(cfg):
    p, mock = planner_with(cfg, [EXAMPLE_PLAN])
    failed = {'step': {'id': 3, 'tool': 'navigate_to'}, 'observation': {'error_code': 'target_not_detected'}}
    r = p.replan('touch red', [{'id': 1, 'tool': 'hold_depth', 'ok': True}], failed, CTX, 1, 2)
    assert r.status == 'plan' and r.calls[0]['kind'] == 'replan'
    msg = mock.calls[0][1]['content']
    assert 'target_not_detected' in msg and 'replan 1 of 2' in msg and 'touch red' in msg


def test_llm_error_becomes_error_result(cfg):
    p, _ = planner_with(cfg, [])
    r = p.plan('x', CTX)
    assert r.status == 'error' and 'LLM call failed' in r.reason


# ---------------------------------------------------------------- providers


def test_factory_and_config():
    llm = load_llm_config()
    assert isinstance(make_provider(llm, 'ollama'), OllamaProvider)
    assert isinstance(make_provider(llm, 'groq'), OpenAICompatProvider)
    assert isinstance(make_provider(llm, 'mock'), MockProvider)
    with pytest.raises(ValueError):
        make_provider(llm, 'nonexistent')
    for name, p in llm['providers'].items():
        assert 'api_key' not in p, f'{name}: keys belong in .env'


def _fake_http(monkeypatch, responses):
    sent = []

    def fake(url, body, headers, timeout_s):
        sent.append((url, body, headers))
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r
    monkeypatch.setattr('auv_mission_planner.llm.providers.http_post_json', fake)
    return sent


def test_ollama_request_uses_schema_and_counts_tokens(monkeypatch):
    sent = _fake_http(monkeypatch, [{'message': {'content': '{"x": 1}'}, 'prompt_eval_count': 100,
                                     'eval_count': 20}])
    p = make_provider(load_llm_config(), 'ollama')
    r = p.complete([{'role': 'user', 'content': 'hi'}], {'type': 'object'})
    url, body, _ = sent[0]
    assert url.endswith('/api/chat') and body['format'] == {'type': 'object'} and body['stream'] is False
    assert r.usage.input_tokens == 100 and r.usage.output_tokens == 20 and r.usage.cost_usd == 0


def test_openai_compat_key_pricing_and_extra_body(monkeypatch):
    monkeypatch.setenv('GROQ_API_KEY', 'sk-test')
    sent = _fake_http(monkeypatch, [{'choices': [{'message': {'content': '{}'}}],
                                     'usage': {'prompt_tokens': 1_000_000, 'completion_tokens': 1_000_000}}])
    p = make_provider(load_llm_config(), 'groq')
    r = p.complete([{'role': 'user', 'content': 'hi'}], {'type': 'object'})
    url, body, headers = sent[0]
    assert url == 'https://api.groq.com/openai/v1/chat/completions'
    assert headers['Authorization'] == 'Bearer sk-test'
    assert body['response_format'] == {'type': 'json_object'} and body['reasoning_effort'] == 'low'
    assert r.usage.cost_usd == 0.0                                  # free tier
    assert r.usage.list_cost_usd == pytest.approx(0.15 + 0.60)     # list-price equivalent


def test_missing_api_key_is_clear_error(monkeypatch):
    monkeypatch.delenv('GROQ_API_KEY', raising=False)
    with pytest.raises(LLMError, match='GROQ_API_KEY'):
        make_provider(load_llm_config(), 'groq').complete([{'role': 'user', 'content': 'x'}])


def test_retry_on_rate_limit():
    calls = {'n': 0}
    slept = []

    def fn():
        calls['n'] += 1
        if calls['n'] < 3:
            e = LLMError('HTTP 429', retryable=True)
            e.retry_after = 1.5
            raise e
        return 'ok'
    out, retries = with_retries(fn, attempts=4, sleep=slept.append)
    assert out == 'ok' and retries == 2 and slept == [1.5, 1.5]
    with pytest.raises(LLMError):
        with_retries(lambda: (_ for _ in ()).throw(LLMError('HTTP 401', retryable=False)), sleep=slept.append)


def test_http_error_mapping(monkeypatch):
    def boom(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 429, 'rate', {'Retry-After': '3'}, io.BytesIO(b'slow down'))
    monkeypatch.setattr(llm_base.urllib.request, 'urlopen', boom)
    with pytest.raises(LLMError) as ei:
        llm_base.http_post_json('https://example.invalid/x', {}, {}, 1)
    assert ei.value.retryable and ei.value.retry_after == 3.0


def test_dotenv_loader(tmp_path, monkeypatch):
    from auv_mission_planner.config import load_dotenv
    f = tmp_path / '.env'
    f.write_text('# c\nAUV_TEST_KEY_A="abc"\nexport AUV_TEST_KEY_B=1\nAUV_TEST_EMPTY=\n')
    for k in ('AUV_TEST_KEY_A', 'AUV_TEST_KEY_B'):
        monkeypatch.setenv(k, 'x')
        monkeypatch.delenv(k)          # registers an undo, so nothing leaks into other tests
    got = load_dotenv(str(f))
    assert got == {'AUV_TEST_KEY_A': 'abc', 'AUV_TEST_KEY_B': '1'} and os.environ['AUV_TEST_KEY_A'] == 'abc'


def test_validation_retry_wording_depends_on_rule():
    from auv_mission_planner.prompts import validation_retry
    limit = validation_retry(['[depth_limit] step 1: depth 50.00 m exceeds the maximum 4.0 m'])
    assert 'never silently' in limit and 'refuse' in limit
    fixable = validation_retry(['[invalid_params] step 2: navigate_to: params: x: Field required'])
    assert 'fixable' in fixable and 'never silently' not in fixable
