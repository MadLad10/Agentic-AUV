"""Vision-language model (VLM) clients for occasional scene queries (not per frame).

Used by the perception agent on demand: describe the scene when a search fails, explain a degraded
camera, or check a sonar contact. Every call is bounded by a latency budget; a late or failed answer
is discarded and the mission carries on with the deterministic perception output.

Providers: groq (OpenAI-compatible vision chat, key in .env), ollama (local VLM, e.g. moondream),
mock (tests).
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.request

import numpy as np

PROMPT = ('You are the camera analyst of an underwater robot in a test pool. The course has an orange '
          'two-post gate and red, green and yellow buoys. Answer in at most three short sentences.')


def _jpeg_b64(img: np.ndarray, max_w: int = 512) -> str:
    from PIL import Image
    im = Image.fromarray(img)
    if im.width > max_w:
        im = im.resize((max_w, int(im.height * max_w / im.width)))
    buf = io.BytesIO()
    im.save(buf, format='JPEG', quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _post(url, body, headers, timeout):
    req = urllib.request.Request(url, json.dumps(body).encode(), {'Content-Type': 'application/json', **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class GroqVLM:
    def __init__(self, model: str, timeout_s: float, base_url='https://api.groq.com/openai/v1'):
        self.model, self.timeout, self.base_url = model, timeout_s, base_url
        self.key = os.environ.get('GROQ_API_KEY', '')
        if not self.key:
            raise RuntimeError('GROQ_API_KEY not set (.env)')

    def describe(self, img, question):
        body = {'model': self.model, 'temperature': 0.1, 'max_tokens': 160, 'messages': [
            {'role': 'system', 'content': PROMPT},
            {'role': 'user', 'content': [{'type': 'text', 'text': question},
                                         {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + _jpeg_b64(img)}}]}]}
        r = _post(f'{self.base_url}/chat/completions', body, {'Authorization': f'Bearer {self.key}'}, self.timeout)
        return r['choices'][0]['message']['content'].strip()


class OllamaVLM:
    def __init__(self, model: str, timeout_s: float, host: str):
        self.model, self.timeout, self.host = model, timeout_s, host.rstrip('/')

    def warm_up(self):
        """Load the model now (~40 s on CPU) so mission-time queries fit the latency budget."""
        import threading

        def go():
            try:
                body = {'model': self.model, 'stream': False, 'keep_alive': '30m',
                        'messages': [{'role': 'user', 'content': 'ok'}], 'options': {'num_predict': 1}}
                _post(f'{self.host}/api/chat', body, {}, 180)
            except Exception:
                pass
        threading.Thread(target=go, daemon=True).start()

    def describe(self, img, question):
        body = {'model': self.model, 'stream': False, 'keep_alive': '30m',
                'options': {'temperature': 0.1, 'num_predict': 120},
                'messages': [{'role': 'system', 'content': PROMPT},
                             {'role': 'user', 'content': question, 'images': [_jpeg_b64(img, 384)]}]}
        r = _post(f'{self.host}/api/chat', body, {}, self.timeout)
        return r['message']['content'].strip()


class MockVLM:
    model = 'mock-vlm'

    def __init__(self, answer='murky water; no buoy visible; an orange post at the left edge', delay_s=0.0):
        self.answer, self.delay = answer, delay_s

    def describe(self, img, question):
        time.sleep(self.delay)
        if img is not None and img.max() == 0:
            return 'the image is completely black: the camera is not producing a picture'
        return self.answer


def make_vlm(cfg: dict):
    p = cfg.get('provider', 'none')
    if p in (None, 'none'):
        return None
    if p == 'mock':
        return MockVLM()
    if p == 'groq':
        return GroqVLM(cfg['groq_model'], cfg['timeout_s'])
    if p == 'ollama':
        return OllamaVLM(cfg['ollama_model'], cfg['timeout_s'], cfg['ollama_host'])
    raise ValueError(f'unknown VLM provider {p!r}')
