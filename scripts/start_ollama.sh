#!/usr/bin/env bash
# Local LLM server via Docker (official ollama/ollama image from Docker Hub; no sudo, no GitHub).
#   scripts/start_ollama.sh          start (create if missing), wait for API, ensure model pulled
#   scripts/start_ollama.sh stop     stop the container (frees RAM)
#   scripts/start_ollama.sh status
set -u
NAME=auv2-ollama           # v2's own container (v1 uses auv-ollama:11436; the plain `ollama` is shared)
PORT="${AUV_OLLAMA_PORT:-11437}"
IMAGE=ollama/ollama:0.12.6
MODEL="${OLLAMA_MODEL:-qwen2.5:3b-instruct}"
MEM="${OLLAMA_MEM:-4g}"
case "${1:-start}" in
  stop) docker stop "$NAME" >/dev/null 2>&1 && echo "[ollama] stopped"; exit 0;;
  status) docker ps --filter name="^${NAME}$" --format '{{.Names}} {{.Status}}'; curl -s localhost:${PORT}/api/version; echo; exit 0;;
esac
if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  :
elif docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  docker start "$NAME" >/dev/null
else
  # one model in memory at a time: planner LLM and VLM together exceeded the memory cap (runner OOM-killed)
  docker run -d --name "$NAME" --restart no --memory "$MEM" -e OLLAMA_MAX_LOADED_MODELS=1 \
    -e OLLAMA_KEEP_ALIVE=10m -p 127.0.0.1:${PORT}:11434 \
    -v auv2-ollama:/root/.ollama "$IMAGE" >/dev/null
fi
for _ in $(seq 1 30); do curl -sf localhost:${PORT}/api/version >/dev/null && break; sleep 1; done
curl -sf localhost:${PORT}/api/version >/dev/null || { echo "[ollama] API not reachable" >&2; exit 1; }
if ! docker exec "$NAME" ollama list | awk '{print $1}' | grep -qx "$MODEL"; then
  echo "[ollama] pulling $MODEL"; docker exec "$NAME" ollama pull "$MODEL"
fi
# optional local vision-language model for the perception agent (e.g. VLM_MODEL=moondream)
if [ -n "${VLM_MODEL:-}" ] && ! docker exec "$NAME" ollama list | awk '{print $1}' | grep -q "^${VLM_MODEL}"; then
  echo "[ollama] pulling $VLM_MODEL"; docker exec "$NAME" ollama pull "$VLM_MODEL"
fi
echo "[ollama] ready: $MODEL at http://127.0.0.1:${PORT}"
