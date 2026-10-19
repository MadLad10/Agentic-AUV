#!/usr/bin/env bash
# Self-hosted trace backend for OpenTelemetry: Jaeger all-in-one (Docker Hub image, ~36 MB, in-memory store).
#   scripts/start_jaeger.sh          start (UI http://127.0.0.1:16686, OTLP http://127.0.0.1:4318)
#   scripts/start_jaeger.sh stop     stop and remove (frees RAM)
set -u
NAME=auv2-jaeger
IMAGE=jaegertracing/all-in-one:1.62.0
case "${1:-start}" in
  stop) docker rm -f "$NAME" >/dev/null 2>&1 && echo "[jaeger] stopped"; exit 0;;
  status) docker ps --filter name="^${NAME}$" --format '{{.Names}} {{.Status}}'; exit 0;;
esac
if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  docker rm -f "$NAME" >/dev/null 2>&1
  docker run -d --name "$NAME" --restart no --memory 512m \
    -e MEMORY_MAX_TRACES=5000 \
    -p 127.0.0.1:16686:16686 -p 127.0.0.1:4318:4318 "$IMAGE" >/dev/null
fi
for _ in $(seq 1 30); do curl -sf localhost:16686/ >/dev/null && break; sleep 1; done
curl -sf localhost:16686/ >/dev/null || { echo "[jaeger] UI not reachable" >&2; exit 1; }
echo "[jaeger] ready: UI http://127.0.0.1:16686  OTLP http://127.0.0.1:4318 (export AUV_OTEL=1)"
