#!/usr/bin/env bash
# Publication-grade evaluation campaign (PLAN.md §8): repeated runs + ablations, one frozen code version.
# Each (arm, seed) writes its own folder, so --resume skips finished missions after any interruption.
#   setsid nohup scripts/run_campaign.sh > eval/results/campaign_v23.log 2>&1 < /dev/null &
# Re-running the script continues where it stopped. Summary: eval/results/campaign_v23/campaign_summary.md
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
OUT=eval/results/campaign_v23
mkdir -p "$OUT"
if [ ! -e "$OUT/CODE_STATE.txt" ]; then
  { echo "base_commit: $(git rev-parse --short HEAD)"
    echo "uncommitted_diff_sha1 (src + eval runner): $(git diff HEAD -- src eval/run_eval_v2.py | sha1sum | cut -c1-12)"
    echo "started: $(date -Iseconds)"
    echo "note: campaign v2.3; code frozen; LLM qwen2.5:3b-instruct, VLM moondream (one model in memory)"
  } > "$OUT/CODE_STATE.txt"
fi
scripts/start_ollama.sh || exit 1
# headroom for the desktop: the laptop hard-froze twice during the v2.2 evaluation (cause unknown;
# scripts/sysmon.py records memory/CPU/temperature to logs/sysmon.log to catch the next one)
docker update --cpus 8 auv2-ollama > /dev/null
pgrep -f "scripts/sysmon[.]py" > /dev/null || setsid nohup python3 scripts/sysmon.py > /dev/null 2>&1 < /dev/null &

COMPLEX_UNSAFE=C1,C2,C3,C4,C5,R1,R2,R3,I1,I2,I3,U1,U2,U3
FAULTS=F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11

run() {   # run <dir> <systems> <seed> <ablation> <only>
  local dir="$OUT/$1"
  echo "[campaign] $(date +%H:%M:%S) $1"
  nice -n 10 ionice -c 3 python eval/run_eval_v2.py --systems "$2" --sets nominal,faults --provider ollama --seed "$3" \
    --ablation "$4" ${5:+--only "$5"} --out "$dir" --resume > "$dir.log" 2>&1
}

# interleave so a partial campaign already has every arm at seed 0
for seed in 0 1 2 3 4; do
  run "full_s$seed" v2 "$seed" none ""
  if [ "$seed" -le 2 ]; then
    run "no_validator_s$seed" v2 "$seed" no_validator "$COMPLEX_UNSAFE"
    run "no_health_s$seed"    v2 "$seed" no_health    "$FAULTS"
    run "no_fusion_s$seed"    v2 "$seed" no_fusion    "$FAULTS"
    run "single_agent_s$seed" v1 "$seed" none         "$FAULTS,U1,U2,U3"
  fi
  python eval/campaign_summary.py "$OUT"/*/runs.jsonl --out "$OUT" > /dev/null
done
echo "[campaign] finished $(date +%H:%M:%S)"
