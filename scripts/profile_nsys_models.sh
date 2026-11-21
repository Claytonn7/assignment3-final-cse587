#!/usr/bin/env bash
set -e
MODE=${1:-inference}   # inference | train_step
OUT_DIR="runs"; mkdir -p "$OUT_DIR"
for MODEL in small medium; do
  for CTX in 128 256 512 1024; do
    OUT="${OUT_DIR}/nsys_${MODE}_${MODEL}_s${CTX}"
    echo ">>> model=$MODEL ctx=$CTX mode=$MODE -> $OUT.nsys-rep"
    uv run nsys profile -o "$OUT" --force-overwrite=true \
      --trace=cuda,nvtx --sample=none --cpuctxsw=none \
      python systems/benchmark.py --model-size "$MODEL" --context-length "$CTX" \
        --nvtx --profile --profile-mode "$MODE" --profile-steps 1 \
      || echo "[failed / OOM] model=$MODEL ctx=$CTX"
  done
done
