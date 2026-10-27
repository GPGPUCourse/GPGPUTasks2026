#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p profiles

# При n = 100000000 и BLOCK_SIZE = 256 пропускаем уровни 6, 1526 и 390625.
# Профилируем финальный scan исходного массива.
ncu \
  --kernel-name-base function \
  --kernel-name prefix_sum_02_prefix_accumulation \
  --launch-skip 3 \
  --launch-count 1 \
  --set full \
  --import-source yes \
  --force-overwrite \
  --export profiles/prefix_accum \
  ./build/main_prefix_sum "${1:-0}"
