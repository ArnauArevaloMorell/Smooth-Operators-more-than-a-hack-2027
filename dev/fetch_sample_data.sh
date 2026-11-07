#!/usr/bin/env bash
# Download Metrica Sports' public sample games (other matches -- allowed for training/tuning by
# RULES.md) into data/sample-data/, which is gitignored like the rest of data/.
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=https://raw.githubusercontent.com/metrica-sports/sample-data/master/data
OUT=data/sample-data
for n in 1 2; do
  mkdir -p "$OUT/Sample_Game_$n"
  for f in RawEventsData RawTrackingData_Home_Team RawTrackingData_Away_Team; do
    curl -sSfL -o "$OUT/Sample_Game_$n/Sample_Game_${n}_$f.csv" "$BASE/Sample_Game_$n/Sample_Game_${n}_$f.csv"
  done
done
mkdir -p "$OUT/Sample_Game_3"
for f in events.json metadata.xml tracking.txt; do
  curl -sSfL -o "$OUT/Sample_Game_3/Sample_Game_3_$f" "$BASE/Sample_Game_3/Sample_Game_3_$f"
done
echo "sample games in $OUT"
