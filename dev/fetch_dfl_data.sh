#!/usr/bin/env bash
# Download the DFL / IDSSE open dataset (other matches -- allowed by RULES.md) into data/dfl/,
# which is gitignored like the rest of data/. Match information + positions only (~2.6 GB).
#
#   Bassek, M. et al. (2025). An integrated dataset of spatiotemporal and event data in elite
#   soccer. Scientific Data 12, 195. doi:10.1038/s41597-025-04505-y
#   figshare doi:10.6084/m9.figshare.28196177 -- licence CC BY 4.0
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/dfl
cd data/dfl
curl -sSf https://api.figshare.com/v2/articles/28196177 | python3 -c "
import json, sys
for f in json.load(sys.stdin)['files']:
    if 'matchinformation' in f['name'] or 'positions' in f['name']:
        print(f['name'], f['download_url'])" | while read -r name url; do
  if [ ! -s "$name" ]; then
    curl -sSfL -o "$name" "$url"
    echo "downloaded $name"
  fi
done
echo "DFL data in data/dfl/"
