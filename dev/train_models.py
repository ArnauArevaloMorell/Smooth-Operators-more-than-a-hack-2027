"""Train the shipped per-frame models on every development case -> solution/models.json.

    bash dev/fetch_sample_data.sh        # once: Metrica's public sample games into data/
    bash dev/fetch_dfl_data.sh           # once: the DFL/IDSSE open matches into data/dfl/
    python dev/train_models.py

Training data: Metrica Sports' public sample games 1-3 (github.com/metrica-sports/sample-data),
labels derived from their event data by dev/sample_games.py, and the seven DFL/IDSSE open matches
(CC BY 4.0, labels from their ball-status and possession flags, dev/dfl_games.py); tracking
degraded by dev/degrade.py. No BAR-ATH data of any kind is read. Deterministic (fixed seeds), so
the JSON is reproducible.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness as hz  # noqa: E402
from solution import logreg, pipeline  # noqa: E402

if __name__ == "__main__":
    models = hz.train(hz.CASES)
    models["_trained_on"] = ("Metrica public sample games 1-3 and DFL/IDSSE open matches 1-7, "
                             "degradations " + ", ".join(sorted({c[1] for c in hz.CASES})))
    logreg.save(models, pipeline.MODELS_PATH)
    print("wrote", os.path.relpath(pipeline.MODELS_PATH, hz.ROOT))
