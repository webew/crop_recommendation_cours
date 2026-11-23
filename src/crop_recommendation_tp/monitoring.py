import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from crop_recommendation_tp.config import ROOT

# FastAPI traite les requêtes en parallèle : un verrou évite que deux lignes se mélangent
_verrou = threading.Lock()


def get_log_path(config: dict) -> Path:
    """Chemin du journal des requêtes ; la variable REQUESTS_LOG_PATH l'emporte sur config.yaml."""
    path = os.environ.get("REQUESTS_LOG_PATH") or config["monitoring"]["requests_log"]
    return ROOT / path


def log_request(path: Path, mesures: dict, culture: str, version: str) -> None:
    """Ajoute une ligne JSON au journal : date, mesures reçues, prédiction, version du modèle."""
    ligne = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        **mesures,
        "prediction": culture,
        "model_version": version,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with _verrou, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(ligne, ensure_ascii=False) + "\n")


def read_request_log(path: Path) -> pd.DataFrame:
    """Lit le journal des requêtes ; tableau vide si aucune requête n'a été reçue."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_json(path, lines=True, dtype={"model_version": str})
