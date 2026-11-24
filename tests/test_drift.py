import numpy as np
import pandas as pd
import pytest

from crop_recommendation_tp import drift
from crop_recommendation_tp.drift import run_report, summarize
from crop_recommendation_tp.monitoring import log_request

MONITORING = {
    "drift_method": "ks",
    "drift_threshold": 0.05,
    "drift_share": 0.1,
    "min_requests": 50,
}


def parcelles(n: int, seed: int, chaleur: float = 0.0) -> pd.DataFrame:
    """Tableau fabriqué de deux variables ; chaleur décale la température."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "temperature": rng.normal(25 + chaleur, 5, n),
            "rainfall": rng.normal(100, 30, n),
        }
    )


@pytest.fixture
def reference():
    return parcelles(1000, seed=0)


def test_meme_distribution_pas_de_derive(reference):
    resume = summarize(run_report(reference, parcelles(200, seed=1), MONITORING))
    assert resume["derive"] is False
    assert not any(c["derive"] for c in resume["colonnes"].values())


def test_temperature_decalee_seule_en_derive(reference):
    courant = parcelles(200, seed=1, chaleur=10)
    resume = summarize(run_report(reference, courant, MONITORING))
    assert resume["derive"] is True
    assert resume["colonnes"]["temperature"]["derive"] is True
    assert resume["colonnes"]["rainfall"]["derive"] is False
    assert resume["part"] == 0.5


def test_pas_de_conclusion_sous_le_minimum(config, monkeypatch, capsys):
    config["monitoring"].update(MONITORING)
    monkeypatch.setattr(drift, "load_config", lambda: config)
    path = drift.get_log_path(config)
    for _ in range(3):
        log_request(path, {"temperature": 25.0, "rainfall": 100.0}, "rice", "1")
    drift.main()  # ni rapport, ni code de sortie d'erreur
    assert "il en faut au moins 50" in capsys.readouterr().out
