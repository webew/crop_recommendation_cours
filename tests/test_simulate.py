import pandas as pd
import pytest

from crop_recommendation_tp.simulate import build_requests


@pytest.fixture
def X_test():
    """Petit jeu de test fabriqué : 10 parcelles."""
    return pd.DataFrame(
        {
            "N": range(10),
            "P": range(10),
            "K": range(10),
            "temperature": [20.0 + i for i in range(10)],
            "humidity": [80.0] * 10,
            "ph": [6.5] * 10,
            "rainfall": [150.0 + 3 * i for i in range(10)],
        }
    )


def test_nombre_de_requetes_et_tirage_reproductible(X_test):
    premier = build_requests(X_test, "normal", 25, seed=42)
    second = build_requests(X_test, "normal", 25, seed=42)
    assert len(premier) == 25  # tirage avec remise : plus de requêtes que de parcelles
    pd.testing.assert_frame_equal(premier, second)


def test_region_chaude_ajoute_10_degres(X_test):
    normal = build_requests(X_test, "normal", 20, seed=42)
    chaud = build_requests(X_test, "region-chaude", 20, seed=42)
    assert (chaud["temperature"] - normal["temperature"]).round(6).eq(10).all()
    pd.testing.assert_frame_equal(
        chaud.drop(columns="temperature"), normal.drop(columns="temperature")
    )


def test_secheresse_divise_la_pluie_par_3(X_test):
    normal = build_requests(X_test, "normal", 20, seed=42)
    sec = build_requests(X_test, "secheresse", 20, seed=42)
    assert (sec["rainfall"] - normal["rainfall"] / 3).abs().max() <= 0.05  # arrondi
    pd.testing.assert_frame_equal(
        sec.drop(columns="rainfall"), normal.drop(columns="rainfall")
    )


def test_concept_ne_change_pas_les_mesures(X_test):
    pd.testing.assert_frame_equal(
        build_requests(X_test, "concept", 20, seed=42),
        build_requests(X_test, "normal", 20, seed=42),
    )
