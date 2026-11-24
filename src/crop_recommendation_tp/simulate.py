import argparse
import json
import os
import urllib.error
import urllib.request
from collections import Counter

import pandas as pd

from crop_recommendation_tp.config import load_config
from crop_recommendation_tp.data import load_dataset, split_dataset

SCENARIOS = {
    "normal": "parcelles comparables à celles de l'entraînement",
    "region-chaude": "nouvelle région, 10 °C plus chaude",
    "secheresse": "sécheresse : trois fois moins de pluie",
    "concept": "mêmes mesures qu'à l'entraînement ; seule la bonne réponse a changé",
}


def build_requests(
    X_test: pd.DataFrame, scenario: str, n: int, seed: int
) -> pd.DataFrame:
    """Tire n parcelles du jeu de test et leur applique la transformation du scénario."""
    mesures = X_test.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)
    if scenario == "region-chaude":
        mesures["temperature"] = mesures["temperature"] + 10
    elif scenario == "secheresse":
        mesures["rainfall"] = mesures["rainfall"] / 3
    # normal et concept : mesures inchangées (le concept drift ne se voit pas dans les entrées)
    return mesures.round(1)


def send(url: str, mesures: dict) -> str | None:
    """Envoie une requête à /predict ; retourne la culture prédite, ou None si elle est rejetée."""
    requete = urllib.request.Request(
        f"{url}/predict",
        data=json.dumps(mesures).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(requete) as reponse:
            return json.load(reponse)["culture"]
    except urllib.error.HTTPError as erreur:
        if erreur.code == 422:
            return None
        raise


def main() -> None:
    """Joue des utilisateurs de l'API : envoie n parcelles selon un scénario."""
    parser = argparse.ArgumentParser(
        description="Envoie des requêtes simulées à l'API."
    )
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--n", type=int, default=200, help="nombre de requêtes")
    parser.add_argument("--seed", type=int, default=42, help="graine du tirage")
    parser.add_argument(
        "--url",
        default=os.environ.get("API_URL", "http://127.0.0.1:8000"),
        help="adresse de l'API (variable API_URL)",
    )
    args = parser.parse_args()

    config = load_config()
    _, X_test, _, _ = split_dataset(load_dataset(config), config)
    requetes = build_requests(X_test, args.scenario, args.n, args.seed)

    cultures = Counter()
    try:
        for mesures in requetes.to_dict(orient="records"):
            cultures[send(args.url, mesures)] += 1
    except urllib.error.URLError as erreur:
        raise SystemExit(
            f"API injoignable à {args.url} ({erreur.reason}) : "
            "démarrer la pile avec docker compose up -d."
        ) from erreur

    rejetees = cultures.pop(None, 0)
    print(f"Scénario {args.scenario} : {SCENARIOS[args.scenario]}.")
    print(f"{args.n} requêtes envoyées, {rejetees} rejetée(s).")
    print("Cultures prédites les plus fréquentes :", dict(cultures.most_common(5)))
