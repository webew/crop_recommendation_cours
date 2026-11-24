import sys

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from crop_recommendation_tp.config import ROOT, load_config
from crop_recommendation_tp.data import get_features, load_dataset, split_dataset
from crop_recommendation_tp.monitoring import get_log_path, read_request_log


def load_reference(config: dict) -> pd.DataFrame:
    """Données de référence : les variables du jeu d'entraînement, celles que le modèle a apprises."""
    X_train, _, _, _ = split_dataset(load_dataset(config), config)
    return X_train.reset_index(drop=True)


def run_report(reference: pd.DataFrame, current: pd.DataFrame, monitoring: dict):
    """Compare chaque variable des requêtes reçues à la même variable de la référence.

    Le test (Kolmogorov-Smirnov) renvoie une p-value par variable : la probabilité
    d'observer un tel écart si les deux échantillons venaient de la même distribution.
    """
    report = Report(
        [
            DataDriftPreset(
                method=monitoring["drift_method"],
                threshold=monitoring["drift_threshold"],
                drift_share=monitoring["drift_share"],
            )
        ]
    )
    return report.run(current_data=current, reference_data=reference)


def summarize(snapshot) -> dict:
    """Extrait du rapport Evidently, pour chaque variable, la p-value et la décision."""
    colonnes = {}
    part = seuil_part = None
    for metric in snapshot.dict()["metrics"]:
        config = metric["config"]
        if config["type"].endswith("ValueDrift"):
            p_value, seuil = float(metric["value"]), config["threshold"]
            colonnes[config["column"]] = {
                "p_value": p_value,
                "seuil": seuil,
                "derive": p_value < seuil,  # écart trop grand pour être dû au hasard
            }
        elif config["type"].endswith("DriftedColumnsCount"):
            part, seuil_part = metric["value"]["share"], config["drift_share"]
    return {"colonnes": colonnes, "part": part, "derive": part >= seuil_part}


def main() -> None:
    """Compare le journal des requêtes aux données d'entraînement ; code de sortie 1 si dérive."""
    config = load_config()
    monitoring = config["monitoring"]
    journal = read_request_log(get_log_path(config))
    if len(journal) < monitoring["min_requests"]:
        print(
            f"{len(journal)} requête(s) dans le journal : il en faut au moins "
            f"{monitoring['min_requests']} pour conclure."
        )
        return

    snapshot = run_report(
        load_reference(config), journal[get_features(config)], monitoring
    )
    resume = summarize(snapshot)
    report_path = ROOT / monitoring["drift_report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(report_path))

    print(f"{len(journal)} requêtes comparées aux données d'entraînement.")
    for nom, colonne in resume["colonnes"].items():
        decision = "DÉRIVE" if colonne["derive"] else "ok"
        print(
            f"  {nom:<12} p-value {colonne['p_value']:<8.3g} "
            f"(seuil {colonne['seuil']}) : {decision}"
        )
    print(
        f"Variables en dérive : {resume['part']:.0%}. "
        f"Rapport : {monitoring['drift_report']}"
    )
    if resume["derive"]:
        print("Dérive détectée : alerter, collecter de nouvelles données, réentraîner.")
        sys.exit(1)
    print("Pas de dérive détectée.")
