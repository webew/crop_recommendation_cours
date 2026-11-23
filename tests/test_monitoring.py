from crop_recommendation_tp.monitoring import (
    get_log_path,
    log_request,
    read_request_log,
)

MESURES = {
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 20.8,
    "humidity": 82.0,
    "ph": 6.5,
    "rainfall": 202.9,
}


def test_journal_absent(tmp_path):
    assert read_request_log(tmp_path / "requests.jsonl").empty


def test_une_ligne_par_requete(tmp_path):
    path = tmp_path / "logs" / "requests.jsonl"  # dossier créé au besoin
    log_request(path, MESURES, "rice", "1")
    log_request(path, {**MESURES, "ph": 7.0}, "maize", "2")
    journal = read_request_log(path)
    assert len(journal) == 2
    assert journal["prediction"].tolist() == ["rice", "maize"]
    assert journal["model_version"].tolist() == ["1", "2"]
    assert journal["ph"].tolist() == [6.5, 7.0]
    assert list(journal.columns) == [
        "timestamp",
        *MESURES,
        "prediction",
        "model_version",
    ]


def test_variable_d_environnement_prioritaire(config, tmp_path, monkeypatch):
    monkeypatch.setenv("REQUESTS_LOG_PATH", (tmp_path / "autre.jsonl").as_posix())
    assert get_log_path(config) == tmp_path / "autre.jsonl"
