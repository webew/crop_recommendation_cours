# Recommandation de cultures : démarche MLOps

Projet support du cours : recommander une culture (22 classes) à partir de 7 mesures du sol et du climat, puis industrialiser la démarche étape par étape.

## Point de départ

Le projet contient :

| Élément | Rôle |
| --- | --- |
| `pyproject.toml`, `uv.lock`, `.python-version` | Projet Python et versions exactes des dépendances |
| `config/config.yaml` | Réglages : données, découpage, MLflow, grille de recherche |
| `src/crop_recommendation_tp/config.py` | Lecture de la configuration |
| `data/raw/crop_recommendation.csv` | Données (2 200 parcelles) |
| `notebooks/01_demarche_modelisation.ipynb` | Démarche de modélisation, essais enregistrés dans MLflow |

## Prérequis

| Outil | Rôle |
| --- | --- |
| [Git](https://git-scm.com/) | Historique du code |
| [uv](https://docs.astral.sh/uv/) | Python, environnement virtuel et dépendances |
| [VS Code](https://code.visualstudio.com/) | Éditeur, avec les extensions *Python* et *Jupyter* |

La version de Python (3.13) est indiquée dans `.python-version` : uv l'installe si nécessaire.

## Installation

```bash
git clone <url-du-dépôt>
cd crop_recommendation_cours
uv sync
```

`uv sync` crée l'environnement virtuel `.venv` et installe les versions exactes de `uv.lock`.

## Démarrer MLflow

Dans un terminal dédié, à la racine du projet :

```bash
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5001
```

- Interface : <http://127.0.0.1:5001>
- Le terminal reste occupé tant que le serveur tourne ; `Ctrl+C` l'arrête.
- Les essais sont conservés dans `mlflow.db` et `mlartifacts/`, à la racine du projet. Ces fichiers ne sont pas versionnés.

## Exécuter le notebook

1. Ouvrir `notebooks/01_demarche_modelisation.ipynb` dans VS Code.
2. Choisir le noyau : *Sélectionner un noyau* → *Environnements Python* → `.venv` du projet.
3. Lancer toutes les cellules (*Run All*).
4. Dans MLflow, ouvrir l'expérience `crop-recommendation-exploration` : un run par modèle comparé, un run de réglage et un run « modèle final ».

## Résolution de problèmes

| Message ou symptôme | Cause | Solution |
| --- | --- | --- |
| `Connection refused` dans le notebook | Serveur MLflow arrêté | Démarrer MLflow (section précédente) |
| `Address already in use` au démarrage de MLflow | Port 5001 déjà utilisé | Choisir un autre port (`--port 5003`) et le reporter dans `mlflow.tracking_uri` de `config/config.yaml` |
| `ModuleNotFoundError: crop_recommendation_tp` | Noyau du notebook différent de `.venv` | Sélectionner le noyau `.venv`, puis relancer |
| `FileNotFoundError: 'config/config.yaml'` | Le notebook ne tourne pas depuis la racine du projet | Redémarrer le noyau et relancer toutes les cellules |
