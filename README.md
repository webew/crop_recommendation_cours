# Recommandation de cultures : démarche MLOps

Projet support du cours : recommander une culture (22 classes) à partir de 7 mesures du sol et du climat, puis industrialiser la démarche étape par étape.

## Étapes

Chaque étape correspond à un commit et à une étiquette Git. Pour passer à une étape donnée :

```bash
git checkout -f etape-2     # -f : abandonne les modifications locales des fichiers suivis
uv sync                     # installe les dépendances de cette étape
uv run dvc pull             # à partir de l'étape 2 : restaure les données gérées par DVC
```

La première fois qu'on atteint l'étape 2, le stockage DVC n'existe pas encore : suivre la section « Première mise en place » avant `dvc pull`.

Le serveur MLflow et ses essais (`mlflow.db`, `mlartifacts/`) ne sont pas suivis par Git : ils sont conservés d'une étape à l'autre.

| Étiquette | Étape | Apport |
| --- | --- | --- |
| `etape-0` | Point de départ | projet uv, configuration, notebook, MLflow local |
| `etape-1` | Validation des données | schéma des données, fonction `validate` |
| `etape-2` | Versionnage des données | DVC : données hors de Git, stockage local |
| `etape-3` | Du notebook au paquet | `data.py` : chargement validé, découpage, plis de validation croisée |

## Contenu

| Élément | Rôle |
| --- | --- |
| `pyproject.toml`, `uv.lock`, `.python-version` | Projet Python et versions exactes des dépendances |
| `config/config.yaml` | Réglages : données, découpage, MLflow, grille de recherche |
| `config/schema.yaml` | Schéma des données : colonnes, types, bornes, cultures autorisées |
| `src/crop_recommendation_tp/config.py` | Lecture des fichiers YAML |
| `src/crop_recommendation_tp/validation.py` | Contrôle d'un tableau par rapport au schéma |
| `src/crop_recommendation_tp/data.py` | Chargement et validation des données, variables explicatives, découpage, plis de validation croisée |
| `data/raw/crop_recommendation.csv.dvc` | Étiquette DVC des données : empreinte md5 et taille |
| `data/raw/crop_recommendation.csv` | Données (2 200 parcelles), gérées par DVC et ignorées par Git |
| `.dvc/config` | Configuration de DVC, dont l'adresse du stockage |
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

## Données (DVC)

Le fichier de données est versionné avec DVC. Git conserve l'étiquette `data/raw/crop_recommendation.csv.dvc` ; le contenu est rangé dans le cache DVC (`.dvc/cache/`) et dans le stockage `../crop_recommendation_cours_storage`, un dossier situé à côté du dépôt.

### Première mise en place (passage à l'étape 2)

En passant de l'étape 1 à l'étape 2, Git supprime le CSV du dossier de travail : il n'est plus suivi par Git, et le stockage DVC n'existe pas encore sur le poste. Le récupérer depuis l'historique Git, puis le confier à DVC :

```bash
git show etape-1:data/raw/crop_recommendation.csv > data/raw/crop_recommendation.csv
uv run dvc add data/raw/crop_recommendation.csv
uv run dvc push
```

`dvc add` retrouve la même empreinte que l'étiquette `.dvc` : `git status` ne signale aucune modification.

### Commandes

| Commande | Effet |
| --- | --- |
| `uv run dvc status` | Compare les données à l'étiquette `.dvc` |
| `uv run dvc add <fichier>` | Calcule l'empreinte, range le fichier dans le cache, met à jour le `.dvc` |
| `uv run dvc push` | Envoie les données du cache vers le stockage |
| `uv run dvc pull` | Récupère les données décrites par les fichiers `.dvc` |
| `uv run dvc checkout --force` | Remet les données dans l'état décrit par les fichiers `.dvc`, en écrasant les modifications locales |

Contrôle : `uv run dvc status` doit afficher `Data and pipelines are up to date.`

## Contrôler les données

Le schéma `config/schema.yaml` décrit les données attendues. La fonction `validate` renvoie la liste des écarts ; une liste vide signifie des données conformes.

```bash
uv run python -c "import pandas as pd; from crop_recommendation_tp.config import load_config, load_yaml; from crop_recommendation_tp.validation import validate; c = load_config(); print(validate(pd.read_csv(c['dataset']['path']), load_yaml(c['dataset']['schema'])))"
```

Résultat attendu : `[]`.

## Résolution de problèmes

| Message ou symptôme | Cause | Solution |
| --- | --- | --- |
| `Connection refused` dans le notebook | Serveur MLflow arrêté | Démarrer MLflow (section précédente) |
| `Address already in use` au démarrage de MLflow | Port 5001 déjà utilisé | Choisir un autre port (`--port 5003`) et le reporter dans `mlflow.tracking_uri` de `config/config.yaml` |
| `ModuleNotFoundError: crop_recommendation_tp` | Noyau du notebook différent de `.venv` | Sélectionner le noyau `.venv`, puis relancer |
| `FileNotFoundError: 'config/config.yaml'` | Le notebook ne tourne pas depuis la racine du projet | Redémarrer le noyau et relancer toutes les cellules |
| `FileNotFoundError: … crop_recommendation.csv` après `git checkout etape-2` | Données retirées de Git, stockage DVC absent | Section « Première mise en place » |
| `Some of the cache files do not exist` lors de `dvc pull` | Stockage DVC absent sur ce poste | Section « Première mise en place » |
