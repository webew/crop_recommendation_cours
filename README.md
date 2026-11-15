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

Les essais MLflow ne sont pas suivis par Git : ils sont conservés d'une étape à l'autre.

| Étiquette | Étape | Apport |
| --- | --- | --- |
| `etape-0` | Point de départ | projet uv, configuration, notebook, MLflow local |
| `etape-1` | Validation des données | schéma des données, fonction `validate` |
| `etape-2` | Versionnage des données | DVC : données hors de Git, stockage local |
| `etape-3` | Du notebook au paquet | `data.py` : chargement validé, découpage, plis de validation croisée |
| `etape-4` | Tests et qualité du code | pytest, Ruff |
| `etape-5` | Entraînement en script | `uv run tune`, `uv run train`, runs tracés dans MLflow |
| `etape-6` | Intégration continue | GitHub Actions : Ruff et tests à chaque push |
| `etape-7` | Serveur MLflow | Docker Compose : MLflow et PostgreSQL |
| `etape-8` | Registre et promotion | `uv run promote`, alias `champion` |

## Contenu

| Élément | Rôle |
| --- | --- |
| `pyproject.toml`, `uv.lock`, `.python-version` | Projet Python et versions exactes des dépendances |
| `config/config.yaml` | Réglages : données, découpage, MLflow, hyperparamètres du modèle, grille de recherche |
| `config/schema.yaml` | Schéma des données : colonnes, types, bornes, cultures autorisées |
| `src/crop_recommendation_tp/config.py` | Lecture des fichiers YAML |
| `src/crop_recommendation_tp/validation.py` | Contrôle d'un tableau par rapport au schéma |
| `src/crop_recommendation_tp/data.py` | Chargement et validation des données, variables explicatives, découpage, plis de validation croisée, empreinte des données |
| `src/crop_recommendation_tp/tracking.py` | Connexion à MLflow, enregistrement des runs |
| `src/crop_recommendation_tp/tune.py` | Commande `tune` : recherche des hyperparamètres |
| `src/crop_recommendation_tp/train.py` | Commande `train` : entraînement et évaluation du modèle |
| `src/crop_recommendation_tp/promote.py` | Commande `promote` : choix du modèle champion |
| `data/raw/crop_recommendation.csv.dvc` | Étiquette DVC des données : empreinte md5 et taille |
| `data/raw/crop_recommendation.csv` | Données (2 200 parcelles), gérées par DVC et ignorées par Git |
| `.dvc/config` | Configuration de DVC, dont l'adresse du stockage |
| `notebooks/01_demarche_modelisation.ipynb` | Démarche de modélisation, essais enregistrés dans MLflow |
| `tests/` | Tests automatisés (pytest) |
| `.vscode/settings.json` | Ruff à l'enregistrement dans VS Code |
| `.github/workflows/ci.yml` | Intégration continue (GitHub Actions) |
| `compose.yaml`, `Dockerfile.mlflow` | Serveur MLflow et base PostgreSQL (Docker Compose) |
| `.env.example` | Modèle du fichier des secrets `.env` (ignoré par Git) |

## Prérequis

| Outil | Rôle |
| --- | --- |
| [Git](https://git-scm.com/) | Historique du code |
| [uv](https://docs.astral.sh/uv/) | Python, environnement virtuel et dépendances |
| [VS Code](https://code.visualstudio.com/) | Éditeur, avec les extensions *Python*, *Jupyter* et *Ruff* |
| [Docker Desktop](https://www.docker.com/products/docker-desktop/) | Serveur MLflow et base PostgreSQL |

La version de Python (3.13) est indiquée dans `.python-version` : uv l'installe si nécessaire.

## Installation

```bash
git clone <url-du-dépôt>
cd crop_recommendation_cours
uv sync
```

`uv sync` crée l'environnement virtuel `.venv` et installe les versions exactes de `uv.lock`.

## Démarrer MLflow

Le serveur MLflow et sa base PostgreSQL tournent dans Docker (`compose.yaml`).

Première fois : créer le fichier des secrets, puis y renseigner `POSTGRES_PASSWORD`.

```bash
cp .env.example .env        # Windows (PowerShell) : Copy-Item .env.example .env
```

Démarrage :

```bash
docker compose up -d --build
docker compose ps            # les deux services doivent être « healthy »
```

- Interface : <http://127.0.0.1:5001>. Le port est publié sur `127.0.0.1` seulement ; la variable `MLFLOW_PORT` permet d'en choisir un autre.
- Les essais sont conservés dans deux volumes Docker : `postgres_data` (paramètres, métriques) et `mlflow_artifacts` (modèles, figures, rapports).
- Les essais enregistrés auparavant dans le serveur local (`mlflow.db`) ne sont pas repris : relancer le notebook et `uv run train`.

| Commande | Effet |
| --- | --- |
| `docker compose ps` | État des services |
| `docker compose logs --tail 20 mlflow` | Dernières lignes du journal |
| `docker compose down` | Arrêt ; les essais sont conservés |
| `docker compose down -v` | Arrêt **et suppression** des essais |

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

## Entraîner un modèle

Le serveur MLflow doit être démarré.

```bash
uv run tune     # occasionnel : compare les combinaisons de tuning.param_grid en validation croisée
uv run train    # entraîne le modèle avec model.params et enregistre un run
```

1. `tune` affiche la meilleure combinaison et enregistre un run « tuning » (un run enfant par combinaison) dans l'expérience `crop-recommendation`.
2. On reporte la meilleure combinaison dans `model.params` de `config/config.yaml`, puis on commite.
3. `train` charge et valide les données, les découpe, entraîne le modèle, mesure le score en validation croisée et sur le jeu de test, puis enregistre un run.

Chaque run de `train` enregistre :

| Élément | Contenu |
| --- | --- |
| Paramètres | réglages du modèle, part du jeu de test |
| Métriques | `cv_f1_macro_mean`, `cv_f1_macro_std`, `accuracy`, `f1_macro` |
| Tags | `data_md5` (empreinte des données), `git_commit` (commit du code), `stage = candidate` |
| Artefacts | `config.yaml`, `schema.yaml`, rapport de classification, matrice de confusion, modèle |

Le suffixe `-dirty` de `git_commit` signale des modifications non commitées ; le suffixe `-modified` de `data_md5`, des données différentes de l'étiquette `.dvc`. Commiter avant d'entraîner un modèle que l'on veut garder.

## Registre et promotion

Le registre MLflow conserve les versions du modèle `crop-recommendation`. L'alias `champion` désigne la version de référence.

```bash
uv run train      # entraîne un candidat (run étiqueté stage = candidate)
uv run promote    # compare le dernier candidat au champion
```

`promote` compare le score de validation croisée (`cv_f1_macro_mean`) du dernier candidat à celui du champion :

| Situation | Résultat |
| --- | --- |
| Aucun run candidat | Message : lancer d'abord `uv run train` |
| Le candidat n'a pas la métrique | Message, aucune promotion |
| Registre sans champion | Candidat promu |
| Candidat déjà champion | Aucune action |
| Score inférieur ou égal à celui du champion | Champion conservé |
| Score strictement supérieur, ou champion sans score | Nouvelle version, qui reçoit l'alias `champion` |

Le nom du modèle et celui de l'alias se trouvent dans la section `mlflow` de `config/config.yaml`.

Dans l'interface MLflow, l'alias se voit dans *Model registry* (barre de navigation de gauche), colonne *Aliased versions*. L'onglet *Models* d'une expérience liste les modèles enregistrés par les runs : ils n'ont jamais d'alias.

## Tests et qualité du code

```bash
uv run pytest                 # tests automatisés
uv run ruff check .           # erreurs probables, ordre des imports
uv run ruff format --check .  # mise en forme
```

| Fichier | Tests | Ce qui est vérifié |
| --- | --- | --- |
| `tests/test_validation.py` | 9 | Règles du schéma : colonnes, types, valeurs manquantes, bornes, cultures autorisées |
| `tests/test_data.py` | 4 | Variables explicatives, découpage stratifié et reproductible, plis de validation croisée |
| `tests/test_train.py` | 1 | `train` et `tune` donnent le même score pour les mêmes réglages |
| `tests/test_promote.py` | 11 | Règle de promotion, dans tous les cas du tableau ci-dessus |

Les tests utilisent de petits tableaux fabriqués et une base MLflow SQLite temporaire (fixture `config` de `tests/conftest.py`) : ils s'exécutent sans les données, sans Docker et sans serveur MLflow.

`uv run ruff check . --fix` et `uv run ruff format .` corrigent automatiquement la plupart des écarts. Dans VS Code, avec l'extension Ruff, `.vscode/settings.json` trie les imports, applique les corrections et met en forme le code à chaque enregistrement.

## Intégration continue

Le workflow `.github/workflows/ci.yml` s'exécute à chaque `push` et `pull_request`, sur une machine neuve. Il installe l'environnement exact (`uv sync --locked`), puis lance :

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

`uv sync --locked` échoue si `uv.lock` ne correspond pas à `pyproject.toml`. Avant chaque commit, lancer les trois mêmes commandes en local. Résultat des exécutions : onglet *Actions* du dépôt GitHub, ou `gh run list`.

## Résolution de problèmes

| Message ou symptôme | Cause | Solution |
| --- | --- | --- |
| `Connection refused` dans le notebook ou les scripts | Serveur MLflow arrêté | `docker compose up -d` |
| `port is already allocated` au démarrage | Port 5001 déjà utilisé, par exemple par le serveur local des étapes précédentes | Arrêter ce serveur (`Ctrl+C`), ou choisir un autre port : `MLFLOW_PORT=5003 docker compose up -d`, à reporter dans `mlflow.tracking_uri` de `config/config.yaml` |
| `env file … .env not found` | Fichier des secrets absent | `cp .env.example .env`, puis renseigner `POSTGRES_PASSWORD` |
| `ModuleNotFoundError: crop_recommendation_tp` | Noyau du notebook différent de `.venv` | Sélectionner le noyau `.venv`, puis relancer |
| `FileNotFoundError: 'config/config.yaml'` | Le notebook ne tourne pas depuis la racine du projet | Redémarrer le noyau et relancer toutes les cellules |
| `FileNotFoundError: … crop_recommendation.csv` après `git checkout etape-2` | Données retirées de Git, stockage DVC absent | Section « Première mise en place » |
| `Some of the cache files do not exist` lors de `dvc pull` | Stockage DVC absent sur ce poste | Section « Première mise en place » |
