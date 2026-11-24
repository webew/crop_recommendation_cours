# Manuel étudiant : la démarche MLOps pas à pas

Ce manuel permet de suivre le cours en autonomie. Il explique les notions, puis fait découvrir le code étape par étape, à l'aide des commits du dépôt.

Projet support : recommander une culture agricole (22 classes) à partir de 7 mesures du sol et du climat. Le modèle est volontairement simple ; le cours porte sur tout ce qui l'entoure.

## Sommaire

1. [Mode d'emploi](#mode-demploi)
2. [La démarche MLOps](#la-démarche-mlops)
3. [Naviguer entre les étapes avec Git](#naviguer-entre-les-étapes-avec-git)
4. [Installation](#installation)
5. Phase 1, en local
   - [Étape 0 : point de départ](#étape-0--point-de-départ)
   - [Étape 1 : validation des données](#étape-1--validation-des-données)
   - [Étape 2 : versionnage des données](#étape-2--versionnage-des-données)
   - [Étape 3 : du notebook au paquet](#étape-3--du-notebook-au-paquet)
   - [Étape 4 : tests et qualité du code](#étape-4--tests-et-qualité-du-code)
   - [Étape 5 : entraînement en script](#étape-5--entraînement-en-script)
   - [Étape 6 : intégration continue](#étape-6--intégration-continue)
   - [Bilan de la phase 1 : retrouver l'origine d'un modèle](#bilan-de-la-phase-1--retrouver-lorigine-dun-modèle)
6. Phase 2, services
   - [Étape 7 : serveur MLflow](#étape-7--serveur-mlflow)
   - [Étape 8 : registre et promotion](#étape-8--registre-et-promotion)
   - [Étape 9 : API de prédiction](#étape-9--api-de-prédiction)
   - [Étape 10 : API en conteneur](#étape-10--api-en-conteneur)
   - [Bilan de la phase 2 : retrouver l'origine d'une prédiction](#bilan-de-la-phase-2--retrouver-lorigine-dune-prédiction)
7. Phase 3, vers la production
   - [Étape 11 : journal des requêtes](#étape-11--journal-des-requêtes)
   - [Étape 12 : surveillance de la dérive](#étape-12--surveillance-de-la-dérive)
   - [Étape 13 : image publiée et dépendances à jour](#étape-13--image-publiée-et-dépendances-à-jour)
8. [Glossaire](#glossaire)
9. [Aide-mémoire des commandes](#aide-mémoire-des-commandes)

---

## Mode d'emploi

### Organisation du dépôt

Le dépôt contient **une étape par commit**. Chaque commit porte une **étiquette** (`etape-0`, `etape-1`, …, `etape-10`). Changer d'étiquette fait apparaître le projet tel qu'il était à cette étape : seuls les fichiers de l'étape sont présents.

### Méthode de travail conseillée

Pour chaque étape :

1. **Lire** la partie « Pourquoi cette étape » et les notions.
2. **Passer** à l'étape avec les commandes indiquées.
3. **Observer** ce qui a changé (`git diff`, voir plus bas).
4. **Lire le code** avec les explications du manuel, fichier ouvert dans VS Code.
5. **Exécuter** les manipulations et comparer avec les résultats attendus.
6. **Répondre** aux questions de compréhension, puis déplier les réponses.

Durée totale indicative : une dizaine d'heures.

### Conventions

- Les commandes se tapent dans un terminal, **à la racine du projet** (le dossier `crop_recommendation_cours`).
- Elles fonctionnent telles quelles dans Git Bash, dans le terminal de macOS et dans PowerShell, sauf indication contraire : la variante PowerShell est alors donnée.
- `<…>` désigne une valeur à remplacer, chevrons compris : `<git_commit>` devient par exemple `2b3e414`.
- `→` annonce le résultat attendu d'une commande.
- Les chemins de fichiers sont relatifs à la racine du projet.

---

## La démarche MLOps

### Le problème

Un notebook est un excellent outil pour explorer des données et essayer des modèles. Il montre vite ses limites dès qu'un modèle doit être **réutilisé**, **partagé** ou **mis en service** :

| Situation | Conséquence |
| --- | --- |
| Un collègue relance le notebook avec d'autres versions de pandas ou de scikit-learn | Le code plante, ou donne d'autres résultats |
| Le fichier de données a été corrigé | Impossible de savoir sur quelle version le modèle a été entraîné |
| Vingt réglages ont été essayés | Impossible de retrouver celui qui donnait le meilleur score |
| Une application doit utiliser le modèle | Il faut recopier du code du notebook, sans garantie |
| Une fonction est modifiée | Rien ne signale qu'une autre partie du code ne fonctionne plus |

Aucun de ces problèmes ne vient du modèle : ils viennent de **tout ce qui l'entoure**.

### Définition

Le **MLOps** (*Machine Learning Operations*) rassemble les pratiques qui rendent un projet de machine learning :

- **reproductible** : le même code, les mêmes données et le même environnement donnent le même modèle ;
- **traçable** : pour tout modèle, on retrouve le code, les données et les réglages qui l'ont produit ;
- **vérifié** : des tests et des contrôles automatiques signalent toute régression ;
- **déployable** : le modèle est utilisable par d'autres applications, et sa mise à jour est maîtrisée.

### Un modèle, trois ingrédients

Un modèle est le produit de trois ingrédients. Pour le reproduire, il faut retrouver les trois :

| Ingrédient | Outil | Ce que l'outil garantit |
| --- | --- | --- |
| Code | Git | historique de chaque modification |
| Environnement | uv, `uv.lock` | mêmes versions de bibliothèques sur chaque poste |
| Données | DVC | version exacte du fichier utilisé |

S'y ajoutent **MLflow**, qui garde la trace de chaque entraînement, et les **tests**, qui vérifient le code en continu.

### La chaîne complète

```text
 LOCAL (poste de travail)          GITHUB                 PRODUCTION (serveur)
 notebook → paquet src/            dépôt Git              API (conteneur)
 uv · configuration · schéma  ──►  CI : Ruff + tests ──►  MLflow + PostgreSQL
 DVC · tune · train · promote      CD : image + déploie   stockage des données
 MLflow · Docker Compose · API                            surveillance de la dérive
 pytest · Ruff                                                     │
        ▲                                                          │
        └──────────── alerte de dérive → nouvel entraînement ──────┘
```

- **En local**, on construit et on expérimente.
- **GitHub** vérifie chaque modification (intégration continue, CI) et livre l'application (déploiement continu, CD).
- **En production**, on sert le modèle, on le surveille et on réagit.

Ce manuel couvre les phases 1 et 2 : tout ce qui se fait en local. La phase 3 (déploiement et automatisation) est en cours de rédaction : ses étapes 11 (journal des requêtes) et 12 (surveillance de la dérive) se font encore en local.

### Le cycle de vie d'un modèle

Un modèle n'est jamais terminé. Le monde change, et des **événements** relancent le cycle données → validation → entraînement → promotion → déploiement → surveillance :

| Événement | Exemple | Réaction |
| --- | --- | --- |
| Nouveau code | correction d'un bug | tests, puis déploiement |
| Nouvelles données | campagne de mesures de l'année suivante | validation, entraînement, promotion |
| Données invalides en entrée | capteur de pH défaillant (pH = 15) | rejet de la requête, vérification de la source |
| Data drift | parcelles d'une région plus chaude | collecte de nouvelles données, nouvel entraînement |
| Concept drift | restriction d'eau : pour les mêmes mesures, le riz n'est plus adapté | nouvel entraînement sur des données récentes |
| Modèle défaillant | un nouveau modèle se trompe en service | retour à la version précédente |

- **Data drift** : la distribution des **entrées** change. Détectable immédiatement, en comparant les entrées reçues aux données d'entraînement.
- **Concept drift** : la **relation** entre les entrées et la bonne réponse change. Détectable seulement avec la vérité terrain (la culture était-elle réellement adaptée ?), qui arrive tard.

---

## Naviguer entre les étapes avec Git

### Notions

| Notion | Définition |
| --- | --- |
| Commit | enregistrement d'un état complet du projet, avec un message |
| Étiquette (*tag*) | nom fixe donné à un commit : `etape-3` désigne toujours le même commit |
| `HEAD` | commit actuellement présent dans le dossier de travail |
| Branche `main` | dernier état du dépôt, qui contient aussi ce manuel |

### Commandes utiles

```bash
git tag                                  # liste des étiquettes
git log --oneline --decorate             # historique, avec les étiquettes
git checkout -f etape-3                  # passer à l'étape 3
git show --stat etape-3                  # fichiers modifiés par l'étape 3
git diff etape-2 etape-3 -- src/         # différences entre deux étapes, pour un dossier
git checkout main                        # revenir au dernier état
```

### Changer d'étape

À chaque changement d'étape :

```bash
git checkout -f etape-N
uv sync
uv run dvc pull        # à partir de l'étape 2
```

- `-f` (*force*) abandonne les modifications locales des fichiers suivis par Git. Sans `-f`, Git refuse de changer d'étape si vous avez modifié un fichier. **Vos modifications sont perdues** : copiez-les ailleurs si vous voulez les garder.
- `uv sync` installe les dépendances de l'étape : certaines étapes en ajoutent.
- `uv run dvc pull` restaure les données, gérées par DVC à partir de l'étape 2. Il suppose la **première mise en place** de l'étape 2 faite sur votre poste : si vous arrivez directement à une étape ultérieure, faites-la d'abord.

### « detached HEAD »

Après `git checkout -f etape-3`, Git affiche un avertissement : `You are in 'detached HEAD' state`. Il signifie que le dossier de travail correspond à un commit précis, et non à une branche. C'est normal pour consulter une étape. Pour revenir à la branche : `git checkout main`.

---

## Installation

### Prérequis

| Outil | Rôle | Phase |
| --- | --- | --- |
| [Git](https://git-scm.com/) | historique du code | 1 |
| [uv](https://docs.astral.sh/uv/) | Python, environnement virtuel, dépendances | 1 |
| [VS Code](https://code.visualstudio.com/), extensions *Python*, *Jupyter*, *Ruff* | éditeur, notebook | 1 |
| [Docker Desktop](https://www.docker.com/products/docker-desktop/) | conteneurs | 2 |

### Récupérer le dépôt

```bash
git clone https://github.com/webew/crop_recommendation_cours.git
cd crop_recommendation_cours
git checkout -f etape-0
uv sync
```

`uv sync` installe Python 3.13 si nécessaire, crée l'environnement virtuel `.venv` et y installe les bibliothèques.

---

## Étape 0 : point de départ

### Pourquoi cette étape

Le point de départ est une démarche de modélisation **complète et correcte**, réalisée dans un notebook, avec les essais enregistrés dans MLflow. Les étapes suivantes l'industrialisent, sans changer la démarche.

### Notions

**Environnement virtuel.** Dossier (`.venv/`) qui contient un Python et des bibliothèques propres au projet, isolés du reste de la machine. Deux projets peuvent ainsi utiliser des versions différentes d'une même bibliothèque.

**Fichier de verrouillage.** `uv.lock` note la version **exacte** de chaque bibliothèque installée, y compris celles dont dépendent vos bibliothèques. `pyproject.toml` indique seulement des versions minimales (`>=`). Avec `uv.lock`, tous les postes ont exactement le même environnement.

**Paquet Python.** Code organisé pour être installé et importé : ici `src/crop_recommendation_tp/`. `uv sync` l'installe dans `.venv`, si bien que le notebook peut écrire `from crop_recommendation_tp.config import load_config`.

**Configuration externalisée.** Tous les réglages (chemins, découpage, paramètres) sont dans un fichier, `config/config.yaml`, et non écrits dans le code. On change un réglage sans toucher au code, et chaque réglage n'est écrit qu'à un seul endroit.

**MLflow.** Outil qui enregistre chaque entraînement : un **run** regroupe les **paramètres** (réglages en entrée), les **métriques** (résultats chiffrés), les **tags** (informations libres) et les **artefacts** (fichiers : figures, rapports, modèle). Les runs sont rangés dans des **expériences**. Une interface web permet de les comparer.

### Passer à l'étape

```bash
git checkout -f etape-0
uv sync
```

Dans un **second terminal**, laissé ouvert, démarrer le serveur MLflow :

```bash
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5001
```

- `--backend-store-uri sqlite:///mlflow.db` : les runs sont enregistrés dans une base SQLite, le fichier `mlflow.db`.
- `--port 5001` : le port 5000 est souvent occupé (AirPlay sur macOS).
- Interface : <http://127.0.0.1:5001>. `Ctrl+C` arrête le serveur.

### Fichiers

| Fichier | Rôle |
| --- | --- |
| `pyproject.toml` | identité du projet, dépendances avec leur version minimale |
| `uv.lock` | versions exactes de toutes les dépendances |
| `.python-version` | version de Python (3.13) |
| `config/config.yaml` | réglages du projet |
| `src/crop_recommendation_tp/config.py` | lecture de la configuration |
| `data/raw/crop_recommendation.csv` | données : 2 200 parcelles |
| `notebooks/01_demarche_modelisation.ipynb` | démarche de modélisation |
| `.gitignore` | fichiers que Git doit ignorer (`.venv`, `mlflow.db`…) |

### Lecture du code : `config/config.yaml`

```yaml
dataset:
    path: data/raw/crop_recommendation.csv
    target: label
split:
    test_size: 0.2
    random_seed: 42
mlflow:
    tracking_uri: http://127.0.0.1:5001
    experiment_name: crop-recommendation
tuning:
    cv_folds: 5
    scoring: f1_macro
    param_grid:
        n_estimators: [50, 100, 200]
        max_depth: [3, 5, 10, null]
```

- **YAML** : format texte lisible, organisé par indentation. `null` correspond à `None` en Python.
- `split` : part du jeu de test (20 %) et **graine aléatoire** (42). La graine fige le hasard : le même découpage est produit sur tous les postes.
- `tuning` : réglages de la recherche d'hyperparamètres (5 plis de validation croisée, mesure F1 macro, combinaisons à essayer).

### Lecture du code : `config.py`

```python
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str) -> dict:
    """Lit un fichier YAML dont le chemin est relatif à la racine du projet."""
    with open(ROOT / path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: str = "config/config.yaml") -> dict:
    """Retourne la configuration du projet."""
    return load_yaml(path)
```

- `__file__` contient le chemin du fichier `config.py` lui-même : `…/src/crop_recommendation_tp/config.py`.
- `.resolve()` en fait un chemin absolu ; `.parents[0]` est `crop_recommendation_tp/`, `.parents[1]` est `src/`, `.parents[2]` est la **racine du projet**.
- `ROOT / path` assemble deux chemins (opérateur `/` de `pathlib`). Les chemins de la configuration fonctionnent ainsi quel que soit le dossier depuis lequel on lance le code : terminal, notebook ou tests.
- `yaml.safe_load` lit le fichier et renvoie un dictionnaire. `safe_load` refuse d'exécuter du code qui serait caché dans le fichier, contrairement à `yaml.load` : bonne pratique de sécurité.

### Lecture du notebook

Ouvrir `notebooks/01_demarche_modelisation.ipynb`, choisir le noyau `.venv` du projet (*Sélectionner un noyau* → *Environnements Python*), puis *Run All*. L'exécution dure une quinzaine de secondes.

| Section | Contenu | À retenir |
| --- | --- | --- |
| Environnement | retour à la racine du projet, imports | le notebook lit les fichiers relativement à la racine |
| 1. Données et objectif | variables, mesure de succès | mesure choisie **avant** de modéliser : F1 macro |
| 2. Qualité des données | valeurs manquantes, doublons, effectifs | 2 200 lignes, 100 par culture, aucune valeur manquante |
| 3. Analyse exploratoire | distributions, profils par culture, corrélations | cultures bien séparées ; seule corrélation notable entre `P` et `K` |
| 4. Protocole d'évaluation | découpage, validation croisée, fonction `evaluer` | protocole fixé **avant** toute modélisation |
| 5. Comparaison de modèles | régression logistique, k plus proches voisins, forêt aléatoire | forêt aléatoire retenue (F1 macro 0,9937) |
| 6. Réglage | grille d'hyperparamètres | 200 arbres, profondeur illimitée (0,9943) |
| 7. Évaluation finale | mesure unique sur le jeu de test | accuracy 0,9955 : 2 erreurs sur 440 parcelles |
| 8. Importance des variables | part de chaque variable dans les décisions | `rainfall`, `humidity`, `K`, `P` en tête |
| 9. Conclusions | modèle retenu, limites, feuille de route | chaque étape du notebook deviendra un composant de l'application |

#### Le F1 macro

Le **score F1** d'une classe combine la précision (parmi les parcelles prédites « riz », combien le sont vraiment) et le rappel (parmi les parcelles « riz », combien ont été trouvées). Le **F1 macro** est la moyenne des F1 des 22 cultures : chaque culture compte autant, même la plus rare. L'accuracy seule peut masquer un modèle qui ignore les classes rares.

#### Le protocole d'évaluation (section 4)

```python
X_train, X_test, y_train, y_test = train_test_split(
    df[features],
    df[target],
    test_size=config["split"]["test_size"],
    random_state=seed,
    stratify=df[target],
)
cv = StratifiedKFold(
    n_splits=config["tuning"]["cv_folds"], shuffle=True, random_state=seed
)
```

- `train_test_split` met de côté 20 % des lignes : le **jeu de test**, qui ne servira qu'une fois, à la fin.
- `stratify=df[target]` : chaque culture garde la même proportion dans les deux jeux (exactement 20 parcelles par culture dans le test).
- **Validation croisée** : le jeu d'entraînement est découpé en 5 parts (*plis*). On entraîne 5 fois, en gardant à chaque fois un pli différent pour évaluer, puis on fait la moyenne des 5 scores. On obtient une mesure fiable, et un écart-type qui en indique la stabilité, **sans toucher au jeu de test**.
- `StratifiedKFold(shuffle=True)` : plis stratifiés, lignes mélangées avant le découpage (le CSV est rangé par culture).

**Règle essentielle** : tous les choix (modèle, hyperparamètres) se font sur le score de validation croisée. Le jeu de test sert **une seule fois**, à l'évaluation finale. S'il servait à choisir, son score deviendrait trop optimiste : il ne mesurerait plus la performance sur des données jamais vues.

#### La fonction `evaluer`

```python
def evaluer(nom, modele):
    """Évalue un modèle en validation croisée et enregistre le résultat dans MLflow."""
    scores = cross_validate(
        modele, X_train, y_train, cv=cv, scoring=["f1_macro", "accuracy"]
    )
    ...
    with mlflow.start_run(run_name=nom):
        mlflow.set_tag("etape", "comparaison")
        mlflow.log_params(...)
        mlflow.log_metrics(...)
```

- `cross_validate` renvoie les scores des 5 plis pour chaque mesure demandée.
- `with mlflow.start_run(...)` ouvre un **run**. Tout ce qui est enregistré dans le bloc `with` lui est rattaché ; le run se ferme à la sortie du bloc.
- `log_params` enregistre les réglages, `log_metrics` les scores, `set_tag` une information libre (ici, l'étape de la démarche).

#### La mise à l'échelle dans un pipeline (section 5)

```python
make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
```

- La régression logistique et les k plus proches voisins sont sensibles à l'échelle des variables : `rainfall` varie jusqu'à 300, `ph` jusqu'à 10. `StandardScaler` ramène chaque variable à une moyenne de 0 et un écart-type de 1.
- Placée dans un **pipeline**, la mise à l'échelle est apprise sur les plis d'entraînement uniquement, puis appliquée au pli d'évaluation. Aucune information du pli d'évaluation ne « fuit » dans l'entraînement.

### Manipulations

1. Exécuter le notebook en entier.
2. Ouvrir <http://127.0.0.1:5001>, expérience `crop-recommendation-exploration`.
3. Cocher les trois runs « comparaison » → *Compare* : comparer `cv_f1_macro_mean`.
4. Ouvrir le run « modèle final » → onglet *Artifacts* : matrice de confusion, rapport de classification, importance des variables.

**Résultat attendu** : cinq runs (trois modèles comparés, le réglage, le modèle final) ; sur le run final, `test_accuracy` = 0,9955.

### Pièges fréquents

| Symptôme | Cause | Solution |
| --- | --- | --- |
| `ModuleNotFoundError: crop_recommendation_tp` | noyau du notebook différent de `.venv` | choisir le noyau `.venv`, relancer |
| `Connection refused` | serveur MLflow arrêté | démarrer le serveur dans un second terminal |
| `Address already in use` | port 5001 occupé | autre port (`--port 5003`), à reporter dans `mlflow.tracking_uri` |

### Questions de compréhension

<details>
<summary>1. Pourquoi <code>uv.lock</code> est-il nécessaire alors que <code>pyproject.toml</code> liste déjà les dépendances ?</summary>

`pyproject.toml` donne des versions minimales (`mlflow>=3.17.2`) : deux installations à des dates différentes peuvent obtenir des versions différentes, y compris des dépendances indirectes. `uv.lock` fige les versions exactes de toutes les bibliothèques : tout le monde obtient le même environnement.
</details>

<details>
<summary>2. Que vaudrait <code>ROOT</code> si <code>config.py</code> était placé directement dans <code>src/</code> ?</summary>

`parents[2]` remonterait d'un niveau de trop : le dossier parent de la racine du projet. Les chemins de la configuration ne seraient plus trouvés. `parents[2]` dépend de l'emplacement du fichier.
</details>

<details>
<summary>3. Pourquoi ne pas comparer les modèles directement sur le jeu de test ?</summary>

Choisir le modèle d'après son score sur le test revient à utiliser le test pour décider : son score devient optimiste et ne mesure plus la performance sur des données nouvelles. Les choix se font en validation croisée sur l'entraînement ; le test sert une seule fois, à la fin.
</details>

<details>
<summary>4. Que se passerait-il sans <code>stratify</code> dans <code>train_test_split</code> ?</summary>

Le hasard pourrait laisser une culture sous-représentée dans le jeu de test (par exemple 12 parcelles au lieu de 20), voire presque absente. Les scores seraient moins fiables, surtout le F1 macro, qui donne le même poids à chaque culture.
</details>

---

## Étape 1 : validation des données

### Pourquoi cette étape

Un modèle entraîné sur des données fausses donne des résultats faux **sans aucun message d'erreur**. Un pH de 75 (l'échelle va de 0 à 14), une culture mal orthographiée (« ricee »), une colonne manquante : ces erreurs passent inaperçues si personne ne les cherche. On écrit donc **ce que l'on attend des données** dans un fichier, le **schéma**, et on vérifie automatiquement chaque fichier avant de s'en servir.

### Notions

**Schéma de données.** Contrat écrit entre les données et le code : liste des colonnes, type de chaque colonne, bornes, valeurs autorisées.

**Échouer tôt.** Mieux vaut arrêter un traitement dès l'entrée, avec un message clair, que produire un résultat faux plus loin.

### Passer à l'étape

```bash
git checkout -f etape-1
uv sync
git show --stat etape-1
```

Recharger le notebook dans VS Code, puis redémarrer le noyau.

### Fichiers

| Fichier | Changement |
| --- | --- |
| `config/schema.yaml` | nouveau : le schéma |
| `src/crop_recommendation_tp/validation.py` | nouveau : la fonction `validate` |
| `config/config.yaml` | ajout de `dataset.schema` (chemin du schéma) |
| notebook | section 2 : validation par le schéma ; variables déduites du schéma |

Pour voir les changements du notebook : `git diff etape-0 etape-1 -- notebooks/`.

### Lecture du code : `config/schema.yaml`

```yaml
columns:
    N: { dtype: int, min: 0 }
    P: { dtype: int, min: 0 }
    K: { dtype: int, min: 0 }
    temperature: { dtype: float, min: -50, max: 60 }
    humidity: { dtype: float, min: 0, max: 100 }
    ph: { dtype: float, min: 0, max: 14 }
    rainfall: { dtype: float, min: 0 }
    label:
        dtype: str
        allowed: [apple, banana, …]
```

- `{ dtype: int, min: 0 }` : notation YAML compacte d'un dictionnaire.
- Les bornes sont des valeurs **physiquement possibles** (humidité entre 0 et 100 %), et non les valeurs observées dans les données.
- `allowed` : les 22 cultures autorisées pour la cible.

### Lecture du code : `validation.py`

```python
def is_number(series: pd.Series) -> bool:
    """Vrai pour une colonne numérique (entiers ou décimaux), booléens exclus."""
    return types.is_numeric_dtype(series) and not types.is_bool_dtype(series)


TYPE_CHECKS = {
    "int": types.is_integer_dtype,
    "float": is_number,
    "str": types.is_string_dtype,
}
```

- `TYPE_CHECKS` est un **dictionnaire de fonctions** : à chaque type du schéma correspond la fonction de pandas qui le vérifie. `TYPE_CHECKS["float"](valeurs)` appelle `is_number(valeurs)`.
- Le type `float` accepte les entiers (une température de 21 est valide) mais refuse les booléens, que pandas considère comme des nombres.

```python
def validate(df: pd.DataFrame, schema: dict) -> list[str]:
    errors = []
    for name, rules in schema["columns"].items():
        if name not in df.columns:
            errors.append(f"Colonne absente : {name}")
            continue
```

- On parcourt les colonnes du schéma : `name` est le nom de la colonne, `rules` son dictionnaire de règles.
- `continue` passe directement à la colonne suivante : inutile de vérifier une colonne absente.

```python
        values = df[name].dropna()
        n_missing = len(df) - len(values)
        if n_missing:
            errors.append(f"{name} : {n_missing} valeur(s) manquante(s)")
        elif not TYPE_CHECKS[rules["dtype"]](values):
            errors.append(f"{name} : type {df[name].dtype}, attendu {rules['dtype']}")
            continue
```

- `dropna()` retire les valeurs manquantes ; la différence de longueur donne leur nombre.
- `if n_missing:` : en Python, 0 est faux et tout autre nombre est vrai.
- Si le type est faux, `continue` évite de contrôler les bornes d'une colonne mal typée.

```python
        if "allowed" in rules:
            unknown = sorted(set(values) - set(rules["allowed"]))
```

- **Différence d'ensembles** : les valeurs présentes dans les données qui ne figurent pas dans la liste autorisée.

```python
        if "min" in rules:
            n = (values < rules["min"]).sum()
```

- `values < rules["min"]` produit une colonne de booléens ; `.sum()` compte les `True` (`True` vaut 1).

```python
    extra = set(df.columns) - set(schema["columns"])
```

- Colonnes présentes dans les données mais absentes du schéma.

**Choix de conception** : `validate` renvoie la **liste de toutes les erreurs**, au lieu de s'arrêter à la première. On voit tous les problèmes d'un coup, une liste vide signifie « conforme », et la fonction est simple à tester (étape 4).

### Manipulations

Dans un terminal :

```bash
uv run python
```

```python
import pandas as pd

from crop_recommendation_tp.config import load_config, load_yaml
from crop_recommendation_tp.validation import validate

c = load_config()
df = pd.read_csv(c["dataset"]["path"])
schema = load_yaml(c["dataset"]["schema"])
validate(df, schema)  # → []

df.loc[0, "ph"] = 15
df.loc[1, "label"] = "ricee"
df.loc[2, "N"] = None
validate(df, schema)
```

**Résultat attendu** :

```text
['N : 1 valeur(s) manquante(s)', 'ph : 1 valeur(s) > 14', "label : valeurs non autorisées ['ricee']"]
```

`exit()` quitte Python. Le fichier CSV n'a pas été modifié : seul le tableau en mémoire l'a été.

### Questions de compréhension

<details>
<summary>1. Pourquoi les bornes du schéma ne sont-elles pas les minimums et maximums observés dans les données ?</summary>

Le schéma décrit ce qui est **possible**. Avec les valeurs observées, la moindre nouvelle parcelle un peu plus humide serait rejetée. Les bornes physiques (pH entre 0 et 14) ne rejettent que les valeurs impossibles, signe d'une erreur de saisie ou de capteur.
</details>

<details>
<summary>2. Pourquoi renvoyer une liste d'erreurs plutôt que lever une exception dès la première ?</summary>

On voit toutes les erreurs en une fois, au lieu de les corriger une par une en relançant. La liste vide donne un critère simple (« conforme »), et chaque règle se teste en comparant la liste obtenue à la liste attendue. Le choix de lever une exception est laissé à l'appelant (étape 3).
</details>

<details>
<summary>3. Une colonne <code>ph</code> contient <code>True</code> et <code>False</code>. Que renvoie <code>validate</code> ?</summary>

`["ph : type bool, attendu float"]` : `is_number` refuse les booléens, même si pandas les considère comme numériques.
</details>

---

## Étape 2 : versionnage des données

### Pourquoi cette étape

Les données sont contrôlées, mais **quelle version** a servi à entraîner un modèle ? Si le CSV est corrigé, plus rien ne permet de le savoir. Git ne convient pas aux données : il garde chaque version de chaque fichier pour toujours, et un dépôt contenant des données volumineuses grossirait sans fin (GitHub refuse les fichiers de plus de 100 Mo).

**DVC** (*Data Version Control*) résout ce problème : Git conserve une petite **étiquette** des données, DVC range le contenu ailleurs.

### Notions

**Empreinte (hash md5).** Courte suite de caractères calculée à partir du contenu d'un fichier. Deux fichiers identiques ont la même empreinte ; la moindre modification (un seul caractère) la change entièrement. L'empreinte identifie donc une version précise des données.

**Étiquette `.dvc`.** Petit fichier texte, versionné par Git, qui contient l'empreinte et la taille des données.

**Cache DVC.** Dossier `.dvc/cache/` du poste, où DVC range une copie des données, nommée d'après leur empreinte.

**Stockage distant (*remote*).** Endroit où DVC envoie les données pour les partager ou les sauvegarder. Ici, un dossier situé à côté du dépôt ; en production, un stockage objet (S3).

### Passer à l'étape

```bash
git checkout -f etape-2
uv sync
```

Le CSV **disparaît** de `data/raw/` : à cette étape, il n'est plus suivi par Git, et Git retire les fichiers qu'il ne suit plus. Le stockage DVC n'existe pas encore sur votre poste. **Première mise en place**, une seule fois :

```bash
git restore --source=etape-1 -- data/raw/crop_recommendation.csv
uv run dvc add data/raw/crop_recommendation.csv
uv run dvc push          # → 1 file pushed
uv run dvc status        # → Data and pipelines are up to date.
git status               # → aucune modification
```

- `git restore --source=etape-1` remet le fichier tel qu'il était dans le commit de l'étape 1, octet pour octet, sans que Git le suive de nouveau.
- `dvc add` calcule l'empreinte du fichier et le range dans le cache. L'empreinte est identique à celle de l'étiquette du dépôt (`5e743dd4…`) : Git ne voit aucune modification. Si `git status` signale `crop_recommendation.csv.dvc` modifié, le fichier restauré n'est pas le bon : supprimez-le et recommencez.
- `dvc push` crée le stockage `../crop_recommendation_cours_storage`, à côté du dépôt, et y envoie les données. Il contient alors un seul fichier : `files/md5/5e/743dd4cf49f84d23a904cd51c9b5b8`.
- N'utilisez pas `git show etape-1:<fichier> > <fichier>` : sous Windows PowerShell, la redirection `>` réécrit le texte en UTF-16. Le fichier change de taille et d'empreinte.

Aux changements d'étape suivants, `uv run dvc pull` suffit.

### Fichiers

| Fichier | Rôle |
| --- | --- |
| `pyproject.toml`, `uv.lock` | ajout de DVC |
| `.dvc/config` | configuration de DVC : stockage par défaut |
| `.dvc/.gitignore`, `.dvcignore` | fichiers internes de DVC, exclus de Git |
| `data/raw/crop_recommendation.csv.dvc` | étiquette des données |
| `data/raw/.gitignore` | exclut le CSV de Git |

### Comment l'étape a été construite

```bash
uv add dvc                                          # dépendance
uv run dvc init                                     # crée .dvc/ et .dvcignore
git rm --cached data/raw/crop_recommendation.csv    # Git cesse de suivre le CSV ; le fichier reste sur le disque
uv run dvc add data/raw/crop_recommendation.csv     # étiquette .dvc et exclusion dans data/raw/.gitignore
uv run dvc remote add -d local_storage ../crop_recommendation_cours_storage
uv run dvc push
git add .dvc .dvcignore data/raw pyproject.toml uv.lock
git commit -m "Étape 2 : versionnage des données"
```

`-d` fait du stockage `local_storage` le stockage **par défaut** : `dvc push` et `dvc pull` l'utilisent sans qu'on le précise.

### Lecture : l'étiquette `.dvc`

```yaml
outs:
- md5: 5e743dd4cf49f84d23a904cd51c9b5b8
  size: 150034
  hash: md5
  path: crop_recommendation.csv
```

C'est tout ce que Git voit des données. Quand les données changent, ce fichier change : la modification apparaît dans l'historique Git comme une modification de code.

### Lecture : `.dvc/config`

```ini
[core]
    remote = local_storage
['remote "local_storage"']
    url = ../../crop_recommendation_cours_storage
```

Le chemin du stockage est relatif au dossier `.dvc/`, d'où le double `../` : `.dvc/` → racine du projet → dossier parent.

### Lecture : cache et stockage

```bash
ls .dvc/cache/files/md5/5e                              # → 743dd4cf49f84d23a904cd51c9b5b8
ls ../crop_recommendation_cours_storage/files/md5/5e    # → 743dd4cf49f84d23a904cd51c9b5b8
```

Le contenu est rangé **par empreinte** : les deux premiers caractères forment le nom du dossier, la suite le nom du fichier. Le stockage distant suit la même organisation.

### Correspondance entre Git et DVC

| DVC | Effet | Équivalent Git |
| --- | --- | --- |
| `dvc add` | empreinte, cache, étiquette | `git add` |
| `dvc push` | envoi vers le stockage | `git push` |
| `dvc pull` | récupération depuis le stockage | `git pull` |
| `dvc status` | comparaison avec l'étiquette | `git status` |
| `dvc checkout` | restauration de la version décrite par l'étiquette | `git checkout` |

### Manipulations

```bash
echo '1,2,3,4,5,6,7,inconnu' >> data/raw/crop_recommendation.csv
uv run dvc status                # → modified: data/raw/crop_recommendation.csv
git status                       # → aucune modification : Git ignore le CSV
uv run dvc checkout --force      # restaure la version décrite par l'étiquette
uv run dvc status                # → Data and pipelines are up to date.
```

- `>>` ajoute une ligne à la fin du fichier (sous PowerShell, encodée en UTF-16 : pour DVC, le fichier est modifié de la même façon).
- `--force` est nécessaire : DVC refuse d'écraser une modification locale sans confirmation explicite.

### Pièges fréquents

| Symptôme | Cause | Solution |
| --- | --- | --- |
| `FileNotFoundError … crop_recommendation.csv` | données retirées par `git checkout` | `uv run dvc pull`, ou la première mise en place |
| `Some of the cache files do not exist` | stockage DVC absent sur ce poste | première mise en place |
| `git status` signale `crop_recommendation.csv.dvc` modifié après la première mise en place | CSV restauré différent de l'original (souvent : `git show … >` sous PowerShell) | supprimer le CSV et le dossier `../crop_recommendation_cours_storage`, `git checkout -- data/raw`, puis refaire la mise en place avec `git restore` |
| `dvc checkout` ne restaure pas le fichier | modification locale protégée | `uv run dvc checkout --force` |

### Questions de compréhension

<details>
<summary>1. Pourquoi Git ne voit-il pas la ligne ajoutée au CSV ?</summary>

Le CSV est listé dans `data/raw/.gitignore` : Git l'ignore. Seul DVC le surveille, en comparant son empreinte à celle de l'étiquette `.dvc`.
</details>

<details>
<summary>2. Les données sont corrigées. Quelles commandes enregistrent cette nouvelle version ?</summary>

`uv run dvc add data/raw/crop_recommendation.csv` (nouvelle empreinte dans l'étiquette), `git add data/raw/crop_recommendation.csv.dvc`, `git commit`, puis `uv run dvc push`. L'historique Git garde l'ancienne et la nouvelle étiquette.
</details>

<details>
<summary>3. Comment revenir aux données utilisées dans un ancien commit ?</summary>

`git checkout <commit> -- data/raw/crop_recommendation.csv.dvc` remet l'ancienne étiquette, puis `uv run dvc checkout` (ou `dvc pull`) remet les données correspondantes.
</details>

---

## Étape 3 : du notebook au paquet

### Pourquoi cette étape

Le code de chargement, de découpage et de validation croisée est écrit dans des cellules du notebook. Pour l'utiliser ailleurs (scripts d'entraînement, tests, API), il faudrait le **recopier** : deux copies finissent toujours par diverger. On le déplace donc dans un **module** du paquet, `data.py`, que le notebook et les scripts importent.

**Règle pratique** : dès qu'un bout de code doit être relancé, partagé ou testé, il quitte le notebook.

### Passer à l'étape

```bash
git checkout -f etape-3
uv sync
uv run dvc pull
git diff etape-2 etape-3 --stat
```

Recharger le notebook, puis redémarrer le noyau.

### Fichiers

| Fichier | Changement |
| --- | --- |
| `src/crop_recommendation_tp/data.py` | nouveau : `load_dataset`, `get_features`, `split_dataset`, `build_cv` |
| notebook | appelle ces fonctions au lieu de son propre code |

### Lecture du code : `data.py`

```python
def load_dataset(config: dict) -> pd.DataFrame:
    """Lit le CSV, le valide avec le schéma et retourne le tableau."""
    df = pd.read_csv(ROOT / config["dataset"]["path"])
    schema = load_yaml(config["dataset"]["schema"])
    errors = validate(df, schema)
    if errors:
        raise ValueError("Données non conformes au schéma :\n" + "\n".join(errors))
    return df
```

- Toute lecture des données passe désormais par la validation.
- `raise ValueError(...)` interrompt le programme avec un message. `"\n".join(errors)` assemble la liste des erreurs, une par ligne.
- C'est ici que s'applique le principe « échouer tôt » : un entraînement sur des données invalides s'arrête dès le chargement.

```python
def get_features(config: dict) -> list[str]:
    """Retourne les variables explicatives : colonnes du schéma, cible exclue."""
    schema = load_yaml(config["dataset"]["schema"])
    return [c for c in schema["columns"] if c != config["dataset"]["target"]]
```

- **Liste en compréhension** : toutes les colonnes du schéma, sauf la cible.
- Les variables viennent du schéma, dans son ordre : **une seule source de vérité**. Ajouter une variable revient à modifier `schema.yaml`.

```python
def split_dataset(df: pd.DataFrame, config: dict):
    """Découpe en jeux d'entraînement et de test (X_train, X_test, y_train, y_test)."""
    y = df[config["dataset"]["target"]]
    return train_test_split(
        df[get_features(config)],
        y,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_seed"],
        stratify=y,
    )
```

- Le même découpage que dans le notebook de l'étape 0, avec les réglages lus dans la configuration.

```python
def build_cv(config: dict) -> StratifiedKFold:
    """Plis de validation croisée stratifiés, lignes mélangées avec la graine."""
    return StratifiedKFold(
        n_splits=config["tuning"]["cv_folds"],
        shuffle=True,
        random_state=config["split"]["random_seed"],
    )
```

- Une seule fonction fabrique les plis. À l'étape 5, la recherche d'hyperparamètres et l'entraînement l'utiliseront toutes deux : elles évalueront un modèle sur **exactement les mêmes plis**, et annonceront donc le même score pour les mêmes réglages.

### Lecture du notebook

`git diff etape-2 etape-3 -- notebooks/` montre les changements :

- `df = load_dataset(config)` remplace `pd.read_csv(...)` : la lecture inclut la validation.
- `features = get_features(config)` remplace la liste construite à la main.
- `split_dataset(df, config)` et `build_cv(config)` remplacent `train_test_split` et `StratifiedKFold`.

### Manipulations

1. Exécuter le notebook : résultats identiques à l'étape 0 (test : 0,9955).
2. Dans un terminal :

   ```bash
   uv run python -c "from crop_recommendation_tp.config import load_config; from crop_recommendation_tp.data import load_dataset, split_dataset; c = load_config(); X_train, X_test, y_train, y_test = split_dataset(load_dataset(c), c); print(X_train.shape, X_test.shape)"
   ```

   **Résultat attendu** : `(1760, 7) (440, 7)`. Le même code sert maintenant hors du notebook.

### Questions de compréhension

<details>
<summary>1. Pourquoi <code>load_dataset</code> lève-t-elle une exception, alors que <code>validate</code> renvoie une liste ?</summary>

`validate` constate ; `load_dataset` décide. Une fonction qui charge les données pour un entraînement ne doit pas continuer avec des données invalides : l'exception arrête tout, avec la liste des erreurs. `validate` reste réutilisable ailleurs, par exemple pour afficher un rapport.
</details>

<details>
<summary>2. Quel est l'intérêt de <code>build_cv</code>, qui ne fait que créer un objet <code>StratifiedKFold</code> ?</summary>

Garantir que tous les programmes qui font de la validation croisée utilisent les mêmes plis. Sans cette fonction unique, deux scripts pourraient découper différemment (avec ou sans mélange, par exemple) et annoncer des scores différents pour un même modèle.
</details>

---

## Étape 4 : tests et qualité du code

### Pourquoi cette étape

Dans un notebook, on vérifie « à l'œil » en lisant les sorties. Quand on modifie une fonction, qui revérifie toutes les autres ? Un **test automatisé** est une petite fonction qui exécute un bout de code dans une situation précise et compare le résultat au résultat attendu. On lance tous les tests en une commande, en quelques secondes.

Les tests servent à trois choses :

- **détecter les régressions** : ce qui marchait et ne marche plus ;
- **documenter** : un test se lit comme un exemple du comportement attendu ;
- **oser modifier** : on améliore le code sans craindre de casser le reste.

**Ruff** complète les tests : il analyse le code **sans l'exécuter**, signale les erreurs probables et uniformise la mise en forme.

### Notions

| Notion | Définition |
| --- | --- |
| pytest | outil qui trouve et exécute les tests |
| Fonction de test | fonction dont le nom commence par `test_`, dans un fichier `test_*.py` |
| `assert` | vérification : le test échoue si la condition est fausse |
| Fixture | fonction qui prépare ce dont un test a besoin ; pytest la fournit au test selon le nom du paramètre |
| Linter | outil qui détecte les erreurs probables dans le code (Ruff) |
| Formateur | outil qui uniformise la mise en forme (Ruff) |

### Passer à l'étape

```bash
git checkout -f etape-4
uv sync
uv run dvc pull
uv run pytest
```

**Résultat attendu** : `13 passed`.

### Fichiers

| Fichier | Rôle |
| --- | --- |
| `pyproject.toml`, `uv.lock` | pytest et Ruff dans les dépendances de développement ; règles de Ruff |
| `tests/test_validation.py` | 9 tests de `validate` |
| `tests/test_data.py` | 4 tests de `data.py` |
| `.vscode/settings.json` | Ruff appliqué à chaque enregistrement dans VS Code |

### Lecture du code : `tests/test_validation.py`

```python
@pytest.fixture
def schema():
    return load_yaml("config/schema.yaml")


@pytest.fixture
def df():
    """Tableau de deux lignes conformes au schéma."""
    return pd.DataFrame(
        {
            "N": [90, 20],
            "P": [42, 60],
            "K": [43, 20],
            "temperature": [20.9, 25.0],
            "humidity": [82.0, 60.5],
            "ph": [6.5, 7.0],
            "rainfall": [202.9, 100.0],
            "label": ["rice", "apple"],
        }
    )


def test_donnees_conformes(df, schema):
    assert validate(df, schema) == []


def test_ph_hors_bornes(df, schema):
    df.loc[0, "ph"] = 75.0
    assert validate(df, schema) == ["ph : 1 valeur(s) > 14"]
```

Déroulement d'un test :

1. **Découverte** : pytest parcourt le projet et repère `tests/test_validation.py`.
2. **Collecte** : il retient les fonctions dont le nom commence par `test_`.
3. **Préparation** : il lit le nom des paramètres (`df`, `schema`), exécute les fixtures du même nom et passe leurs résultats au test.
4. **Exécution** : le test réussit si chaque `assert` est vrai.

Chaque test reçoit un **tableau neuf** : la modification du pH dans un test n'affecte pas les autres. Les tests utilisent de **petits tableaux fabriqués** : ils tournent sans le vrai fichier de données.

### Lecture du code : `tests/test_data.py`

```python
def test_decoupage_taille_et_stratification(df, config):
    X_train, X_test, _y_train, y_test = split_dataset(df, config)
    assert len(X_train) == 16
    assert len(X_test) == 4
    assert y_test.value_counts().to_dict() == {"rice": 2, "apple": 2}
```

- Tableau de 20 lignes, deux cultures : 20 % de test donne 4 lignes, 2 de chaque culture grâce à la stratification.
- `_y_train` : le tiret bas signale une variable volontairement inutilisée.

```python
def test_plis_reproductibles_et_melanges(config):
    X = pd.DataFrame({"a": range(40)})
    y = pd.Series(["rice"] * 20 + ["apple"] * 20)  # rangé par culture, comme le CSV
    premiers = [test.tolist() for _, test in build_cv(config).split(X, y)]
    seconds = [test.tolist() for _, test in build_cv(config).split(X, y)]
    assert len(premiers) == 5
    assert premiers == seconds
    # sans mélange, le premier pli prendrait les 4 premières lignes de chaque culture
    assert premiers[0] != [0, 1, 2, 3, 20, 21, 22, 23]
```

- `.split(X, y)` produit, pour chaque pli, les indices d'entraînement et d'évaluation ; on garde ceux d'évaluation.
- Trois vérifications : 5 plis, mêmes plis à chaque appel (reproductibilité), lignes mélangées.

### Lecture : règles de Ruff

```toml
[tool.ruff.lint]
select = ["E4", "E7", "E9", "F", "I", "SIM", "RUF"]
ignore = ["RUF001", "RUF002", "RUF003"]
```

| Règle | Vérification |
| --- | --- |
| `F` (Pyflakes) | imports inutilisés ou en double, noms non définis |
| `I` (isort) | ordre des imports : bibliothèque standard, bibliothèques externes, projet |
| `E4`, `E7`, `E9` | erreurs de syntaxe et de style graves |
| `SIM` | code simplifiable |
| `RUF` | règles propres à Ruff |

Les règles `RUF001` à `RUF003`, qui signalent certains caractères ambigus, sont ignorées : elles se déclencheraient sur les guillemets et apostrophes du français.

### Manipulations

1. **Casser une règle** : dans `validation.py`, remplacer `"float": is_number` par `"float": types.is_float_dtype`, puis :

   ```bash
   uv run pytest
   ```

   **Résultat attendu** : `test_entiers_acceptes_pour_un_float` échoue. pytest affiche la valeur obtenue et la valeur attendue. Le test protège la règle « une température entière est valide ». Annuler la modification : `git checkout -- src`.

2. **Ruff** : dans `data.py`, déplacer `import pandas as pd` sous les imports du projet, puis :

   ```bash
   uv run ruff check .
   ```

   **Résultat attendu** : erreur `I001` (imports non triés). `uv run ruff check . --fix` corrige ; dans VS Code, l'enregistrement du fichier suffit.

### Questions de compréhension

<details>
<summary>1. Pourquoi les tests n'utilisent-ils pas le vrai fichier de données ?</summary>

Pour qu'ils soient rapides, lisibles (on voit les données du test) et exécutables partout, y compris sur une machine qui n'a pas les données, comme celle de l'intégration continue (étape 6).
</details>

<details>
<summary>2. Que se passerait-il si la fixture <code>df</code> renvoyait toujours le même tableau, sans le recréer ?</summary>

Les tests se perturberaient : la modification du pH dans un test resterait visible dans le suivant, qui échouerait ou réussirait selon l'ordre d'exécution. Un test doit donner le même résultat seul ou avec les autres.
</details>

<details>
<summary>3. Ruff et pytest font-ils double emploi ?</summary>

Non. Ruff lit le code sans l'exécuter : il trouve des erreurs de forme (import inutilisé, nom mal écrit) et uniformise la présentation. pytest exécute le code et vérifie son comportement. Un code peut être parfaitement présenté et faux, ou juste et mal présenté.
</details>

---

## Étape 5 : entraînement en script

### Pourquoi cette étape

L'entraînement vit encore dans le notebook : résultat dépendant de l'ordre d'exécution des cellules, aucune trace de la version des données ni du code. On le remplace par deux **commandes** reproductibles :

- `uv run tune` : recherche des meilleurs hyperparamètres ;
- `uv run train` : entraînement du modèle retenu.

Chaque run de `train` enregistre l'**empreinte des données** et le **commit du code** : c'est la base de la traçabilité.

### Notions

**Hyperparamètre.** Réglage d'un modèle choisi avant l'entraînement : nombre d'arbres, profondeur maximale.

**Recherche sur grille.** Essai de toutes les combinaisons d'une liste de valeurs (la grille), chacune évaluée en validation croisée.

**Signature d'un modèle.** Description des entrées et des sorties attendues (colonnes, types). MLflow l'enregistre avec le modèle.

**Run imbriqué.** Run enfant rattaché à un run parent : la recherche crée un parent et un enfant par combinaison.

### Passer à l'étape

```bash
git checkout -f etape-5
uv sync
uv run dvc pull
```

Serveur MLflow démarré (étape 0).

### Fichiers

| Fichier | Rôle |
| --- | --- |
| `src/crop_recommendation_tp/tracking.py` | connexion à MLflow, enregistrement des runs |
| `src/crop_recommendation_tp/train.py` | commande `train` |
| `src/crop_recommendation_tp/tune.py` | commande `tune` |
| `src/crop_recommendation_tp/data.py` | ajout de `get_data_version` |
| `config/config.yaml` | ajout de `model.params` |
| `pyproject.toml` | commandes `train` et `tune` |
| `tests/test_train.py` | `train` et `tune` donnent le même score |

### Lecture : les commandes

```toml
[project.scripts]
train = "crop_recommendation_tp.train:main"
tune = "crop_recommendation_tp.tune:main"
```

« La commande `train` appelle la fonction `main` du module `crop_recommendation_tp.train` ». `uv sync` installe ces commandes dans l'environnement ; `uv run train` les exécute.

### Lecture : `config/config.yaml`

```yaml
model:
    params:
        n_estimators: 200
        max_depth: null
```

Hyperparamètres du modèle entraîné par `train`. Ce sont ceux trouvés par la recherche du notebook (section 6) et par `tune`.

### Lecture du code : `train.py`

```python
def build_model(config: dict) -> RandomForestClassifier:
    """Crée une forêt aléatoire non entraînée avec les paramètres de config.yaml."""
    return RandomForestClassifier(
        **config["model"]["params"],
        random_state=config["split"]["random_seed"],
    )
```

- `**dictionnaire` **déplie** le dictionnaire en arguments nommés : l'appel équivaut à `RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)`.
- Modèle **non entraîné** : la validation croisée en a besoin, car elle l'entraîne elle-même 5 fois.

```python
def train_model(X_train, y_train, config: dict) -> RandomForestClassifier:
    """Entraîne une forêt aléatoire avec les paramètres de config.yaml."""
    return build_model(config).fit(X_train, y_train)
```

- Une seule fonction crée le modèle : impossible que la validation croisée et l'entraînement final utilisent des réglages différents.

```python
def cross_validate_model(X_train, y_train, config: dict) -> dict:
    """Mesure le score en validation croisée sur le jeu d'entraînement."""
    tuning = config["tuning"]
    scores = cross_val_score(
        build_model(config),
        X_train,
        y_train,
        scoring=tuning["scoring"],
        cv=build_cv(config),
    )
    metric = f"cv_{tuning['scoring']}"
    return {
        f"{metric}_mean": float(scores.mean()),
        f"{metric}_std": float(scores.std()),
    }
```

- Score de validation croisée sur l'entraînement : il sert à **comparer** les modèles (étape 8).
- Le nom de la métrique est construit à partir de la configuration : `cv_f1_macro_mean`, `cv_f1_macro_std`.

```python
def main() -> None:
    config = load_config()
    X_train, X_test, y_train, y_test = split_dataset(load_dataset(config), config)
    model = train_model(X_train, y_train, config)
    cv_metrics = cross_validate_model(X_train, y_train, config)
    predictions, test_metrics = evaluate(model, X_test, y_test)
    metrics = {**cv_metrics, **test_metrics}
    run_id = log_run("train", config, model, metrics, predictions, X_test, y_test)
```

- `main` **assemble** les fonctions : charger et valider, découper, entraîner, mesurer, enregistrer.
- `{**a, **b}` fusionne deux dictionnaires.
- Le score sur le test **estime** la performance ; il ne sert à aucun choix.

### Lecture du code : `tracking.py`

```python
def setup_mlflow(config: dict) -> None:
    uri = os.environ.get("MLFLOW_TRACKING_URI") or config["mlflow"]["tracking_uri"]
    ...
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
```

- `os.environ.get(...)` lit une **variable d'environnement** ; si elle n'existe pas, elle renvoie `None`, et `or` prend alors la valeur de la configuration. La variable l'emporte : les tests et les conteneurs s'en servent pour changer de serveur sans modifier la configuration.
- `set_experiment` crée l'expérience si elle n'existe pas.

```python
def get_git_commit() -> str:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()

    try:
        return git("rev-parse", "HEAD") + (
            "-dirty" if git("status", "--porcelain") else ""
        )
    except (OSError, subprocess.CalledProcessError):
        return "inconnu"
```

- `subprocess.run` exécute une commande du système : ici `git rev-parse HEAD`, qui affiche l'identifiant du commit courant.
- `git status --porcelain` liste les fichiers modifiés ; si la liste n'est pas vide, on ajoute `-dirty` : le code exécuté ne correspond exactement à aucun commit.
- Si Git est absent ou si le dossier n'est pas un dépôt, la fonction renvoie `"inconnu"` au lieu de planter.

```python
with mlflow.start_run(run_name=run_name) as run:
    mlflow.log_params(params)
    mlflow.set_tags({**trace_tags(config), **(tags or {})})
    mlflow.log_dict(config, "config.yaml")
    mlflow.log_dict(load_yaml(config["dataset"]["schema"]), "schema.yaml")
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(
        model,
        name="model",
        signature=infer_signature(X_test.astype("float64"), predictions),
        input_example=X_test.head().astype("float64"),
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )
```

| Ligne | Ce qui est enregistré |
| --- | --- |
| `log_params` | tous les réglages du modèle et la part du jeu de test |
| `set_tags` | `data_md5`, `git_commit` (fonction `trace_tags`) et les tags passés en paramètre |
| `log_dict(config, …)` | la configuration **telle qu'elle était** au moment de l'entraînement |
| `log_dict(schema, …)` | le schéma utilisé |
| `log_metrics` | scores de validation croisée et de test |
| `log_model` | le modèle, sa signature (7 colonnes `float64`) et un exemple d'entrée |

- `skops_trusted_types` : MLflow enregistre les modèles scikit-learn au format *skops*, qui exige la liste des types autorisés au chargement. Sans cette liste, l'enregistrement échoue (`UntrustedTypesFoundException`).
- `matplotlib.use("Agg")`, en tête du module : les figures sont produites sans fenêtre, ce qui permet de les créer dans un script ou sur un serveur.

### Lecture du code : `get_data_version` (`data.py`)

```python
def get_data_version(config: dict) -> str:
    dvc_file = load_yaml(config["dataset"]["path"] + ".dvc")
    expected = dvc_file["outs"][0]["md5"]
    actual = hashlib.md5((ROOT / config["dataset"]["path"]).read_bytes()).hexdigest()
    return actual if actual == expected else actual + "-modified"
```

- Le fichier `.dvc` est du YAML : `load_yaml` le lit, et `["outs"][0]["md5"]` extrait l'empreinte attendue.
- `hashlib.md5(...).hexdigest()` calcule l'empreinte du fichier réellement lu.
- Suffixe `-modified` si les deux diffèrent : quelqu'un a modifié le CSV sans `dvc add`.

### Lecture du code : `tune.py` et `log_tuning`

```python
search = GridSearchCV(
    RandomForestClassifier(random_state=config["split"]["random_seed"]),
    param_grid=tuning["param_grid"],
    scoring=tuning["scoring"],
    cv=build_cv(config),
    n_jobs=-1,
)
```

- 3 valeurs de `n_estimators` × 4 valeurs de `max_depth` = 12 combinaisons, chacune sur 5 plis : 60 entraînements.
- `n_jobs=-1` utilise tous les cœurs du processeur.
- `cv=build_cv(config)` : les mêmes plis que `train`.

`log_tuning` crée un run parent « tuning » et un run enfant par combinaison (`nested=True`), puis note la meilleure combinaison sur le parent (`best_n_estimators`, `best_max_depth`).

`tune` n'enregistre aucun modèle et ne modifie pas la configuration : **chercher** et **décider** restent deux étapes. On reporte soi-même la meilleure combinaison dans `model.params`, puis on commite : le choix est tracé dans Git.

### Lecture du code : `tests/test_train.py`

```python
def test_train_et_tune_donnent_le_meme_score(config, donnees):
    X, y = donnees
    score_train = cross_validate_model(X, y, config)["cv_f1_macro_mean"]
    score_tune = tune(X, y, config).best_score_
    assert score_train == pytest.approx(score_tune)
```

- La configuration du test contient une seule combinaison dans la grille, identique à `model.params` : `tune` et `train` évaluent donc le même modèle.
- `pytest.approx` compare deux décimaux avec une tolérance, pour éviter les écarts d'arrondi.
- Si l'une des deux commandes cessait d'utiliser `build_cv`, ce test échouerait.

### Manipulations

```bash
uv run tune
```

**Résultat attendu** :

```text
Meilleurs paramètres : {'max_depth': None, 'n_estimators': 200}
```

```bash
uv run train
```

**Résultat attendu** :

```text
Run … enregistré : {'cv_f1_macro_mean': 0.9942…, 'cv_f1_macro_std': 0.0047…, 'accuracy': 0.9954…, 'f1_macro': 0.9954…}
```

Dans MLflow, expérience `crop-recommendation` :

1. Run « tuning » : déplier ses 12 runs enfants, trier par `cv_f1_macro_mean`.
2. Run « train » : onglet *Overview* (paramètres, métriques, tags `data_md5` et `git_commit`), onglet *Artifacts* (configuration, schéma, rapport, matrice de confusion).
3. Comparer `git_commit` au résultat de `git rev-parse HEAD` : identiques, sans `-dirty`.

### Pièges fréquents

| Symptôme | Cause | Solution |
| --- | --- | --- |
| `git_commit` se termine par `-dirty` | fichiers modifiés ou non suivis : `git status` les liste | commiter avant d'entraîner un modèle que l'on veut garder ; notebook enregistré après exécution : `git checkout -- notebooks` |
| `-dirty` avec `?? .env` dans `git status` | `.env` créé à l'étape 7, et `.gitignore` ne l'exclut qu'à partir de cette étape | supprimer `.env` pour revenir aux étapes 0 à 6, le recréer à l'étape 7 |
| `data_md5` se termine par `-modified` | CSV différent de l'étiquette `.dvc` | `uv run dvc checkout --force`, ou `dvc add` si la modification est voulue |
| `Données non conformes au schéma` | CSV invalide | lire la liste des erreurs |

### Questions de compréhension

<details>
<summary>1. Pourquoi la configuration est-elle enregistrée dans le run, alors qu'elle est déjà versionnée par Git ?</summary>

Le run garde la configuration réellement utilisée, même si elle n'était pas commitée (`-dirty`). On la consulte directement dans MLflow, sans fouiller l'historique Git.
</details>

<details>
<summary>2. Pourquoi <code>tune</code> ne met-il pas lui-même à jour <code>model.params</code> ?</summary>

Pour séparer la recherche de la décision. La recherche est occasionnelle ; l'entraînement doit pouvoir être relancé à l'identique à partir de la seule configuration. Le report manuel garde une personne dans la boucle et laisse une trace dans Git.
</details>

<details>
<summary>3. Le score de test sert-il à quelque chose, s'il ne sert à aucun choix ?</summary>

Oui : c'est l'estimation honnête de la performance sur des données nouvelles, à communiquer. C'est justement parce qu'il ne sert à aucun choix qu'il reste honnête.
</details>

---

## Étape 6 : intégration continue

### Pourquoi cette étape

Les tests et Ruff ne servent que si on les lance. L'**intégration continue** (CI) les lance automatiquement à chaque envoi de code sur GitHub, sur une **machine neuve**. Elle prouve que le projet s'installe et fonctionne ailleurs que sur votre poste, et signale immédiatement le commit qui casse quelque chose.

### Notions

**GitHub Actions.** Service de GitHub qui exécute des tâches sur des machines virtuelles, décrites dans un fichier YAML du dossier `.github/workflows/`.

**Workflow.** Suite d'étapes déclenchée par un événement (envoi de code, demande de fusion).

### Passer à l'étape

```bash
git checkout -f etape-6
uv sync
uv run dvc pull
```

### Lecture : `.github/workflows/ci.yml`

```yaml
name: CI

on:
    push:
    pull_request:

permissions:
    contents: read

jobs:
    tests:
        runs-on: ubuntu-latest
        steps:
            - uses: actions/checkout@v7
            - uses: astral-sh/setup-uv@v10.3.3
            - run: uv sync --locked
            - run: uv run ruff check .
            - run: uv run ruff format --check .
            - run: uv run pytest
```

| Élément | Rôle |
| --- | --- |
| `on` | déclencheurs : chaque `push`, chaque `pull_request` |
| `permissions: contents: read` | la CI peut lire le code, pas le modifier : droits minimaux |
| `runs-on: ubuntu-latest` | machine Linux neuve, fournie par GitHub |
| `actions/checkout` | récupère le code du dépôt |
| `astral-sh/setup-uv` | installe uv |
| `uv sync --locked` | installe l'environnement exact ; échoue si `uv.lock` ne correspond pas à `pyproject.toml` |
| `ruff check`, `ruff format --check` | vérifications de Ruff, sans modifier le code |
| `pytest` | tests |

Les tests n'ont besoin ni des données ni de MLflow : ils s'exécutent sur cette machine vierge.

### Manipulations

1. Sur GitHub, onglet *Actions* du dépôt de cours : exécution sur le commit de l'étape 6, toutes les étapes en vert. Cliquer sur une étape pour lire son journal.
2. Pour voir votre propre CI : *Fork* du dépôt sur votre compte GitHub, puis un commit sur votre fork ; l'onglet *Actions* de votre fork montre l'exécution.
3. Avant chaque commit, lancer les trois mêmes vérifications en local :

   ```bash
   uv run ruff check .
   uv run ruff format --check .
   uv run pytest
   ```

### Questions de compréhension

<details>
<summary>1. Pourquoi la CI utilise-t-elle <code>uv sync --locked</code> plutôt que <code>uv sync</code> ?</summary>

`uv sync` mettrait à jour `uv.lock` s'il ne correspondait plus à `pyproject.toml`. `--locked` échoue dans ce cas : on détecte un fichier de verrouillage oublié, au lieu de tester un environnement différent de celui des développeurs.
</details>

<details>
<summary>2. Les tests passent sur votre poste mais échouent en CI. Causes possibles ?</summary>

Un test qui dépend d'un fichier absent du dépôt (données, `.env`), d'une variable d'environnement locale, d'un service lancé sur votre poste, ou d'un fichier modifié non commité.
</details>

---

## Bilan de la phase 1 : retrouver l'origine d'un modèle

Objectif : partir d'un run de `train` et retrouver le code, les données et les réglages qui l'ont produit, puis le reproduire.

1. **Le run** : dans MLflow, ouvrir le dernier run « train » de l'expérience `crop-recommendation`.
2. **Les réglages** : onglet *Artifacts*, `config.yaml` et `schema.yaml`, tels qu'ils étaient lors de l'entraînement.
3. **Le code** : tag `git_commit`.

   ```bash
   git log -1 --format='%h %s' <git_commit>
   git describe --tags --exact-match <git_commit>
   ```

   **Résultat attendu** : `2b3e414 Étape 5 : entraînement en script` et `etape-5`, si votre dernier `train` date de l'étape 5 (l'étape 6 n'entraîne rien). Si `git_commit` se termine par `-dirty`, utiliser l'identifiant sans le suffixe : il désigne le commit de départ, mais le code exécuté en différait.

4. **Les données** : tag `data_md5`, à comparer à l'étiquette DVC de ce commit.

   ```bash
   git show <git_commit>:data/raw/crop_recommendation.csv.dvc
   ```

   **Résultat attendu** : `md5: 5e743dd4…`, identique à `data_md5`.

5. **Reproduire** :

   ```bash
   git checkout -f <git_commit>
   uv sync
   uv run dvc checkout --force
   uv run train
   ```

   **Résultat attendu** : les mêmes scores (0,9943 en validation croisée, 0,9955 sur le test).

| Ingrédient | Où le retrouver |
| --- | --- |
| Code | tag `git_commit` → Git |
| Données | tag `data_md5` → étiquette `.dvc` → DVC |
| Environnement | `uv.lock` du commit |
| Réglages | artefacts `config.yaml` et `schema.yaml` du run |

Le modèle est **reproductible** : chaque ingrédient est enregistré automatiquement.

---

## Étape 7 : serveur MLflow

### Pourquoi cette étape

Le serveur MLflow des étapes précédentes tourne dans un terminal, avec une base SQLite : il s'arrête avec le terminal, et ses données sont dans un fichier du projet. On le remplace par un serveur **durable** : conteneurs Docker, base PostgreSQL, données conservées dans des volumes. C'est l'architecture qu'on retrouvera en production.

### Notions

| Notion | Définition |
| --- | --- |
| Docker | outil qui fait tourner des logiciels dans des conteneurs isolés, identiques sur tous les ordinateurs |
| Image | environnement logiciel figé (système, Python, bibliothèques), décrit par un `Dockerfile` |
| Conteneur | image en cours d'exécution |
| Volume | espace de stockage géré par Docker, qui survit à l'arrêt et à la suppression des conteneurs |
| Docker Compose | outil qui décrit plusieurs conteneurs dans un seul fichier (`compose.yaml`) et les lance ensemble |
| Port publié | port du poste relié à un port du conteneur, pour y accéder depuis le poste |
| PostgreSQL | système de base de données, conçu pour plusieurs utilisateurs simultanés |

### Passer à l'étape

1. Arrêter le serveur MLflow local (`Ctrl+C` dans son terminal) : il occupe le port 5001.
2. Changer d'étape, créer le fichier des secrets, démarrer les services :

   ```bash
   git checkout -f etape-7
   uv sync
   uv run dvc pull
   cp .env.example .env
   ```

   Ouvrir `.env` et renseigner un mot de passe après `POSTGRES_PASSWORD=`, en lettres et chiffres seulement : il est inséré dans une adresse (`postgresql+psycopg2://…`) où `@`, `:`, `/` ou `#` la casseraient. Puis :

   ```bash
   docker compose up -d --build
   docker compose ps
   ```

**Résultat attendu** : services `postgres` et `mlflow` à l'état `healthy`. La première construction de l'image prend une à deux minutes.

L'adresse du serveur ne change pas (<http://127.0.0.1:5001>) : `config.yaml`, le notebook et les scripts fonctionnent sans modification. Les runs de l'ancienne base SQLite ne sont pas repris : relancer le notebook et `uv run train`.

### Fichiers

| Fichier | Rôle |
| --- | --- |
| `compose.yaml` | services `postgres` et `mlflow`, volumes |
| `Dockerfile.mlflow` | image du serveur MLflow |
| `.env.example` | modèle du fichier des secrets |
| `.gitignore` | ajout de `.env` |

### Lecture du code : `Dockerfile.mlflow`

```dockerfile
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /uvx /bin/

RUN uv pip install --system mlflow==3.18.0 psycopg2-binary

EXPOSE 5000
```

| Instruction | Effet |
| --- | --- |
| `FROM` | image de départ : Python 3.13, version allégée |
| `COPY --from=…` | copie l'exécutable uv depuis l'image officielle d'uv |
| `RUN` | exécute une commande pendant la construction : installation de MLflow et du pilote PostgreSQL |
| `EXPOSE` | documente le port utilisé par le conteneur |

La version de MLflow est **figée** (`==3.18.0`), identique à celle du projet (`uv.lock`) : le serveur et les scripts qui lui parlent sont compatibles.

### Lecture du code : `compose.yaml`

```yaml
services:
    postgres:
        image: postgres:18
        env_file: .env
        volumes:
            - postgres_data:/var/lib/postgresql
        healthcheck:
            test: ["CMD-SHELL", "pg_isready -U $${POSTGRES_USER} -d $${POSTGRES_DB}"]
            interval: 5s
            retries: 10
```

- `image: postgres:18` : image officielle, téléchargée automatiquement.
- `env_file: .env` : les variables du fichier `.env` (utilisateur, mot de passe, nom de la base) sont transmises au conteneur.
- `volumes` : le dossier des données de PostgreSQL est placé dans le volume `postgres_data`.
- `healthcheck` : toutes les 5 secondes, Docker vérifie que la base accepte les connexions (`pg_isready`).
- `$$` : Compose transmet un `$` littéral ; la variable est lue **dans** le conteneur.

```yaml
    mlflow:
        build:
            context: .
            dockerfile: Dockerfile.mlflow
        depends_on:
            postgres:
                condition: service_healthy
        ports:
            - "127.0.0.1:${MLFLOW_PORT:-5001}:5000"
        volumes:
            - mlflow_artifacts:/mlartifacts
        command: >
            mlflow server
            --host 0.0.0.0
            --port 5000
            --backend-store-uri postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
            --artifacts-destination /mlartifacts
```

| Élément | Rôle |
| --- | --- |
| `build` | l'image est construite à partir de `Dockerfile.mlflow` |
| `depends_on … service_healthy` | MLflow attend que la base soit prête |
| `ports` | le port 5000 du conteneur est publié sur le port 5001 du poste, **seulement** sur `127.0.0.1` (inaccessible depuis le réseau) |
| `${MLFLOW_PORT:-5001}` | valeur de la variable `MLFLOW_PORT`, ou 5001 par défaut |
| `--host 0.0.0.0` | le serveur écoute sur toutes les interfaces du conteneur, condition pour être joignable depuis l'extérieur du conteneur |
| `--backend-store-uri` | paramètres, métriques et registre dans PostgreSQL |
| `@postgres:5432` | Compose crée un réseau où chaque service est joignable **par son nom** : `postgres` désigne le conteneur de la base |
| `--artifacts-destination` | fichiers (modèles, figures) dans le volume `mlflow_artifacts` ; les scripts envoient leurs fichiers **au serveur**, qui les range |

### Les secrets

Les identifiants de la base ne sont **jamais** écrits dans `compose.yaml` : Compose les lit dans `.env`, exclu de Git. Le dépôt ne contient que `.env.example`, sans mot de passe, qui documente les variables à fournir.

### Manipulations

1. `docker compose ps` : deux services en bonne santé.
2. `docker volume ls` : volumes `crop_recommendation_cours_postgres_data` et `crop_recommendation_cours_mlflow_artifacts`.
3. Relancer le notebook, puis `uv run train` : les runs apparaissent dans le nouveau serveur.
4. `docker compose down`, puis `docker compose up -d` : les runs sont conservés, grâce aux volumes.

| Commande | Effet |
| --- | --- |
| `docker compose ps` | état des services |
| `docker compose logs --tail 20 mlflow` | dernières lignes du journal |
| `docker compose down` | arrêt ; les volumes sont conservés |
| `docker compose down -v` | arrêt **et suppression** des volumes : tous les runs sont perdus |

### Pièges fréquents

| Symptôme | Cause | Solution |
| --- | --- | --- |
| `port is already allocated` | serveur local encore démarré | `Ctrl+C` dans son terminal |
| `env file … .env not found` | fichier des secrets absent | `cp .env.example .env` |
| PostgreSQL redémarre en boucle | mot de passe vide dans `.env` | renseigner `POSTGRES_PASSWORD` |
| `Cannot connect to the Docker daemon` | Docker Desktop arrêté | lancer Docker Desktop |

### Questions de compréhension

<details>
<summary>1. Pourquoi publier le port sur <code>127.0.0.1</code> plutôt que sur toutes les interfaces ?</summary>

Avec `127.0.0.1`, seul votre poste accède au serveur. Sans cette précision, n'importe qui sur le même réseau (une salle de cours, un café) pourrait ouvrir MLflow, qui n'a pas d'authentification.
</details>

<details>
<summary>2. Dans <code>--backend-store-uri</code>, pourquoi l'hôte est-il <code>postgres</code> et non <code>localhost</code> ?</summary>

Dans un conteneur, `localhost` désigne le conteneur lui-même. Les services d'un même fichier Compose communiquent par un réseau interne, où chacun est joignable par son nom de service.
</details>

<details>
<summary>3. Que perd-on avec <code>docker compose down -v</code> ?</summary>

Les volumes : la base PostgreSQL (paramètres, métriques, registre) et les artefacts (modèles, figures). Tous les runs disparaissent.
</details>

---

## Étape 8 : registre et promotion

### Pourquoi cette étape

Chaque `uv run train` produit un modèle. Lequel mettre en service ? Il faut distinguer **tous les essais** des **modèles retenus**, et une **règle** appliquée automatiquement pour décider qu'un nouveau modèle remplace l'ancien.

### Notions

| Notion | Définition |
| --- | --- |
| Registre de modèles | catalogue des modèles retenus, sous un nom (`crop-recommendation`) |
| Version | chaque modèle ajouté au registre reçoit un numéro : 1, 2, 3… |
| Alias | étiquette mobile posée sur une version ; `@champion` désigne le modèle en service |
| Candidat | dernier modèle entraîné, en attente de comparaison |
| Promotion | le candidat devient champion : nouvelle version, et l'alias y est déplacé |

L'API demandera « le champion », et non « la version 3 » : changer de modèle en service revient à **déplacer l'alias**.

### Passer à l'étape

```bash
git checkout -f etape-8
uv sync
uv run dvc pull
uv run promote      # → Aucun run candidat : lancer d'abord uv run train.
uv run train
uv run promote      # → Version 1 promue champion (score 0.9943).
uv run promote      # → Le candidat est déjà le champion (version 1).
```

Le premier `promote` ne trouve aucun candidat : les runs des étapes précédentes ne portaient pas l'étiquette `stage = candidate`.

### Fichiers

| Fichier | Changement |
| --- | --- |
| `src/crop_recommendation_tp/promote.py` | nouveau |
| `src/crop_recommendation_tp/train.py` | runs étiquetés `stage = candidate` |
| `config/config.yaml` | `registered_model_name`, `champion_alias` |
| `pyproject.toml` | commande `promote` |
| `tests/conftest.py` | fixture partagée : base MLflow temporaire |
| `tests/test_promote.py` | 11 tests de la règle de promotion |

### La règle de promotion

| Situation | Résultat |
| --- | --- |
| Aucun run candidat | message : lancer d'abord `uv run train` |
| Le candidat n'a pas la métrique | message, aucune promotion |
| Registre sans champion | candidat promu |
| Candidat déjà champion | aucune action |
| Score inférieur ou égal à celui du champion | champion conservé |
| Score strictement supérieur, ou champion sans score | nouvelle version, qui reçoit l'alias `champion` |

Le score comparé est `cv_f1_macro_mean`, la validation croisée sur l'entraînement. À égalité, on garde le modèle en place : changer de modèle a un coût (vérifications, redémarrage du service).

### Lecture du code : `promote.py`

```python
def find_candidate_run_id(client: MlflowClient, config: dict) -> str | None:
    experiment = client.get_experiment_by_name(config["mlflow"]["experiment_name"])
    if experiment is None:
        return None
    for run in client.search_runs([experiment.experiment_id]):
        if run.data.tags.get("stage") == "candidate":
            return run.info.run_id
    return None
```

- `MlflowClient` interroge le serveur MLflow.
- `search_runs` renvoie les runs du plus récent au plus ancien : le premier run candidat est le dernier entraîné.
- `tags.get("stage")` renvoie `None` si le tag n'existe pas, au lieu de lever une erreur.
- `str | None` : la fonction renvoie un texte, ou `None` s'il n'y a pas de candidat.

```python
def get_champion(client: MlflowClient, config: dict) -> ModelVersion | None:
    registry = config["mlflow"]
    try:
        return client.get_model_version_by_alias(
            registry["registered_model_name"], registry["champion_alias"]
        )
    except MlflowException:
        return None
```

- Sans champion, MLflow lève une exception. `try` / `except` la convertit en `None`, que `main` sait traiter.

```python
def promote_candidate(client: MlflowClient, config: dict, run_id: str) -> ModelVersion:
    registry = config["mlflow"]
    name = registry["registered_model_name"]
    version = mlflow.register_model(f"runs:/{run_id}/model", name)
    client.set_registered_model_alias(name, registry["champion_alias"], version.version)
    return version
```

- `runs:/<run_id>/model` désigne le modèle enregistré par ce run.
- `register_model` en crée une nouvelle version dans le registre (et crée le modèle enregistré s'il n'existe pas).
- `set_registered_model_alias` déplace l'alias sur cette version.

```python
champion = get_champion(client, config)
if champion is not None:
    if champion.run_id == run_id:
        print(f"Le candidat est déjà le champion (version {champion.version}).")
        return
    champion_score = get_run_score(client, champion.run_id, metric)
    if champion_score is not None and candidate_score <= champion_score:
        print(f"Candidat non promu : {candidate_score:.4f} <= {champion_score:.4f}.")
        return
version = promote_candidate(client, config, run_id)
```

- Chaque cas particulier du tableau a son message et son `return`. On n'atteint la promotion que si aucun cas ne l'empêche.
- `{candidate_score:.4f}` affiche le nombre avec 4 décimales.

### Lecture du code : `tests/conftest.py`

```python
@pytest.fixture
def config(tmp_path, monkeypatch):
    """Configuration qui pointe vers une base SQLite temporaire."""
    monkeypatch.chdir(tmp_path)
    db = (tmp_path / "mlflow.db").as_posix()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{db}")
    mlflow.set_tracking_uri(f"sqlite:///{db}")
    yield {
        "mlflow": {...},
        "tuning": {"scoring": "f1_macro"},
    }
    mlflow.set_tracking_uri(None)
```

| Élément | Rôle |
| --- | --- |
| `conftest.py` | fichier spécial : ses fixtures sont disponibles dans tous les fichiers de tests |
| `tmp_path` | fixture de pytest : un dossier temporaire neuf pour chaque test |
| `monkeypatch` | fixture de pytest : modifie temporairement le dossier courant ou une variable d'environnement, puis rétablit l'état initial après le test |
| `yield` | coupe la fixture en deux : ce qui précède s'exécute avant le test, ce qui suit après |
| `set_tracking_uri(None)` | efface un réglage **global** de MLflow ; sans cette ligne, le test suivant hériterait de la base du précédent |

Chaque test de `promote` reçoit ainsi une base MLflow **neuve**, sans serveur. Les tests enregistrent de petits modèles factices (`DummyClassifier`) : ils vérifient la **logique** de promotion, pas la qualité d'un modèle.

### Manipulations

1. Interface MLflow, *Model registry* (barre de navigation de gauche) : modèle `crop-recommendation`, version 1, colonne *Aliased versions* : `@champion`.
2. **Piège** : l'onglet *Models* d'une expérience liste les modèles enregistrés par les runs ; ils n'ont **jamais** d'alias. L'alias se voit dans *Model registry*.
3. **Candidat moins bon** : dans `config/config.yaml`, mettre `n_estimators: 20` et `max_depth: 3`, puis :

   ```bash
   uv run train
   uv run promote      # → Candidat non promu : 0.8472 <= 0.9943.
   git checkout -- config
   ```

### Questions de compréhension

<details>
<summary>1. Pourquoi comparer les scores de validation croisée plutôt que les scores de test ?</summary>

Promouvoir un modèle est un choix. Faire ce choix sur le jeu de test le rendrait optimiste (voir l'étape 0). La validation croisée sur l'entraînement sert aux choix ; le test reste une estimation honnête.
</details>

<details>
<summary>2. Pourquoi un alias plutôt qu'un numéro de version dans l'API ?</summary>

Avec un numéro, chaque nouveau modèle obligerait à modifier l'API. Avec un alias, l'API demande toujours « le champion » ; seule la promotion décide quelle version il désigne.
</details>

<details>
<summary>3. Que vérifierait un test qui échouerait si l'on supprimait <code>mlflow.set_tracking_uri(None)</code> de la fixture ?</summary>

Un test qui dépend de l'ordre d'exécution : par exemple, « aucun candidat dans une base vide ». Lancé après un autre test, il lirait la base du test précédent, qui contient déjà un candidat.
</details>

---

## Étape 9 : API de prédiction

### Pourquoi cette étape

Le champion est désigné ; il faut le rendre utilisable par d'autres applications. Une **API web** reçoit les 7 mesures d'une parcelle et renvoie la culture recommandée. Les entrées sont contrôlées avec le schéma avant d'atteindre le modèle.

### Notions

| Notion | Définition |
| --- | --- |
| API web | interface qui permet à un programme d'en appeler un autre par le réseau, avec le protocole HTTP |
| Route | adresse et méthode d'appel : `GET /health`, `POST /predict` |
| JSON | format texte d'échange de données : `{"N": 90, "P": 42}` |
| Code de réponse | 200 : succès ; 422 : données invalides ; 500 : erreur du serveur |
| FastAPI | bibliothèque Python pour écrire des API ; valide les données et produit une documentation interactive |
| Pydantic | bibliothèque de validation de données, utilisée par FastAPI |

### Passer à l'étape

```bash
git checkout -f etape-9
uv sync
uv run dvc pull
uv run fastapi dev
```

Prérequis : services Docker démarrés et champion existant (étape 8). Documentation interactive : <http://127.0.0.1:8000/docs>. `Ctrl+C` arrête l'API.

### Fichiers

| Fichier | Rôle |
| --- | --- |
| `src/crop_recommendation_tp/api.py` | l'API |
| `pyproject.toml` | FastAPI ; section `[tool.fastapi]` qui désigne l'application |
| `tests/test_api.py` | 5 tests |

### Lecture du code : le démarrage

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    setup_mlflow(config)
    registry = config["mlflow"]
    name = registry["registered_model_name"]
    alias = registry["champion_alias"]
    try:
        version = MlflowClient().get_model_version_by_alias(name, alias)
    except MlflowException as erreur:
        raise RuntimeError(
            f"Alias {alias} introuvable pour {name} : vérifier que le serveur "
            "MLflow tourne, puis lancer uv run train et uv run promote."
        ) from erreur
    app.state.model = mlflow.pyfunc.load_model(f"models:/{name}/{version.version}")
    app.state.version = str(version.version)
    app.state.run_id = version.run_id
    yield
```

- `lifespan` s'exécute **une seule fois**, au démarrage de l'API. Ce qui précède `yield` prépare l'API ; elle répond ensuite aux requêtes.
- Le modèle est chargé **en mémoire** une fois pour toutes : le charger à chaque requête serait beaucoup trop lent.
- L'alias est résolu en numéro de version : on sait exactement quelle version est chargée.
- `app.state` : espace où FastAPI conserve des objets partagés entre les requêtes.
- Sans champion, l'API **refuse de démarrer** avec un message qui dit quoi faire. `from erreur` conserve l'erreur d'origine dans le message.

### Lecture du code : la validation des entrées

```python
TYPES = {"int": int, "float": float}


def build_input_model(config):
    schema = load_yaml(config["dataset"]["schema"])
    target = config["dataset"]["target"]
    fields = {}
    for name, rules in schema["columns"].items():
        if name == target:
            continue
        fields[name] = (
            TYPES[rules["dtype"]],
            Field(ge=rules.get("min"), le=rules.get("max")),
        )
    return create_model("CropInput", **fields)


CropInput = build_input_model(load_config())
```

- `create_model` construit une classe Pydantic **à partir du schéma** : un champ par variable, avec son type et ses bornes.
- `ge` : supérieur ou égal ; `le` : inférieur ou égal. `rules.get("min")` vaut `None` si le schéma ne fixe pas de borne.
- Les règles des données d'entraînement et celles de l'API restent identiques : **une seule source de vérité**, `schema.yaml`.

### Lecture du code : les routes

```python
@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": app.state.version,
        "run_id": app.state.run_id,
    }


@app.post("/predict")
def predict(entree: CropInput):  # pyright: ignore[reportInvalidTypeForm]
    donnees = pd.DataFrame([entree.model_dump()]).astype("float64")
    prediction = app.state.model.predict(donnees)
    return {"culture": str(prediction[0]), "version": app.state.version}
```

- `@app.get("/health")` est un **décorateur** : il associe la fonction à la route `GET /health`.
- L'annotation `entree: CropInput` est lue par **FastAPI pendant l'exécution** : il lit le corps JSON de la requête, le valide avec `CropInput`, renvoie une erreur 422 si les données sont invalides, et documente les champs dans `/docs`. La fonction n'est appelée qu'avec des données valides.
- Le commentaire `pyright: ignore` concerne seulement l'analyseur de l'éditeur (Pylance), qui ne peut pas connaître une classe créée pendant l'exécution.
- `model_dump()` transforme l'entrée en dictionnaire ; `astype("float64")` respecte la signature du modèle, qui attend des décimaux.
- La réponse contient la **version** du modèle : chaque prédiction est traçable.

### Manipulations

1. <http://127.0.0.1:8000/docs> → `POST /predict` → *Try it out*, avec :

   ```json
   {"N": 90, "P": 42, "K": 43, "temperature": 20.8, "humidity": 82.0, "ph": 6.5, "rainfall": 202.9}
   ```

   **Résultat attendu** : `{"culture": "rice", "version": "1"}`.

2. Même requête avec `"ph": 15` : code **422**, message `Input should be less than or equal to 14`.
3. En ligne de commande :

   ```bash
   curl http://127.0.0.1:8000/health      # PowerShell : curl.exe http://127.0.0.1:8000/health
   ```

   **Résultat attendu** : `{"status":"ok","version":"1","run_id":"…"}`. Sous PowerShell, `curl` sans `.exe` désigne une autre commande (`Invoke-WebRequest`).

4. Après une nouvelle promotion, l'API sert toujours l'ancien modèle : il faut la relancer (`Ctrl+C`, puis `uv run fastapi dev`).

### Questions de compréhension

<details>
<summary>1. Pourquoi l'API charge-t-elle le modèle au démarrage plutôt qu'à chaque requête ?</summary>

Charger un modèle prend du temps (téléchargement depuis MLflow, lecture). Le faire à chaque requête rendrait l'API très lente. En contrepartie, une promotion n'est prise en compte qu'au redémarrage.
</details>

<details>
<summary>2. Une requête envoie <code>"N": 12.5</code>. Que se passe-t-il ?</summary>

Erreur 422 : `N` est déclaré entier (`int`) dans le schéma, et Pydantic refuse un nombre à décimales pour un champ entier.
</details>

<details>
<summary>3. À quoi sert la version dans la réponse de <code>/predict</code> ?</summary>

À tracer la prédiction : la version mène au run, et le run au code, aux données et aux réglages (bilan de la phase 2).
</details>

---

## Étape 10 : API en conteneur

### Pourquoi cette étape

L'API tourne sur le poste, lancée à la main. Pour la déployer, il faut l'empaqueter dans une **image Docker**, comme le serveur MLflow. Toute la pile (base, serveur MLflow, API) démarre alors en une commande, identique sur tous les postes. C'est ce qui sera déployé en production.

### Passer à l'étape

1. Arrêter `uv run fastapi dev` s'il tourne : il occupe le port 8000.
2. Changer d'étape et reconstruire :

   ```bash
   git checkout -f etape-10
   uv sync
   uv run dvc pull
   docker compose up -d --build
   docker compose ps
   ```

**Résultat attendu** : `postgres`, `mlflow` et `api` à l'état `healthy`. La construction de l'image de l'API prend plusieurs minutes la première fois (image d'environ 2,6 Go).

Sans champion dans le registre, le conteneur `api` s'arrête : lancer `uv run train` et `uv run promote`, puis `docker compose restart api`.

### Fichiers

| Fichier | Rôle |
| --- | --- |
| `Dockerfile.api` | image de l'API |
| `.dockerignore` | fichiers exclus des images |
| `compose.yaml` | service `api` ; option `--allowed-hosts` du serveur MLflow |

### Lecture du code : `Dockerfile.api`

```dockerfile
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# 1. Les dépendances seules : cette étape reste en cache tant que uv.lock ne change pas
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

# 2. Le code et la configuration, puis le projet lui-même
COPY README.md ./
COPY src/ src/
COPY config/ config/
RUN uv sync --locked --no-dev

# 3. Exécution sans les droits d'administrateur
RUN useradd --create-home app
USER app

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]

CMD ["fastapi", "run", "--host", "0.0.0.0", "--port", "8000"]
```

| Partie | Explication |
| --- | --- |
| `WORKDIR /app` | dossier de travail dans l'image |
| Étape 1 | Docker garde chaque instruction en cache. Les dépendances sont installées **avant** de copier le code : tant que `uv.lock` ne change pas, cette longue installation n'est pas refaite |
| `--no-dev` | ni Jupyter, ni pytest, ni Ruff dans l'image |
| `--no-install-project` | dépendances seules, sans le projet, dont le code n'est pas encore copié |
| Étape 2 | copie du code et de la configuration, puis installation du projet. `README.md` est nécessaire à la construction du paquet |
| Étape 3 | un utilisateur `app` sans droits d'administrateur exécute l'API : si elle était compromise, l'attaquant aurait moins de pouvoir |
| `ENV PATH` | les commandes de `.venv` (dont `fastapi`) sont trouvées directement |
| `HEALTHCHECK` | Docker interroge `/health` ; le conteneur n'est « healthy » que si l'API répond |
| `CMD` | commande de démarrage : `fastapi run`, le mode production |

Le projet est installé en mode modifiable dans `/app` : la constante `ROOT` de `config.py` y retrouve `config/config.yaml` et `config/schema.yaml`.

### Lecture : `.dockerignore`

Liste des fichiers qui n'entrent **jamais** dans une image : secrets (`.env`), `.git`, `.venv`, données, essais MLflow, notebooks, tests. L'image ne contient que ce qui sert à l'API.

### Lecture du code : `compose.yaml`

```yaml
    api:
        build:
            context: .
            dockerfile: Dockerfile.api
        depends_on:
            mlflow:
                condition: service_healthy
        environment:
            MLFLOW_TRACKING_URI: http://mlflow:5000
        ports:
            - "127.0.0.1:${API_PORT:-8000}:8000"
```

- Dans le conteneur, `127.0.0.1` désigne le conteneur lui-même : l'adresse de `config.yaml` ne convient pas. L'API joint MLflow **par son nom de service**, `mlflow`, sur son port interne 5000.
- `MLFLOW_TRACKING_URI` l'emporte sur `config.yaml` : c'est la fonction `setup_mlflow` de l'étape 5.

```yaml
            --allowed-hosts mlflow:5000,localhost:*,127.0.0.1:*
```

Depuis la version 3, le serveur MLflow n'accepte que les noms d'hôte autorisés, pour se protéger d'une attaque appelée *DNS rebinding*. Par défaut, seuls `localhost` et les adresses privées sont acceptés. Sans cette option, il refuse l'API, qui l'appelle sous le nom `mlflow` : erreur 403, `Invalid Host header`.

### Manipulations

1. `docker compose ps` : trois services en bonne santé.
2. `curl http://127.0.0.1:8000/health` (PowerShell : `curl.exe`), puis une prédiction depuis <http://127.0.0.1:8000/docs>.
3. `docker compose exec api whoami` → `app`.
4. L'API charge le champion à son démarrage : après une promotion, `docker compose restart api` est nécessaire pour que `/health` affiche la nouvelle version. Avec la grille de réglages actuelle, aucun candidat ne dépasse le champion (0,9943) : la promotion se verra en phase 3, avec de nouvelles données.

### Pièges fréquents

| Symptôme | Cause | Solution |
| --- | --- | --- |
| Le conteneur `api` s'arrête : `Alias champion introuvable…` | registre sans champion | `uv run train`, `uv run promote`, `docker compose restart api` |
| `port is already allocated` pour 8000 | `fastapi dev` encore lancé | `Ctrl+C` dans son terminal |
| `403` et `Invalid Host header` dans `docker compose logs api` | nom d'hôte non autorisé | ajouter le nom à `--allowed-hosts` |

### Questions de compréhension

<details>
<summary>1. Pourquoi copier <code>pyproject.toml</code> et <code>uv.lock</code> avant le code ?</summary>

Pour profiter du cache de Docker. Une modification du code ne touche pas `uv.lock` : la couche d'installation des dépendances est réutilisée, et seule la fin de la construction est refaite.
</details>

<details>
<summary>2. L'image contient-elle le modèle ?</summary>

Non : elle contient le code et la configuration. Le modèle est chargé depuis le registre MLflow au démarrage. Changer de champion ne demande donc pas de reconstruire l'image, seulement de redémarrer le conteneur.
</details>

<details>
<summary>3. Pourquoi l'API ne peut-elle pas utiliser <code>http://127.0.0.1:5001</code>, l'adresse de <code>config.yaml</code> ?</summary>

Dans le conteneur, `127.0.0.1` désigne le conteneur de l'API, où aucun serveur MLflow n'écoute. Le port 5001 n'existe que sur le poste. Les conteneurs se joignent par leur nom de service, sur leur port interne.
</details>

---

## Bilan de la phase 2 : retrouver l'origine d'une prédiction

Objectif : partir d'une prédiction de l'API et remonter jusqu'au code, aux données et aux réglages qui l'ont produite.

1. **La prédiction** : `POST /predict` → `{"culture": "rice", "version": "1"}`. La réponse contient la version du modèle.
2. **Le modèle chargé** :

   ```bash
   curl http://127.0.0.1:8000/health      # PowerShell : curl.exe http://127.0.0.1:8000/health
   ```

   **Résultat attendu** : `{"status":"ok","version":"1","run_id":"b1049297…"}` (votre `run_id` sera différent).

3. **Le registre** : *Model registry* → `crop-recommendation` → version 1 (`@champion`) → lien vers le run source.
4. **Le run** : tags `data_md5`, `git_commit` et `stage` ; métriques ; artefacts `config.yaml` et `schema.yaml`. En ligne de commande :

   ```bash
   MLFLOW_TRACKING_URI=http://127.0.0.1:5001 uv run mlflow artifacts download --run-id <run_id> --artifact-path config.yaml
   ```

   PowerShell :

   ```powershell
   $env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5001"
   uv run mlflow artifacts download --run-id <run_id> --artifact-path config.yaml
   ```

   La commande affiche le chemin du fichier téléchargé.

5. **Le code** :

   ```bash
   git log -1 --format='%h %s' <git_commit>
   git describe --tags --exact-match <git_commit>
   ```

   **Résultat attendu** : le commit et l'étiquette de l'étape où le modèle a été entraîné, par exemple `etape-8`.

6. **Les données** :

   ```bash
   git show <git_commit>:data/raw/crop_recommendation.csv.dvc
   ```

   **Résultat attendu** : `md5: 5e743dd4…`, identique au tag `data_md5`.

```text
prédiction → version du modèle → run → réglages (artefacts)
                                     → code (git_commit → Git)
                                     → données (data_md5 → .dvc → DVC)
```

Chaque maillon est enregistré automatiquement ; aucun ne dépend de la mémoire d'une personne.

---

## Étape 11 : journal des requêtes

### Pourquoi cette étape

Le modèle a été entraîné sur des données d'une période et d'une région précises. Une fois en service, il reçoit des mesures qui peuvent s'en éloigner : c'est le **data drift** (diapo 12). Pour le détecter, il faut comparer ce que l'API **reçoit** à ce qui a servi à l'entraînement. Or, jusqu'ici, l'API répond puis oublie la requête.

Cette étape garde une trace de chaque prédiction. Sans ce journal, aucune surveillance de la dérive n'est possible : c'est le premier maillon de la phase 3.

### Notions

**Journal (*log*) des requêtes.** Fichier où l'API ajoute une ligne à chaque prédiction : date, mesures reçues, culture prédite, version du modèle.

**JSON Lines (`.jsonl`).** Un objet JSON par ligne. On ajoute une ligne sans relire le fichier, et pandas le lit directement (`pd.read_json(..., lines=True)`).

**UTC.** Heure universelle, sans fuseau ni changement d'heure : deux serveurs dans des pays différents écrivent des dates comparables.

### Passer à l'étape

```bash
git checkout -f etape-11
uv sync
uv run dvc pull
docker compose up -d --build
```

Le service `api` est reconstruit : son image contient le nouveau code.

### Fichiers

| Fichier | Changement |
| --- | --- |
| `src/crop_recommendation_tp/monitoring.py` | nouveau : `get_log_path`, `log_request`, `read_request_log` |
| `src/crop_recommendation_tp/api.py` | `/predict` journalise chaque prédiction |
| `config/config.yaml` | ajout de `monitoring.requests_log` |
| `compose.yaml`, `Dockerfile.api` | volume `api_logs` monté sur `/app/logs` |
| `.gitignore`, `.dockerignore` | `logs/` exclu de Git et des images |
| `tests/test_monitoring.py`, `tests/test_api.py`, `tests/conftest.py` | 4 nouveaux tests, un test complété |

### Lecture du code : `monitoring.py`

```python
def log_request(path: Path, mesures: dict, culture: str, version: str) -> None:
    ligne = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        **mesures,
        "prediction": culture,
        "model_version": version,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with _verrou, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(ligne, ensure_ascii=False) + "\n")
```

- `**mesures` déplie les 7 mesures dans le dictionnaire : une colonne par variable, comme dans les données d'entraînement. La comparaison des deux (étape suivante) en sera directe.
- `model_version` : on saura quel modèle a répondu, même après une promotion.
- `open(path, "a")` : mode **ajout**, les lignes précédentes sont conservées.
- `_verrou` (`threading.Lock`) : FastAPI traite plusieurs requêtes en même temps ; le verrou empêche deux écritures de se mélanger dans une même ligne.

```python
def get_log_path(config: dict) -> Path:
    path = os.environ.get("REQUESTS_LOG_PATH") or config["monitoring"]["requests_log"]
    return ROOT / path
```

- Même principe que `MLFLOW_TRACKING_URI` (étape 5) : la variable d'environnement l'emporte sur `config.yaml`.
- Un chemin absolu reste absolu : `ROOT / "/tmp/x.jsonl"` donne `/tmp/x.jsonl`. Les tests en profitent pour écrire dans un dossier temporaire.

### Lecture du code : `api.py`

```python
@app.post("/predict")
def predict(entree: CropInput):
    mesures = entree.model_dump()
    donnees = pd.DataFrame([mesures]).astype("float64")
    culture = str(app.state.model.predict(donnees)[0])
    log_request(app.state.log_path, mesures, culture, app.state.version)
    return {"culture": culture, "version": app.state.version}
```

- Le chemin du journal est calculé **une fois**, au démarrage (`lifespan`), comme le chargement du modèle.
- Une requête invalide (pH = 15) est rejetée par FastAPI **avant** d'atteindre `predict` : elle n'est pas journalisée. Le journal ne contient que des mesures plausibles, comparables aux données d'entraînement.

### Lecture : le volume `api_logs`

```yaml
api:
    volumes:
        - api_logs:/app/logs
```

```dockerfile
RUN useradd --create-home app && mkdir logs && chown app:app logs
```

- Sans volume, le journal disparaîtrait avec le conteneur, à chaque `docker compose up --build`.
- Le dossier `logs/` est créé dans l'image et confié à l'utilisateur `app`. À sa création, le volume reprend ce propriétaire : l'API, qui tourne sans droits d'administrateur, peut y écrire.

### Manipulations

1. Envoyer deux prédictions valides et une invalide depuis <http://127.0.0.1:8000/docs> (ou avec `curl`, voir le README).
2. Lire le journal du conteneur :

   ```bash
   docker compose exec api tail -n 3 logs/requests.jsonl
   ```

   **Résultat attendu** : deux lignes, avec `"prediction"` et `"model_version"` ; aucune ligne pour la requête invalide.

3. Redémarrer l'API, puis relire le journal :

   ```bash
   docker compose restart api
   docker compose exec api wc -l logs/requests.jsonl     # → 2 logs/requests.jsonl
   ```

   Les lignes sont conservées : elles vivent dans le volume, pas dans le conteneur.

4. Tests : `uv run pytest` → **34 passed**.

### Pièges fréquents

| Symptôme | Cause | Solution |
| --- | --- | --- |
| `Permission denied: '/app/logs/requests.jsonl'` | volume `api_logs` créé par une autre version de l'image, avec un autre propriétaire | `docker compose down`, `docker volume rm crop_recommendation_cours_api_logs`, puis `docker compose up -d` |
| `ports are not available … 8000` | port 8000 occupé par une autre application | choisir un autre port : `API_PORT=8001 docker compose up -d` (PowerShell : `$env:API_PORT = "8001"`) |
| Git Bash : `No such file or directory` sur `/app/logs/…` | Git Bash convertit les chemins qui commencent par `/` | utiliser un chemin relatif (`logs/requests.jsonl`), ou `MSYS_NO_PATHCONV=1` devant la commande |

### Limites, à traiter dans les étapes suivantes

- Le fichier grossit sans fin : en production, il faudra le faire tourner (un fichier par jour) ou l'envoyer dans une base.
- Les requêtes rejetées (422) ne sont comptées nulle part. Leur nombre est pourtant un signal utile (« Données invalides en entrée », diapo 11) : il relève de la surveillance du service (Prometheus).
- Le journal contient les données des utilisateurs : en production, il faut décider combien de temps on le garde.

### Questions de compréhension

<details>
<summary>1. Pourquoi enregistrer la version du modèle dans chaque ligne ?</summary>

Après une promotion, le journal mêle des réponses de deux modèles. La version permet d'analyser chaque modèle séparément, et de relier une réponse surprenante au run qui l'a produite (bilan de la phase 2).
</details>

<details>
<summary>2. Pourquoi ne pas journaliser les requêtes rejetées dans le même fichier ?</summary>

Le journal sert à comparer les entrées reçues aux données d'entraînement. Des valeurs impossibles (pH = 15) fausseraient cette comparaison : ce sont des erreurs de saisie ou de capteur, pas une évolution du monde. Elles se surveillent à part, en comptant les erreurs 422.
</details>

<details>
<summary>3. Que se passerait-il sans le volume <code>api_logs</code> ?</summary>

Le journal serait écrit dans le conteneur. Il survivrait à un `docker compose restart`, mais disparaîtrait au premier `docker compose up --build`, qui recrée le conteneur à partir de la nouvelle image.
</details>

---

## Étape 12 : surveillance de la dérive

### Pourquoi cette étape

Le modèle a appris sur 1 760 parcelles. Il n'est fiable que pour des parcelles qui leur ressemblent, mais il ne le sait pas : il répond toujours, avec la même assurance. Envoyez à l'API une parcelle de riz à 58 °C (valide pour le schéma, qui autorise jusqu'à 60 °C ; jamais vue à l'entraînement, où la température ne dépasse pas 44 °C) : la réponse est `rice`, sans aucun avertissement.

Le schéma (étape 1) contrôle chaque requête **une à une** : il rejette l'impossible, pas l'inhabituel. Pour voir que les requêtes, **dans leur ensemble**, s'éloignent de ce que le modèle connaît, il faut comparer des **distributions** : celle du journal (étape 11) et celle des données d'entraînement. C'est la surveillance du **data drift** (diapo 12).

Côté métier, une alerte précoce évite de découvrir le problème six mois plus tard, à la récolte.

### Notions

**Référence et données courantes.** La référence est ce que le modèle a appris : les 7 variables du jeu d'entraînement (pas le jeu de test, ni la cible). Les données courantes sont les mesures reçues par l'API, lues dans le journal. On compare uniquement les **entrées** : le journal ne contient pas la bonne réponse.

**Test de Kolmogorov-Smirnov (KS).** Compare deux échantillons d'une même variable en mesurant le plus grand écart entre leurs courbes de fréquences cumulées. Il renvoie une **p-value**.

**p-value.** Probabilité d'observer un écart au moins aussi grand si les deux échantillons venaient de la même distribution. Petite p-value (< 0,05) : l'écart est trop grand pour être dû au hasard, on conclut à une dérive.

**Fausse alerte.** Avec un seuil de 0,05, un test sur deux échantillons de même origine conclut à tort à une dérive une fois sur vingt. Avec 7 variables testées à chaque rapport, une fausse alerte de temps en temps est inévitable.

**Evidently.** Bibliothèque de surveillance de modèles : elle applique le test à chaque variable et produit un rapport HTML.

### Le journal n'est pas un jeu d'entraînement

Les requêtes reçues ne sont pas de nouvelles données d'entraînement : elles ne contiennent pas la culture **réellement adaptée**, connue seulement après la récolte. Le journal sert à **détecter** que les entrées changent. Une alerte déclenche la suite de la diapo 12 : collecter de nouvelles données **étiquetées**, les versionner avec DVC (étape 2), réentraîner (`train`, `promote`).

### Passer à l'étape

```bash
git checkout -f etape-12
uv sync
uv run dvc pull
docker compose up -d --build
```

- `uv sync` installe Evidently et passe pandas de la version 3 à la version 2 (voir « Fichiers »).
- L'image de l'API est reconstruite : son `Dockerfile` change.

### Fichiers

| Fichier | Changement |
| --- | --- |
| `src/crop_recommendation_tp/simulate.py` | nouveau : commande `simulate`, des utilisateurs simulés |
| `src/crop_recommendation_tp/drift.py` | nouveau : commande `drift`, comparaison du journal à l'entraînement |
| `config/config.yaml` | réglages de la surveillance dans `monitoring` |
| `pyproject.toml`, `uv.lock` | commandes `simulate` et `drift` ; groupe `monitoring` (Evidently) ; pandas 2 |
| `Dockerfile.api` | `--no-default-groups` au lieu de `--no-dev` |
| `logs/.gitkeep`, `.gitignore`, `.dockerignore` | dossier `logs/` présent mais vide dans Git ; `reports/` exclu |
| `tests/test_simulate.py`, `tests/test_drift.py` | 7 nouveaux tests |

**Pourquoi pandas 2.** La version actuelle d'Evidently (0.7) utilise une fonction interne de pandas supprimée dans pandas 3 : l'import échoue. On contraint donc pandas (`pandas>=2.3,<3`). C'est une situation courante : une bibliothèque en retard sur une autre impose sa contrainte au projet. `uv.lock` garantit que tout le monde a la même combinaison qui fonctionne.

### Lecture : la configuration

```yaml
monitoring:
    requests_log: logs/requests.jsonl
    drift_report: reports/drift_report.html
    min_requests: 50
    drift_method: ks
    drift_threshold: 0.05
    drift_share: 0.1
```

- `min_requests` : en dessous de 50 requêtes, pas de conclusion. Sur 3 requêtes, un test ne peut rien dire.
- `drift_threshold` : une variable est en dérive si sa p-value est inférieure à 0,05.
- `drift_share` : part des variables en dérive qui déclenche l'alerte. 1 variable sur 7 (14 %) dépasse 0,1 : une seule suffit.

### Lecture du code : `simulate.py`

```python
def build_requests(X_test, scenario, n, seed):
    mesures = X_test.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)
    if scenario == "region-chaude":
        mesures["temperature"] = mesures["temperature"] + 10
    elif scenario == "secheresse":
        mesures["rainfall"] = mesures["rainfall"] / 3
    return mesures.round(1)
```

- Les parcelles viennent du **jeu de test** : réalistes, et jamais vues à l'entraînement.
- `replace=True` : tirage avec remise, on peut demander plus de requêtes qu'il n'y a de parcelles.
- `random_state=seed` : même tirage sur tous les postes, d'où des résultats identiques à ceux de ce manuel.
- Scénario `concept` : aucune transformation. Dans l'histoire, la bonne réponse a changé (restriction d'eau, diapo 13), mais les mesures, elles, sont les mêmes.

Le simulateur passe par l'API, comme une vraie application : les requêtes sont validées, prédites, puis journalisées. Il ne touche pas au journal.

### Lecture du code : `drift.py`

```python
def load_reference(config):
    X_train, _, _, _ = split_dataset(load_dataset(config), config)
    return X_train.reset_index(drop=True)
```

- La référence est recalculée avec `split_dataset` (étape 3) : exactement les lignes d'entraînement, grâce à la graine.

```python
report = Report([DataDriftPreset(method="ks", threshold=0.05, drift_share=0.1)])
snapshot = report.run(current_data=current, reference_data=reference)
```

- `DataDriftPreset` applique le test à chaque variable. Les valeurs viennent de la configuration.
- `snapshot.save_html(...)` écrit le rapport ; `snapshot.dict()` donne les mêmes résultats sous forme de dictionnaire, lus par `summarize`.

```python
if resume["derive"]:
    print("Dérive détectée : alerter, collecter de nouvelles données, réentraîner.")
    sys.exit(1)
```

- **Code de sortie 1** en cas de dérive. Un programme qui lance `drift` chaque jour (tâche planifiée, CI) sait ainsi, sans lire le texte, qu'il faut alerter. C'est ce qui rendra la surveillance automatique en production.

### Lecture : Evidently hors de l'image de l'API

```toml
[dependency-groups]
monitoring = ["evidently>=0.7.23"]

[tool.uv]
default-groups = ["dev", "monitoring"]
```

```dockerfile
RUN uv sync --locked --no-default-groups --no-install-project
```

- `uv sync` installe les groupes par défaut : `dev` et `monitoring`. Sur le poste et en CI, `drift` et ses tests fonctionnent.
- L'API n'a pas besoin d'Evidently. `--no-dev` n'exclurait que le groupe `dev` ; `--no-default-groups` les exclut tous. L'image reste légère, et une bibliothèque de moins dans l'API, c'est une source de failles de moins.
- `drift.py` est un module séparé de `monitoring.py` : l'API importe `monitoring.py` (journal), jamais `drift.py` (Evidently).

### Manipulations

Pile démarrée (`docker compose up -d`) et champion existant (étape 8). Pour chaque scénario :

```bash
docker compose exec api rm -f logs/requests.jsonl           # 1. vider le journal
uv run simulate --scenario normal                             # 2. 200 requêtes
docker compose cp api:app/logs/requests.jsonl logs/           # 3. copier le journal sur le poste
uv run drift                                                  # 4. comparer
```

- Étape 1 : sans elle, les scénarios s'accumulent dans le journal et le rapport les mélange.
- Étape 3 : le journal est dans le volume du conteneur ; `drift` tourne sur le poste. Dans `docker compose cp`, le chemin du conteneur part de sa racine (`app/logs/…`).

**Résultats attendus** (graine 42) :

| Scénario | Variables en dérive | Code de sortie | Cultures prédites |
| --- | --- | --- | --- |
| `normal` | aucune | 0 | banana, pomegranate, maize… |
| `region-chaude` | `temperature` (p-value ≈ 3e-114) | 1 | **les mêmes** que `normal` |
| `secheresse` | `rainfall` (p-value ≈ 2e-83) | 1 | changent : mothbeans, lentil… |
| `concept` | aucune | 0 | les mêmes que `normal` |

Ouvrir `reports/drift_report.html` dans un navigateur : histogramme de chaque variable, référence et données courantes superposées.

Puis :

5. **Fausse alerte** : vider le journal, `uv run simulate --scenario normal --seed 7`, copier, `uv run drift`. La variable `N` est signalée (p-value ≈ 0,05), alors que les parcelles viennent du même jeu que l'entraînement.
6. **Trop peu de requêtes** : vider le journal, `uv run simulate --scenario region-chaude --n 20`, copier, `uv run drift` → « il en faut au moins 50 pour conclure ».
7. Tests : `uv run pytest` → **41 passed**.

### À observer

- **Une dérive ne change pas forcément les prédictions.** Avec 10 °C de plus, le modèle prédit les mêmes cultures : il s'appuie peu sur la température. Avec trois fois moins de pluie, ses prédictions changent : `rainfall` est sa variable la plus importante (notebook, section 8). Une dérive dit que le modèle sort de son domaine connu, pas qu'il se trompe.
- **Le concept drift est invisible.** Scénario `concept` : aucune alerte, alors que dans l'histoire, le riz n'est plus adapté. Seule la vérité terrain, après la récolte, le révélerait. C'est la limite de la diapo 13 : on surveille d'abord les entrées parce que c'est le seul signal disponible tôt.

### Pièges fréquents

| Symptôme | Cause | Solution |
| --- | --- | --- |
| `ImportError: cannot import name 'infer_dtype_from_object'` | pandas 3 installé avec Evidently 0.7 | `uv sync` (le verrou impose pandas 2) |
| `API injoignable à http://127.0.0.1:8000` | pile arrêtée, ou API sur un autre port | `docker compose up -d` ; sinon `uv run simulate … --url http://127.0.0.1:8001` |
| `0 requête(s) dans le journal` | journal non copié depuis le conteneur | `docker compose cp api:app/logs/requests.jsonl logs/` |
| Le rapport mélange deux scénarios | journal non vidé entre les deux | `docker compose exec api rm -f logs/requests.jsonl` |

### Limites, à discuter

- **Tout le journal est comparé.** Dans un an, avec des milliers de requêtes, l'arrivée d'une nouvelle coopérative serait noyée dans l'historique, et une dérive passée resterait visible pour toujours. En production, on compare plutôt les requêtes récentes (par exemple les 30 derniers jours).
- **Les requêtes sont saisonnières** (avant les semis). Une dérive détectée n'est pas toujours un problème : c'est peut-être le calendrier agricole.
- **La référence n'a ni date ni région.** On ne sait pas de quand datent les données d'entraînement, ni d'où elles viennent : on constate seulement que les entrées s'en éloignent.

### Questions de compréhension

<details>
<summary>1. Pourquoi comparer le journal au jeu d'entraînement, et pas au jeu de test ?</summary>

La référence décrit ce que le modèle a appris. Le jeu de test, mis de côté, n'a pas servi à l'entraînement : ce n'est pas lui qui délimite le domaine connu du modèle. Le simulateur, lui, tire ses parcelles du jeu de test, pour envoyer des mesures réalistes mais jamais vues.
</details>

<details>
<summary>2. Avec 10 °C de plus, les prédictions ne changent pas. La dérive est-elle alors sans importance ?</summary>

Non. Le modèle n'a jamais vu de telles températures : personne ne sait si ses recommandations restent bonnes, et des cultures sensibles à la chaleur pourraient être recommandées à tort. La dérive signale une perte de garantie, pas une erreur constatée. Il faut des données étiquetées de cette région pour trancher.
</details>

<details>
<summary>3. Pourquoi <code>drift</code> se termine-t-il avec un code de sortie 1 en cas de dérive ?</summary>

Pour être utilisable par un autre programme : une tâche planifiée ou une CI teste le code de sortie, sans lire le texte, et déclenche une alerte. Par convention, 0 signifie « tout va bien ».
</details>

<details>
<summary>4. Le scénario <code>concept</code> ne déclenche aucune alerte. Evidently est-il en défaut ?</summary>

Non : il compare les entrées, et elles n'ont pas changé. Le concept drift change la relation entre les entrées et la bonne réponse ; il ne se voit qu'avec la vérité terrain, qui arrive tard. D'où l'intérêt de surveiller d'abord les entrées, seul signal disponible tôt, sans s'en contenter.
</details>

## Étape 13 : image publiée et dépendances à jour

### Pourquoi cette étape

Jusqu'ici, l'image de l'API n'existe que sur le poste qui l'a construite (`docker compose up -d --build`). Pour la faire tourner sur un serveur (étape suivante), il faudrait y copier le code et reconstruire, avec le risque d'obtenir une image différente de celle qui a été testée.

La bonne pratique : **construire une seule fois, dans la CI, après les tests**, puis publier l'image dans un **registre d'images**. Le serveur ne construit rien, il télécharge exactement l'image vérifiée. C'est le même principe que le registre de modèles (étape 8), appliqué au logiciel qui sert le modèle.

Deuxième problème : `uv.lock` fige les versions des dépendances. C'est ce qui rend le projet reproductible (étape 0), mais les versions vieillissent : des failles de sécurité sont découvertes, des corrections publiées. **Dependabot** propose ces mises à jour sous forme de pull requests, que la CI teste avant toute fusion. On reste reproductible, sans rester figé.

### Notions

**Registre d'images.** Serveur qui stocke des images Docker, comme Docker Hub. GitHub en fournit un, **GitHub Container Registry** (`ghcr.io`), rattaché au dépôt : l'image s'appelle `ghcr.io/webew/crop_recommendation_cours`.

**Étiquette d'image.** Nom d'une version de l'image, après les deux-points : `:latest`, `:etape-13`, `:sha-9d9ce6c`. `latest` bouge à chaque publication ; `sha-<commit>` ne bouge jamais et relie l'image au code exact qui l'a produite.

**`GITHUB_TOKEN`.** Jeton temporaire que GitHub fournit à chaque exécution de la CI, valable le temps du job. Aucun mot de passe à créer ni à stocker. Ses droits sont déclarés dans `permissions:`.

**Dependabot.** Robot de GitHub qui lit les fichiers de dépendances, compare aux dernières versions publiées et ouvre une pull request par mise à jour, avec les notes de version.

### Passer à l'étape

```bash
git checkout -f etape-13
```

Aucune dépendance Python ne change : ni `uv sync`, ni reconstruction d'image ne sont nécessaires.

### Fichiers

| Fichier | Changement |
| --- | --- |
| `.github/workflows/ci.yml` | nouveau job `image` : construit et publie l'image si les tests passent |
| `.github/dependabot.yml` | nouveau : mises à jour proposées chaque semaine |
| `compose.ghcr.yaml` | nouveau : variante de `compose.yaml`, API téléchargée au lieu d'être construite |

### Lecture du code : le job `image`

```yaml
image:
    needs: tests
    if: github.event_name == 'push' && (github.ref == 'refs/heads/main' || startsWith(github.ref, 'refs/tags/etape-'))
    permissions:
        contents: read
        packages: write
```

- `needs: tests` : le job attend la fin des tests et ne démarre que s'ils sont verts. Une image cassée n'est jamais publiée.
- `if:` : seulement pour un push sur `main` ou d'un tag d'étape. Une pull request, celle d'un étudiant ou de Dependabot, est testée mais ne publie rien.
- `packages: write` : le droit d'écrire dans le registre, accordé à ce job seul. Le job `tests` garde `contents: read` (principe du moindre privilège).

Les étapes du job :

| Action | Rôle |
| --- | --- |
| `docker/setup-buildx-action` | installe le constructeur d'images |
| `docker/login-action` | se connecte à `ghcr.io` avec `GITHUB_TOKEN` |
| `docker/metadata-action` | calcule les étiquettes (`latest`, `etape-13`, `sha-…`) et les métadonnées (commit, dépôt) |
| `docker/build-push-action` | construit `Dockerfile.api` et pousse l'image ; `cache-from/to: type=gha` réutilise les couches d'une exécution à l'autre |

**Sur `main`, après l'étape 13.** Les déclencheurs ont été resserrés : `push` seulement sur `main` et les tags, en plus de `pull_request`. Avant, une PR de Dependabot lançait la CI deux fois, une fois pour sa branche (`push`) et une fois pour la PR (`pull_request`).

```yaml
on:
    push:
        branches: [main]
        tags: ["*"]
    pull_request:
```

C'est le même `Dockerfile.api` qu'en local (étape 10) : la CI ne fait rien de plus que `docker compose build api`, mais sur une machine neuve, à partir du code commité.

### Lecture : `compose.ghcr.yaml`

```yaml
services:
    api:
        build: !reset null
        image: ghcr.io/webew/crop_recommendation_cours:${API_TAG:-latest}
        pull_policy: always
```

Ce fichier ne se lance pas seul : il **complète** `compose.yaml`. Compose fusionne les deux, dans l'ordre :

```bash
docker compose -f compose.yaml -f compose.ghcr.yaml up -d
```

- `build: !reset null` efface la construction locale héritée de `compose.yaml`.
- `image:` désigne l'image publiée ; `API_TAG` choisit l'étiquette.
- `pull_policy: always` vérifie à chaque démarrage s'il existe une version plus récente de l'étiquette.

PostgreSQL et MLflow ne changent pas. C'est ce que fera le serveur de l'étape suivante.

### Lecture : `.github/dependabot.yml`

| Écosystème | Fichiers surveillés | Fréquence |
| --- | --- | --- |
| `uv` | `pyproject.toml`, `uv.lock` | chaque semaine |
| `docker` | `Dockerfile.api`, `Dockerfile.mlflow` (`python:3.13-slim`, uv) | chaque semaine |
| `docker-compose` | `compose.yaml` (`postgres:18`) | chaque semaine |
| `github-actions` | versions des actions de la CI | chaque mois |

Deux réglages limitent le bruit :

- `groups` : toutes les mises à jour mineures et correctives des bibliothèques Python arrivent dans **une seule** PR.
- `ignore` : pandas 3 n'est pas proposé, puisqu'Evidently 0.7 ne le prend pas en charge (étape 12). Sans cette règle, Dependabot proposerait la mise à jour chaque semaine et la CI échouerait chaque fois.

### Manipulations

1. Sur GitHub, onglet *Actions* : ouvrir l'exécution du tag `etape-13`. Le job `image` démarre après `tests` (flèche entre les deux).
2. Page d'accueil du dépôt, colonne de droite, *Packages* : ouvrir `crop_recommendation_cours` et lire les étiquettes publiées.
3. Télécharger l'image et lire son commit d'origine :
   ```bash
   docker pull ghcr.io/webew/crop_recommendation_cours:etape-13
   docker image inspect ghcr.io/webew/crop_recommendation_cours:etape-13 --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}'
   git rev-parse etape-13
   ```
   Les deux identifiants sont identiques.
4. Lancer la pile avec l'image publiée :
   ```bash
   API_TAG=etape-13 docker compose -f compose.yaml -f compose.ghcr.yaml up -d
   docker compose images api
   curl http://127.0.0.1:8000/health
   ```
5. Revenir à l'image construite en local : `docker compose up -d --build`.
6. Onglet *Pull requests* (ou `gh pr list --author app/dependabot`) : lire une PR de Dependabot, ses notes de version et le résultat de la CI.

### Pièges fréquents

| Symptôme | Cause | Solution |
| --- | --- | --- |
| `denied` ou `unauthorized` au `docker pull` | paquet privé : il hérite de la visibilité du dépôt (public ici) | sur un dépôt privé : `docker login ghcr.io`, ou *Package settings* → *Change visibility* → *Public* |
| Le premier `docker pull` est long | environ 570 Mo à télécharger, 2,6 Go une fois décompressée (bibliothèques scientifiques) | patienter ; les téléchargements suivants ne reprennent que les couches modifiées |
| `docker compose up -d` ne prend pas l'image publiée | `compose.ghcr.yaml` oublié | toujours passer les deux fichiers avec `-f` |
| Le job `image` est absent d'une exécution | événement `pull_request`, ou branche autre que `main` | normal : seul un push sur `main` ou d'un tag `etape-*` publie |

### Questions de compréhension

<details>
<summary>1. Pourquoi construire l'image dans la CI plutôt que sur le serveur de production ?</summary>

Pour que l'image déployée soit exactement celle qui a été testée, construite à partir du code commité, sur une machine neuve. Construire sur le serveur demanderait d'y copier le code et les outils de construction, et pourrait donner une image différente (cache, fichiers locaux, versions téléchargées ce jour-là).
</details>

<details>
<summary>2. Pourquoi publier l'étiquette <code>sha-&lt;commit&gt;</code> en plus de <code>latest</code> ?</summary>

`latest` change à chaque publication : elle ne dit pas quel code tourne. `sha-<commit>` ne change jamais : elle permet de déployer une version précise, de revenir à la précédente, et de retrouver le code d'une image en production. C'est la traçabilité de la phase 1, appliquée au déploiement.
</details>

<details>
<summary>3. Une PR de Dependabot a une CI rouge. Que faire ?</summary>

Ne pas fusionner. Lire l'erreur : la nouvelle version casse quelque chose (comme pandas 3 avec Evidently). Soit on adapte le code, soit on refuse la mise à jour (fermer la PR, ou ajouter une règle `ignore` en expliquant pourquoi). C'est la CI qui rend les mises à jour sûres : sans tests, on n'oserait fusionner aucune de ces PR.
</details>

<details>
<summary>4. Pourquoi le job <code>image</code> ne tourne-t-il jamais pour une pull request ?</summary>

Une PR contient du code non relu : elle pourrait venir de n'importe qui, ou de Dependabot. La publier écraserait `latest` avec un code non validé. On publie seulement ce qui est fusionné sur `main`, ou marqué d'un tag.
</details>

---

## Glossaire

| Terme | Définition |
| --- | --- |
| Accuracy | part des prédictions correctes |
| Alias | étiquette mobile posée sur une version du registre (`champion`) |
| API | interface qui permet à un programme d'en appeler un autre |
| Artefact | fichier joint à un run MLflow : modèle, figure, rapport, configuration |
| Cache DVC | dossier local où DVC range les données, par empreinte |
| Candidat | dernier modèle entraîné, en attente de comparaison au champion |
| CI | intégration continue : vérification automatique de chaque modification |
| CD | déploiement continu : livraison automatique d'une modification vérifiée |
| Commit | état enregistré du projet dans Git |
| Concept drift | changement de la relation entre les entrées et la bonne réponse |
| Conteneur | image Docker en cours d'exécution |
| Data drift | changement de la distribution des entrées |
| Dependabot | robot de GitHub qui propose les mises à jour des dépendances en pull requests |
| Docker Compose | description et lancement de plusieurs conteneurs |
| DVC | outil de versionnage des données |
| Empreinte md5 | identifiant calculé à partir du contenu d'un fichier |
| Environnement virtuel | dossier `.venv` contenant Python et les bibliothèques du projet |
| Étiquette d'image | nom d'une version d'une image Docker : `latest`, `etape-13`, `sha-<commit>` |
| Étiquette Git (*tag*) | nom fixe donné à un commit |
| Evidently | bibliothèque de surveillance de modèles : tests de dérive, rapports |
| Expérience MLflow | groupe de runs comparables |
| F1 macro | moyenne des scores F1 de chaque classe |
| Fausse alerte | dérive signalée par le hasard, sans changement réel |
| Fixture | fonction pytest qui prépare ce dont un test a besoin |
| `GITHUB_TOKEN` | jeton temporaire fourni par GitHub à chaque exécution de la CI |
| Graine aléatoire | valeur qui rend le hasard reproductible |
| Hyperparamètre | réglage d'un modèle choisi avant l'entraînement |
| Image Docker | environnement logiciel figé, décrit par un `Dockerfile` |
| Jeu de test | données mises de côté, utilisées une seule fois à la fin |
| Journal des requêtes | fichier où l'API enregistre chaque prédiction : mesures, culture, version du modèle |
| JSON Lines | format texte : un objet JSON par ligne |
| Kolmogorov-Smirnov (test) | test qui compare deux échantillons d'une variable ; renvoie une p-value |
| Métrique | résultat chiffré enregistré dans un run |
| MLOps | pratiques qui rendent un projet de ML reproductible, traçable, vérifié et déployable |
| Paramètre (MLflow) | réglage d'entrée enregistré dans un run |
| p-value | probabilité d'un écart au moins aussi grand si rien n'a changé ; petite p-value : dérive |
| Pli (*fold*) | une des parts du jeu d'entraînement en validation croisée |
| Promotion | désignation d'un nouveau champion |
| Référence (surveillance) | données auxquelles on compare les requêtes reçues : le jeu d'entraînement |
| Registre d'images | serveur qui stocke et distribue des images Docker : `ghcr.io`, Docker Hub |
| Registre de modèles | catalogue des versions de modèles retenues |
| Run | exécution enregistrée dans MLflow |
| Schéma | description des données attendues |
| Signature | description des entrées et sorties d'un modèle |
| Stockage distant (*remote*) | endroit où DVC envoie les données |
| Stratification | découpage qui conserve la proportion de chaque classe |
| Tag (MLflow) | information libre attachée à un run |
| Test automatisé | fonction qui exécute du code et vérifie le résultat |
| uv | outil de gestion de Python et des dépendances |
| Validation croisée | évaluation répétée sur plusieurs plis du jeu d'entraînement |
| Volume Docker | stockage qui survit aux conteneurs |
| YAML | format texte lisible, utilisé pour la configuration |

---

## Aide-mémoire des commandes

### Git

| Commande | Effet |
| --- | --- |
| `git checkout -f etape-N` | passer à l'étape N |
| `git show --stat etape-N` | fichiers modifiés par l'étape N |
| `git diff etape-A etape-B -- <chemin>` | différences entre deux étapes |
| `git log --oneline --decorate` | historique avec les étiquettes |
| `git rev-parse HEAD` | identifiant du commit courant |

### uv

| Commande | Effet |
| --- | --- |
| `uv sync` | installer l'environnement depuis `uv.lock` |
| `uv add <paquet>` | ajouter une dépendance |
| `uv run <commande>` | exécuter une commande dans l'environnement du projet |

### DVC

| Commande | Effet |
| --- | --- |
| `uv run dvc status` | comparer les données à l'étiquette `.dvc` |
| `uv run dvc add <fichier>` | versionner un fichier de données |
| `uv run dvc push` / `pull` | envoyer / récupérer les données |
| `uv run dvc checkout --force` | restaurer les données décrites par l'étiquette |

### Projet

| Commande | Effet | Étape |
| --- | --- | --- |
| `uv run mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5001` | serveur MLflow local | 0 à 6 |
| `uv run pytest` | tests | 4 |
| `uv run ruff check .` / `uv run ruff format .` | vérification / mise en forme | 4 |
| `uv run tune` | recherche des hyperparamètres | 5 |
| `uv run train` | entraînement d'un candidat | 5 |
| `uv run promote` | promotion du meilleur modèle | 8 |
| `uv run fastapi dev` | API en mode développement | 9 |
| `uv run simulate --scenario <scénario>` | requêtes simulées envoyées à l'API | 12 |
| `uv run drift` | comparaison du journal aux données d'entraînement | 12 |

### Docker

| Commande | Effet |
| --- | --- |
| `docker compose up -d --build` | construire les images et démarrer les services |
| `docker compose ps` | état des services |
| `docker compose logs --tail 20 <service>` | journal d'un service |
| `docker compose restart api` | recharger le champion dans l'API |
| `docker compose exec api tail -n 3 logs/requests.jsonl` | dernières requêtes journalisées (étape 11) |
| `docker compose exec api rm -f logs/requests.jsonl` | vider le journal (étape 12) |
| `docker compose cp api:app/logs/requests.jsonl logs/` | copier le journal sur le poste (étape 12) |
| `docker compose -f compose.yaml -f compose.ghcr.yaml up -d` | démarrer avec l'image publiée sur `ghcr.io` (étape 13) |
| `docker compose images api` | image utilisée par le service `api` (étape 13) |
| `docker compose down` | arrêter ; les données sont conservées |
| `docker compose down -v` | arrêter et supprimer les données |
