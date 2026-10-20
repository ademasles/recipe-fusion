# recipe-fusion

[![CI](https://github.com/ademasles/recipe-fusion/actions/workflows/ci.yml/badge.svg)](https://github.com/ademasles/recipe-fusion/actions/workflows/ci.yml)

**Recipe Fusion — Comparing Image–Text Fusion Strategies for Food Recognition**

Classer une recette parmi 101 plats à partir de sa photo et du titre de sa page web.

Projet 1 du cours de deep learning avancé (Mines Nancy) : classification multimodale texte et
image, avec prétraitement, extraction par modalité, fusion, entraînement et évaluation
comparative. Les données sont UPMC Food-101 (Wang et al., 2015) : 90 688 paires image et titre
réparties en 101 classes.

Le plan de travail est dans [docs/plan.md](docs/plan.md).

## Démarrage

Prérequis : Linux, macOS sur puce Apple ou WSL2, avec git, [uv](https://docs.astral.sh/uv/) et
[just](https://just.systems). Un GPU NVIDIA accélère les calculs ; 8 Go suffisent pour développer.

```bash
just sync        # crée .venv depuis uv.lock
just gpu         # vérifie que PyTorch voit le GPU
just check       # ruff + basedpyright + pytest
just download    # UPMC Food-101 depuis Kaggle (8,3 Go)
uv run pre-commit install   # une fois après le clone
```

Pour n'explorer que le texte, `just download data.texts_only=true` récupère les deux fichiers de
titres (7,6 Mo) en quelques secondes.

## Installation des outils

Une seule fois par machine, sans droits administrateur. Python n'a pas besoin d'être installé :
uv télécharge la version fixée dans `.python-version`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # uv, macOS et Linux
uv tool install rust-just                         # just, distribué par uv
```

## Organisation

| Chemin | Contenu |
|---|---|
| `src/recipe_fusion/data/` | Index du jeu (`upmc.py`) et téléchargement (`download.py`) |
| `src/recipe_fusion/seed.py` | Graines et choix du périphérique |
| `configs/` | Configuration Hydra : chemins, données |
| `scripts/` | Scripts ponctuels (`gpu_check.py`) |
| `notebooks/` | Exploration ; `just kernel` enregistre l'environnement dans Jupyter |
| `tests/` | pytest |
| `docs/` | Plan du projet |
| `data/`, `outputs/` | Données et sorties d'expériences, ignorées par git |

## Données

- Les données vont dans `data/upmc-food101/`. Sur une autre machine,
  `export F101_DATA_DIR=/chemin/vers/data`.
- Source : la copie Kaggle `gianmarco96/upmcfood101`, version 1, téléchargée par `kagglehub`
  sans compte.
- `texts/{train,test}_titles.csv` : sans en-tête, trois colonnes (fichier image, titre, classe).
  Images dans `images/{train,test}/<classe>/<fichier>.jpg`.
- 67 972 paires d'entraînement et 22 716 de test. Il n'y a pas de jeu de validation : il faut
  le prélever dans l'entraînement.
- Les titres sont gardés tels quels, entités HTML comprises (`&#8217;`).

## Conventions

- Dépendances : `uv add <paquet>`, outils de développement : `uv add --dev <paquet>`. Jamais
  `pip install`.
- PyTorch vient de l'index CUDA déclaré dans `pyproject.toml` sous Linux, de PyPI sur Mac.
- Toute expérience est seedée et configurée par Hydra.
- Code sans GPU : `pick_device()`, jamais `"cuda"` en dur, tests sur processeur.
- Ni données, ni points de contrôle, ni secrets dans git ; `gitleaks` tourne en pre-commit.
- Code, commentaires et messages de commit en anglais, au format des commits conventionnels.

## Licences

Le code est sous licence MIT (voir [LICENSE](LICENSE)).

La copie Kaggle d'UPMC Food-101 n'indique pas de licence. Les images et les titres ont été
collectés sur le web par les auteurs : usage de recherche et d'enseignement seulement, sans
redistribution. Citer Wang, Kumar, Thome, Cord et Precioso (2015), *Recipe recognition with
large multimodal food dataset*, ICME Workshops.
