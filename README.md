# MMedAgent-Lite Jurida — Pipeline D1+D2 (FR/AR)

Pipeline de données et stratification pour le pôle juridique de MMedAgent-Lite (cohorte ENS Martil 2026).

## Contexte

Le projet MMedAgent-Lite original couvre le domaine médical. Ce module transpose la fondation données au **droit marocain** à partir du dataset [Jurida](https://www.kaggle.com/datasets/ouaillaamiri02/jurida-dataset) (24 973 Q&A bilingues FR/AR, 1 267 documents juridiques de 1913 à 2024).

- (Groupe D1 Juridique)
- **Encadrant :** Pr. Abdellatif EL AFIA
- **Livrable principal :** `data/jurida_processed/` — HuggingFace `DatasetDict` à 6 splits (`{fr,ar} × {easy, medium, hard}`), consommable par les équipes downstream (D3 à D10).

## Ce que fait le pipeline

```
archive/*.csv
    │
    ▼  download_jur.py         → data/raw/
    ▼  preprocess_jur_fr.py    → data/interim/qa_fr.parquet   (mojibake fix, clean, tokenize)
    ▼  preprocess_jur_ar.py    → data/interim/qa_ar.parquet   (normalisation alef/ya, tashkeel)
    ▼  stratify_jur.py         → data/interim/splits.json     (score de difficulté → tertiles E/M/H)
    ▼  build_hf_dataset.py     → data/jurida_processed/       (DatasetDict final)
    ▼  rapport_jurida.ipynb    → figures/ + rapport DS
```

## Installation

```bash
pip install -r requirements.txt
```

CPU-only, 8 Go RAM suffisants. Aucune dépendance GPU ni API payante.

## Utilisation

Régénération complète depuis `archive/` :

```bash
make all         # download → fr → ar → stratify → build
make test        # tests unitaires + intégration
make notebook    # exécute rapport_jurida.ipynb
make clean       # supprime data/raw, data/interim, data/jurida_processed, figures/*.png
```

Chargement du dataset côté downstream :

```python
from datasets import load_from_disk

ds = load_from_disk("data/jurida_processed")
train_fr_easy = ds["fr_easy_train"]
# champs : qa_id, lang, question, answer, context, doc_id, doc_type,
#          doc_date, long_title, difficulty, difficulty_score,
#          q_tokens, a_tokens, c_tokens, truncated, split
```

## Structure du dépôt

```
.
├── archive/                    # données brutes Jurida (qa.csv, train_with_file_content.csv)
├── config.yaml                 # feature flags, poids de stratification, seed
├── dataset/                    # scripts du pipeline (download, preprocess FR/AR, stratify, build)
├── data/
│   ├── raw/                    # copies vérifiées
│   ├── interim/                # parquet intermédiaires + splits.json
│   └── jurida_processed/       # DatasetDict final (livrable)
├── tests/                      # tests unitaires + fixtures
├── notebooks/rapport_jurida.ipynb
├── figures/                    # produit par le notebook
├── report/                     # rapport DS 3 pages
└── docs/superpowers/specs/     # spécification technique complète
```

## Configuration

Tous les paramètres (seed, seuils, flags FR/AR, poids de stratification) sont centralisés dans [`config.yaml`](config.yaml). Aucune valeur n'est codée en dur dans les scripts.

## Pour aller plus loin

La spécification technique détaillée (architecture, heuristiques de stratification, schéma complet, stratégie de tests, risques) se trouve dans [`docs/superpowers/specs/2026-04-20-jurida-d1-d2-design.md`](docs/superpowers/specs/2026-04-20-jurida-d1-d2-design.md).
