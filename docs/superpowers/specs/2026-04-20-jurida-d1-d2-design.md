# Spécification Technique — D1 + D2 Jurida

## Pipeline de Données et Stratification pour MMedAgent-Lite (Pôle Juridique)

**Date :** 2026-04-20
**Équipe :** Groupe D1 Juridique (3 personnes : Mohammed Azan, Hashem Emran, Imad)
**Encadrant :** Pr. Abdellatif EL AFIA
**Cadre parent :** MMedAgent-Lite — transposition au domaine légal (cohorte ENS Martil 2026)
**Dataset :** Jurida — Moroccan Legal QA Dataset (FR/AR) — Kaggle `ouaillaamiri02/jurida-dataset`
**Livrables :** scripts Python modulaires + `rapport_jurida.ipynb` + `data/jurida_processed/` (HuggingFace DatasetDict) + rapport DS de 3 pages

---

## 1. Contexte et Objectif

### 1.1 Positionnement dans MMedAgent-Lite (Pôle Juridique)

Le projet MMedAgent-Lite original couvre le domaine médical (VQA-RAD + PathVQA). Le pôle juridique transpose la même architecture (GP1 triage → S1/S2 spécialistes → GP2 synthèse C-MARL + RAG + FL) au droit marocain. Notre module **D1 + D2 Jurida** produit la fondation données sur laquelle reposent toutes les équipes downstream (D3 encodeur, D4 GP1, D5 agents Avocat/Juge, D6 GRPO, D7 GP2, D8 RAG Juridique, D9 FL, D10 intégration).

### 1.2 Périmètre retenu : D1 + D2 étendu

Comme le pôle juridique dispose d'**un seul dataset** (Jurida) bilingue, notre équipe couvre le périmètre combiné :
- **D1 médical (analogue FR)** : téléchargement + prétraitement + statistiques sur la partie française
- **D2 médical (analogue AR)** : téléchargement + prétraitement + statistiques + stratification Easy/Medium/Hard sur la partie arabe
- **Splits E/M/H unifiés** pour les deux langues (livrables communs)

### 1.3 Dataset Jurida — caractéristiques observées

| Propriété                     | Valeur                                              |
|-------------------------------|-----------------------------------------------------|
| Paires Q&A                    | 24 973                                              |
| Documents sources uniques     | 1 267 (référencés via `file_name`)                  |
| Documents avec métadonnées    | 5 000 (`train_with_file_content.csv`)               |
| Langues                       | FR 52,2 % (13 039) / AR 47,8 % (11 934)             |
| Types de documents            | Arrêté (1973), Dahir (1810), Décret (1096), Décision (90), Loi, Instruction, Ordre, Code, Convention... |
| Période couverte              | 1913-07-13 → 2024-01-31 (111 ans)                   |
| Longueur médiane question     | 109 caractères                                      |
| Longueur médiane réponse      | 1 028 caractères                                    |
| Longueur médiane contexte     | 18 487 caractères                                   |
| Q&A moyennes par document     | 19,7 (max : 317)                                    |
| **Problème encodage**         | `train_with_file_content.csv` : mojibake cp1252 (`D�cret` au lieu de `Décret`) |

---

## 2. Architecture et Flux de Données

### 2.1 Vue d'ensemble

```
archive/*.csv
    │
    ▼
[1] download_jur.py       → data/raw/qa.csv, documents.csv (vérif MD5)
    │
    ▼
[2] preprocess_jur_fr.py  → data/interim/qa_fr.parquet + docs_fr.parquet
    │         (mojibake→UTF-8, clean HTML résiduel, tokenize, dédup)
    │
    ▼
[2'] preprocess_jur_ar.py → data/interim/qa_ar.parquet + docs_ar.parquet
    │         (normalisation alef/ya, tashkeel optionnel, tokenize)
    │
    ▼
[3] stratify_jur.py       → data/interim/splits.json + difficulty_score par qa_id
    │         (heuristiques B, feature flag hybride C activable)
    │
    ▼
[4] build_hf_dataset.py   → data/jurida_processed/ (DatasetDict 6 splits)
    │         {fr_easy, fr_medium, fr_hard, ar_easy, ar_medium, ar_hard}
    │
    ▼
[5] rapport_jurida.ipynb  → figures/*.png + rapport_ds.pdf (3 pages)
          (importe depuis les scripts, ne duplique aucune logique)
```

Orchestration via `run_all.py` (chaînage + vérification des sorties) ou `Makefile`.

### 2.2 Feature flags (`config.yaml`)

```yaml
USE_HYBRID_STRATIFICATION: false   # active le passage modèle C (après D3)
REMOVE_TASHKEEL: true               # AR : suppression diacritiques
DEDUP_QUESTIONS: true               # élimine questions strictement identiques
MAX_CONTEXT_CHARS: 30000            # troncature sécurité
MIN_QUESTION_CHARS: 10              # rejet questions triviales
LANG_DETECT_THRESHOLD: 0.30         # seuil ratio chars arabes pour classer AR
RANDOM_SEED: 42                     # reproductibilité splits
```

### 2.3 Contraintes techniques

- **Matériel :** CPU-only, 8 Go RAM (cohérent MMedAgent-Lite)
- **Lecture par chunks :** `pandas.read_csv(chunksize=1000)` sur `qa.csv` (7,3 M lignes brutes multi-lignes)
- **Budget RAM :** ≤ 2 Go au pic (qa_fr + qa_ar en mémoire simultanément)
- **Durée cible pipeline complet :** ≤ 10 min (hors stratification hybride C)

---

## 3. Préprocessing Français — `preprocess_jur_fr.py`

### 3.1 Étapes

1. **Chargement** : `pd.read_csv("archive/qa.csv", chunksize=1000, encoding="utf-8")`
2. **Détection langue** par ratio de caractères arabes `U+0600–U+06FF` : ratio < 0.30 → FR
3. **Correction mojibake** sur `Context` (et `file_content` via jointure avec `train_with_file_content.csv`) :
   - Tentative `bytes.decode("cp1252").encode("latin-1").decode("utf-8")` (chaîne standard pour récupérer UTF-8 encodé en cp1252)
   - Fallback : laisser tel quel si échec, logger l'ID pour investigation
   - Vérification : présence de `é è à ç ù` dans ≥ 80 % des documents après correction
4. **Nettoyage HTML résiduel** : suppression balises `<...>`, entités HTML (`&amp;`, `&nbsp;`), espaces multiples, caractères de contrôle
5. **Normalisation Unicode** : NFC (forme composée)
6. **Tokenisation** : `HuggingFace Qwen2Tokenizer` (cohérent avec le modèle downstream Qwen2.5-VL-3B) pour compter les tokens (`q_tokens`, `a_tokens`, `c_tokens`)
7. **Filtrage** :
   - Rejet si `question` < `MIN_QUESTION_CHARS` après clean
   - Rejet si `answer` vide ou placeholder (`"N/A"`, `"-"`, null)
   - Rejet si `context` vide
8. **Déduplication** (si `DEDUP_QUESTIONS=true`) : hash `(question, file_name)` normalisé (lowercase, whitespace collapsé)
9. **Troncature sécurité** : `context[:MAX_CONTEXT_CHARS]` avec drapeau `truncated=True`
10. **Sortie** :
    - `data/interim/qa_fr.parquet` — colonnes `qa_id`, `lang="fr"`, `question`, `answer`, `context`, `doc_id`, `doc_type`, `doc_date`, `q_tokens`, `a_tokens`, `c_tokens`, `truncated`
    - `data/interim/docs_fr.parquet` — colonnes `doc_id`, `long_title`, `date`, `doc_type`, `file_content`

### 3.2 Identifiants

- `qa_id` : `f"fr_{index:05d}"` (5 chiffres, indexé global sur la partition FR)
- `doc_id` : valeur de `file_name` sans extension `.html`

---

## 4. Préprocessing Arabe — `preprocess_jur_ar.py`

### 4.1 Étapes spécifiques AR

1. **Filtrage langue** : ratio chars arabes ≥ 0.30
2. **Normalisation des lettres** (standard pour l'IR arabe) :
   - Alef : `أ إ آ` → `ا`
   - Ya / Alef maqsura : `ى` → `ي`
   - Ta marbuta : `ة` → `ه` (optionnel, drapeau `NORMALIZE_TA_MARBUTA`, défaut `false` pour préserver la précision juridique)
3. **Suppression tashkeel** (diacritiques) si `REMOVE_TASHKEEL=true` :
   - Plage `U+064B–U+0652` + `U+0670` + tatweel `U+0640`
4. **Normalisation Unicode** : NFC puis NFKC pour présentation arabe (letter forms)
5. **Nettoyage** : balises HTML résiduelles, caractères de contrôle, espaces multiples
6. **Tokenisation** : même `Qwen2Tokenizer` (support multilingue)
7. **Filtrage qualité, déduplication, troncature** : identiques à FR (MIN_QUESTION_CHARS, DEDUP_QUESTIONS, MAX_CONTEXT_CHARS)
8. **Sortie** :
    - `data/interim/qa_ar.parquet`
    - `data/interim/docs_ar.parquet`

### 4.2 Identifiants

- `qa_id` : `f"ar_{index:05d}"`

### 4.3 Note sur les métadonnées AR

Les `train_with_file_content.csv` n'ont pas de version arabe des `long_title`/`doc_type`. Deux options :
- Conserver les métadonnées FR (français) même pour les Q&A arabes (les deux pointent vers le même `file_name`)
- Ne pas inclure `long_title` pour la partition AR

**Choix retenu** : conserver les métadonnées FR (jointure sur `file_name`). Les types de documents (Dahir, Décret, etc.) sont des termes techniques utilisables pour la stratification sans poser de problème de langue.

---

## 5. Stratification Easy/Medium/Hard — `stratify_jur.py`

### 5.1 Heuristique retenue (Option B)

On calcule un **score de difficulté** `d ∈ [0, 1]` par Q&A, puis on partitionne par tertiles (33 % / 33 % / 34 %) **indépendamment par langue**.

### 5.2 Features (normalisées [0, 1] sur la partition de la langue)

| Feature                         | Formule                                               | Contribution |
|--------------------------------|-------------------------------------------------------|--------------|
| `f1_ans_len`                    | `min(a_tokens / 500, 1)`                              | + (long = hard) |
| `f2_ctx_len`                    | `min(c_tokens / 8000, 1)`                             | + (long = hard) |
| `f3_lexical_overlap`            | 1 − Jaccard(tokens(answer), tokens(context))          | + (faible overlap = hard) |
| `f4_has_negation`               | présence `ne ... pas`, `sans`, `لا`, `غير`, `ليس`     | + (pondéré 0,5) |
| `f5_has_quantifier`             | présence nombres, `tous`, `aucun`, `كل`, `جميع`, `بعض`| + (pondéré 0,3) |
| `f6_cross_refs`                 | nombre de `article N`, `المادة N`, `chapitre N` (normalisé /10) | + (inférence = hard) |
| `f7_question_type`              | classification question : factuelle (0,1), descriptive (0,5), raisonnement (0,9) | + |

### 5.3 Score composite

```
d = 0.20·f1 + 0.15·f2 + 0.25·f3 + 0.10·f4 + 0.05·f5 + 0.10·f6 + 0.15·f7
```

Poids calibrables via `config.yaml` (`stratification_weights`).

### 5.4 Seuils par tertile

```python
easy_threshold   = np.percentile(d, 33.33)
medium_threshold = np.percentile(d, 66.67)
difficulty = np.where(d <= easy_threshold, "easy",
             np.where(d <= medium_threshold, "medium", "hard"))
```

### 5.5 Classification du type de question (f7)

Heuristique lexicale :
- **Factuelle** : débute par `quand`, `où`, `qui`, `combien`, `متى`, `أين`, `من`, `كم`
- **Descriptive** : débute par `quel(le)(s)`, `ما`, `ماهي`, `اذكر`, `عدد`
- **Raisonnement** : contient `pourquoi`, `comment`, `expliquez`, `justifiez`, `لماذا`, `كيف`, `اشرح`, `برر`

### 5.6 Feature flag hybride C (désactivé par défaut)

Si `USE_HYBRID_STRATIFICATION=true` et checkpoint Qwen2.5-VL-3B disponible (produit par D3) :
- Appliquer K=1 passage du modèle sur les **cas ambigus uniquement** (d ∈ [0,30 ; 0,40] ∪ [0,63 ; 0,70])
- Score binaire correct/incorrect ajusté : `d_final = 0,7·d_heur + 0,3·(1 − acc_model)`

### 5.7 Sortie `splits.json`

```json
{
  "version": "1.0",
  "date": "2026-04-20",
  "seed": 42,
  "config_hash": "sha256:abc123...",
  "fr": {
    "easy":   ["fr_00001", "fr_00007", ...],
    "medium": ["fr_00002", ...],
    "hard":   ["fr_00003", ...],
    "thresholds": {"easy": 0.28, "medium": 0.54}
  },
  "ar": { ... }
}
```

### 5.8 Train/Test split

Split supplémentaire **80/20 train/test** dans chaque bucket E/M/H, stratifié par `doc_type` et `doc_date_decade` pour éviter les fuites temporelles.

---

## 6. Construction du HuggingFace DatasetDict — `build_hf_dataset.py`

### 6.1 Schéma de chaque ligne

```python
{
  "qa_id":            "fr_00012",
  "lang":             "fr",                # {"fr", "ar"}
  "question":         str,
  "answer":           str,
  "context":          str,
  "doc_id":           "11944",
  "doc_type":         "Dahir",
  "doc_date":         "1958-02-24",
  "long_title":       str,
  "difficulty":       "easy",              # {"easy", "medium", "hard"}
  "difficulty_score": 0.23,
  "q_tokens":         42,
  "a_tokens":         180,
  "c_tokens":         3210,
  "truncated":        False,
  "split":            "train",             # {"train", "test"}
}
```

### 6.2 Structure `DatasetDict`

```
data/jurida_processed/
├── dataset_dict.json
├── fr_easy_train/   fr_easy_test/
├── fr_medium_train/ fr_medium_test/
├── fr_hard_train/   fr_hard_test/
├── ar_easy_train/   ar_easy_test/
├── ar_medium_train/ ar_medium_test/
└── ar_hard_train/   ar_hard_test/
```

Chargement downstream :
```python
from datasets import load_from_disk
ds = load_from_disk("data/jurida_processed")
gp1_train = ds["fr_easy_train"]  # exemple
```

---

## 7. Rapport Statistiques — `rapport_jurida.ipynb`

Importe les fonctions de `jur_utils.py` — **aucune logique dupliquée**. Génère dans `figures/` :

1. **fig1_lang_distribution.png** — camembert FR/AR + par doc_type
2. **fig2_doc_types.png** — barplot horizontal des types de documents
3. **fig3_temporal_distribution.png** — histogramme dates par décennie, par doc_type
4. **fig4_length_distributions.png** — violons `q_tokens`, `a_tokens`, `c_tokens` par langue
5. **fig5_difficulty_distribution.png** — histogramme `difficulty_score` + seuils E/M/H
6. **fig6_difficulty_by_doctype.png** — heatmap `doc_type × difficulty`
7. **fig7_stratification_coherence.png** — corrélation chaque feature `f1–f7` vs score global
8. **fig8_top_docs_qa_count.png** — top-20 documents par nombre de Q&A
9. **Exemples qualitatifs** (tableaux Markdown) : 3 exemples Easy, 3 Medium, 3 Hard par langue

### 7.1 Rapport texte (3 pages)

- **Page 1 :** Contexte, source du dataset, positionnement dans MMedAgent-Lite, choix du périmètre étendu D1+D2
- **Page 2 :** Pipeline de prétraitement (FR + AR), tableaux statistiques clés, justification des heuristiques de stratification
- **Page 3 :** Résultats, cas limites rencontrés (mojibake, caractères arabes cassés, doublons), recommandations pour les équipes downstream (formats, accès, pièges)

---

## 8. Structure du Dépôt (cohérente `mmedagent-lite/`)

```
PL-Chatbot/
├── README.md
├── requirements.txt
├── config.yaml
├── Makefile
├── archive/                           # données brutes (déjà présentes)
│   ├── qa.csv
│   ├── train_with_file_content.csv
│   └── data/data/*.html.csv
├── data/
│   ├── raw/                           # copies vérifiées MD5
│   ├── interim/                       # parquet intermédiaires + splits.json
│   └── jurida_processed/              # DatasetDict final
├── dataset/
│   ├── __init__.py
│   ├── download_jur.py                # (D1-P1) Mohammed Azan
│   ├── preprocess_jur_fr.py           # (D1-P1) Mohammed Azan
│   ├── preprocess_jur_ar.py           # (D1-P2) Hashem Emran
│   ├── stratify_jur.py                # (D1-P3) Imad
│   ├── build_hf_dataset.py            # (D1-P3) Imad
│   └── jur_utils.py                   # utilitaires partagés
├── tests/
│   ├── test_preprocess_fr.py
│   ├── test_preprocess_ar.py
│   ├── test_stratify.py
│   └── fixtures/                      # mini-corpus pour tests
├── figures/                           # produit par rapport_jurida.ipynb
├── notebooks/
│   └── rapport_jurida.ipynb           # (D1-P3) Imad, intègre tout
├── report/
│   └── rapport_ds_jurida.pdf          # 3 pages, produit depuis le notebook
└── docs/superpowers/specs/
    └── 2026-04-20-jurida-d1-d2-design.md   # (ce document)
```

---

## 9. Répartition à 3 personnes

| Personne     | Responsabilités                                                  | Livrables principaux                                         |
|--------------|------------------------------------------------------------------|--------------------------------------------------------------|
| **Mohammed Azan** | Téléchargement + pipeline FR + tests FR                     | `download_jur.py`, `preprocess_jur_fr.py`, `tests/test_preprocess_fr.py` |
| **Hashem Emran**  | Pipeline AR + normalisation arabe + tests AR                | `preprocess_jur_ar.py`, `tests/test_preprocess_ar.py`, section AR du rapport |
| **Imad**          | Stratification + DatasetDict + notebook + intégration       | `stratify_jur.py`, `build_hf_dataset.py`, `jur_utils.py`, `rapport_jurida.ipynb`, `Makefile` |

Intégration : revues croisées et merge sur branche `master` par pull requests.

---

## 10. Dépendances (`requirements.txt`)

```
pandas>=2.0
pyarrow>=15.0                # parquet I/O
datasets>=2.18               # HuggingFace DatasetDict
transformers>=4.40           # Qwen2Tokenizer
ftfy>=6.2                    # backup mojibake fixer
pyyaml>=6.0                  # config.yaml
matplotlib>=3.8              # figures
seaborn>=0.13                # figures
jupyter>=1.0                 # notebook
pytest>=8.0                  # tests
tqdm>=4.66                   # progress bars
```

Pas de GPU, pas d'API payante, pas de téléchargement de modèle lourd (tokenizer uniquement).

---

## 11. Stratégie de Tests

### 11.1 Tests unitaires (`tests/`)

- `test_preprocess_fr.py` : mojibake fixer sur 10 cas type, HTML cleaner, filtrage, déduplication, tokenisation
- `test_preprocess_ar.py` : normalisation alef/ya, suppression tashkeel, détection langue borderline
- `test_stratify.py` : calcul de chaque feature `f1–f7`, score composite, reproductibilité avec seed fixe, cohérence des tertiles (33/33/34)

### 11.2 Tests d'intégration

Script `tests/test_integration.py` qui exécute le pipeline complet sur un mini-corpus (100 Q&A FR + 100 Q&A AR en fixture) et vérifie :
- Structure finale du `DatasetDict`
- Somme des splits = total nettoyé
- Aucun `qa_id` dupliqué cross-split

### 11.3 Critères d'acceptation

- ≥ 90 % des Q&A FR passent le filtre (après mojibake fix)
- ≥ 90 % des Q&A AR passent le filtre
- Ratio final E/M/H : 33 ± 2 % chacun par langue
- Pipeline complet : < 10 min sur CPU 8 Go
- Zéro exception non gérée sur les 24 973 Q&A

---

## 12. Risques et Mitigations

| Risque                                            | Probabilité | Impact  | Mitigation                                                      |
|---------------------------------------------------|-------------|---------|-----------------------------------------------------------------|
| Mojibake non récupérable sur certains documents   | Moyenne     | Moyen   | Double passe : `ftfy.fix_text()` en fallback, log IDs, rapport  |
| Détection de langue incorrecte sur Q&A mixtes     | Faible      | Faible  | Logger les cas ambigus (ratio 0,25–0,40), revue manuelle sur 50 |
| Stratification déséquilibrée par doc_type         | Moyenne     | Faible  | Stratification par tertile global ; vérification heatmap §7     |
| Mémoire insuffisante sur chargement qa.csv        | Faible      | Haut    | `chunksize=1000`, traitement streaming, libération explicite    |
| Désalignement avec MMedAgent-Lite format attendu  | Moyenne     | Moyen   | Réplique exacte structure `mmedagent-lite/dataset/` (§8)        |
| Modèle Qwen absent → option C inapplicable        | Haute       | Faible  | Feature flag désactivé par défaut ; B suffit comme baseline     |

---

## 13. Critères de Succès

1. **Reproductibilité** : `make all` régénère l'intégralité des livrables à partir de `archive/` avec seed fixe.
2. **Intégration** : une équipe downstream charge le dataset via `load_from_disk("data/jurida_processed")` sans code spécifique.
3. **Qualité** : ≥ 90 % de taux de rétention après nettoyage par langue.
4. **Interprétabilité** : chaque Q&A annotée avec `difficulty_score` et composantes `f1–f7` (audit possible).
5. **Documentation** : notebook de rapport + spec + README suffisent pour qu'un nouvel étudiant reprenne le module en 2 h.

---

## 14. Hors-périmètre (explicitement exclu)

- Entraînement de modèles (domaine de D3–D7)
- RAG Juridique et index vectoriel (domaine de D8)
- Federated Learning (domaine de D9)
- Traduction automatique FR ↔ AR (non requis — les deux langues traitées indépendamment)
- OCR ou traitement d'images (les données sont déjà textuelles)
- Annotation manuelle de la difficulté (heuristiques automatiques suffisent pour un baseline défendable)
