# NLP Disaster Tweet Classification

> **Course**: CSC-4260 Advanced Data Science — Spring 2025  
> **Team**: Thomas D. Robertson II, Sharon Colson, Caleb Smith, Tania Perdomo-Flores  
> **Original Repository**: [GitHub](https://github.com/CSC-4260-Advanced-Data-Science-Project/NLP_Disaster_Tweets)  
> **Data Source**: [Kaggle — NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started/data)

A systematic machine learning pipeline for classifying tweets as disaster-related or non-disaster. This project investigates how preprocessing strategy choices affect classification performance across 6 machine learning algorithms and 30 dataset variants — totaling over 180 experiments — with hyperparameter tuning, transformer model evaluation, and comprehensive visualization.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Research Question](#research-question)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Preprocessing and Data Variants](#preprocessing-and-data-variants)
- [Modeling Approach](#modeling-approach)
- [Feature Engineering](#feature-engineering)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [HPC Execution](#hpc-execution)
- [Running Locally](#running-locally)
- [Evaluation Metrics](#evaluation-metrics)
- [Results Summary](#results-summary)
- [Visualizations](#visualizations)
- [Team Contributions](#team-contributions)
- [Future Work](#future-work)
- [Authors and Acknowledgments](#authors-and-acknowledgments)

---

## Project Overview

When a disaster occurs, social media becomes one of the fastest information channels available. Twitter in particular sees a surge of disaster-related content — but the same language ("fire," "flood," "explosion") appears in non-disaster contexts constantly. Disaster monitoring agencies, news organizations, and emergency responders need reliable automated systems to separate genuine disaster signals from noise in real time.

This project builds a complete NLP classification pipeline to address that challenge. We train and evaluate models to predict whether a given tweet is reporting a real disaster (`target=1`) or not (`target=0`), using data from the [Kaggle NLP Getting Started competition](https://www.kaggle.com/competitions/nlp-getting-started/data).

Beyond basic classification, the project treats preprocessing strategy as a primary research variable. Rather than selecting a single cleaning approach and testing models against it, we constructed 30 distinct dataset variants across three preprocessing baselines and 10 processing strategies — then evaluated all 6 models against all 30 variants before and after hyperparameter tuning. This systematic structure allows us to isolate the effect of preprocessing decisions on model performance, not just model architecture choices.

---

## Research Question

> **How do different text preprocessing strategies, combined with different machine learning algorithms and hyperparameter configurations, affect disaster tweet classification performance?**

We operationalized this question by:
1. Creating 30 preprocessing variants from a single raw dataset
2. Establishing baseline performance for 6 classifiers across all 30 variants (180+ model-dataset combinations)
3. Hyperparameter-tuning each model on each dataset variant using GridSearchCV
4. Comparing pre- and post-tuning performance across the full matrix
5. Evaluating transformer models (BERT, BERTweet) as a high-capability benchmark

The central finding: **minimal preprocessing (lowercase-only) consistently outperformed aggressive cleaning strategies** across most model types — a counterintuitive result that demonstrates why preprocessing assumptions require empirical validation.

---

## Dataset

| Field | Value |
|:------|:------|
| **Source** | [Kaggle NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started/data) (originally from Figure-Eight's *Data for Everyone* dataset) |
| **Training set size** | 7,613 tweets |
| **Test set size** | 3,263 tweets (unlabeled) |
| **Positive class (disaster)** | ~43% of training set |
| **Negative class (non-disaster)** | ~57% of training set |

### Features

| Column | Type | Description |
|:-------|:-----|:------------|
| `id` | Integer | Unique tweet identifier |
| `keyword` | String (nullable) | A keyword from the tweet (e.g., "explosion", "wildfire") |
| `location` | String (nullable) | User-reported location (unreliable; often blank or fictional) |
| `text` | String | Raw tweet content |
| `target` | Integer (0 or 1) | Label: 1 = disaster, 0 = non-disaster |

### Notes on the Data

- The `keyword` field is present for roughly 99% of tweets and can be prepended to the tweet text as a feature enrichment strategy (this is the basis for the `prepended` preprocessing baseline).
- The `location` field was excluded from modeling — it is too sparse and unreliable to be a useful feature.
- Class distribution is near-balanced (~43/57), so accuracy alone is a reasonable indicator, though F1 macro is used as the primary metric to avoid any subtle class bias.

---

## Project Structure

```
NLP_Disaster_Tweets/
│
├── Data/                              # Raw dataset files from Kaggle
│   ├── train.csv                      # 7,613 labeled tweets
│   ├── test.csv                       # 3,263 unlabeled tweets (Kaggle test set)
│   └── sample_submission.csv          # Submission format reference
│
├── processed_data/                    # Initial preprocessing output (early exploration)
│
├── final_processed/                   # 30 finalized dataset variants used in all modeling
│   ├── kept_v1_basic_clean.csv        # Variants with hashtags/mentions retained
│   ├── kept_v2_no_emojis_mentions.csv
│   ├── ...                            # kept_v1 through kept_v10
│   ├── dropped_v1_basic_clean.csv     # Variants with hashtags/mentions removed
│   ├── ...                            # dropped_v1 through dropped_v10
│   ├── prepended_v1_basic_clean.csv   # Variants with keywords prepended to text
│   └── ...                            # prepended_v1 through prepended_v10
│
├── baseline_performance/              # Baseline (pre-tuning) model evaluation outputs
│   ├── ALL_results.csv                # Merged performance metrics across all runs
│   └── [dataset]_[model]/             # Per-run subdirectories with visualizations
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       └── learning_curve.png
│
├── final_ht_performance_metrics/      # Post-hypertuning best model outputs (30 files)
│   └── [dataset]_best_model.txt       # Best params + metrics for each dataset variant
│
├── final_complete_ht_slurm_files/     # SLURM job output logs from HPC execution
│
├── Deliverables/                      # Academic deliverables
│   ├── CSC-4260-NLP Tweets.pdf
│   ├── NLP Tweets Final Report.pdf
│   ├── NLP Tweets Initial Report.pdf
│   ├── Group 2 NLP Tweets Final Presentation.pptx
│   ├── ProjectPoster.pdf
│   └── Project Plan.pdf
│
├── Images/                            # Summary visualization outputs
│   ├── baseline_vs_tuned_f1_by_dataset.png
│   ├── baseline_vs_tuned_top5_models.png
│   ├── F1_Comparison.png
│   └── f1_score_comparison_chart.png
│
├── Sandbox/                           # Exploratory notebooks (not part of main pipeline)
│   ├── NLP-Preprocessing-and-Model-Analysis_main.ipynb
│   └── eda_approach_2.ipynb
│
├── NLP_DS_Pipeline.ipynb              # PRIMARY NOTEBOOK: Full implementation pipeline — data cleaning,
│                                      #   EDA, preprocessing variants, all modeling, hypertuning,
│                                      #   BERT integration, performance aggregation, visualizations,
│                                      #   and research paper writeup (10 sections, ~4,900 lines)
├── bert_model.ipynb                   # BERT fine-tuning notebook (Caleb Smith)
├── bert_model.py                      # BERT pipeline script for HPC execution
├── bertweet_model.ipynb               # BERTweet fine-tuning notebook
├── bertweet_model.py                  # BERTweet pipeline script for HPC execution
│
├── pipeline_modules.py                # Baseline model evaluation functions
├── pipeline_modules_ht.py             # Hyperparameter tuning evaluation functions (core engine)
│
├── run_pipeline.py                    # Entry point: baseline model runs
├── run_pipeline_ht.py                 # Entry point: hypertuned model runs
├── run_bert_cv.py                     # Entry point: BERT cross-validation runs
├── run_bertweet_cv.py                 # Entry point: BERTweet cross-validation runs
│
├── run_pipeline.sh                    # SLURM job script: baseline execution
├── run_pipeline_ht.sh                 # SLURM job script: hypertuned execution
├── run_bert_cv.sh                     # SLURM job script: BERT execution
├── run_bertweet_cv.sh                 # SLURM job script: BERTweet execution
│
├── merge_results.py                   # Merges per-dataset results into ALL_results.csv
├── dataset_list.txt                   # List of 30 dataset names for SLURM array indexing
│
└── README.md                          # This file
```

---

## Primary Notebook: NLP_DS_Pipeline.ipynb

`NLP_DS_Pipeline.ipynb` is the central implementation artifact for this project. It drives the full pipeline from raw data to final results and contains the research paper writeup inline as Section 10. All major analysis, modeling, and visualization work lives here.

The notebook is structured into 10 sections:

| Section | Contents |
|:--------|:---------|
| **1. Import Data** | Load raw training and test CSVs with encoding detection |
| **2. Initial Data Cleaning** | Non-ASCII character detection and normalization; replacement of corrupted byte sequences; dropping unnecessary columns; handling missing values; removing conflicting labels; removing duplicates; saving cleaned datasets |
| **3. Exploratory Data Analysis** | Dataset overview (shape, types, unique values, text length stats); target class distribution and balance analysis; tweet length histograms and boxplots per class; @mention frequency analysis by class; URL distribution analysis; emoji and ASCII emoticon detection; special character frequency visualization; sample tweet inspection including edge cases |
| **4. Text Preprocessing Variants** | Definition and application of all 10 text cleaning strategies; generation and saving of all 30 final processed dataset CSVs (`final_processed/`) |
| **5. Word Clouds & Frequency Comparison** | Wordcloud generation per class (disaster vs. non-disaster); visual comparison of top tokens across classes |
| **6. Baseline Modeling Pipeline** | Load all 30 preprocessed datasets; define TF-IDF + classifier pipelines for all 6 models; 80/20 train/test split; cross-validation evaluation; save per-model performance metrics and visualizations |
| **7. Hyperparameter Tuning Pipeline** | Define GridSearchCV search spaces per model; run hypertuning on all models; evaluate best-tuned models on test set; save best model summaries and tuned metrics |
| **8. Pre-Trained Models: BERT and BERTweet** | Integration of Caleb Smith's BERT and BERTweet cross-validation results; side-by-side comparison with classical model performance |
| **9. Performance Aggregation and Visualization** | Merge all result files; top model per algorithm (baseline vs. tuned); F1 improvement per model; top model per dataset; greatest F1 improvement per model; best hypertuned model comparison charts |
| **10. Research Paper** | Inline research paper writeup synthesizing methodology, results, and conclusions |

---

## Preprocessing and Data Variants

To investigate how preprocessing choices affect model performance, we created **30 dataset variants** from the original training set. These are organized as three preprocessing baselines, each applied with ten different text cleaning strategies.

### Three Preprocessing Baselines

| Baseline | Prefix | Description |
|:---------|:-------|:------------|
| `kept` | `kept_` | Hashtags and @mentions are retained and processed normally as part of the token stream |
| `dropped` | `dropped_` | Hashtags, @mentions, and URLs are removed entirely before text processing |
| `prepended` | `prepended_` | The tweet's `keyword` field is prepended to the tweet text before processing |

The rationale: hashtags and mentions carry signal in disaster tweets (e.g., `#wildfire`, `@FEMA`) — retaining them may help, but they may also add noise. The `prepended` strategy tests whether the keyword metadata adds discriminative power when fused directly into the text.

### Ten Text Processing Strategies (v1–v10)

Each baseline was processed using all 10 strategies below, producing 30 total variants:

| Version | Name | Cleaning Applied |
|:--------|:-----|:----------------|
| `v1` | `basic_clean` | Lowercasing, stopword removal, punctuation stripping |
| `v2` | `no_emojis_mentions` | v1 + emoji removal + @mention removal |
| `v3` | `lemmatized` | v1 + WordNet lemmatization |
| `v4` | `stemmed` | v1 + Porter stemming |
| `v5` | `lemma_stem` | v1 + both lemmatization and stemming |
| `v6` | `custom_stopwords` | v1 + custom extended stopword list |
| `v7` | `lowercase_words_only` | **Lowercase alphabetic tokens only; no stemming, lemmatization, or stopword removal** |
| `v8` | `keep_hashtags` | v1 + hashtag text preserved (# stripped, word retained) |
| `v9` | `minimal_processing` | Lowercase only, punctuation stripped, no stopword removal |
| `v10` | `lemma_stem_custom` | v1 + lemmatization + stemming + custom stopwords |

### Key Insight

`kept_v7_lowercase_words_only` — the simplest strategy — produced the highest F1 scores after hyperparameter tuning. This suggests that aggressive cleaning discards contextual tokens that carry disaster-signal information. Stemming and lemmatization, in particular, may collapse important distinctions between word forms in this domain.

---

## Modeling Approach

### Baseline Classifiers

All six classifiers were evaluated using a consistent scikit-learn `Pipeline` combining TF-IDF vectorization with each model. Each was run on all 30 dataset variants before hyperparameter tuning.

| Model | Why Included |
|:------|:-------------|
| **Multinomial Naive Bayes (MNB)** | Strong baseline for text classification; fast; well-suited to TF-IDF feature distributions |
| **Logistic Regression (LR)** | Linear model with strong performance on text tasks; interpretable; good regularization support |
| **Passive Aggressive Classifier (PA)** | Online learning algorithm well-suited for text; handles large feature spaces efficiently |
| **Support Vector Machine (SVM)** | Strong generalization on high-dimensional feature spaces; linear kernel is common for text |
| **K-Nearest Neighbors (KNN)** | Instance-based learner; included for contrast with generative/discriminative models |
| **Multi-Layer Perceptron (MLP)** | Feed-forward neural network; captures non-linear patterns that linear models miss |

### Transformer Models

Two pre-trained transformer models were evaluated as high-performance benchmarks:

| Model | Source | Notes |
|:------|:-------|:------|
| **BERT Base-Uncased** | `bert-base-uncased` (Google) | General-purpose transformer pre-trained on BooksCorpus + Wikipedia |
| **BERTweet Base** | `vinai/bertweet-base` (VinAI) | BERT variant pre-trained specifically on 850M English tweets — designed for social media text |

Both transformer models were fine-tuned using 5-fold cross-validation on select high-performing dataset variants. They achieved comparable or slightly better classification performance than the best classical models, but at significantly higher computational cost — requiring GPU allocation on the HPC cluster versus CPU-only for the classical models.

---

## Feature Engineering

All classical models use a **TF-IDF vectorizer** with the following configuration:

```python
TfidfVectorizer(
    stop_words='english',   # English stopword list applied at vectorization stage
    max_df=0.8,             # Ignore terms appearing in more than 80% of documents
    ngram_range=(1, 3)      # Unigrams, bigrams, and trigrams
)
```

**Why TF-IDF?** Term frequency–inverse document frequency balances term frequency against how common a term is across all documents. This down-weights generic high-frequency words and up-weights domain-specific signals — well-suited for differentiating disaster language from casual language.

**Why n-grams up to trigrams?** Multi-word phrases like "wildfire spreading," "search and rescue," or "building collapse" carry more signal than individual words. Trigrams capture these patterns that unigrams miss.

**Why `max_df=0.8`?** Terms appearing in more than 80% of documents are unlikely to discriminate between classes. Filtering them reduces noise in the feature space.

---

## Hyperparameter Tuning

Hyperparameter tuning was applied to all 6 baseline models across all 30 dataset variants using **GridSearchCV** with 5-fold stratified cross-validation.

### Configuration

| Parameter | Value |
|:----------|:------|
| Search method | `GridSearchCV` |
| Cross-validation | 5-fold |
| Scoring metric | `f1_macro` |
| Parallelization | `n_jobs=-1` (all available cores) |

### Parameter Grids

Each model was given a tailored search space:

**Multinomial Naive Bayes**
```python
{'clf__alpha': [0.1, 0.5, 1.0, 1.5, 2.0]}
```

**Logistic Regression**
```python
{
    'clf__C': [0.01, 0.1, 1.0, 10.0],
    'clf__penalty': ['l2'],
    'clf__solver': ['lbfgs'],
    'clf__max_iter': [300, 500, 1000]
}
```

**Passive Aggressive Classifier**
```python
{
    'clf__C': [0.01, 0.1, 1.0, 10.0],
    'clf__max_iter': [500, 1000, 2000],
    'clf__tol': [1e-4, 1e-3, 1e-2]
}
```

**Support Vector Machine**
```python
{
    'clf__C': [0.1, 1.0, 10.0],
    'clf__kernel': ['linear', 'rbf'],
    'clf__gamma': ['scale', 'auto']
}
```

**K-Nearest Neighbors**
```python
{
    'clf__n_neighbors': [3, 5, 7, 9],
    'clf__weights': ['uniform', 'distance'],
    'clf__metric': ['euclidean', 'manhattan']
}
```

**Multi-Layer Perceptron**
```python
{
    'clf__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'clf__activation': ['relu', 'tanh'],
    'clf__solver': ['adam', 'sgd'],
    'clf__alpha': [0.0001, 0.001, 0.01],
    'clf__learning_rate': ['constant', 'adaptive'],
    'clf__early_stopping': [True],
    'clf__n_iter_no_change': [5, 10],
    'clf__validation_fraction': [0.1, 0.2]
}
```

### Best Hyperparameters (Top Model)

For the best-performing configuration — Passive Aggressive Classifier on `kept_v7_lowercase_words_only`:

| Parameter | Best Value |
|:----------|:-----------|
| `clf__C` | `0.01` |
| `clf__max_iter` | `500` |
| `clf__tol` | `0.001` |

---

## HPC Execution

The full hypertuning pipeline was too computationally expensive to run locally across all 30 datasets. Execution was parallelized on a university HPC cluster using **SLURM array jobs**.

### Cluster Configuration

| Resource | Allocation |
|:---------|:-----------|
| CPU cores per job | 28 |
| RAM per job | 16 GB |
| GPU cores (BERT/BERTweet only) | 4 |
| Time limit per job | 4 hours |
| Job manager | SLURM |

### How the Array Job Works

The `dataset_list.txt` file contains all 30 dataset names, one per line. The SLURM array scripts read the dataset name at the array index corresponding to the job ID:

```bash
# From run_pipeline_ht.sh
DATASET=$(sed -n "${SLURM_ARRAY_TASK_ID}p" dataset_list.txt)
python run_pipeline_ht.py --dataset "$DATASET"
```

Submitting the job array:
```bash
sbatch --array=1-30 run_pipeline_ht.sh
```

Each job processes one dataset variant independently. All 30 jobs can run in parallel (subject to cluster queue availability).

### Output Files

Each completed job writes a best-model summary to:
```
final_ht_performance_metrics/[dataset_name]_best_model.txt
```

Example output file (`kept_v7_lowercase_words_only_best_model.txt`):
```
Best Model: PassiveAggressive
Dataset: kept_v7_lowercase_words_only
Best Params: {'clf__C': 0.01, 'clf__max_iter': 500, 'clf__tol': 0.001}
Accuracy:  0.7926
Precision: 0.7650
Recall:    0.7392
F1 Score:  0.7518
ROC AUC:   0.8588
```

After all jobs complete, run the merge script to consolidate results:
```bash
python merge_results.py
```

This produces `baseline_performance/ALL_results.csv` with all model-dataset metrics in a single file for analysis.

---

## Running Locally

### Prerequisites

- Python 3.8+
- pip

### Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk
pip install transformers datasets torch  # Only needed for BERT/BERTweet models
```

### Download NLTK Data

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

### Run the Main Notebook

`NLP_DS_Pipeline.ipynb` is the primary implementation artifact and the recommended starting point. Open it in Jupyter:

```bash
jupyter notebook NLP_DS_Pipeline.ipynb
```

The notebook runs the full pipeline end-to-end across all 10 sections:
- **Section 2**: Initial data cleaning (non-ASCII normalization, conflict/duplicate removal)
- **Section 3**: Full EDA (class distribution, tweet length analysis, mention/URL/emoji frequency)
- **Section 4**: Generates and saves all 30 preprocessed dataset variants to `final_processed/`
- **Section 5**: Wordcloud generation per class
- **Section 6**: Baseline modeling across all 30 datasets
- **Section 7**: Hyperparameter tuning with GridSearchCV
- **Section 9**: Performance aggregation and comparison visualizations
- **Section 10**: Inline research paper writeup

> **Note**: Sections 6 and 7 in the notebook demonstrate the pipeline on a subset. Full-scale execution across all 30 datasets is handled via the HPC scripts described below.

### Run Baseline Models (Script)

To run the baseline pipeline without HPC:
```bash
python run_pipeline.py
```

Output is written to `baseline_performance/`.

### Run Hyperparameter Tuning (Script, Single Dataset)

To run hypertuning for a specific dataset variant locally:
```bash
python run_pipeline_ht.py --dataset kept_v7_lowercase_words_only
```

Note: Running all 30 datasets locally without HPC will take significant time depending on your hardware. The full grid search for MLP in particular is computationally expensive.

### Run BERT Cross-Validation (Script)

```bash
python run_bert_cv.py --dataset kept_v7_lowercase_words_only
```

Requires a CUDA-capable GPU for practical runtimes. Falls back to CPU if no GPU is available (very slow).

---

## Evaluation Metrics

All models were evaluated on the following metrics. F1 macro was used as the **primary optimization target** for hyperparameter tuning.

| Metric | What It Measures | Why It Was Used |
|:-------|:----------------|:----------------|
| **Accuracy** | Fraction of all predictions that are correct | General performance indicator; reasonable for near-balanced classes |
| **Precision** | Of predicted disaster tweets, how many are actually disasters | Minimizes false alarms — important for alerting systems |
| **Recall** | Of actual disaster tweets, how many were correctly identified | Minimizes missed disasters — critical for emergency response |
| **F1 Score (macro)** | Harmonic mean of precision and recall, averaged across classes | Balances precision and recall; preferred for class-imbalanced evaluation |
| **ROC AUC** | Area under the ROC curve | Measures discriminative ability independent of classification threshold |

Performance comparisons were made **before and after hyperparameter tuning** to quantify the impact of the tuning step.

---

## Results Summary

### Baseline Model Performance (Pre-Tuning)

Results from 5-fold cross-validation on the three baseline datasets before any hyperparameter tuning:

| Metric | MNB | PA | LR | SVM | KNN | MLP |
|:-------|----:|---:|---:|----:|----:|----:|
| **Accuracy** | 0.7969 | 0.7846 | 0.7864 | 0.6110 | 0.6849 | 0.7951 |
| **Precision** | 0.8607 | 0.7609 | 0.7609 | 1.0000 | 0.8775 | 0.7903 |
| **Recall** | 0.6304 | 0.7288 | 0.5885 | 0.0964 | 0.3340 | 0.7136 |
| **F1 Score** | 0.7276 | 0.7443 | 0.7034 | 0.1757 | 0.4458 | 0.7499 |
| **ROC AUC** | 0.8481 | 0.8399 | 0.8465 | 0.8524 | 0.7383 | 0.8463 |

> **SVM baseline anomaly**: At baseline, SVM achieved precision=1.0 with recall=0.097 — it was classifying nearly everything as non-disaster. This was due to overfitting in the default configuration. Hyperparameter tuning corrected this substantially (see post-tuning results below).

### Post-Tuning Performance (Best Dataset per Model)

After GridSearchCV hyperparameter tuning, best results per model on each model's optimal dataset:

| Model | Best Dataset | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|:------|:------------|:--------:|:---------:|:------:|:--------:|:-------:|
| **Passive Aggressive** | `kept_v7_lowercase_words_only` | 79.26% | 76.50% | 73.92% | **75.18%** | 85.88% |
| Logistic Regression | `kept_v5_lemma_stem` | 79.36% | 78.16% | 71.39% | 74.62% | 86.18% |
| SVM | `kept_v9_minimal_processing` | 79.65% | 79.13% | 70.82% | 74.74% | 86.13% |
| MLP | `prepended_v10_lemma_stem_custom_stopwords` | 78.93% | 77.14% | 71.68% | 74.28% | 85.17% |
| MNB | `prepended_v4_stemmed` | 78.83% | 79.64% | 67.32% | 72.95% | 84.90% |
| KNN | `dropped_v1_basic_clean` | 76.76% | 79.97% | 60.57% | 68.84% | 81.49% |

### Top 5 Dataset-Model Combinations (Passive Aggressive, by Accuracy)

| Dataset | Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|:--------|:------|:--------:|:---------:|:------:|:--------:|:-------:|
| `kept_v7_lowercase_words_only` | Passive Aggressive | 0.793 | 0.765 | 0.734 | 0.752 | 0.859 |
| `kept_v2_no_emojis_mentions` | Passive Aggressive | 0.793 | 0.765 | 0.739 | 0.752 | 0.856 |
| `kept_v9_minimal_processing` | Passive Aggressive | 0.792 | 0.764 | 0.739 | 0.751 | 0.861 |
| `kept_v1_basic_clean` | Passive Aggressive | 0.791 | 0.762 | 0.737 | 0.750 | 0.860 |
| `kept_v6_custom_stopwords` | Passive Aggressive | 0.789 | 0.759 | 0.741 | 0.750 | 0.864 |

### Transformer Model Performance (BERT / BERTweet)

BERT and BERTweet were evaluated on the three top-performing dataset variants (`kept_v7`, `kept_v2`, `kept_v9`) using 5-fold cross-validation and a held-out test set:

| Model | Accuracy | Precision | Recall | F1 Score |
|:------|:--------:|:---------:|:------:|:--------:|
| **BERTweet-Base** (best) | **84.08%** | **82.89%** | **78.83%** | **80.81%** |
| BERT-Base-Uncased | 83.14% | 81.83% | 77.57% | 79.64% |
| PA Classifier (best classical) | 79.26% | 76.50% | 73.92% | 75.18% |

BERTweet-Base outperformed BERT-Base-Uncased across all metrics, as expected given its pre-training on tweet data. However, the performance advantage over the best classical model is modest: approximately +5% accuracy and +6% F1. Given the significant difference in computational cost, the PA Classifier remains the practical choice for deployment scenarios without GPU infrastructure.

### Key Findings

1. **Preprocessing had a larger impact than hypertuning**: The choice of preprocessing strategy affected performance more than hyperparameter optimization. Simple preprocessing approaches (lowercasing, minimal token filtering) consistently outperformed complex pipelines — likely because over-processing a short text like a tweet removes more signal than noise.

2. **Simple preprocessing wins**: `kept_v7_lowercase_words_only` — lowercase only, no stopword removal, no stemming, no lemmatization, keeping all tokens — outperformed variants with aggressive cleaning. The `v7` preprocessing specifically removes numerical digits, punctuation, special characters, emojis, and mentions while retaining all alphabetic word tokens.

3. **Hyperparameter tuning improved all models**: Post-tuning F1 scores were higher than baseline across every model-dataset combination. The PA Classifier benefited most; SVM recovered from a degenerate baseline (F1=0.18 → 0.75) after tuning fixed its default overfitting.

4. **PA Classifier is the efficiency winner**: Highest F1 among classical models, training orders of magnitude faster than SVM or MLP — the practical choice for real-time tweet classification.

5. **Transformer models are competitive but expensive**: BERTweet-Base achieved ~5–6% higher accuracy and F1 than the best classical model. The computational cost (GPU required, significantly longer training) may not justify this marginal gain for most deployment scenarios.

6. **`kept` baseline beat `dropped` and `prepended`**: Retaining hashtags and mentions in the token stream outperformed removing them — these social media features carry real discriminative signal, not just noise.

7. **Surprising emoji finding**: EDA revealed that disaster tweets contained *more* emojis than non-disaster tweets — counterintuitive, but interpretable: not all disaster tweeters are in panic, and emojis help convey urgency or emotion when reporting an event.

---

## Visualizations

Generated charts are stored in `Images/`. The following were produced during analysis:

| File | What It Shows |
|:-----|:-------------|
| `baseline_vs_tuned_f1_by_dataset.png` | F1 score improvement per dataset variant after hyperparameter tuning, organized by preprocessing strategy |
| `baseline_vs_tuned_top5_models.png` | Pre- vs. post-tuning comparison for the top 5 model-dataset combinations |
| `F1_Comparison.png` | Overall F1 comparison chart across all 30 datasets and 6 models |
| `f1_score_comparison_chart.png` | Bar chart comparison of F1 scores across model types |

Per-model-dataset visualizations (ROC curves, confusion matrices, learning curves) are stored in `baseline_performance/[dataset]_[model]/`.

To regenerate summary visualizations, run the relevant cells in `NLP_DS_Pipeline.ipynb`.

---

## Team Contributions

| Team Member | Contributions |
|:------------|:--------------|
| **Thomas D. Robertson II** | Primary notebook (`NLP_DS_Pipeline.ipynb`) — full 10-section implementation pipeline including data cleaning, EDA, preprocessing variant generation, all baseline and hypertuned modeling, visualizations, and inline research paper; hyperparameter tuning module (`pipeline_modules_ht.py`) and execution scripts (`run_pipeline_ht.py`, `run_pipeline_ht.sh`); dataset integration and results aggregation; wordcloud generation and F1 comparison visualization suite; co-contributor to academic report and presentation materials |
| **Sharon Colson** | Data Cleaning and Pre-processing methods for 30 different dataset variants; Baseline model execution pipeline (`pipeline_modules.py`, `run_pipeline.py`); HPC execution management and SLURM job scripts for baseline runs (`run_pipeline.sh`); results merge script (`merge_results.py`); co-author of BERT cross-validation runner (`run_bert_cv.py`) |
| **Caleb Smith** | BERT and BERTweet model implementation (`bert_model.py`, `bertweet_model.py`, `bert_model.ipynb`, `bertweet_model.ipynb`); 5-fold cross-validation for transformer models; BERT SLURM execution script (`run_bert_cv.sh`) |
| **Tania Perdomo-Flores** | Academic deliverables lead (final report, initial report, poster, project plan); group-to-professor communication; report structure and presentation organization; contributor to report content and analysis write-up |

---

## Future Work

| Area | Description |
|:-----|:------------|
| **Ensemble Methods** | Combine the PA Classifier with MLP or LR using soft voting or stacking — the two models have complementary error profiles that an ensemble could exploit |
| **Advanced Embeddings** | Replace TF-IDF with dense word embeddings (Word2Vec, FastText, GloVe) or contextual embeddings (sentence-BERT) for richer feature representations |
| **Efficient Hypertuning** | Replace GridSearchCV with RandomizedSearchCV or Bayesian optimization (Optuna, Hyperopt) to reduce the combinatorial search cost, especially for MLP |
| **Larger Transformer Models** | Fine-tune larger models (RoBERTa, DeBERTa, or tweet-specific models) with a proper GPU budget — the current BERT experiments were limited by HPC allocation |
| **Real-Time Deployment** | Build a streaming inference prototype using the PA Classifier with a lightweight Flask or FastAPI endpoint, consuming live tweets via the Twitter API |
| **First Responder Notification** | Area-based threshold alerting: when disaster tweet classifications for a given location exceed a threshold, trigger an amber-alert-style notification to first responders — a direct real-world application of this classification pipeline |
| **Keyword Feature Engineering** | Investigate keyword as a structured feature (categorical encoding or embedding) rather than only as prepended text |
| **Error Analysis** | Systematic review of misclassified tweets — particularly false positives where metaphorical disaster language ("this traffic is a disaster") fools the classifier |

---

## Authors and Acknowledgments

**Authors**  
- Thomas D. Robertson II
- Sharon Colson
- Caleb Smith
- Tania Perdomo-Flores

**Course**: CSC-4260 Advanced Data Science — Spring 2025

**Data Source**: [Kaggle NLP with Disaster Tweets Competition](https://www.kaggle.com/competitions/nlp-getting-started/data)

**Original Repository**: [CSC-4260-Advanced-Data-Science-Project/NLP_Disaster_Tweets](https://github.com/CSC-4260-Advanced-Data-Science-Project/NLP_Disaster_Tweets)

**Acknowledgments**  
- **Tennessee Tech ITS Services and the Warp 1 HPC Facility** (NSF Award #2127188) for computational resources used in model training and hyperparameter tuning
- **Dr. William Eberle** for guidance throughout CSC-4260
- The BERT cross-validation implementation (`run_bert_cv.py`) includes structural patterns assisted by OpenAI ChatGPT. The base transformer fine-tuning workflow was adapted from standard Hugging Face `Trainer` API examples.
