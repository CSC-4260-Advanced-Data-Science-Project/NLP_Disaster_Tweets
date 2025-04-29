# NLP Disaster Tweet Classification

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Preprocessing and Data Variants](#preprocessing-and-data-variants)
- [Modeling Approach](#modeling-approach)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Evaluation Metrics](#evaluation-metrics)
- [Results Summary](#results-summary)
- [Tools and Technologies](#tools-and-technologies)
- [Future Work](#future-work)

---

## Project Overview

This project builds a pipeline to classify tweets as **disaster-related** or **non-disaster**. The goal is to create an accurate, efficient model capable of real-time classification to aid disaster monitoring agencies.

Our work progressed through **exploratory data analysis (EDA)**, **baseline modeling**, **systematic preprocessing experiments**, **extensive model hypertuning**, and **transformer-based model testing**.

---

## Dataset

- Source: [Kaggle NLP Disaster Tweets Competition](https://www.kaggle.com/competitions/nlp-getting-started/data)
- 7,613 labeled tweets
- Features:
  - `id`: Tweet ID
  - `text`: Tweet content
  - `location`, `keyword`: Metadata (optional)
  - `target`: 1 (disaster) or 0 (non-disaster)

---

## Project Structure

```
baseline_performance/
Data/
Deliverables/
final_complete_ht_slurm_files/
final_ht_performance_metrics/
final_processed/
Images/
processed_data/
Sandbox/
wordclouds/
.gitignore
README.md
OG_README.md
NLP_DS_Pipeline.ipynb
merge_results.py
pipeline_modules.py
pipeline_modules_ht.py
run_pipeline.py
run_pipeline.sh
run_pipeline_ht.py
run_pipeline_ht.sh
dataset_list.txt
```

---

## Preprocessing and Data Variants

To test different cleaning strategies, we created **30 dataset variants** by:
- Lowercasing, stemming, lemmatization
- Removing emojis, mentions, hashtags selectively
- Custom stopword filtering

Each baseline (`kept`, `dropped`, `prepended`) had 10 cleaning strategies (`v1` to `v10`).

Example Variants:
- `kept_v7_lowercase_words_only`: Kept only lowercase alphabetic tokens
- `prepended_v4_stemmed`: Stemming applied, keywords prepended
- `dropped_v1_basic_clean`: Basic stopword removal, punctuation stripped

**Vectorization**: TF-IDF with unigrams, bigrams, trigrams.

---

## Modeling Approach

### Baseline Models
- Multinomial Naive Bayes (MNB)
- Passive Aggressive Classifier (PA)
- Logistic Regression (LR)
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Multi-Layer Perceptron (MLP Neural Net)

### Advanced Models
- **BERT Base-Uncased**
- **BERTweet Base** (trained on tweets)

---

## Hyperparameter Tuning

We implemented **GridSearchCV** for each model with tailored search spaces:

| Model | Key Parameters Tuned |
|:------|:---------------------|
| MNB | `alpha` (Laplace smoothing) |
| Logistic Regression | `C`, `solver`, `penalty`, `max_iter` |
| Passive Aggressive | `C`, `max_iter`, `tol` |
| SVM | `C`, `kernel`, `gamma` |
| KNN | `n_neighbors`, `weights`, `metric` |
| MLP | `hidden_layer_sizes`, `activation`, `solver`, `alpha`, `learning_rate`, `early_stopping` |

**Execution**:
- Parallelized using HPC cluster (28 cores, 16GB RAM per job)
- Managed with `run_pipeline_ht.sh` and array jobs

---

## Evaluation Metrics

Each model was evaluated with:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score** (Primary focus)
- **ROC AUC**

Performance comparison was done **before** and **after** hypertuning.

---

## Results Summary

### Top-Performing Hypertuned Model
- **Model**: Passive Aggressive Classifier
- **Dataset**: `kept_v7_lowercase_words_only`
- **Metrics**:
  - Accuracy: 79.26%
  - Precision: 76.50%
  - Recall: 73.92%
  - F1 Score: 75.18%
  - ROC AUC: 85.88%

### Key Insights
- **Simple preprocessing** (lowercasing only) worked better than aggressive cleaning.
- **Hyperparameter tuning** significantly boosted F1 scores across all models.
- **Transformer models** (BERT/BERTweet) performed slightly better, but at high computational cost.
- **PA Classifier** emerged as the best balance of performance and efficiency.

Visual comparisons were created for:
- Baseline vs. Tuned F1 Scores per Model
- Baseline vs. Tuned F1 Scores per Dataset

---

## Tools and Technologies

| Tool | Purpose |
|:----|:--------|
| Pandas | Data manipulation |
| Scikit-learn | Modeling, GridSearchCV, evaluation |
| NLTK | Tokenization, stemming |
| Matplotlib, Seaborn | Visualization |
| Hugging Face Transformers | Pretrained BERT models |
| HPC Cluster | Parallelized model training and hypertuning |

---

## Future Work

- **Ensemble Methods**: Combine models (e.g., PA + MLP) for better performance.
- **Advanced Feature Engineering**: Use Word2Vec, FastText embeddings.
- **Efficient Hypertuning**: Explore RandomizedSearchCV or Bayesian optimization.
- **Larger Transformer Models**: Test more advanced LLMs with tweet-specific fine-tuning.
- **Real-Time Deployment**: Build a prototype for streaming tweet classification.

---

> **Authors**: Sharon Colson, Thomas Robertson, Caleb Smith, Tania Perdomo-Flores  
> **Original Repository**: [GitHub Repo](https://github.com/CSC-4260-Advanced-Data-Science-Project/NLP_Disaster_Tweets)  
> **Data Source**: [Kaggle - NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started/data)


