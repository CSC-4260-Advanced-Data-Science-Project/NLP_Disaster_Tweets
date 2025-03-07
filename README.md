# NLP for Disaster Tweets

## Project Overview

This repository contains the code, data preprocessing scripts, and model implementations for the classification of disaster-related tweets. The goal is to develop an NLP model that can effectively distinguish genuine disaster-related tweets from non-disaster tweets. The project leverages machine learning and deep learning techniques for text classification, integrating traditional models such as Multinomial Naive Bayes, Logistic Regression, etc, alongside potential future integration of transformer-based models.

## Team Members

- Sharon Colson
- Thomas D. Robertson
- Caleb Smith
- Tania Flores

### Notebook Contributions
Visualizations: Sharon Colson, Tania Perdomo Flores, Thomas Robertson, Caleb Smith
Data Preprocessing: Sharon Colson, Tania Perdomo Flores, Thomas Robertson, Caleb Smith
Model Selection: Sharon Colson, Tania Perdomo Flores, Thomas Robertson, Caleb Smith
Modeling: Thomas Robertson, Caleb Smith
Model Tuning: Thomas Robertson, Caleb Smith
Model Evaluation: Thomas Robertson
Model Comparison: Thomas Robertson

### Report Contributions
All team members contributed to the report by providing content, reviewing, and editing sections.

## Problem Statement

Twitter serves as a crucial source of real-time information during emergencies. However, not all tweets that mention disaster-related keywords pertain to actual events, making it challenging for disaster response teams to filter relevant information. This project aims to build an accurate and efficient classifier that can identify disaster-related tweets, ultimately assisting emergency management organizations in prioritizing response efforts.

## Dataset

The dataset is sourced from Kaggle's competition "Natural Language Processing with Disaster Tweets":
- **Train Data:** `train.csv` – Contains labeled tweets (1 for disaster-related, 0 for non-disaster).
- **Test Data:** `test.csv` – Contains unlabeled tweet data for evaluation.
- **Sample Submission:** `sample_submission.csv` – Example format for submitting predictions.

Key Characteristics:
- Binary classification labels.
- Features include tweet text, location, and keyword metadata.
- Feature of interest for testing was the text content of the tweets.

## Project Structure

```
├── Data
│   ├── train.csv                # Training dataset
│   ├── test.csv                 # Test dataset
│   └── sample_submission.csv    # Submission example
├── notebooks
│   ├── eda_approach_2.ipynb       # Exploratory Data Analysis (EDA)
│   ├── NLP_Preprocessing-and-Model-Analysis.ipynb  # Text preprocessing and initial models
│   ├── Multiple_Model_Analysis.ipynb  # Advanced model analysis (upcoming)
├── reports                      
│   ├── NLP Tweets Initial Report.pdf  # Detailed project report
├── README.md                    # Project documentation
└── requirements.txt             # Required Python packages (upcoming)
```

## Methods

### Data Cleaning and Preprocessing

- **Text Normalization:** Lowercasing, removal of special characters, stopword filtering.
- **Tokenization & Stemming:** Using NLTK
- **Vectorization:** TF-IDF and word embeddings.
- **Feature Engineering:** Bi-grams and tri-grams for improved classification.

### Machine Learning Models Implemented

1. **Baseline Model:** Multinomial Naive Bayes.
2. **Traditional ML Models:** Logistic Regression, Support Vector Machines (SVM), Passive Aggressive Classifier, and K-Nearest Neighbors.
3. **Neural Network:** Multi-Layer Perceptron (MLP).
4. **Future Work:** Transformer-based models (BERT, DistilBERT, RoBERTa) and ensemble learning approaches.

## Results

| Model                          | Accuracy | Precision | Recall | F1 Score |
|--------------------------------|----------|------------|---------|------------|
| Multinomial Naive Bayes        | 0.7969   | 0.8607     | 0.6304  | 0.7276     |
| Passive Aggressive Classifier  | 0.7846   | 0.7609     | 0.7288  | 0.7443     |
| Logistic Regression            | 0.7864   | 0.7609     | 0.5885  | 0.7034     |
| Support Vector Machine         | 0.6110   | 1.0000     | 0.0964  | 0.1757     |
| K-Nearest Neighbors            | 0.6849   | 0.8775     | 0.3340  | 0.4458     |
| MLP Classifier (Neural Network) | 0.7951   | 0.7903     | 0.7136  | 0.7499     |

## Tools and Technologies

- **Programming Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **NLP Processing:** NLTK, SpaCy (upcoming)
- **Machine Learning:** Scikit-learn, Transformer Models (upcoming)
- **Visualization:** Matplotlib, Seaborn, pyLDAvis (upcoming)

## Future Work

- Implement Transformer-based models for improved contextual understanding.
- Hyperparameter tuning for performance optimization.
- Exploration of ensemble methods.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone git@github.com:CSC-4260-Advanced-Data-Science-Project/NLP_Disaster_Tweets.git
   cd NLP_Disaster_Tweets
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv ./venv
   source venv/bin/activate  # On Mac/Linux
   .\venv\Scripts\activate  # On Windows
   ```

3. **Install Required Packages:** (upcoming)
   ```bash
   pip install -r requirements.txt
   ```

### Running the Notebooks
Simply open the Jupyter notebooks in the root directory and run all the cells to execute the code.

## Data Links

- **Kaggle Dataset:** [NLP Disaster Tweets Dataset](https://www.kaggle.com/competitions/nlp-getting-started/data)

## Acknowledgements

This project was developed as part of the CSC-4260 course.
