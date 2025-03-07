# NLP for Disaster Tweets
README IS CURRENTLY A WORK IN PROGRESS
This repository contains the code, data preprocessing scripts, and initial model implementation for our project aimed at distinguishing genuine disaster-related tweets from non-disaster tweets. The project leverages natural language processing (NLP) techniques to clean, preprocess, and classify tweet data for effective disaster detection.

## Project Overview

Twitter has become an essential channel for real-time information during emergencies. However, separating tweets that report an actual disaster from those that merely reference disaster-related terms is a challenging task. This project addresses that challenge by building a classification model that accurately identifies disaster-related tweets. Our current baseline model uses a Multinomial Naive Bayes classifier, and future work will incorporate additional models (e.g., logistic regression, support vector machines, and transformer-based approaches) to improve performance.

## Dataset

The dataset used in this project is sourced from a Kaggle competition:
- **Train Data:** `train.csv` – Contains tweets with the target label (1 for disaster-related, 0 for non-disaster).
- **Test Data:** `test.csv` – Contains tweet data without target labels for final evaluation.
- **Sample Submission:** `sample_submission.csv` – An example format for submitting predictions.

## Project Structure

```
├── Data
│   ├── train.csv                # Training dataset with target labels
│   ├── test.csv                 # Test dataset without target labels
│   └── sample_submission.csv    # Example submission file
├── notebooks (Only has a singular notebook for now)
│   ├── eda_approach_2.ipynb       # Exploratory Data Analysis and visualization
│   ├── NLP_Preprocessing-and-Model-Analysis.ipynb  # Text cleaning, tokenization, and vectorization scripts and pipelined models for analysis
│   └── Multiple_Model_Analysis.ipynb     # Not yet implemented will contain new model analysis methods on differently prepared data
├── reports                      # Contains the PDF Reports of the project
│   ├── CSC-4260-NLP Tweets-Initial-Report.pdf       # Initial report
├── README.md                    # Project documentation
└── requirements.txt             # List of required Python packages (not yet implemented)
```

## Installation

1. **Clone the Repository:**

   ```bash
   git clone git@github.com:CSC-4260-Advanced-Data-Science-Project/NLP_Disaster_Tweets.git
   cd NLP_Disaster_Tweets
   ```

2. **Create a Virtual Environment (optional but recommended):**

   ```bash
   python -m venv ./venv
   .\venv\Scripts\activate

   ```

3. **Install Required Packages:**

   ```bash
   pip install pandas numpy matplotlib seaborn nltk scikit-learn spacy pyldavis wordcloud
   ```

   Or use requirements.txt (not yet implemented)
   ```bash
   pip install -r requirements.txt
   ```

4. **Notebooks:**  (Not yet needed)
   For interactive exploration and further analysis, open the Jupyter notebooks located in the `notebooks` directory.

## Results

The baseline Multinomial Naive Bayes model currently implemented shows the following performance based on cross-validation:
- **Mean Accuracy:** 0.7939
- **Mean Precision:** 0.8805
- **Mean Recall:** 0.6037
- **Mean F1 Score:** 0.7159

- 

These results indicate that while the model is highly precise, it misses some disaster-related tweets (lower recall). Future iterations will focus on improving recall by incorporating and comparing additional models.

## Future Work

- **Model Expansion:** Implement and evaluate additional classifiers such as logistic regression, support vector machines, and transformer-based models.
- **Hyperparameter Tuning:** Optimize model parameters for improved performance.
- **Ensemble Methods:** Explore ensemble strategies to balance precision and recall.

## Authors

- Sharon Colson
- Thomas Robertson
- Caleb Smith
- Tania Flores

## Acknowledgements

This project was developed as part of the CSC-4260-NLP course. We thank our instructors and peers for their valuable feedback and support throughout the project.


