# Author: Sharon Colson
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from matplotlib import cm  # for colormap
from matplotlib.colors import to_hex
from matplotlib.colors import ListedColormap

from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, learning_curve, train_test_split
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

def load_xy_datasets(folder_path="final_processed", text_col="processed_text", target_col="target"):
    xy_datasets = {}
    nan_report = {}

    for fname in os.listdir(folder_path):
        if fname.endswith(".csv"):
            path = os.path.join(folder_path, fname)
            name = fname.replace(".csv", "")
            df = pd.read_csv(path)

            nans_before = df[[text_col, target_col]].isna().sum().sum()
            df = df.dropna(subset=[text_col, target_col]).reset_index(drop=True)
            nans_after = df[[text_col, target_col]].isna().sum().sum()

            if nans_before > 0:
                nan_report[name] = nans_before

            if text_col in df.columns and target_col in df.columns:
                xy_datasets[name] = (df[text_col], df[target_col])

    print(f"\n📥 Loaded {len(xy_datasets)} datasets.")
    if nan_report:
        print("\n⚠️ Datasets with NaNs (before cleaning):")
        for name, count in nan_report.items():
            print(f" - {name}: {count} NaNs removed")

    return xy_datasets

tfidf = TfidfVectorizer(stop_words='english', max_df=0.8, ngram_range=(1, 3))

pipelines = {
    'MultinomialNB': Pipeline([
        ('tfidf', tfidf),
        ('clf', MultinomialNB())
    ]),
    'LogisticRegression': Pipeline([
        ('tfidf', tfidf),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'PassiveAggressive': Pipeline([
        ('tfidf', tfidf),
        ('clf', PassiveAggressiveClassifier(max_iter=1000, random_state=42))
    ]),
    'SVM': Pipeline([
        ('tfidf', tfidf),
        ('clf', SVC(kernel='linear', C=1.0, probability=True, random_state=42))
    ]),
    'KNN': Pipeline([
        ('tfidf', tfidf),
        ('clf', KNeighborsClassifier(n_neighbors=5))
    ]),
    'NeuralNetwork': Pipeline([
        ('tfidf', tfidf),
        ('clf', MLPClassifier(hidden_layer_sizes=(50,), max_iter=300, verbose=True, random_state=42))
    ])
}

def plot_learning_curve(estimator, title, X, y, cv, scoring='accuracy', n_jobs=-1, save_path=None):

    # Use viridis colormap to get consistent, soft shades
    viridis = cm.get_cmap('viridis')
    train_color = to_hex(viridis(0.25))  # bluish-teal
    test_color = to_hex(viridis(0.65))   # greenish

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, train_sizes=np.linspace(0.1, 1.0, 10)
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(facecolor='white')
    plt.title(title)
    plt.xlabel("Training Examples")
    plt.ylabel(scoring.capitalize())
    plt.grid(alpha=0.3)

    # Soft shading
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2, color=train_color)
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2, color=test_color)

    plt.plot(train_sizes, train_scores_mean, 'o-', color=train_color, label="Training Score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color=test_color, label="Validation Score")

    plt.legend(loc="best")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def evaluate_models(xy_datasets, pipelines, cv_folds=5, random_state=42, output_dir="performance_metrics3"):
    
    results = []

    viridis = cm.get_cmap('viridis', 256)
    viridis_light = ListedColormap(viridis(np.linspace(0.2, 0.8)))  # focus on mid-range

    
    for dataset_name, (X, y) in xy_datasets.items():
        print(f"\n📦 Dataset: {dataset_name} ({len(X)} samples)")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=random_state
        )

        for model_name, pipeline in pipelines.items():
            print(f"  🔍 Evaluating: {model_name}")
            combo_name = f"{dataset_name}_{model_name}"
            save_dir = os.path.join(output_dir, combo_name)
            os.makedirs(save_dir, exist_ok=True)

            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            acc = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy').mean()
            prec = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='precision').mean()
            rec = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='recall').mean()
            f1 = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1').mean()

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            try:
                if hasattr(pipeline.named_steps['clf'], 'predict_proba'):
                    y_scores = pipeline.predict_proba(X_test)[:, 1]
                else:
                    y_scores = pipeline.decision_function(X_test)
                roc = roc_auc_score(y_test, y_scores)
            except Exception:
                roc = np.nan
                y_scores = None

            results.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1 Score': f1,
                'ROC AUC': roc
            })

            # ROC Curve
            if y_scores is not None:
                fpr, tpr, _ = roc_curve(y_test, y_scores)
                plt.figure()
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc:.2f})', color=to_hex(viridis(0.6)))
                plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
                plt.grid(alpha=0.3)
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve — {model_name} on {dataset_name}')
                plt.legend(loc='lower right')
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, "roc_curve.png"))
                plt.close()

            # Confusion Matrix
            cmatrix = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(
                cmatrix,
                annot=True,
                fmt='d',
                # cmap=sns.light_palette("seagreen", as_cmap=True),
                cmap=viridis_light,
                cbar=False,
                linewidths=0.5,
                linecolor='white',
                square=True,
                annot_kws={"size": 12},
                xticklabels=pipeline.classes_,
                yticklabels=pipeline.classes_
            )
            plt.title(f'Confusion Matrix — {model_name} on {dataset_name}')
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
            plt.close()

            # Learning Curve
            print(f"📈 Plotting learning curve for {model_name} on {dataset_name}...")
            plot_learning_curve(
                pipeline,
                f"Learning Curve — {model_name} on {dataset_name}",
                X_train,
                y_train,
                cv=cv,
                save_path=os.path.join(save_dir, "learning_curve.png")
            )

    # Sort by F1 Score in descending order
    results_df = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False).reset_index(drop=True)

    return results_df