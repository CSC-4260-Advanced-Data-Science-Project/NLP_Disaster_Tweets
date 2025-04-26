# Author: Thomas D. Robertson II
import sys
from pipeline_modules_ht import load_xy_datasets, evaluate_models, run_grid_searches, pipelines
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

# Grab the dataset name from the command-line
dataset_name = sys.argv[1]

xy_all = load_xy_datasets("final_processed")

if dataset_name not in xy_all:
    print(dataset_name)
    raise ValueError(f"{dataset_name} not found in final_processed/")

xy_subset = {dataset_name: xy_all[dataset_name]}
X, y = xy_all[dataset_name]  # Add this
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Run the hypertuning function
best_models = run_grid_searches(X_train, y_train, pipelines)

# Evaluate best models on test set
for name, model in best_models.items():
    print(f"\n📊 Evaluation Report for {name}")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

# Only returns results_df now
results_df = evaluate_models(xy_subset, best_models)

# Save raw results to CSV
results_df.to_csv(f"performance_metrics4/{dataset_name}_results.csv", index=False)

# Find the best model
best_row = results_df.iloc[0]  # because you already sorted by F1 descending
best_model_name = best_row['Model']
best_f1_score = best_row['F1 Score']

# Get best hyperparameters
best_model = best_models[best_model_name]
best_params = best_model.best_params_ if hasattr(best_model, 'best_params_') else {}

# Save best model summary
summary_path = f"performance_metrics4/{dataset_name}_best_model.txt"
with open(summary_path, "w") as f:
    f.write(f"Dataset: {dataset_name}\n")
    f.write(f"Best Model: {best_model_name}\n")
    f.write(f"Best Hyperparameters:\n")
    for param, value in best_params.items():
        f.write(f"  {param}: {value}\n")
    f.write("\n")
    f.write(f"Best Metrics:\n")
    f.write(f"  Accuracy: {best_row['Accuracy']:.4f}\n")
    f.write(f"  Precision: {best_row['Precision']:.4f}\n")
    f.write(f"  Recall: {best_row['Recall']:.4f}\n")
    f.write(f"  F1 Score: {best_row['F1 Score']:.4f}\n")
    f.write(f"  ROC AUC: {best_row['ROC AUC']:.4f}\n")

print(f"\n🏆 Best model summary saved to {summary_path}")

