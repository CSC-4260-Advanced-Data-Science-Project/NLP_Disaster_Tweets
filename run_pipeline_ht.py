# Author: Thomas D. Robertson II
import sys
from pipeline_modules_ht import load_xy_datasets, evaluate_models, run_grid_searches
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
results_df.to_csv(f"performance_metrics3/{dataset_name}_results.csv", index=False)