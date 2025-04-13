import sys
from pipeline_modules import load_xy_datasets, pipelines, evaluate_models

# Grab the dataset name from the command-line
dataset_name = sys.argv[1]

xy_all = load_xy_datasets("final_processed")

if dataset_name not in xy_all:
    print(dataset_name)
    raise ValueError(f"{dataset_name} not found in final_processed/")

xy_subset = {dataset_name: xy_all[dataset_name]}

# Only returns results_df now
results_df = evaluate_models(xy_subset, pipelines)

# Save raw results to CSV
results_df.to_csv(f"performance_metrics3/{dataset_name}_results.csv", index=False)
