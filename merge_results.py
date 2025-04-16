import os
import pandas as pd

def merge_all_results(input_dir="performance_metrics4", output_file="performance_metrics4/ALL_results.csv"):
    all_dfs = []

    for fname in sorted(os.listdir(input_dir)):
        if fname.endswith("_results.csv") and fname != "ALL_results.csv":
            path = os.path.join(input_dir, fname)
            try:
                df = pd.read_csv(path)
                all_dfs.append(df)
            except Exception as e:
                print(f"❌ Could not read {fname}: {e}")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(output_file, index=False)
        print(f"✅ Merged {len(all_dfs)} result files into {output_file}")
    else:
        print("⚠️ No valid result files found.")

if __name__ == "__main__":
    merge_all_results()
