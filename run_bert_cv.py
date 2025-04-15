# run_bert_cv.py
import sys
import os
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.model_selection import KFold
from sklearn import metrics
import torch

def run_bert_pipeline(dataset_name):
    print(f"🚀 Starting BERT CV for {dataset_name}...")

    # Load and preprocess data
    df = pd.read_csv(f"final_processed/{dataset_name}.csv")
    df.drop(columns=[col for col in ['text', 'text_length', 'length'] if col in df], inplace=True)
    df.rename(columns={'processed_text': 'text', 'target': 'label'}, inplace=True)
    df = df[["text", "label"]].copy()
    df["text"] = df["text"].astype(str)

    dataset = Dataset.from_pandas(df)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

    # Set up K-Fold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    metrics_summary = {
        "train": {"acc": [], "f1": [], "prec": [], "rec": []},
        "val": {"acc": [], "f1": [], "prec": [], "rec": []},
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n📁 Fold {fold + 1}/5")

        train = Dataset.from_dict(dataset[train_idx])
        val = Dataset.from_dict(dataset[val_idx])

        train = train.map(tokenize_function, batched=True)
        val = val.map(tokenize_function, batched=True)

        train = train.rename_column("label", "labels")
        val = val.rename_column("label", "labels")

        train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        val.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

        training_args = TrainingArguments(
            output_dir=f"./results/{dataset_name}/fold{fold}",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            save_total_limit=1,
            load_best_model_at_end=True,
            logging_dir=f"./logs/{dataset_name}/fold{fold}",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train,
            eval_dataset=val,
            tokenizer=tokenizer,
        )

        trainer.train()

        # Evaluate on train/val sets
        for split, data in zip(["train", "val"], [train, val]):
            output = trainer.predict(data)
            preds = np.argmax(output.predictions, axis=1)
            labels = output.label_ids

            acc = metrics.accuracy_score(labels, preds)
            f1 = metrics.f1_score(labels, preds)
            prec = metrics.precision_score(labels, preds)
            rec = metrics.recall_score(labels, preds)

            metrics_summary[split]["acc"].append(acc)
            metrics_summary[split]["f1"].append(f1)
            metrics_summary[split]["prec"].append(prec)
            metrics_summary[split]["rec"].append(rec)

    # Save final mean metrics
    result_path = f"performance_metrics_bert/{dataset_name}_bert_results.csv"
    os.makedirs("performance_metrics_bert", exist_ok=True)
    df_out = pd.DataFrame({
        "Split": ["Train", "Validation"],
        "Accuracy": [np.mean(metrics_summary["train"]["acc"]), np.mean(metrics_summary["val"]["acc"])],
        "F1 Score": [np.mean(metrics_summary["train"]["f1"]), np.mean(metrics_summary["val"]["f1"])],
        "Precision": [np.mean(metrics_summary["train"]["prec"]), np.mean(metrics_summary["val"]["prec"])],
        "Recall": [np.mean(metrics_summary["train"]["rec"]), np.mean(metrics_summary["val"]["rec"])],
    })
    df_out.to_csv(result_path, index=False)
    print(f"✅ Saved results to {result_path}")

if __name__ == "__main__":
    dataset_name = sys.argv[1]
    run_bert_pipeline(dataset_name)
