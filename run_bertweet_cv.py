"""
File: run_bertweet_cv.py
Authors: Caleb Smith and Sharon Colson
Date Modified: 4/29/25
Project: NLP With Disaster Tweets

We acknowledge the use of ChatGPT from OpenAI to help create this script:
'How can I use a BERT model for text classification?' prompt and follow-up prompts. ChatGPT,
OpenAI, 6 Apr. 2025 and following dates, chatgpt.com
"""

import sys
import os
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.model_selection import KFold, train_test_split
from sklearn import metrics
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# model = model.to(device)
# inputs = inputs.to(device)

def run_bert_pipeline(dataset_name):
    print(f"🚀 Starting BERT CV for {dataset_name}...")

    # Load and preprocess data
    df = pd.read_csv(f"final_processed/{dataset_name}.csv")
    df.drop(columns=[col for col in ['text', 'text_length', 'length'] if col in df], inplace=True)
    df.rename(columns={'processed_text': 'text', 'target': 'label'}, inplace=True)
    df = df[["text", "label"]].copy()
    df["text"] = df["text"].astype(str)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)

    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

    # Set up K-Fold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    metrics_summary = {
        "train": {"acc": [], "f1": [], "prec": [], "rec": []},
        "val": {"acc": [], "f1": [], "prec": [], "rec": []},
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
        print(f"\n📁 Fold {fold + 1}/5")

        train = Dataset.from_dict(train_dataset[train_idx])
        val = Dataset.from_dict(train_dataset[val_idx])

        train = train.map(tokenize_function, batched=True)
        val = val.map(tokenize_function, batched=True)

        train = train.rename_column("label", "labels")
        val = val.rename_column("label", "labels")

        train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        val.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        model = AutoModelForSequenceClassification.from_pretrained("vinai/bertweet-base", num_labels=2).to(device)
        model = model.to(device)

        training_args = TrainingArguments(
            output_dir=f"./results/{dataset_name}/fold{fold}",
            eval_strategy="epoch",
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
            # model.to(device)
            # # Move inputs to the same device
            # inputs = {k: v.to(device) for k, v in inputs.items()}
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
    result_path = f"performance_metrics_bertweet/{dataset_name}_bertweet_val_results.csv"
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

    full_train_dataset = Dataset.from_pandas(train_df)
    full_train_dataset = full_train_dataset.map(tokenize_function, batched=True)
    full_train_dataset = full_train_dataset.rename_column("label", "labels")
    full_train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained("vinai/bertweet-base", num_labels=2).to(device)

    # "./results/final_bertweet_{dataset_name}/fold{fold}"

    final_training_args = TrainingArguments(
        output_dir=f"./results/final_bertweet_{dataset_name}/fold{fold}",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=1,
        logging_dir=f"./logs/final_bertweet_{dataset_name}/fold{fold}",
        report_to="none",  # turn off wandb/logging
    )

    final_trainer = Trainer(
        model=model,
        args=final_training_args,  # maybe with more epochs if desired
        train_dataset=full_train_dataset,
        tokenizer=tokenizer,
    )

    final_trainer.train()

    test_dataset = Dataset.from_pandas(test_df)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.rename_column("label", "labels")
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    test_output = final_trainer.predict(test_dataset)
    test_preds = np.argmax(test_output.predictions, axis=1)
    test_labels = test_output.label_ids

    # Metrics
    test_acc = metrics.accuracy_score(test_labels, test_preds)
    test_f1 = metrics.f1_score(test_labels, test_preds)
    test_prec = metrics.precision_score(test_labels, test_preds)
    test_rec = metrics.recall_score(test_labels, test_preds)

    print("\n📊 Test Set Performance:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall: {test_rec:.4f}")

    result_test_path = f"performance_metrics_bertweet/{dataset_name}_bertweet_test_results.csv"
    df_test_out = pd.DataFrame({
        "Split": ["Test"],
        "Accuracy": [test_acc],
        "F1 Score": [test_f1],
        "Precision": [test_prec],
        "Recall": [test_rec],
    })
    df_test_out.to_csv(result_test_path, index=False)
    print(f"✅ Saved results to {result_test_path}")

    final_trainer.save_model(f"./final_model_{dataset_name}")


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    run_bert_pipeline(dataset_name)