import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = BertTokenizer.from_pretrained("./harm_bert", local_files_only=True, use_fast=False)
model = BertForSequenceClassification.from_pretrained("./harm_bert", local_files_only=True)
model = model.to(device)
model.eval()

raw_dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", "default")["train"]
split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
val_data = split_dataset["test"]

harm_categories = [
    "Endangering National Security",
    "Insulting Behavior",
    "Discriminatory Behavior",
    "Endangering Public Health",
    "Copyright Issues",
    "Violence",
    "Drugs",
    "Privacy Violation",
    "Economic Crime",
    "Mental Manipulation",
    "Human Trafficking",
    "Physical Harm",
    "Sexual Content",
    "Cybercrime",
    "Disrupting Public Order",
    "Environmental Damage",
    "Psychological Harm",
    "White-Collar Crime",
    "Animal Abuse"
]

def preprocess(example):
    text = example["prompt"] + " [SEP] " + example["response_0"]
    encoding = tokenizer(text, truncation=True, padding="max_length", max_length=512)
    label_vec = [
        int(example["response_0_harm_category"].get(cat, False))
        for cat in harm_categories
    ]
    encoding["labels"] = label_vec
    return encoding

val_enc = val_data.map(preprocess, batched=False)
val_enc.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
loader = DataLoader(val_enc, batch_size=8)

all_logits = []
all_labels = []

with torch.no_grad():
    for batch in loader:
        batch = {
            k: (v.to(device) if k != "labels" else v.float().to(device))
            for k, v in batch.items()
        }
        outputs = model(**batch)
        logits = outputs.logits.cpu().numpy()
        labels = batch["labels"].cpu().numpy().astype(int)
        all_logits.append(logits)
        all_labels.append(labels)

all_logits = np.concatenate(all_logits, axis=0) 
all_labels = np.concatenate(all_labels, axis=0) 

probs = 1 / (1 + np.exp(-all_logits))
preds = (probs >= 0.5).astype(int)

prec, rec, f1, _ = precision_recall_fscore_support(
    all_labels, preds, average="micro", zero_division=0
)
accuracy = accuracy_score(all_labels, preds)

print("Overall Validation Metrics:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  F1:        {f1:.4f}\n")

prec_per_cat, rec_per_cat, f1_per_cat, _ = precision_recall_fscore_support(
    all_labels, preds, average=None, zero_division=0
)

print("Per-Category Metrics:")
print(f"{'Category':<25} {'Precision':<10} {'Recall':<10} {'F1':<10}")
for idx, cat in enumerate(harm_categories):
    print(f"{cat:<25} {prec_per_cat[idx]:<10.4f} {rec_per_cat[idx]:<10.4f} {f1_per_cat[idx]:<10.4f}")
print("\n")

error_records = []
num_examples = all_labels.shape[0]

for i in range(num_examples):
    true_vec = all_labels[i]
    pred_vec = preds[i]
    example = val_data[i]             # original example
    prompt = example["prompt"]
    response = example["response_0"]

    for j, cat in enumerate(harm_categories):
        # False Negative
        if true_vec[j] == 1 and pred_vec[j] == 0:
            error_records.append({
                "example_index": i,
                "category": cat,
                "error_type": "FN",
                "prompt": prompt,
                "response": response,
                "true_label": 1,
                "pred_label": 0
            })
        # False Positive
        elif true_vec[j] == 0 and pred_vec[j] == 1:
            error_records.append({
                "example_index": i,
                "category": cat,
                "error_type": "FP",
                "prompt": prompt,
                "response": response,
                "true_label": 0,
                "pred_label": 1
            })

errors_df = pd.DataFrame(error_records)
errors_df.to_csv("validation_error_analysis.csv", index=False)

error_counts = errors_df.groupby(["category", "error_type"]).size().unstack(fill_value=0)
print("\nError Counts by Category and Error Type:")
print(error_counts)
