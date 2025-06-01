import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = BertTokenizer.from_pretrained("./harm_bert/harm_bert", local_files_only=True, use_fast=False)
model = BertForSequenceClassification.from_pretrained("./harm_bert/harm_bert", local_files_only=True)
model = model.to(device)
model.eval()

raw_dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", "default")["train"]
split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
val_data = split_dataset["test"]

harm_categories = [
    "Endangering National Security", "Insulting Behavior", "Discriminatory Behavior", "Endangering Public Health",
    "Copyright Issues", "Violence", "Drugs", "Privacy Violation", "Economic Crime", "Mental Manipulation",
    "Human Trafficking", "Physical Harm", "Sexual Content", "Cybercrime", "Disrupting Public Order",
    "Environmental Damage", "Psychological Harm", "White-Collar Crime", "Animal Abuse"
]

def preprocess(example):
    text = example["prompt"] + " [SEP] " + example["response_0"]
    encoding = tokenizer(text, truncation=True, padding="max_length", max_length=512)
    label_vec = [int(example["response_0_harm_category"].get(cat, False)) for cat in harm_categories]
    encoding["labels"] = label_vec
    return encoding

val_enc = val_data.map(preprocess, batched=False)
val_enc.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
loader = DataLoader(val_enc, batch_size=8)

all_logits = []
all_labels = []
with torch.no_grad():
    for batch in loader:
        # Move tensors to device; cast labels to float for the model
        batch = {k: (v.to(device) if k != "labels" else v.float().to(device)) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits.cpu().numpy()
        labels = batch["labels"].cpu().numpy()
        all_logits.append(logits)
        all_labels.append(labels)

all_logits = np.concatenate(all_logits, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

probs = 1 / (1 + np.exp(-all_logits))
preds = (probs >= 0.5).astype(int)

f1 = f1_score(all_labels, preds, average="micro")
precision = precision_score(all_labels, preds, average="micro")
recall = recall_score(all_labels, preds, average="micro")
accuracy = accuracy_score(all_labels, preds)

print("Finetuned Model Performance:")
print("Accuracy:", accuracy)
print("F1:", f1)
print("Precision:", precision)
print("Recall:", recall)