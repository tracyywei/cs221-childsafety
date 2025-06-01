import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import f1_score, precision_score, recall_score

tokenizer = BertTokenizer.from_pretrained("./harm_bert (1)")
model     = BertForSequenceClassification.from_pretrained("./harm_bert (1)")
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

mapping = {
  "Psychological Harm":        "Psychological",
  "Sexual Content":            "Sexual",
  "Privacy Violation":         "Danger",
  "Cybercrime":                "Danger",
  "Insulting Behavior":        "Profanities",
  "Misinformation":            "Danger",
  "Endangering Public Health": "Danger",
  "Physical Harm":             "Danger",
  "Financial Fraud":           "Danger",
  "Terrorism":                 "Danger",
  "Hate Speech":               "Hateful",
  "Self‑Harm Encouragement":   "Self‑harm",
  "Drug Advice":               "Substance use",
  "Violent Extremism":         "Hateful",
  "Underage Substance Use":    "Substance use",
  "Economic Crime":            "Danger",
  "Gambling Advice":           "Danger",
  "Animal Cruelty":            "Danger",
  "Other":                     "Danger"
}

# pku categories
harm_categories = [
    "Endangering National Security","Insulting Behavior","Discriminatory Behavior",
    "Endangering Public Health","Copyright Issues","Violence","Drugs",
    "Privacy Violation","Economic Crime","Mental Manipulation","Human Trafficking",
    "Physical Harm","Sexual Content","Cybercrime","Disrupting Public Order",
    "Environmental Damage","Psychological Harm","White-Collar Crime","Animal Abuse"
]

# MinorBench categories
minorbench_cats = ["Sexual", "Profanities", "Hateful", "Danger", "Self-harm", "Substance use"]
mb_index = {c: i for i, c in enumerate(minorbench_cats)}

mb = load_dataset("govtech/MinorBench", split="test")

all_true = []
all_pred = []

for ex in mb:
    enc = tokenizer(
        ex["Prompt"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    ).to(device)

    # run the model, logits → probs
    with torch.no_grad():
        logits = model(**enc).logits     # shape (1,19)
    probs = torch.sigmoid(logits).cpu().numpy()[0]  # (19,)

    # threshold at 0.5 
    pku_pred = (probs >= 0.5).astype(int)

    # map PKU preds -> 6 child categories
    child_pred = [0]*len(minorbench_cats)
    for i, flag in enumerate(pku_pred):
        if flag:
            cat19 = harm_categories[i]
            mapped = mapping.get(cat19)
            # only keep if it's one of the 6 MinorBench labels
            if mapped in mb_index:
                child_pred[mb_index[mapped]] = 1

    true_vec = [0]*len(minorbench_cats)
    if ex["Category"] not in mb_index:
        raise KeyError(f"Category '{ex['Category']}' normalized to '{ex["Category"]}' not found in mb_index")
    true_vec[mb_index[ex["Category"]]] = 1
    all_true.append(true_vec)
    all_pred.append(child_pred)


all_true = np.array(all_true)
all_pred = np.array(all_pred)
print("F1:       ", f1_score(all_true, all_pred, average="micro"))
print("F1:       ", f1_score(all_true, all_pred, average="macro"))
print("Precision:  ", precision_score(all_true, all_pred, average="micro"))
print("Recall:     ", recall_score(all_true, all_pred, average="micro"))
