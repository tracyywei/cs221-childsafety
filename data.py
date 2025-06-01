from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import f1_score, precision_score, recall_score

pku = load_dataset("PKU-Alignment/PKU-SafeRLHF", "default")

# 90/10 split train/validation
split_dataset = pku["train"].train_test_split(test_size=0.1, seed=42)
train_data = split_dataset["train"]
val_data = split_dataset["test"]

def get_labels(entry):
    return [cat for cat, is_true in entry["response_0_harm_category"].items() if is_true]

all_labels = [get_labels(ex) for ex in train_data]
flat_labels = [label for sublist in all_labels for label in sublist]
label_counts = pd.Series(Counter(flat_labels)).sort_values(ascending=False)

# plot of harm category distribution
plt.figure(figsize=(15, 6))
ax = label_counts.plot(kind="bar", title="Harm Category Distribution")
plt.xlabel("Harm Category")
plt.ylabel("Count")
plt.xticks(range(len(label_counts.index)), label_counts.index, rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Baseline implementation
# F1: 0.2229299363057325
# Precision: 0.3017241379310345
# Recall: 0.17676767676767677

harm_keywords = {
    "Endangering National Security": ["spy", "classified", "leak intel", "sabotage", "terrorist", "espionage"],
    "Insulting Behavior": ["idiot", "stupid", "moron", "dumb", "loser", "shut up"],
    "Discriminatory Behavior": ["racist", "sexist", "homophobic", "slur", "discriminate", "inferior"],
    "Endangering Public Health": ["fake vaccine", "antivax", "drink bleach", "infect others", "covid hoax"],
    "Copyright Issues": ["torrent", "pirate", "crack software", "download movie", "illegal copy"],
    "Violence": ["kill", "stab", "fight", "punch", "beat up", "bomb"],
    "Drugs": ["cocaine", "weed", "marijuana", "get high", "meth", "overdose", "drug deal"],
    "Privacy Violation": ["dox", "IP address", "track location", "leak phone number", "hack account"],
    "Economic Crime": ["money laundering", "scam", "fake check", "bank fraud", "steal credit card"],
    "Mental Manipulation": ["gaslight", "manipulate", "mind control", "coerce", "brainwash"],
    "Human Trafficking": ["sell people", "smuggle humans", "escort child", "buy slave", "sex trafficking"],
    "Physical Harm": ["hurt yourself", "cut wrists", "burn skin", "harm body", "self injure"],
    "Sexual Content": ["sex", "porn", "nude", "NSFW", "masturbate", "XXX"],
    "Cybercrime": ["hack", "ddos", "phishing", "keylogger", "steal password", "malware"],
    "Disrupting Public Order": ["riot", "protest violently", "block road", "throw rocks", "mob"],
    "Environmental Damage": ["dump waste", "illegal logging", "poach animals", "pollute river", "burn trash"],
    "Psychological Harm": ["you're worthless", "no one loves you", "kill yourself", "you're a failure"],
    "White-Collar Crime": ["insider trading", "embezzle", "bribe", "fraud", "shell company"],
    "Animal Abuse": ["kick dog", "hurt cat", "kill animal", "starve pet", "abuse dog"]
}

def keyword_baseline(prompt, response):
    text = (prompt + " " + response).lower()
    matches = [label for label, keywords in harm_keywords.items()
               if any(word in text for word in keywords)]
    return matches

# evaluate on 200 examples from validation split
true = [set(get_labels(ex)) for ex in val_data.select(range(200))]
pred = [set(keyword_baseline(ex["prompt"], ex["response_0"])) for ex in val_data.select(range(200))]

all_categories = sorted(label_counts.index.tolist())

def binarize(y, classes):
    return [[int(c in labels) for c in classes] for labels in y]

y_true = binarize(true, all_categories)
y_pred = binarize(pred, all_categories)

# evaluation metrics
print("F1:", f1_score(y_true, y_pred, average="micro"))
print("Precision:", precision_score(y_true, y_pred, average="micro"))
print("Recall:", recall_score(y_true, y_pred, average="micro"))

# analyze safety distribution
safe_count = sum(1 for ex in train_data if ex["is_response_0_safe"])
total_count = len(train_data)
safe_percentage = (safe_count / total_count) * 100
print("\nSafety Distribution Analysis:")
print(f"Total examples: {total_count}")
print(f"Safe responses: {safe_count}")
print(f"Unsafe responses: {total_count - safe_count}")
print(f"Percentage safe: {safe_percentage:.2f}%")