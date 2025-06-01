import numpy as np
import torch
from transformers import BertTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt

MODEL_NAME_OR_PATH = "./harm_bert/harm_bert"
SHAP_FILE         = "shap_values.npz"
OUTPUT_PLOT       = "category_avg_shap.png"
MAX_LENGTH        = 256
N_EXPLAIN         = 5               # how many validation samples explained

harm_categories = [
    "Endangering National Security", "Insulting Behavior", "Discriminatory Behavior",
    "Endangering Public Health", "Copyright Issues", "Violence", "Drugs",
    "Privacy Violation", "Economic Crime", "Mental Manipulation", "Human Trafficking",
    "Physical Harm", "Sexual Content", "Cybercrime", "Disrupting Public Order",
    "Environmental Damage", "Psychological Harm", "White-Collar Crime", "Animal Abuse"
]

loaded = np.load(SHAP_FILE)
all_keys = list(loaded.files)
print("Keys found in", SHAP_FILE, ":", all_keys)

cat_keys = [k for k in all_keys if k.startswith("cat_")]

first_arr = loaded[cat_keys[0]]
shap_values_list = []  # final list of per‐category arrays

n_labels = first_arr.shape[-1]
for lbl_idx in range(n_labels):
    sub_arr = first_arr[..., lbl_idx]
    shap_values_list.append(sub_arr)

num_cats = n_labels
used_cat_keys = [f"cat_{i}" for i in range(n_labels)]

per_token_importances = []  
avg_per_token         = [] 
avg_overall           = []

for idx in range(num_cats):
    arr = shap_values_list[idx]

    if arr.ndim == 3:
        norms = np.linalg.norm(arr, axis=-1)
    else:
        norms = arr.copy()

    per_token_importances.append(norms)
    avg_per_token.append(norms.mean(axis=0))
    avg_overall.append(norms.mean())

avg_overall = np.array(avg_overall)

raw_dataset   = load_dataset("PKU-Alignment/PKU-SafeRLHF", "default", split="train")
split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
val_data      = split_dataset["test"]

texts = [f"{ex['prompt']} {ex['response_0']}" for ex in val_data]

if "labels" in val_data.column_names:
    labels_arr = np.array(val_data["labels"])
else:
    labels_arr = np.array([
        [int(ex["response_0_harm_category"].get(cat, False)) for cat in harm_categories]
        for ex in val_data
    ])

positive_indices = np.where(labels_arr.sum(axis=1) > 0)[0]
sample_indices   = positive_indices[:N_EXPLAIN].tolist()
texts_sample     = [texts[i] for i in sample_indices]

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
batch_enc = tokenizer(
    texts_sample,
    padding="max_length",
    truncation=True,
    max_length=MAX_LENGTH,
    return_tensors="np"
)
sample_ids = batch_enc["input_ids"]
sample_tokens = []
for i in range(N_EXPLAIN):
    toks = tokenizer.convert_ids_to_tokens(sample_ids[i])
    sample_tokens.append(toks)

# find top 5 tokens per category
top_tokens_per_cat = {}

for idx in range(num_cats):
    if idx < len(harm_categories):
        cat_name = harm_categories[idx]
    else:
        cat_name = f"cat_{idx}"

    scores = avg_per_token[idx]
    rep_toks = sample_tokens[0]

    pairs = list(zip(rep_toks, scores))
    pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
    top5 = pairs_sorted[:5]
    top_tokens_per_cat[cat_name] = top5

# plot for avg SHAP magnitude
labels_to_plot = harm_categories[:num_cats]
plt.figure(figsize=(10, 6))
plt.barh(labels_to_plot, avg_overall[:num_cats])
plt.xlabel("Average SHAP magnitude (all tokens × all samples)")
plt.title("Average SHAP Importance by Category")
plt.gca().invert_yaxis() 
plt.tight_layout()
plt.savefig(OUTPUT_PLOT)
plt.close()

print("\nTop‑5 Tokens (by average SHAP) for each category:\n")
print(f"{'Category':<35}  Top‑5 tokens (token : avg_SHAP)")
print("-" * 80)
for idx in range(num_cats):
    if idx < len(harm_categories):
        cat_name = harm_categories[idx]
    else:
        cat_name = f"cat_{idx}"
    top5 = top_tokens_per_cat[cat_name]
    top5_str = ", ".join([f"{tkn} ({score:.4f})" for tkn, score in top5])
    print(f"{cat_name:<35}  {top5_str}")
print()
