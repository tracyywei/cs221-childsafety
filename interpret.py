import torch
import numpy as np
import shap
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm  # for progress bars

# =============================================================================
# interpretability_and_thresholds_hf.py
#
# Same as before, up through computing `token_level_importances`.
# After that, we loop through all categories and show top‐5 tokens for sample 0.
# =============================================================================

# --------------------------------------
# 1. Load PKU-SafeRLHF & Split into Train/Val
# --------------------------------------
raw_dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", "default", split="train")
split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
val_data = split_dataset["test"]

# --------------------------------------
# 2. Load Model & Tokenizer
# --------------------------------------
MODEL_NAME_OR_PATH = "./harm_bert"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
base_model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME_OR_PATH,
    num_labels=19,
    problem_type="multi_label_classification"
)
base_model.to(DEVICE)
base_model.eval()

# --------------------------------------
# 3. Prepare Validation Texts & Labels
# --------------------------------------
texts = [f"{ex['prompt']} {ex['response_0']}" for ex in val_data]

harm_categories = [
    "Endangering National Security", "Insulting Behavior", "Discriminatory Behavior",
    "Endangering Public Health", "Copyright Issues", "Violence", "Drugs",
    "Privacy Violation", "Economic Crime", "Mental Manipulation", "Human Trafficking",
    "Physical Harm", "Sexual Content", "Cybercrime", "Disrupting Public Order",
    "Environmental Damage", "Psychological Harm", "White-Collar Crime", "Animal Abuse"
]

if "labels" in val_data.column_names:
    labels = np.array(val_data["labels"])
else:
    labels = np.array([
        [int(ex["response_0_harm_category"].get(cat, False)) for cat in harm_categories]
        for ex in val_data
    ])

# --------------------------------------
# 4. Tokenization Helper (pad to max_length)
# --------------------------------------
def tokenize_batch(text_list, max_length=256):
    return tokenizer(
        text_list,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

# --------------------------------------
# 5. SHAP Interpretability via GradientExplainer (on GPU, very small sets)
# --------------------------------------

# 5a) Select up to 5 validation examples with ≥1 positive label
positive_indices = np.where(labels.sum(axis=1) > 0)[0]
sample_indices = positive_indices[:5].tolist()
texts_sample = [texts[i] for i in sample_indices]

# 5b) Build a background set of 5 random validation examples
background_indices = np.random.choice(len(texts), size=5, replace=False)
background_texts = [texts[i] for i in background_indices]

# 5c) Tokenize background & sample on GPU, cast attention_mask to float
background_inputs = tokenize_batch(background_texts).to(DEVICE)
sample_inputs     = tokenize_batch(texts_sample).to(DEVICE)

background_ids   = background_inputs["input_ids"]                 # (5, 256)
background_att   = background_inputs["attention_mask"].float()    # (5, 256)

sample_ids       = sample_inputs["input_ids"]                     # (5, 256)
sample_att       = sample_inputs["attention_mask"].float()        # (5, 256)

# 5d) Define a GPU wrapper for SHAP: forward(inputs_embeds, attention_mask)
class BertForSHAP_GPU(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, inputs_embeds, attention_mask):
        logits = self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        ).logits
        return logits  # (batch, 19)

wrapped_model_gpu = BertForSHAP_GPU(base_model).to(DEVICE)
wrapped_model_gpu.eval()

# 5e) Compute embeddings on GPU using base_model’s BERT embeddings
with torch.no_grad():
    background_embeds = base_model.bert.embeddings(background_ids)  # (5, 256, embed_dim)
    sample_embeds     = base_model.bert.embeddings(sample_ids)      # (5, 256, embed_dim)

# 5f) Build GradientExplainer on GPU
print("🔍 Computing SHAP values on GPU (very small sets)…")
background_data = [background_embeds, background_att]
explainer = shap.GradientExplainer(wrapped_model_gpu, background_data)

# 5g) Compute SHAP values for the 5 samples
sample_data = [sample_embeds, sample_att]
shap_values = explainer.shap_values(sample_data)
print("✅ SHAP values computation complete.")

# Save SHAP values to an external file (one array per category)
shap_values_dict = {f"cat_{i}": shap_values[i] for i in range(len(shap_values))}
np.savez("shap_values.npz", **shap_values_dict)
print("✅ Saved SHAP values to shap_values.npz")

# 5h) Aggregate across embedding dimension to get token‑level importances
# shap_values is a list of 19 arrays, each (5, 256, embed_dim)
token_level_importances = []
for cat_idx in range(len(shap_values)):
    vals = shap_values[cat_idx]               # (5, 256, embed_dim)
    per_token = np.linalg.norm(vals, axis=-1)  # (5, 256)
    token_level_importances.append(per_token)

# --------------------------------------
# 5i) PRINT TOP‑5 TOKENS FOR EACH CATEGORY (for sample_idx = 0)
# --------------------------------------
sample_idx = 0  # choose which of the 5 explained samples to inspect (0..4)
print("\nTop‑5 SHAP tokens for sample #0 in each category:\n")
for cat_idx, cat_name in enumerate(harm_categories):
    importances = token_level_importances[cat_idx][sample_idx]   # (256,)
    token_ids    = sample_ids[sample_idx].cpu().numpy()          # (256,)
    tokens       = tokenizer.convert_ids_to_tokens(token_ids)    # list of 256 strings

    # Pair (token, importance) and sort descending by importance
    importances = importances.tolist()  # now each element is a Python float
    token_imp = list(zip(tokens, importances))
    token_imp_sorted = sorted(token_imp, key=lambda x: x[1], reverse=True)


    # Take top-5 (skip “[PAD]” or “[UNK]” if you like, but this grabs top by raw SHAP score)
    top5 = token_imp_sorted[:5]

    print(f"Category: {cat_name}")
    for tkn, score in top5:
        # if score is a list, take its first element; otherwise, use it directly.
        score_val = float(score[0]) if isinstance(score, list) else float(score)
        print(score)
        print(f"   {tkn:>12} : {score_val:.4f}")
    print()  # blank line between categories

# --------------------------------------
# 6. Threshold Tuning on Entire Validation Set (on GPU)
# --------------------------------------
# BATCH_SIZE = 32
# all_probs = []
# all_true  = []

# print("🔧 Collecting probabilities on GPU…")
# for i in range(0, len(texts), BATCH_SIZE):
#     batch_texts  = texts[i : i + BATCH_SIZE]
#     batch_inputs = tokenize_batch(batch_texts).to(DEVICE)
#     with torch.no_grad():
#         logits = base_model(
#             input_ids=batch_inputs["input_ids"],
#             attention_mask=batch_inputs["attention_mask"]
#         ).logits  # (batch, 19)
#         probs = torch.sigmoid(logits).cpu().numpy()  # (batch, 19)
#     all_probs.append(probs)
#     all_true.append(labels[i : i + BATCH_SIZE])

# all_probs = np.vstack(all_probs)  # (N_val, 19)
# all_true  = np.vstack(all_true)   # (N_val, 19)
# print("✅ Probability collection complete.")

# def evaluate_thresholds(probs, true_labels, thresholds):
#     """
#     Returns per-category (precision, recall, F1) given a threshold vector.
#     """
#     if np.isscalar(thresholds):
#         thr = np.full(probs.shape[1], thresholds)
#     else:
#         thr = np.array(thresholds)
#     preds = (probs >= thr).astype(int)
#     precision, recall, f1, _ = precision_recall_fscore_support(
#         true_labels, preds, average=None, zero_division=0
#     )
#     return precision, recall, f1

# # 7. Sweep thresholds per category (0.1 to 0.9) with a tqdm bar
# threshold_range = np.arange(0.1, 0.91, 0.05)
# best_thresholds = {}
# metrics_at_best = {}

# print("🔧 Tuning thresholds for each category:")
# for cat_name in tqdm(harm_categories, desc="Categories"):
#     cat_idx = harm_categories.index(cat_name)
#     best_f1 = 0.0
#     best_thr = 0.5
#     best_prec, best_rec = 0.0, 0.0

#     for thr in threshold_range:
#         thr_vector = np.full(len(harm_categories), 0.5)
#         thr_vector[cat_idx] = thr
#         precision, recall, f1 = evaluate_thresholds(all_probs, all_true, thr_vector)
#         if f1[cat_idx] > best_f1:
#             best_f1 = f1[cat_idx]
#             best_thr = thr
#             best_prec = precision[cat_idx]
#             best_rec = recall[cat_idx]

#     best_thresholds[cat_name] = best_thr
#     metrics_at_best[cat_name] = {
#         "precision": best_prec,
#         "recall": best_rec,
#         "f1": best_f1
#     }

# print("✅ Threshold tuning complete.")

# # 8. Print thresholds & metrics for ALL categories
# print("\nPer-Category Threshold Selection (All Categories):")
# print(f"{'Category':<30}{'Threshold':<10}{'Precision':<10}{'Recall':<10}{'F1':<10}")
# for cat in harm_categories:
#     thr  = best_thresholds[cat]
#     prec = metrics_at_best[cat]["precision"]
#     rec  = metrics_at_best[cat]["recall"]
#     f1   = metrics_at_best[cat]["f1"]
#     print(f"{cat:<30}{thr:<10.2f}{prec:<10.2f}{rec:<10.2f}{f1:<10.2f}")



# --------------------------------------
# 7. (Optional) Map PKU Categories -> Child Taxonomy
# --------------------------------------
# Define your own mapping from PKU category names to child-focused taxonomy
# PKU_TO_CHILD = {
#     "Psychological Harm": "Psychological Harm",
#     "Insulting Behavior": "Psychological Harm",
#     "False Information": "Misinformation",
#     "Endangering Public Health": "Misinformation",
#     "Self-Harm": "Self-Harm",
#     "Sexual Content": "Sexual Content",
#     "Violence": "Physical Harm",
#     "Terrorism": "Physical Harm",
#     "Cybercrime": "Cybercrime",
#     "Bullying or Harassment": "Psychological Harm",
#     "Privacy Violation": "Privacy Violation",
#     # ... fill in remaining mappings ...
# }

# def map_to_child_taxonomy(row_labels, category_names, mapping_dict):
#     """
#     Convert a PKU multi-label vector into a set of child-focused categories.
#     """
#     child_set = set()
#     for idx, val in enumerate(row_labels):
#         if val == 1:
#             pku_cat = category_names[idx]
#             child_set.add(mapping_dict.get(pku_cat, "Other"))
#     if not child_set:
#         child_set.add("None")
#     return child_set

# # Build a list of child-taxonomy sets for all validation examples
# child_taxonomies = [
#     map_to_child_taxonomy(label_row, CATEGORY_NAMES, PKU_TO_CHILD)
#     for label_row in labels
# ]

# # Count examples per child-focused category
# from collections import Counter
# flattened = [c for sublist in child_taxonomies for c in sublist]
# child_counts = Counter(flattened)
# print("\nCounts per child-focused category (validation set):")
# for child_cat, count in child_counts.items():
#     print(f"  {child_cat:<20} : {count}")

# # Optionally: Save a pilot subset for manual labeling
# pilot_indices = np.random.choice(len(texts), size=500, replace=False)
# pilot_subset = {
#     "text": [texts[i] for i in pilot_indices],
#     "labels": [labels[i].tolist() for i in pilot_indices],
#     "child_taxonomy": [child_taxonomies[i] for i in pilot_indices]
# }
# pilot_df = pd.DataFrame(pilot_subset)
# pilot_df.to_csv("child_focused_pilot.csv", index=False)

# =============================================================================
# End of script
# =============================================================================
