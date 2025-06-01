import torch
import numpy as np
import shap
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm  # for progress bars
raw_dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", "default", split="train")
split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
val_data = split_dataset["test"]

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

def tokenize_batch(text_list, max_length=256):
    return tokenizer(
        text_list,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

# SHAP interpretability
positive_indices = np.where(labels.sum(axis=1) > 0)[0]
sample_indices = positive_indices[:5].tolist()
texts_sample = [texts[i] for i in sample_indices]

background_indices = np.random.choice(len(texts), size=5, replace=False)
background_texts = [texts[i] for i in background_indices]
background_inputs = tokenize_batch(background_texts).to(DEVICE)
sample_inputs     = tokenize_batch(texts_sample).to(DEVICE)

background_ids   = background_inputs["input_ids"]
background_att   = background_inputs["attention_mask"].float()

sample_ids       = sample_inputs["input_ids"]        
sample_att       = sample_inputs["attention_mask"].float()  

class BertForSHAP_GPU(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, inputs_embeds, attention_mask):
        logits = self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        ).logits
        return logits 

wrapped_model_gpu = BertForSHAP_GPU(base_model).to(DEVICE)
wrapped_model_gpu.eval()

with torch.no_grad():
    background_embeds = base_model.bert.embeddings(background_ids)  
    sample_embeds     = base_model.bert.embeddings(sample_ids)      

background_data = [background_embeds, background_att]
explainer = shap.GradientExplainer(wrapped_model_gpu, background_data)

sample_data = [sample_embeds, sample_att]
shap_values = explainer.shap_values(sample_data)

shap_values_dict = {f"cat_{i}": shap_values[i] for i in range(len(shap_values))}
np.savez("shap_values.npz", **shap_values_dict)

token_level_importances = []
for cat_idx in range(len(shap_values)):
    vals = shap_values[cat_idx]              
    per_token = np.linalg.norm(vals, axis=-1)  
    token_level_importances.append(per_token)

sample_idx = 0
print("\nTop‑5 SHAP tokens for sample #0 in each category:\n")
for cat_idx, cat_name in enumerate(harm_categories):
    importances = token_level_importances[cat_idx][sample_idx]   
    token_ids    = sample_ids[sample_idx].cpu().numpy()          
    tokens       = tokenizer.convert_ids_to_tokens(token_ids)    

    importances = importances.tolist()  
    token_imp = list(zip(tokens, importances))
    token_imp_sorted = sorted(token_imp, key=lambda x: x[1], reverse=True)

    top5 = token_imp_sorted[:5]

    print(f"Category: {cat_name}")
    for tkn, score in top5:
        score_val = float(score[0]) if isinstance(score, list) else float(score)
        print(score)
        print(f"   {tkn:>12} : {score_val:.4f}")
    print()

BATCH_SIZE = 32
all_probs = []
all_true  = []

for i in range(0, len(texts), BATCH_SIZE):
    batch_texts  = texts[i : i + BATCH_SIZE]
    batch_inputs = tokenize_batch(batch_texts).to(DEVICE)
    with torch.no_grad():
        logits = base_model(
            input_ids=batch_inputs["input_ids"],
            attention_mask=batch_inputs["attention_mask"]
        ).logits
        probs = torch.sigmoid(logits).cpu().numpy()
    all_probs.append(probs)
    all_true.append(labels[i : i + BATCH_SIZE])

all_probs = np.vstack(all_probs)
all_true  = np.vstack(all_true)

def evaluate_thresholds(probs, true_labels, thresholds):
    if np.isscalar(thresholds):
        thr = np.full(probs.shape[1], thresholds)
    else:
        thr = np.array(thresholds)
    preds = (probs >= thr).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, preds, average=None, zero_division=0
    )
    return precision, recall, f1

threshold_range = np.arange(0.1, 0.91, 0.05)
best_thresholds = {}
metrics_at_best = {}

for cat_name in tqdm(harm_categories, desc="Categories"):
    cat_idx = harm_categories.index(cat_name)
    best_f1 = 0.0
    best_thr = 0.5
    best_prec, best_rec = 0.0, 0.0

    for thr in threshold_range:
        thr_vector = np.full(len(harm_categories), 0.5)
        thr_vector[cat_idx] = thr
        precision, recall, f1 = evaluate_thresholds(all_probs, all_true, thr_vector)
        if f1[cat_idx] > best_f1:
            best_f1 = f1[cat_idx]
            best_thr = thr
            best_prec = precision[cat_idx]
            best_rec = recall[cat_idx]

    best_thresholds[cat_name] = best_thr
    metrics_at_best[cat_name] = {
        "precision": best_prec,
        "recall": best_rec,
        "f1": best_f1
    }

print("\nPer-Category Threshold Selection (All Categories):")
print(f"{'Category':<30}{'Threshold':<10}{'Precision':<10}{'Recall':<10}{'F1':<10}")
for cat in harm_categories:
    thr  = best_thresholds[cat]
    prec = metrics_at_best[cat]["precision"]
    rec  = metrics_at_best[cat]["recall"]
    f1   = metrics_at_best[cat]["f1"]
    print(f"{cat:<30}{thr:<10.2f}{prec:<10.2f}{rec:<10.2f}{f1:<10.2f}")
