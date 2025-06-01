import torch
import numpy as np
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

mapping = {
    "Endangering National Security": "Danger",
    "Insulting Behavior": "Profanities",
    "Discriminatory Behavior": "Hateful",
    "Endangering Public Health": "Danger",
    "Copyright Issues": "Danger",
    "Violence": "Danger",
    "Drugs": "Substance use",
    "Privacy Violation": "Danger",
    "Economic Crime": "Danger",
    "Mental Manipulation": "Psychological",
    "Human Trafficking": "Danger",
    "Physical Harm": "Danger",
    "Sexual Content": "Sexual",
    "Cybercrime": "Danger",
    "Disrupting Public Order": "Danger",
    "Environmental Damage": "Danger",
    "Psychological Harm": "Psychological",
    "White-Collar Crime": "Danger",
    "Animal Abuse": "Danger"
}
harm_categories = [
    "Endangering National Security","Insulting Behavior","Discriminatory Behavior",
    "Endangering Public Health","Copyright Issues","Violence","Drugs",
    "Privacy Violation","Economic Crime","Mental Manipulation","Human Trafficking",
    "Physical Harm","Sexual Content","Cybercrime","Disrupting Public Order",
    "Environmental Damage","Psychological Harm","White-Collar Crime","Animal Abuse"
]
mb_categories = ["Sexual", "Profanities", "Hateful", "Danger", "Self-harm", "Substance use"]

def prepare_minorbench_splits(test_size: float = 0.2, seed: int = 42):
    mb = load_dataset("govtech/MinorBench", split="test")
    mb = mb.shuffle(seed=seed)
    split = mb.train_test_split(test_size=test_size, seed=seed)
    return DatasetDict({'dev': split['train'], 'test': split['test']})

def get_raw_scores(model, tokenizer, dataset, batch_size=16, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    texts = dataset['Prompt']
    cats = dataset.features['Category'].names if hasattr(dataset.features['Category'], 'names') else sorted(set(dataset['Category']))
    mb_index = {c: i for i, c in enumerate(cats)}
    all_true, all_scores = [], []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start+batch_size]
        enc = tokenizer(batch_texts, truncation=True, padding=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad(): logits = model(**enc).logits
        probs = torch.sigmoid(logits).cpu().numpy()
        all_scores.append(probs)
        true_vecs = [[1 if i == mb_index[c] else 0 for i in range(len(cats))]
                     for c in dataset['Category'][start:start+batch_size]]
        all_true.extend(true_vecs)
    return np.vstack(all_true), np.vstack(all_scores)

def map_to_minorbench(pku_preds, harm_categories, mapping, mb_categories):
    idx = {c: i for i, c in enumerate(mb_categories)}
    results = []
    for p in pku_preds:
        out = [0]*len(mb_categories)
        for j, flag in enumerate(p):
            if flag:
                m = mapping[harm_categories[j]]  # direct lookup
                if m in idx:
                    out[idx[m]] = 1
        results.append(out)
    return np.array(results)

def error_analysis(true, pred, dataset, mb_categories):
    fn = np.where((true == 1) & (pred == 0))
    fp = np.where((true == 0) & (pred == 1))
    return {
        'false_negatives': [{'idx': i, 'label': mb_categories[j], 'prompt': dataset[i]['Prompt']} for i, j in zip(*fn)],
        'false_positives': [{'idx': i, 'label': mb_categories[j], 'prompt': dataset[i]['Prompt']} for i, j in zip(*fp)]
    }

def main():
    splits = prepare_minorbench_splits()
    dev, test = splits['dev'], splits['test']
    tokenizer = BertTokenizer.from_pretrained("./harm_bert")
    model = BertForSequenceClassification.from_pretrained("./harm_bert")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    true_test, scores_test = get_raw_scores(model, tokenizer, test, batch_size=16, device=device)
    preds = scores_test.astype(int)
    mapped = map_to_minorbench(preds, harm_categories, mapping, mb_categories)
    print("Results:")
    print("Micro F1:", f1_score(true_test, mapped, average='micro'))
    print("Macro F1:", f1_score(true_test, mapped, average='macro'))
    print("Precision (μ):", precision_score(true_test, mapped, average='micro'))
    print("Recall (μ):", recall_score(true_test, mapped, average='micro'))
    errors = error_analysis(true_test, mapped, test, mb_categories)
    print(f"FNs: {len(errors['false_negatives'])}, FPs: {len(errors['false_positives'])}")

if __name__ == '__main__':
    main()
