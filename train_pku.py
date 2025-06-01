import math
import logging
from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from datasets import load_dataset, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

raw_dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", "default")["train"]
split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
train_data = split_dataset["train"]
val_data = split_dataset["test"]

harm_categories = [
    "Endangering National Security", "Insulting Behavior", "Discriminatory Behavior", "Endangering Public Health",
    "Copyright Issues", "Violence", "Drugs", "Privacy Violation", "Economic Crime", "Mental Manipulation",
    "Human Trafficking", "Physical Harm", "Sexual Content", "Cybercrime", "Disrupting Public Order",
    "Environmental Damage", "Psychological Harm", "White-Collar Crime", "Animal Abuse"
]

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# multi-label encoding function
def preprocess(example):
    # text input: prompt + response_0
    text = example["prompt"] + " [SEP] " + example["response_0"]
    encoding = tokenizer(text, truncation=True, padding="max_length", max_length=512)
    # multi-label binary vector
    label_vec = [int(example["response_0_harm_category"].get(cat, False)) for cat in harm_categories]
    encoding["labels"] = label_vec
    return encoding

train_enc = train_data.map(preprocess, batched=False)
val_enc = val_data.map(preprocess, batched=False)

train_enc.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_enc.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# BERT for multi-label classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=19, 
                                                      problem_type="multi_label_classification")

class TrainerConfig:
    def __init__(self, **kwargs):
        # optimization parameters
        self.max_epochs = kwargs.get("max_epochs", 3)
        self.batch_size = kwargs.get("batch_size", 8)
        self.learning_rate = kwargs.get("learning_rate", 2e-5)
        self.weight_decay = kwargs.get("weight_decay", 0.01)
        self.grad_norm_clip = kwargs.get("grad_norm_clip", 1.0)
        # lr decay parameters (set False if not used)
        self.lr_decay = kwargs.get("lr_decay", False)
        self.warmup_tokens = kwargs.get("warmup_tokens", 375e6)
        self.final_tokens = kwargs.get("final_tokens", 260e9)
        # checkpoint and dataloader parameters
        self.ckpt_path = kwargs.get("ckpt_path", None)
        self.num_workers = kwargs.get("num_workers", 0)
        self.writer = kwargs.get("writer", None)

class Trainer:
    def __init__(self, model, train_dataset, dev_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.config = config

        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = self.model.to(self.device)
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.model = self.model.to(self.device)
        else:
            self.device = "cpu"
            self.model = self.model.to(self.device)
        self.tokens = 0 

    def save_checkpoint(self):
        if self.config.ckpt_path is not None:
            ckpt_model = self.model.module if hasattr(self.model, "module") else self.model
            logger.info("Saving checkpoint to %s", self.config.ckpt_path)
            torch.save(ckpt_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        step = 0

        for epoch in range(config.max_epochs):
            model.train()
            train_loader = DataLoader(self.train_dataset, batch_size=config.batch_size,
                                      shuffle=True, num_workers=config.num_workers)
            pbar = tqdm(enumerate(train_loader), total=len(train_loader))
            for it, batch in pbar:
                batch = {k: (v.to(self.device) if k != "labels" else v.to(self.device).float()) for k, v in batch.items()}
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                optimizer.step()

                if config.lr_decay:
                    self.tokens += (batch["attention_mask"] >= 0).sum().item()
                    if self.tokens < config.warmup_tokens:
                        lr_mult = float(self.tokens) / max(1, config.warmup_tokens)
                    else:
                        progress = float(self.tokens - config.warmup_tokens) / max(1, config.final_tokens - config.warmup_tokens)
                        lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    lr = config.learning_rate * lr_mult
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    lr = config.learning_rate

                pbar.set_description(f"Epoch {epoch+1} Iter {it}: Train loss {loss.item():.5f}, lr {lr:e}")
                if config.writer is not None:
                    config.writer.add_scalar('train/loss', loss.item(), step)
                    config.writer.add_scalar('train/lr', lr, step)
                step += 1

            if self.dev_dataset is not None:
                self.evaluate()
            self.save_checkpoint()

    def evaluate(self):
        model = self.model
        model.eval()
        dev_loader = DataLoader(self.dev_dataset, batch_size=self.config.batch_size,
                                shuffle=False, num_workers=self.config.num_workers)
        losses = []
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc="Evaluating"):
                batch = {k: (v.to(self.device) if k != "labels" else v.to(self.device).float()) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                losses.append(loss.item())
                all_logits.append(outputs.logits.cpu())
                all_labels.append(batch['labels'].cpu())
        avg_loss = np.mean(losses)
        logger.info(f"Dev Loss: {avg_loss:.5f}")
        print(f"Dev Loss: {avg_loss:.5f}")
        all_logits = torch.cat(all_logits, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
        preds = (probs >= 0.5).astype(int)
        f1_micro = f1_score(all_labels, preds, average="micro")
        precision_micro = precision_score(all_labels, preds, average="micro")
        recall_micro = recall_score(all_labels, preds, average="micro")
        logger.info(f"Metrics - f1_micro: {f1_micro:.5f}, precision_micro: {precision_micro:.5f}, recall_micro: {recall_micro:.5f}")
        print(f"Metrics - f1_micro: {f1_micro:.5f}, precision_micro: {precision_micro:.5f}, recall_micro: {recall_micro:.5f}")
        model.train()


trainer_config = TrainerConfig(
    max_epochs=3,
    batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    grad_norm_clip=1.0,
    num_workers=0,
    ckpt_path=None,
    lr_decay=False
)

trainer = Trainer(
    model=model,
    train_dataset=train_enc,
    dev_dataset=val_enc,
    config=trainer_config
)

trainer.train()

model.save_pretrained("./harm_bert")
tokenizer.save_pretrained("./harm_bert")
