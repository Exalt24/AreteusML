"""
AreteusML - ModernBERT Fine-tuning on Banking77
Run this on Kaggle with GPU T4 enabled.

Setup:
1. Create new Kaggle notebook
2. Enable GPU: Settings > Accelerator > GPU T4 x2
3. Paste this entire file into a single cell
4. Run it
5. Download the outputs from /kaggle/working/areteusml_artifacts/
"""

# ---- Cell 1: Install dependencies ----
import subprocess
subprocess.run(["pip", "install", "-q", "transformers", "accelerate", "scikit-learn", "pandas", "matplotlib"], check=True)

# ---- Cell 2: Imports ----
import gc
import json
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# ---- Config ----
SEED = 42
MODEL_NAME = "answerdotai/ModernBERT-base"
MAX_LENGTH = 64
OUTPUT_DIR = Path("/kaggle/working/areteusml_artifacts")
MODEL_DIR = OUTPUT_DIR / "model"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---- Reproducibility ----
import random
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")
else:
    print("WARNING: No GPU detected! Enable GPU in Kaggle settings.")

# ---- Dataset class ----
class Banking77Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

# ---- Weighted Trainer ----
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self._loss_fn = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self._loss_fn is None or self._loss_fn.weight.device != logits.device:
            self._loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = self._loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ---- Load data ----
print("Loading Banking77 dataset from HF parquet export...")
train_url = "https://huggingface.co/datasets/PolyAI/banking77/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
test_url = "https://huggingface.co/datasets/PolyAI/banking77/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet"

hf_train = pd.read_parquet(train_url)
hf_test = pd.read_parquet(test_url)
full_df = pd.concat([hf_train, hf_test], ignore_index=True)
print(f"Total samples: {len(full_df)}")

# Get label names from the data
label_names = [
    "activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support",
    "automatic_top_up", "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed",
    "cancel_transfer", "card_about_to_expire", "card_acceptance",
    "card_arrival", "card_delivery_estimate", "card_linking",
    "card_not_working", "card_payment_fee_charged",
    "card_payment_not_recognised", "card_payment_wrong_exchange_rate",
    "card_swallowed", "cash_withdrawal_charge", "cash_withdrawal_not_recognised",
    "change_pin", "compromised_card", "contactless_not_working",
    "country_support", "declined_card_payment", "declined_cash_withdrawal",
    "declined_transfer", "direct_debit_payment_not_recognised",
    "disposable_card_limits", "edit_personal_details",
    "exchange_charge", "exchange_rate", "exchange_via_app",
    "extra_charge_on_statement", "failed_transfer", "fiat_currency_support",
    "get_disposable_virtual_card", "get_physical_card", "getting_spare_card",
    "getting_virtual_card", "lost_or_stolen_card", "lost_or_stolen_phone",
    "order_physical_card", "passcode_forgotten", "pending_card_payment",
    "pending_cash_withdrawal", "pending_top_up", "pending_transfer",
    "pin_blocked", "receiving_money", "Refund_not_showing_up",
    "request_refund", "reverted_card_payment?", "supported_cards_and_currencies",
    "terminate_account", "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge", "top_up_by_cash_or_cheque", "top_up_failed",
    "top_up_limits", "top_up_reverted", "topping_up_by_card",
    "transaction_charged_twice", "transfer_fee_charged",
    "transfer_into_account", "transfer_not_received_by_recipient",
    "transfer_timing", "unable_to_verify_identity", "verify_my_identity",
    "verify_source_of_funds", "verify_top_up", "virtual_card_not_working",
    "visa_or_mastercard", "why_verify_identity", "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
]
num_labels = len(label_names)
print(f"Classes: {num_labels}")

# Stratified train/val/test split (70/15/15)
from sklearn.model_selection import train_test_split as sklearn_split

train_df, temp_df = sklearn_split(full_df, test_size=0.30, random_state=SEED, stratify=full_df["label"])
val_df, test_df = sklearn_split(temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["label"])

del full_df, temp_df, hf_train, hf_test
gc.collect()

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ---- Compute class weights ----
class_weights = compute_class_weight("balanced", classes=np.arange(num_labels), y=train_df["label"].values)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
print(f"Class weight range: [{class_weights.min():.3f}, {class_weights.max():.3f}]")

# ---- Tokenize ----
print("Tokenizing...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_enc = tokenizer(train_df["text"].tolist(), truncation=True, padding=True, max_length=MAX_LENGTH)
val_enc = tokenizer(val_df["text"].tolist(), truncation=True, padding=True, max_length=MAX_LENGTH)

train_dataset = Banking77Dataset(train_enc, train_df["label"].tolist())
val_dataset = Banking77Dataset(val_enc, val_df["label"].tolist())

# Free memory
del train_enc, val_enc, train_df, val_df
gc.collect()

# ---- Model ----
print("Loading model...")
id2label = {i: name for i, name in enumerate(label_names)}
label2id = {name: i for i, name in enumerate(label_names)}

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=num_labels, id2label=id2label, label2id=label2id
)

# ---- Metrics ----
def compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
    }

# ---- Training args (optimized for T4 16GB) ----
training_args = TrainingArguments(
    output_dir=str(CHECKPOINT_DIR),
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1_macro",
    greater_is_better=True,
    save_total_limit=2,
    save_only_model=True,
    seed=SEED,
    report_to="none",
    dataloader_num_workers=2,
)

# ---- Train ----
model.gradient_checkpointing_enable()
trainer = WeightedTrainer(
    class_weights=class_weights_tensor,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    compute_metrics=compute_metrics_fn,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

if torch.cuda.is_available():
    alloc = torch.cuda.memory_allocated() / 1024**3
    print(f"VRAM before training: {alloc:.2f} GB")

print("Starting training...")
trainer.train()
print("Training complete!")

if torch.cuda.is_available():
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Peak VRAM: {peak:.2f} GB")

# ---- Save best model ----
trainer.save_model(str(MODEL_DIR))
tokenizer.save_pretrained(str(MODEL_DIR))
print(f"Model saved to {MODEL_DIR}")

# ---- Evaluate on test set ----
print("Evaluating on test set...")
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

test_enc = tokenizer(test_df["text"].tolist(), truncation=True, padding=True, max_length=MAX_LENGTH)
test_dataset = Banking77Dataset(test_enc, test_df["label"].tolist())
test_preds = trainer.predict(test_dataset)

y_pred = np.argmax(test_preds.predictions, axis=-1)
y_true = test_df["label"].values
y_prob = torch.softmax(torch.tensor(test_preds.predictions), dim=-1).numpy()

# Metrics
test_acc = accuracy_score(y_true, y_pred)
test_f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
test_f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
test_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
test_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

print(f"\n{'='*50}")
print(f"TEST RESULTS")
print(f"{'='*50}")
print(f"Accuracy:         {test_acc:.4f}")
print(f"F1 (macro):       {test_f1_macro:.4f}")
print(f"F1 (weighted):    {test_f1_weighted:.4f}")
print(f"Precision (macro): {test_precision:.4f}")
print(f"Recall (macro):   {test_recall:.4f}")
print(f"{'='*50}\n")

# Save metrics JSON
metrics = {
    "accuracy": float(test_acc),
    "f1_macro": float(test_f1_macro),
    "f1_weighted": float(test_f1_weighted),
    "precision_macro": float(test_precision),
    "recall_macro": float(test_recall),
}
with open(OUTPUT_DIR / "test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# ---- Confusion matrix ----
cm = confusion_matrix(y_true, y_pred)
fig_size = max(10, num_labels * 0.25)
fig, ax = plt.subplots(figsize=(fig_size, fig_size))
im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set(
    xticks=np.arange(num_labels),
    yticks=np.arange(num_labels),
    xticklabels=label_names,
    yticklabels=label_names,
    ylabel="True label",
    xlabel="Predicted label",
    title="Confusion Matrix",
)
plt.setp(ax.get_xticklabels(), rotation=90, ha="right", fontsize=5)
plt.setp(ax.get_yticklabels(), fontsize=5)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "test_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close("all")
print("Saved confusion matrix")

# ---- Per-class report ----
report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).T.loc[label_names]
report_df.to_csv(OUTPUT_DIR / "test_per_class_report.csv")
print("Saved per-class report")

# ---- Training history ----
history_df = pd.DataFrame(trainer.state.log_history)
history_df.to_csv(OUTPUT_DIR / "training_history.csv", index=False)
print("Saved training history")

# ---- Cleanup checkpoints ----
if CHECKPOINT_DIR.exists():
    shutil.rmtree(CHECKPOINT_DIR)
    print("Cleaned up checkpoints")

# ---- Summary ----
print(f"\n{'='*50}")
print(f"All artifacts saved to {OUTPUT_DIR}/")
print(f"Files:")
for f in sorted(OUTPUT_DIR.rglob("*")):
    if f.is_file():
        size_mb = f.stat().st_size / 1024**2
        print(f"  {f.relative_to(OUTPUT_DIR)} ({size_mb:.1f} MB)")
print(f"{'='*50}")
print(f"\nDownload the 'model/' folder and place it at:")
print(f"  AreteusML/ml/models/production/")
