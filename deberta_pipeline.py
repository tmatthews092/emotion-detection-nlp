# Import standard libraries: numpy, pandas, re, os
import numpy as np
import pandas as pd
import re
import os

# Import PyTorch: torch, torch.nn
import torch
import torch.nn as nn

# Import HuggingFace libraries:
from transformers import (AutoTokenizer, TrainingArguments, Trainer,
                          EarlyStoppingCallback, DebertaV2ForSequenceClassification)

# Dataset (from datasets library)
from datasets import load_dataset, Dataset


# Import sklearn metrics: f1_score, classification_report, confusion_matrix
from sklearn.metrics import (classification_report, confusion_matrix, f1_score)
from sklearn.utils.class_weight import compute_class_weight

# Import visualization: matplotlib, seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Import emoji for preprocessing
import emoji

# Define constants
MODEL_NAME = "microsoft/deberta-v3-base"
NUM_LABELS = 6
EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
LABEL2ID = {e: i for i, e in enumerate(EMOTIONS)}
ID2LABEL = {i: e for i, e in enumerate(EMOTIONS)}
MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# load GoEmotions simplified dataset from HuggingFace
# specify cache_dir to use local data/ folder
dataset = load_dataset("google-research-datasets/go_emotions", "simplified", cache_dir="./data")

# Convert train, validation, and test splits to pandas DataFrames
train_df = pd.DataFrame(dataset["train"])
val_df = pd.DataFrame(dataset["validation"])
test_df = pd.DataFrame(dataset["test"])

# Maps each of the 6 Ekman emotions to its GoEmotions sub-labels
ekman_map = {
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": [
        "joy", "amusement", "approval", "excitement", "gratitude", "love", "optimism", "relief", "pride", "admiration",
        "desire", "caring"
    ],
    "sadness":  ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"]
}
EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]


def map_to_ekmans(label_ids, id_to_label):
    # label_ids from dataset is numbers as positions in the id_emotions array
    labels = [id_to_label[i] for i in label_ids]
    for ekman_emotion, sub_labels in ekman_map.items():
        if any(label in sub_labels for label in labels):  # if the label is in the sub_label group return parent
            return ekman_emotion
    return None  # no match


# ['admiration', 'amusement'], position in array is the id of the emotion
id_emotions = dataset["train"].features["labels"].feature.names

for df in [train_df, val_df, test_df]:
    df["ekman_label"] = df["labels"].apply(
        lambda x: map_to_ekmans(x, id_emotions)
    )

train_df = train_df.dropna(subset=["ekman_label"]).reset_index(drop=True)
val_df = val_df.dropna(subset=["ekman_label"]).reset_index(drop=True)
test_df = test_df.dropna(subset=["ekman_label"]).reset_index(drop=True)

for df in [train_df, val_df, test_df]:
    df["label_id"] = df["ekman_label"].map(LABEL2ID).astype(int)

assert train_df["label_id"].isnull().sum() == 0, "NaN label_ids in train"
assert val_df["label_id"].isnull().sum() == 0,   "NaN label_ids in val"
assert test_df["label_id"].isnull().sum() == 0,  "NaN label_ids in test"
print("Label verification passed  no nulls")
print(train_df["label_id"].value_counts().sort_index())


# preproocessing section
def preprocess_deberta(text):
    text = emoji.demojize(text, delimiters=(" ", " "))

    text = re.sub(r"http\S+|www\S+", "[URL]", text)

    text = re.sub(r"@\w+", "@USER", text)

    text = re.sub(r"#(\w+)", lambda m: " ".join(
        re.findall(r'[A-Z][a-z]*|[a-z]+', m.group(1))
    ), text)

    text = re.sub(r'(.)\1{3,}', r'\1\1\1', text)

    return text.strip()


for df in [train_df, val_df, test_df]:
    df["clean_text"] = df["text"].apply(preprocess_deberta)

# tokenizer setup

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# register GoEmotions special tokens
special_tokens = {"additional_special_tokens": ["[NAME]", "[RELIGION]"]}
tokenizer.add_special_tokens(special_tokens)

def tokenize(batch):
    return tokenizer(
        batch["clean_text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )


def make_hf_dataset(df):
    ds = Dataset.from_dict({
        "clean_text": df["clean_text"].tolist(),
        "labels": df["label_id"].astype(int).tolist()  # force int
    })
    return ds.map(tokenize, batched=True)


train_ds = make_hf_dataset(train_df)
val_ds = make_hf_dataset(val_df)
test_ds = make_hf_dataset(test_df)

# Set format for PyTorch
cols = ["input_ids", "attention_mask", "token_type_ids", "labels"]
for ds in [train_ds, val_ds, test_ds]:
    ds.set_format(type="torch", columns=[c for c in cols if c in ds.column_names])

# model and WeightedTrainer section

classes = np.array([0, 1, 2, 3, 4, 5])
weights = compute_class_weight(
    "balanced",
    classes=classes,
    y=train_df["label_id"].values
)
# keep on CPU as float32 cast inside compute_loss
class_weights_cpu = torch.tensor(weights, dtype=torch.float32)


def load_fresh_model():
    m = DebertaV2ForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
        torch_dtype=torch.float32 # force float32  prevents NaN from half precision
    )
    m.resize_token_embeddings(len(tokenizer))
    return m.to(DEVICE)


# WeightedTrainer class replacing the default trainer from deberta
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # move weights to same device and dtype as logits
        w = class_weights_cpu.to(device=logits.device, dtype=logits.dtype)
        loss = nn.CrossEntropyLoss(weight=w)(logits, labels.long())

        if torch.isnan(loss):
            print(f"WARNING: NaN loss  logits dtype: {logits.dtype}, "
                  f"range: {logits.min().item():.3f} to {logits.max().item():.3f}")

        return (loss, outputs) if return_outputs else loss



# training section
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "macro_f1": f1_score(labels, preds, average="macro"),
        "micro_f1": f1_score(labels, preds, average="micro"),
        "accuracy": float((preds == labels).mean())
    }


# training arguments factory
def make_training_args(output_dir, epochs, save_strategy="epoch",
                       load_best=True):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy=save_strategy,
        load_best_model_at_end=load_best,
        metric_for_best_model="eval_macro_f1",
        max_grad_norm=1.0,
        logging_steps=10,
        fp16=True,
        bf16=False, # must be False
        report_to="none"
    )


full_model = load_fresh_model()

trainer = WeightedTrainer(
    model=full_model,
    args=make_training_args(
        "outputs/models/deberta",
        epochs=EPOCHS
    ),
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()

# save training loss curve
log_history = pd.DataFrame(trainer.state.log_history)
train_logs = log_history.dropna(subset=["loss"])

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(train_logs["step"], train_logs["loss"], label="Training loss")
ax.set_xlabel("Step")
ax.set_ylabel("Loss")
ax.set_title("DeBERTa-v3 training loss")
ax.legend()
plt.tight_layout()
os.makedirs("outputs/figures", exist_ok=True)
plt.savefig("outputs/figures/deberta_training_loss.png", dpi=150)
plt.close()

# evalution
predictions = trainer.predict(test_ds)
test_preds_deberta = np.argmax(predictions.predictions, axis=-1)
y_test = test_df["label_id"].values

report = classification_report(y_test, test_preds_deberta, target_names=EMOTIONS, output_dict=True)
report_df = pd.DataFrame(report).T
os.makedirs("outputs/results", exist_ok=True)
report_df.to_csv("outputs/results/deberta_classification_report.csv")
print(report_df)

cm = confusion_matrix(y_test, test_preds_deberta, normalize="true")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt=".2f", xticklabels=EMOTIONS, yticklabels=EMOTIONS, cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("DeBERTa-v3  Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/figures/deberta_confusion_matrix.png", dpi=150)
plt.close()

test_df["deberta_pred_id"] = test_preds_deberta
test_df["deberta_pred"] = test_df["deberta_pred_id"].map(ID2LABEL)
test_df[["text", "ekman_label", "deberta_pred"]].to_csv("outputs/results/deberta_predictions.csv", index=False)
print("Predictions saved.")

macro_f1 = f1_score(y_test, test_preds_deberta, average="macro")
micro_f1 = f1_score(y_test, test_preds_deberta, average="micro")
accuracy = float((test_preds_deberta == y_test).mean())
print(f"\n--- Final Test Metrics ---")
print(f"Macro-F1:  {macro_f1:.4f}")
print(f"Micro-F1:  {micro_f1:.4f}")
print(f"Accuracy:  {accuracy:.4f}")