# Import standard libraries: pandas, numpy, re, os
import pandas as pd
import numpy as np
import re
import os

# Import NLP libraries: nltk, spacy
import nltk
import spacy

# Import NRCLex for emotion lexicon lookup
from nrclex import NRCLex

# Import sklearn components
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (classification_report, confusion_matrix, f1_score, accuracy_score)
from sklearn.utils.class_weight import compute_class_weight

# Import scipy
from scipy.sparse import hstack, csr_matrix

# Import visualization libraries: matplotlib, seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Import emoji library for demojizing
import emoji

# Import datasets from HuggingFace
from datasets import load_dataset

# Load spaCy English model for lemmatization
nlp = spacy.load("en_core_web_sm")

# Download required NLTK data packages:
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")

# load the GoEmotions simplified dataset from HuggingFace
# Specify cache_dir to save to local data/ folder
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

# Apply the mapping function to all three splits
for df in [train_df, val_df, test_df]:
    df["ekman_label"] = df["labels"].apply(
        lambda x: map_to_ekmans(x, id_emotions)
    )

# Drop rows where the mapped label is None (neutral examples)
train_df = train_df.dropna(subset=["ekman_label"])
val_df = val_df.dropna(subset=["ekman_label"])
test_df = test_df.dropna(subset=["ekman_label"])


# Generate and save a bar chart of class distribution
# Save to outputs/figures/class_distribution.png
train_df["ekman_label"].value_counts().plot(kind="bar", color="steelblue")
plt.title("GoEmotions: Ekman label distribution (train)")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/figures/class_distribution.png", dpi=150)
plt.close()



# preprocessing
def preprocessing_nrc(text):
    # demojize
    if emoji.is_emoji(text):
        text = emoji.demojize(text)

    # remove urls
    text = re.sub(r"http\S+|www\S+", "", text)

    # @user -> ""
    text = re.sub(r"@\w+", "", text)

    # segment hashtags
    text = re.sub(r"#(\w+)", lambda m: " ".join(re.findall(r'[A-Z][a-z]*|[a-z]+', m.group(1))).lower(), text)

    contractions = {
        "don't": "do not",
        "can't": "cannot",
        "won't": "will not",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "doesn't": "does not",
        "didn't": "did not",
        "haven't": "have not",
        "hasn't": "has not",
        "hadn't": "had not",
        "wouldn't": "would not",
        "couldn't": "could not",
        "shouldn't": "should not",
        "i'm": "i am",
        "i've": "i have",
        "i'll": "i will",
        "i'd": "i would",
        "it's": "it is",
        "that's": "that is",
        "there's": "there is"
    }

    # normalize elongated words looong -> long
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    text = text.lower()

    # lemmatization
    doc = nlp(text)
    text = " ".join([token.lemma_ for token in doc if not token.is_space])

    return text.strip()


# apply preprocessed text
for df in [train_df, val_df, test_df]:
    df["clean_text"] = df["text"].apply(preprocessing_nrc)

# feature extraction section
ALL_NRC_EMOTIONS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "anticipation",
    "trust",
    "positive",
    "negative"
]


def extract_nrc_features(text):
    emotion = NRCLex()
    emotion.load_raw_text(text)

    # get frequency dictionary
    freqs = emotion.affect_frequencies

    return [freqs.get(e, 0.0) for e in ALL_NRC_EMOTIONS]


def extract_punctuation_features(text):
    exclaim = text.count("!")
    question = text.count("?")
    caps = sum(1 for c in text if c.isupper())
    total = max(len(text), 1)
    return [exclaim, question, caps / total]


def extract_elongation_features(original_text):
    matches = re.findall(r'\b\w*(.)\1{2,}\w*\b', original_text)
    return [len(matches)]


def build_feature_matrix(df):
    nrc_features = np.array([extract_nrc_features(t) for t in df["clean_text"]])
    punctuation_features = np.array([extract_punctuation_features(t) for t in df["text"]])
    elongation_features = np.array([extract_elongation_features(t) for t in df["text"]])
    return np.hstack([nrc_features, punctuation_features, elongation_features])

# Initialize a TfidfVectorizer
# Use unigrams and bigrams (ngram_range=(1, 2))
# Limit to top 10,000 features
# Set minimum document frequency to 2
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=10000,
    min_df=2
)

X_train_tfidf = tfidf.fit_transform(train_df["clean_text"])
X_val_tfidf = tfidf.transform(val_df["clean_text"])
X_test_tfidf = tfidf.transform(test_df["clean_text"])


# combine features and encode labels
# fit the vectorizer on the training clean_text only
# transform train, validation, and test sets
X_train_manual = csr_matrix(build_feature_matrix(train_df))
X_val_manual = csr_matrix(build_feature_matrix(val_df))
X_test_manual = csr_matrix(build_feature_matrix(test_df))

X_train = hstack([X_train_tfidf, X_train_manual])
X_val = hstack([X_val_tfidf,   X_val_manual])
X_test = hstack([X_test_tfidf,  X_test_manual])

y_train = train_df["ekman_label"].values
y_val = val_df["ekman_label"].values
y_test = test_df["ekman_label"].values

# Compute class weights for imbalance
classes = np.unique(y_train)
weights = compute_class_weight("balanced", classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))

# training with logisticregression
lr_model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    class_weight=class_weight_dict,
    solver="lbfgs"
)
lr_model.fit(X_train, y_train)

val_preds = lr_model.predict(X_val)
print("Validation macro-F1:", f1_score(y_val, val_preds, average="macro"))

test_preds_nrc = lr_model.predict(X_test)

# classification report
report = classification_report(
    y_test, test_preds_nrc,
    target_names=EMOTIONS,
    output_dict=True
)
report_df = pd.DataFrame(report).T
report_df.to_csv("outputs/results/nrclex_classification_report.csv")
print(report_df)

# Confusion matrix
cm = confusion_matrix(y_test, test_preds_nrc, labels=EMOTIONS, normalize="true")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt=".2f", xticklabels=EMOTIONS, yticklabels=EMOTIONS, cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("NRCLex + Logistic Regression  Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/figures/nrclex_confusion_matrix.png", dpi=150)
plt.close()

# Save predictions for error analysis
test_df["nrclex_pred"] = test_preds_nrc
test_df[["text", "clean_text", "ekman_label", "nrclex_pred"]].to_csv("outputs/results/nrclex_predictions.csv", index=False)

# Generate predictions on the test feature matrix
test_preds = lr_model.predict(X_test)

report = classification_report(y_test, test_preds, target_names=EMOTIONS, output_dict=True)
report_df = pd.DataFrame(report).T
print(report_df)
os.makedirs("outputs/results", exist_ok=True)
report_df.to_csv("outputs/results/nrclex_classification_report.csv")

cm = confusion_matrix(y_test, test_preds, labels=EMOTIONS, normalize="true")

os.makedirs("outputs/figures", exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt=".2f", xticklabels=EMOTIONS, yticklabels=EMOTIONS, cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("NRCLex + Logistic Regression  Normalized Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/figures/nrclex_confusion_matrix.png", dpi=150)
plt.close()
print("Confusion matrix saved.")

test_df["nrclex_pred"] = test_preds
test_df[["text", "clean_text", "ekman_label", "nrclex_pred"]].to_csv("outputs/results/nrclex_predictions.csv",index=False)
print("Predictions saved to outputs/results/nrclex_predictions.csv")

#   macro-F1, micro-F1, and accuracy
macro_f1 = f1_score(y_test, test_preds, average="macro")
micro_f1 = f1_score(y_test, test_preds, average="micro")
accuracy = accuracy_score(y_test, test_preds)

print("\n--- NRCLex + Logistic Regression Final Test Metrics ---")
print(f" Macro-F1: {macro_f1:.4f}")
print(f" Micro-F1: {micro_f1:.4f}")
print(f" Accuracy: {accuracy:.4f}")