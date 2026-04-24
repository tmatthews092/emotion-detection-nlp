# Emotion Detection in Text: NRC EmoLex vs DeBERTa-v3

**Author:** Theodore Matthews \
**Course:** CSPB 3828 Natural Language Processing \
**Date:** April 2026

## Overview

This project compare the two approaches to emotion detection on the GoEmotions dataset:
**NRC EmoLex + Logistic Regression**: lexicon-based pipeline using emotion word matching
**DeBERTa-v3 fine-tuned**: a transformer-based pipeline using contextual representations

The 27 GoEmotions labels are consolidated to Ekman's six universal emotions: anger, disgust, fear, joy, sadness, and surprise

---

### 1. Install dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```
**Note** The cu version will depend on the GPU version you have.

If you do not have a GPU, then use the CPU version.
```bash
pip install torch torchvision torchaudio
```

```bash
pip install transformers datasets accelerate evaluate
pip install scikit-learn numpy pandas scipy
pip install NRCLex spacy emoji nltk
pip install matplotlib seaborn
pip install mlxtend sentencepiece protobuf
```

### 2. Download required models and data packages

```bash
# spaCy English model for lemmatization
python -m spacy download en_core_web_sm
```

NLTK data packages are downloaded automatically on first run. If you prefer to download them manually:

```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
```

### 3. Create output directories

```bash
mkdir -p outputs/figures outputs/results outputs/models/deberta
```

---

## Running the Project

Run the three files in order. Each file depends on the outputs from the previous one.

### 1. NRCLex Pipeline

```bash
python nrclex_pipeline.py
```

**This pipeline**:
- Downloads GoEmotions from HuggingFace and caches it in the data/ folder
- Preprocesses text: demojize, URL remove, contraction expansion, lemmatization
- Extracts NRC emotion features, TF-IDF features, punctuation and elongation counts
- Trains logistic regression with class weights
- Evaluates on test set and saves results

**Note**: This should take around 5-10 minutes to complete.

**Outputs** \
outputs/figures/class_distribution.png \
outputs/figures/nrclex_confusion_matrix.png \
outputs/results/nrclex_classification_report.csv \
outputs/results/nrclex_predictions.csv

### 2. DeBERTa-v3

```bash
python deberta_pipeline.py
```

**This pipeline**:
- Downloads `microsoft/deberta-v3-base` from HuggingFace
- Preprocesses text: demojize, URL -> `[URL]`, hashtag segmentation
- Tokenizes using SentencePiece tokenizer (max length 128)
- Fine-tunes DeBERTa-v3 using WeightedTrainer with class-weighted cross-entropy loss
- Evaluates on test set and saves results

**Note**: I used a RTX 3060, which took ~70 minutes with fp16 = true 

**Outputs:** \
outputs/figures/deberta_training_loss.png \
outputs/figures/deberta_confusion_matrix.png \
outputs/results/deberta_classification_report.csv \
outputs/results/deberta_predictions.csv \
outputs/models/deberta/

### 3. Error Analysis

```bash
python error_analysis.py
```

**Outputs:** \
outputs/figures/per_class_f1_comparison.png \
outputs/results/model_comparison.csv \
outputs/results/negation_errors.csv \
outputs/results/implicit_emotion_examples.csv \
outputs/results/nrclex_confusion_pairs.csv

### Folders

**data**: GoEmotion dataset \
**outputs/figures**: generated images from the predicted results. \
**outputs/models**: deberta-v3 checkpoints, which are snapshots of the model weights at the end of each epoch. \
**outputs/results**: csvs of predicted results, negation errors, implicit emotion examples and nrclex confusion pairs.
