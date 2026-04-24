import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from mlxtend.evaluate import mcnemar_table, mcnemar
from nrclex import NRCLex

nrc_df = pd.read_csv("outputs/results/nrclex_predictions.csv")
deberta_df = pd.read_csv("outputs/results/deberta_predictions.csv")

results = nrc_df[["text", "ekman_label", "nrclex_pred"]].copy()
results["deberta_pred"] = deberta_df["deberta_pred"]
results["nrc_correct"] = results["ekman_label"] == results["nrclex_pred"]
results["deberta_correct"] = results["ekman_label"] == results["deberta_pred"]

EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

metrics = {}
for name, pred_col in [("NRCLex + LR", "nrclex_pred"), ("DeBERTa-v3", "deberta_pred")]:
    metrics[name] = {
        "macro_f1": f1_score(results["ekman_label"], results[pred_col], average="macro"),
        "micro_f1": f1_score(results["ekman_label"], results[pred_col], average="micro"),
        "accuracy": (results["ekman_label"] == results[pred_col]).mean()
    }

comparison_df = pd.DataFrame(metrics).T
comparison_df.to_csv("outputs/results/model_comparison.csv")
print(comparison_df)

# Per-class F1 bar chart side by side
nrc_f1s = f1_score(results["ekman_label"], results["nrclex_pred"], labels=EMOTIONS, average=None)
deberta_f1s = f1_score(results["ekman_label"], results["deberta_pred"], labels=EMOTIONS, average=None)

x = np.arange(len(EMOTIONS))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width/2, nrc_f1s, width, label="NRCLex + LR", color="#4C72B0")
ax.bar(x + width/2, deberta_f1s, width, label="DeBERTa-v3", color="#DD8452")
ax.set_xticks(x)
ax.set_xticklabels(EMOTIONS, rotation=20)
ax.set_ylabel("F1 Score")
ax.set_ylim(0, 1)
ax.set_title("Per-class F1: NRCLex + LR vs DeBERTa-v3")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/figures/per_class_f1_comparison.png", dpi=150)
plt.close()

negation_words = ["not", "never", "no", "n't", "cannot", "without", "nobody", "nothing", "neither"]

pattern = r'(?:' + '|'.join(negation_words) + r')'
results["has_negation"] = results["text"].str.lower().str.contains(pattern, regex=True)

for subset_name, mask in [("all", pd.Series([True]*len(results))), ("negation", results["has_negation"]), ("no negation", ~results["has_negation"])]:
    sub = results[mask]
    nrc_f1 = f1_score(sub["ekman_label"], sub["nrclex_pred"], average="macro")
    deb_f1 = f1_score(sub["ekman_label"], sub["deberta_pred"], average="macro")
    print(f"{subset_name:15s} n={len(sub):5d} " f"NRC={nrc_f1:.3f}  DeBERTa={deb_f1:.3f} " f"gap={deb_f1-nrc_f1:+.3f}")

# Save negation error examples
neg_errors = results[results["has_negation"] & ~results["nrc_correct"]][["text", "ekman_label", "nrclex_pred", "deberta_pred"]]
neg_errors.to_csv("outputs/results/negation_errors.csv", index=False)


def nrc_hit_count(text):
    emotion = NRCLex()
    emotion.load_raw_text(text)
    # get frequency dictionary
    return sum(emotion.affect_frequencies.values())

results["nrc_hits"] = results["text"].apply(nrc_hit_count)
implicit = results[results["nrc_hits"] == 0]

print(f"\nImplicit emotion examples (zero NRC hits): {len(implicit)}")
print(f"NRCLex accuracy on these: " f"{(implicit['ekman_label']==implicit['nrclex_pred']).mean():.3f}")
print(f"DeBERTa accuracy on these: " f"{(implicit['ekman_label']==implicit['deberta_pred']).mean():.3f}")

implicit[["text", "ekman_label", "nrclex_pred", "deberta_pred"]].to_csv("outputs/results/implicit_emotion_examples.csv", index=False)

errors = results[~results["nrc_correct"]].copy()

confusion_pairs = (errors
    .groupby(["ekman_label", "nrclex_pred"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)
print("\nTop NRCLex confusion pairs:")
print(confusion_pairs.to_string(index=False))
confusion_pairs.to_csv("outputs/results/nrclex_confusion_pairs.csv", index=False)

tb = mcnemar_table(
    y_target = (results["ekman_label"] == results["ekman_label"]).astype(int).values,
    y_model1 = results["nrc_correct"].astype(int).values,
    y_model2 = results["deberta_correct"].astype(int).values
)

# testing whether the two models difference in performance is statistically significant, will DeBERTa or NRCLex always out perform the other
chi2, p = mcnemar(tb, corrected=True)
print(f"\nMcNemar's test  chi2: {chi2:.3f}  p-value: {p:.4f}")
if p < 0.05:
    print("The performance difference is statistically significant (p < 0.05)")
else:
    print("No statistically significant difference detected")