import sys
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Allow imports from image-model
# ============================================================

sys.path.append(
    str(Path(__file__).resolve().parent)
)

from behsof_dataset import (
    BEHSOFDataset,
    get_samples
)

from behsof_model import (
    create_behsof_model,
    get_device
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "ultrasound"
    / "BEHSOF"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "behsof_vit.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "behsof"
)

GRAPH_DIR = (
    OUTPUT_DIR
    / "graphs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

GRAPH_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METRICS_PATH = (
    OUTPUT_DIR
    / "test_metrics.json"
)

PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "test_predictions.csv"
)


# ============================================================
# Configuration
# ============================================================

SEED = 42

IMAGE_SIZE = 224
BATCH_SIZE = 8

NUM_WORKERS = 0


# ============================================================
# Device
# ============================================================

device = get_device()


# ============================================================
# Reproducibility
# ============================================================

torch.manual_seed(SEED)

np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Header
# ============================================================

print("=" * 65)
print("BEHSOF ViT — FINAL TEST EVALUATION")
print("=" * 65)

print("Device:", device)
print("Dataset:", DATASET_ROOT)
print("Model:", MODEL_PATH)

print("=" * 65)


# ============================================================
# Check model
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model checkpoint not found:\n{MODEL_PATH}"
    )


# ============================================================
# Load samples
# ============================================================

samples = get_samples(
    DATASET_ROOT,
    remove_duplicates=True
)

paths = [
    x[0]
    for x in samples
]

labels = [
    x[1]
    for x in samples
]


print()
print(
    "Unique images:",
    len(samples)
)


# ============================================================
# Recreate EXACT same split
# ============================================================

train_paths, temp_paths, train_labels, temp_labels = (
    train_test_split(
        paths,
        labels,
        test_size=0.30,
        stratify=labels,
        random_state=SEED
    )
)


val_paths, test_paths, val_labels, test_labels = (
    train_test_split(
        temp_paths,
        temp_labels,
        test_size=0.50,
        stratify=temp_labels,
        random_state=SEED
    )
)


test_samples = list(
    zip(
        test_paths,
        test_labels
    )
)


print()
print("Test set:")
print(
    f"Total      : {len(test_samples)}"
)

print(
    f"Non-NAFLD  : {test_labels.count(0)}"
)

print(
    f"NAFLD      : {test_labels.count(1)}"
)


# ============================================================
# Transform
# ============================================================

mean = [
    0.485,
    0.456,
    0.406
]

std = [
    0.229,
    0.224,
    0.225
]


test_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean,
        std
    )
])


# ============================================================
# Test dataset
# ============================================================

test_dataset = BEHSOFDataset(
    test_samples,
    transform=test_transform
)


# ============================================================
# Test loader
# ============================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


# ============================================================
# Create model
# ============================================================

model = create_behsof_model(
    num_classes=2
)

model = model.to(device)


# ============================================================
# Load checkpoint
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model.eval()


print()
print(
    "Loaded checkpoint:"
)

print(
    "Best validation AUC:",
    f"{checkpoint['best_val_auc']:.4f}"
)

print(
    "Best epoch:",
    checkpoint["epoch"]
)


# ============================================================
# Inference
# ============================================================

all_labels = []
all_probabilities = []
all_predictions = []
all_paths = []


print()
print("=" * 65)
print("RUNNING TEST INFERENCE")
print("=" * 65)


with torch.no_grad():

    for batch_index, (
        images,
        labels_batch
    ) in enumerate(test_loader):

        images = images.to(
            device,
            non_blocking=True
        )

        labels_batch = labels_batch.to(
            device,
            non_blocking=True
        )


        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda")
        ):

            outputs = model(images)


        probabilities = torch.softmax(
            outputs,
            dim=1
        )[:, 1]


        predictions = (
            probabilities >= 0.5
        ).long()


        all_labels.extend(
            labels_batch
            .cpu()
            .numpy()
            .tolist()
        )

        all_probabilities.extend(
            probabilities
            .cpu()
            .numpy()
            .tolist()
        )

        all_predictions.extend(
            predictions
            .cpu()
            .numpy()
            .tolist()
        )


# ============================================================
# Convert to numpy
# ============================================================

y_true = np.array(
    all_labels
)

y_prob = np.array(
    all_probabilities
)

y_pred = np.array(
    all_predictions
)


# ============================================================
# Metrics
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

sensitivity = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

balanced_accuracy = balanced_accuracy_score(
    y_true,
    y_pred
)

roc_auc = roc_auc_score(
    y_true,
    y_prob
)

pr_auc = average_precision_score(
    y_true,
    y_prob
)


# ============================================================
# Confusion matrix
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()


# ============================================================
# Specificity
# ============================================================

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0.0
)


# ============================================================
# Classification report
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=[
        "Non-NAFLD",
        "NAFLD"
    ],
    output_dict=True,
    zero_division=0
)


# ============================================================
# Print results
# ============================================================

print()
print("=" * 65)
print("FINAL TEST RESULTS")
print("=" * 65)

print(
    f"Accuracy           : {accuracy:.4f}"
)

print(
    f"Precision          : {precision:.4f}"
)

print(
    f"Sensitivity/Recall : {sensitivity:.4f}"
)

print(
    f"Specificity        : {specificity:.4f}"
)

print(
    f"F1 Score           : {f1:.4f}"
)

print(
    f"Balanced Accuracy  : {balanced_accuracy:.4f}"
)

print(
    f"ROC-AUC            : {roc_auc:.4f}"
)

print(
    f"PR-AUC              : {pr_auc:.4f}"
)

print()
print("Confusion Matrix:")
print(cm)

print()
print(
    "TN:",
    tn,
    "| FP:",
    fp,
    "| FN:",
    fn,
    "| TP:",
    tp
)

print("=" * 65)


# ============================================================
# Save metrics JSON
# ============================================================

metrics = {

    "dataset": "BEHSOF",

    "task": "NAFLD vs Non-NAFLD",

    "evaluation_level": "image-level",

    "seed": SEED,

    "test_samples": len(y_true),

    "class_distribution": {
        "Non-NAFLD": int(np.sum(y_true == 0)),
        "NAFLD": int(np.sum(y_true == 1))
    },

    "checkpoint_epoch": int(
        checkpoint["epoch"]
    ),

    "best_validation_auc": float(
        checkpoint["best_val_auc"]
    ),

    "test_metrics": {

        "accuracy": float(
            accuracy
        ),

        "precision": float(
            precision
        ),

        "sensitivity_recall": float(
            sensitivity
        ),

        "specificity": float(
            specificity
        ),

        "f1_score": float(
            f1
        ),

        "balanced_accuracy": float(
            balanced_accuracy
        ),

        "roc_auc": float(
            roc_auc
        ),

        "pr_auc": float(
            pr_auc
        )
    },

    "confusion_matrix": {

        "TN": int(tn),

        "FP": int(fp),

        "FN": int(fn),

        "TP": int(tp)
    },

    "classification_report": report
}


with open(
    METRICS_PATH,
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# ============================================================
# Save predictions
# ============================================================

import csv


with open(
    PREDICTIONS_PATH,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "image_path",
        "true_label",
        "predicted_label",
        "nafld_probability"
    ])

    for index in range(
        len(y_true)
    ):

        writer.writerow([
            test_paths[index],
            int(y_true[index]),
            int(y_pred[index]),
            float(y_prob[index])
        ])


# ============================================================
# GRAPH 1 — CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "BEHSOF Test Confusion Matrix"
)

plt.colorbar()

plt.xticks(
    [0, 1],
    ["Non-NAFLD", "NAFLD"]
)

plt.yticks(
    [0, 1],
    ["Non-NAFLD", "NAFLD"]
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)


for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )


plt.tight_layout()

plt.savefig(
    GRAPH_DIR
    / "confusion_matrix.png",
    dpi=300
)

plt.close()


# ============================================================
# GRAPH 2 — ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_true,
    y_prob
)


plt.figure(
    figsize=(8, 6)
)

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "BEHSOF Test ROC Curve"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    GRAPH_DIR
    / "roc_curve.png",
    dpi=300
)

plt.close()


# ============================================================
# GRAPH 3 — PRECISION-RECALL CURVE
# ============================================================

precision_curve, recall_curve, pr_thresholds = (
    precision_recall_curve(
        y_true,
        y_prob
    )
)


plt.figure(
    figsize=(8, 6)
)

plt.plot(
    recall_curve,
    precision_curve,
    label=f"PR-AUC = {pr_auc:.4f}"
)

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "BEHSOF Test Precision-Recall Curve"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    GRAPH_DIR
    / "precision_recall_curve.png",
    dpi=300
)

plt.close()


# ============================================================
# GRAPH 4 — PROBABILITY DISTRIBUTION
# ============================================================

nafld_probabilities = y_prob[
    y_true == 1
]

non_nafld_probabilities = y_prob[
    y_true == 0
]


plt.figure(
    figsize=(8, 6)
)

plt.hist(
    non_nafld_probabilities,
    bins=20,
    alpha=0.6,
    label="True Non-NAFLD"
)

plt.hist(
    nafld_probabilities,
    bins=20,
    alpha=0.6,
    label="True NAFLD"
)

plt.axvline(
    0.5,
    linestyle="--",
    label="Decision threshold = 0.5"
)

plt.xlabel(
    "Predicted NAFLD Probability"
)

plt.ylabel(
    "Number of Images"
)

plt.title(
    "BEHSOF Test Probability Distribution"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    GRAPH_DIR
    / "probability_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# Finish
# ============================================================

print()
print("=" * 65)
print("EVALUATION COMPLETE")
print("=" * 65)

print(
    "Metrics:",
    METRICS_PATH
)

print(
    "Predictions:",
    PREDICTIONS_PATH
)

print(
    "Graphs:",
    GRAPH_DIR
)

print()
print("Generated test graphs:")

print(
    "  ✓ confusion_matrix.png"
)

print(
    "  ✓ roc_curve.png"
)

print(
    "  ✓ precision_recall_curve.png"
)

print(
    "  ✓ probability_distribution.png"
)

print("=" * 65)