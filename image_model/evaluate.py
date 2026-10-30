import os
import sys
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
    roc_auc_score,
    confusion_matrix,
    classification_report
)

sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

from dataset import (
    load_patients,
    LiverUltrasoundDataset,
    create_samples
)

from model import create_model, get_device


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "data",
    "ultrasound",
    "dataset_liver_bmodes_steatosis_assessment_IJCARS.mat"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "saved_models",
    "liver_vit.pt"
)


# ============================================================
# CONFIG
# ============================================================

SEED = 42

IMAGE_SIZE = 224
BATCH_SIZE = 8


# ============================================================
# DEVICE
# ============================================================

device = get_device()

print("=" * 70)
print("DEVICE")
print("=" * 70)

print(device)

if device.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# TRANSFORM
# ============================================================

test_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD PATIENTS
# ============================================================

patients = load_patients(
    DATASET_PATH
)

patient_ids = [
    patient["patient_id"]
    for patient in patients
]

patient_labels = [
    patient["label"]
    for patient in patients
]


# ============================================================
# RECREATE EXACT SAME SPLIT
# ============================================================

train_ids, temp_ids, train_labels, temp_labels = train_test_split(
    patient_ids,
    patient_labels,
    test_size=0.30,
    random_state=SEED,
    stratify=patient_labels
)

val_ids, test_ids, _, _ = train_test_split(
    temp_ids,
    temp_labels,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_labels
)


print("\n" + "=" * 70)
print("TEST PATIENTS")
print("=" * 70)

print(
    "Test patient IDs:",
    test_ids
)


# ============================================================
# TEST DATA
# ============================================================

patient_lookup = {
    patient["patient_id"]: patient
    for patient in patients
}


test_patients = [
    patient_lookup[pid]
    for pid in test_ids
]


test_samples = create_samples(
    test_patients
)


test_dataset = LiverUltrasoundDataset(
    test_samples,
    transform=test_transform
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING MODEL")
print("=" * 70)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model = create_model(
    num_classes=2
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()

print(
    "Checkpoint validation AUC:",
    checkpoint.get("val_auc")
)


# ============================================================
# IMAGE-LEVEL PREDICTIONS
# ============================================================

all_labels = []
all_probabilities = []
all_patient_ids = []


with torch.no_grad():

    sample_index = 0

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[:, 1]

        batch_size = images.size(0)

        for i in range(batch_size):

            all_labels.append(
                int(labels[i])
            )

            all_probabilities.append(
                float(probabilities[i])
            )

            patient_id = test_samples[
                sample_index
            ][0]

            all_patient_ids.append(
                patient_id
            )

            sample_index += 1


# ============================================================
# IMAGE-LEVEL METRICS
# ============================================================

image_predictions = [
    1 if p >= 0.5 else 0
    for p in all_probabilities
]


image_accuracy = accuracy_score(
    all_labels,
    image_predictions
)

image_precision = precision_score(
    all_labels,
    image_predictions,
    zero_division=0
)

image_recall = recall_score(
    all_labels,
    image_predictions,
    zero_division=0
)

image_f1 = f1_score(
    all_labels,
    image_predictions,
    zero_division=0
)

image_auc = roc_auc_score(
    all_labels,
    all_probabilities
)


# ============================================================
# PATIENT-LEVEL AGGREGATION
# ============================================================

patient_results = {}

for patient_id, label, probability in zip(
    all_patient_ids,
    all_labels,
    all_probabilities
):

    if patient_id not in patient_results:

        patient_results[patient_id] = {
            "label": label,
            "probabilities": []
        }

    patient_results[
        patient_id
    ]["probabilities"].append(
        probability
    )


patient_ids_sorted = sorted(
    patient_results.keys()
)

patient_true = []
patient_probabilities = []


print("\n" + "=" * 70)
print("PATIENT-LEVEL RESULTS")
print("=" * 70)


for patient_id in patient_ids_sorted:

    result = patient_results[
        patient_id
    ]

    probabilities = result[
        "probabilities"
    ]

    true_label = result[
        "label"
    ]

    mean_probability = np.mean(
        probabilities
    )

    prediction = (
        1
        if mean_probability >= 0.5
        else 0
    )

    patient_true.append(
        true_label
    )

    patient_probabilities.append(
        mean_probability
    )

    print(
        f"Patient {patient_id:02d} | "
        f"True={true_label} | "
        f"Probability={mean_probability:.4f} | "
        f"Prediction={prediction}"
    )


# ============================================================
# PATIENT METRICS
# ============================================================

patient_predictions = [
    1 if p >= 0.5 else 0
    for p in patient_probabilities
]


patient_accuracy = accuracy_score(
    patient_true,
    patient_predictions
)

patient_precision = precision_score(
    patient_true,
    patient_predictions,
    zero_division=0
)

patient_recall = recall_score(
    patient_true,
    patient_predictions,
    zero_division=0
)

patient_f1 = f1_score(
    patient_true,
    patient_predictions,
    zero_division=0
)

patient_auc = roc_auc_score(
    patient_true,
    patient_probabilities
)

cm = confusion_matrix(
    patient_true,
    patient_predictions
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("IMAGE-LEVEL TEST RESULTS")
print("=" * 70)

print(
    f"Accuracy:  {image_accuracy:.4f}"
)

print(
    f"Precision: {image_precision:.4f}"
)

print(
    f"Recall:    {image_recall:.4f}"
)

print(
    f"F1:        {image_f1:.4f}"
)

print(
    f"ROC-AUC:   {image_auc:.4f}"
)


print("\n" + "=" * 70)
print("PATIENT-LEVEL TEST RESULTS")
print("=" * 70)

print(
    f"Accuracy:  {patient_accuracy:.4f}"
)

print(
    f"Precision: {patient_precision:.4f}"
)

print(
    f"Recall:    {patient_recall:.4f}"
)

print(
    f"F1:        {patient_f1:.4f}"
)

print(
    f"ROC-AUC:   {patient_auc:.4f}"
)


print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)


print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        patient_true,
        patient_predictions,
        target_names=[
            "Normal",
            "Steatosis"
        ],
        zero_division=0
    )
)