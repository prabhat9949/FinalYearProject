# train.py the file for training the model

import os
import sys
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Allow imports from this folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import (
    load_patients,
    LiverUltrasoundDataset,
    create_samples
)

from model import create_model, get_device


# ============================================================
# CONFIG
# ============================================================

SEED = 42

BATCH_SIZE = 16
NUM_EPOCHS = 20

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

IMAGE_SIZE = 224

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "saved_models",
    "liver_vit.pt"
)

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "ultrasound",
    "dataset_liver_bmodes_steatosis_assessment_IJCARS.mat"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# DEVICE
# ============================================================

device = get_device()

print("=" * 70)
print("DEVICE")
print("=" * 70)

print(device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# LOAD PATIENTS
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATASET")
print("=" * 70)

patients = load_patients(DATASET_PATH)

print("Patients:", len(patients))


# ============================================================
# PATIENT-LEVEL SPLIT
# ============================================================

patient_ids = [
    patient["patient_id"]
    for patient in patients
]

patient_labels = [
    patient["label"]
    for patient in patients
]


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


print("\nPatient split:")

print("Train:", len(train_ids))
print("Validation:", len(val_ids))
print("Test:", len(test_ids))


# ============================================================
# CREATE LOOKUP
# ============================================================

patient_lookup = {
    patient["patient_id"]: patient
    for patient in patients
}


def get_patient_samples(patient_ids):

    selected_patients = [
        patient_lookup[patient_id]
        for patient_id in patient_ids
    ]

    return create_samples(selected_patients)


train_samples = get_patient_samples(train_ids)
val_samples = get_patient_samples(val_ids)
test_samples = get_patient_samples(test_ids)


print("\nImage split:")

print("Train images:", len(train_samples))
print("Validation images:", len(val_samples))
print("Test images:", len(test_samples))


# ============================================================
# DATASETS
# ============================================================

train_dataset = LiverUltrasoundDataset(
    train_samples,
    transform=train_transform
)

val_dataset = LiverUltrasoundDataset(
    val_samples,
    transform=val_transform
)

test_dataset = LiverUltrasoundDataset(
    test_samples,
    transform=val_transform
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

train_labels = [
    sample[2]
    for sample in train_samples
]

class_counts = np.bincount(
    train_labels,
    minlength=2
)

print("\nClass counts:")

print("Class 0:", class_counts[0])
print("Class 1:", class_counts[1])


weights = len(train_labels) / (
    2 * class_counts
)

class_weights = torch.tensor(
    weights,
    dtype=torch.float32
).to(device)

print("Class weights:", class_weights)


# ============================================================
# MODEL
# ============================================================

model = create_model(num_classes=2)

model = model.to(device)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(labels, probabilities):

    predictions = [
        1 if probability >= 0.5 else 0
        for probability in probabilities
    ]

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    try:
        auc = roc_auc_score(
            labels,
            probabilities
        )
    except ValueError:
        auc = 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }


# ============================================================
# TRAIN / VALIDATION
# ============================================================

best_auc = 0.0

os.makedirs(
    os.path.dirname(MODEL_PATH),
    exist_ok=True
)


for epoch in range(NUM_EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0

    for images, labels in train_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        train_loss += (
            loss.item()
            * images.size(0)
        )

    train_loss /= len(train_dataset)


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0

    val_labels = []
    val_probabilities = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss += (
                loss.item()
                * images.size(0)
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )[:, 1]

            val_labels.extend(
                labels.cpu().numpy()
            )

            val_probabilities.extend(
                probabilities.cpu().numpy()
            )

    val_loss /= len(val_dataset)

    metrics = calculate_metrics(
        val_labels,
        val_probabilities
    )

    scheduler.step()


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(
        f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Val Loss: {val_loss:.4f}"
    )

    print(
        f"Accuracy: {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: {metrics['precision']:.4f}"
    )

    print(
        f"Recall: {metrics['recall']:.4f}"
    )

    print(
        f"F1: {metrics['f1']:.4f}"
    )

    print(
        f"AUC: {metrics['auc']:.4f}"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if metrics["auc"] > best_auc:

        best_auc = metrics["auc"]

        torch.save(
            {
                "model_state_dict": model.state_dict(),

                "val_auc": best_auc,

                "class_names": [
                    "normal",
                    "steatosis"
                ],

                "image_size": IMAGE_SIZE
            },
            MODEL_PATH
        )

        print(
            f"✓ Best model saved → {MODEL_PATH}"
        )


print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print("Best validation AUC:", best_auc)
print("Model:", MODEL_PATH)