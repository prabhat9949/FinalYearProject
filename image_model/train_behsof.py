import sys
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Allow imports from image-model
# ============================================================

sys.path.append(str(Path(__file__).resolve().parent))

from behsof_dataset import BEHSOFDataset, get_samples
from behsof_model import create_behsof_model, get_device


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "ultrasound"
    / "BEHSOF"
)

MODEL_DIR = PROJECT_ROOT / "saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = MODEL_DIR / "behsof_vit.pt"


# ============================================================
# BEHSOF OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "behsof"
)

GRAPH_DIR = OUTPUT_DIR / "graphs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_HISTORY_PATH = (
    OUTPUT_DIR / "training_history.json"
)


# ============================================================
# Configuration
# ============================================================

SEED = 42

IMAGE_SIZE = 224
BATCH_SIZE = 8

EPOCHS = 15
PATIENCE = 4

HEAD_LR = 5e-5
BACKBONE_LR = 5e-6

WEIGHT_DECAY = 0.05

# Windows-friendly
NUM_WORKERS = 0


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# Device
# ============================================================

device = get_device()


# ============================================================
# Print experiment information
# ============================================================

print("=" * 60)
print("BEHSOF ViT TRAINING")
print("=" * 60)

print("Device:", device)
print("Dataset:", DATASET_ROOT)
print("Output:", OUTPUT_DIR)

print("=" * 60)


# ============================================================
# Load samples
# ============================================================

samples = get_samples(
    DATASET_ROOT,
    remove_duplicates=True
)

paths = [x[0] for x in samples]
labels = [x[1] for x in samples]


print()
print("Total unique images:", len(samples))


# ============================================================
# Train / Validation / Test split
# ============================================================

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    paths,
    labels,
    test_size=0.30,
    stratify=labels,
    random_state=SEED
)


val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths,
    temp_labels,
    test_size=0.50,
    stratify=temp_labels,
    random_state=SEED
)


train_samples = list(
    zip(train_paths, train_labels)
)

val_samples = list(
    zip(val_paths, val_labels)
)

test_samples = list(
    zip(test_paths, test_labels)
)


# ============================================================
# Print split information
# ============================================================

print()
print("Dataset split:")

print(
    f"Train      : {len(train_samples)} "
    f"(Non-NAFLD={train_labels.count(0)}, "
    f"NAFLD={train_labels.count(1)})"
)

print(
    f"Validation : {len(val_samples)} "
    f"(Non-NAFLD={val_labels.count(0)}, "
    f"NAFLD={val_labels.count(1)})"
)

print(
    f"Test       : {len(test_samples)} "
    f"(Non-NAFLD={test_labels.count(0)}, "
    f"NAFLD={test_labels.count(1)})"
)


# ============================================================
# ImageNet normalization
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


# ============================================================
# Data augmentation
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=7
    ),

    transforms.ColorJitter(
        brightness=0.10,
        contrast=0.10
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean,
        std
    )
])


val_transform = transforms.Compose([

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
# Datasets
# ============================================================

train_dataset = BEHSOFDataset(
    train_samples,
    transform=train_transform
)

val_dataset = BEHSOFDataset(
    val_samples,
    transform=val_transform
)

test_dataset = BEHSOFDataset(
    test_samples,
    transform=val_transform
)


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


# ============================================================
# Model
# ============================================================

model = create_behsof_model(
    num_classes=2
)

model = model.to(device)


# ============================================================
# Class weights
# ============================================================

class_0_count = train_labels.count(0)
class_1_count = train_labels.count(1)

total_train = (
    class_0_count +
    class_1_count
)


class_weights = torch.tensor(
    [
        total_train /
        (2 * class_0_count),

        total_train /
        (2 * class_1_count)
    ],
    dtype=torch.float32,
    device=device
)


print()
print("Class weights:")

print(
    "Non-NAFLD:",
    class_weights[0].item()
)

print(
    "NAFLD:",
    class_weights[1].item()
)


# ============================================================
# Loss
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# Optimizer
# ============================================================

backbone_parameters = []
head_parameters = []


for name, parameter in model.named_parameters():

    if not parameter.requires_grad:
        continue

    if name.startswith("head"):

        head_parameters.append(
            parameter
        )

    else:

        backbone_parameters.append(
            parameter
        )


optimizer = torch.optim.AdamW(
    [
        {
            "params": backbone_parameters,
            "lr": BACKBONE_LR
        },

        {
            "params": head_parameters,
            "lr": HEAD_LR
        }
    ],

    weight_decay=WEIGHT_DECAY
)


# ============================================================
# Scheduler
# ============================================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)


# ============================================================
# Mixed precision
# ============================================================

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(device.type == "cuda")
)


# ============================================================
# Validation function
# ============================================================

def evaluate(model, loader):

    model.eval()

    all_labels = []
    all_probs = []

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for images, labels_batch in loader:

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

                loss = criterion(
                    outputs,
                    labels_batch
                )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )[:, 1]

            total_loss += (
                loss.item()
                * labels_batch.size(0)
            )

            total_samples += (
                labels_batch.size(0)
            )

            all_labels.extend(
                labels_batch
                .cpu()
                .numpy()
            )

            all_probs.extend(
                probabilities
                .cpu()
                .numpy()
            )


    average_loss = (
        total_loss /
        total_samples
    )


    auc = roc_auc_score(
        all_labels,
        all_probs
    )


    return average_loss, auc


# ============================================================
# Training history
# ============================================================

history = {

    "epoch": [],

    "train_loss": [],

    "val_loss": [],

    "val_auc": [],

    "learning_rate": []
}


# ============================================================
# Training
# ============================================================

best_auc = -float("inf")

epochs_without_improvement = 0


print()
print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)


for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    running_loss = 0.0

    total_samples = 0


    # --------------------------------------------------------
    # Learning rate used for this epoch
    # --------------------------------------------------------

    current_lr = (
        optimizer.param_groups[0]["lr"]
    )


    # --------------------------------------------------------
    # Training batches
    # --------------------------------------------------------

    for images, labels_batch in train_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels_batch = labels_batch.to(
            device,
            non_blocking=True
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda")
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels_batch
            )


        scaler.scale(
            loss
        ).backward()


        scaler.unscale_(
            optimizer
        )


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )


        scaler.step(
            optimizer
        )

        scaler.update()


        running_loss += (
            loss.item()
            * labels_batch.size(0)
        )

        total_samples += (
            labels_batch.size(0)
        )


    # --------------------------------------------------------
    # Training loss
    # --------------------------------------------------------

    train_loss = (
        running_loss /
        total_samples
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_loss, val_auc = evaluate(
        model,
        val_loader
    )


    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history["epoch"].append(
        epoch
    )

    history["train_loss"].append(
        float(train_loss)
    )

    history["val_loss"].append(
        float(val_loss)
    )

    history["val_auc"].append(
        float(val_auc)
    )

    history["learning_rate"].append(
        float(current_lr)
    )


    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler.step()


    # --------------------------------------------------------
    # Print epoch
    # --------------------------------------------------------

    print(
        f"Epoch [{epoch:02d}/{EPOCHS}] "
        f"| Train Loss: {train_loss:.4f} "
        f"| Val Loss: {val_loss:.4f} "
        f"| Val AUC: {val_auc:.4f} "
        f"| LR: {current_lr:.2e}"
    )


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_auc > best_auc:

        best_auc = val_auc

        epochs_without_improvement = 0


        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "best_val_auc":
                    best_auc,

                "epoch":
                    epoch,

                "seed":
                    SEED
            },

            CHECKPOINT_PATH
        )


        print(
            f"  ✓ Best model saved "
            f"(Val AUC: {best_auc:.4f})"
        )


    else:

        epochs_without_improvement += 1


        print(
            f"  No improvement "
            f"({epochs_without_improvement}/{PATIENCE})"
        )


    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print()
        print(
            "Early stopping triggered."
        )

        break


# ============================================================
# Save training history
# ============================================================

with open(
    TRAINING_HISTORY_PATH,
    "w"
) as f:

    json.dump(
        history,
        f,
        indent=4
    )


# ============================================================
# GRAPH 1 — TRAINING / VALIDATION LOSS
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history["epoch"],
    history["train_loss"],
    marker="o",
    label="Training Loss"
)

plt.plot(
    history["epoch"],
    history["val_loss"],
    marker="o",
    label="Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "BEHSOF Training and Validation Loss"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "loss_curve.png",
    dpi=300
)

plt.close()


# ============================================================
# GRAPH 2 — VALIDATION AUC
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history["epoch"],
    history["val_auc"],
    marker="o",
    label="Validation ROC-AUC"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "ROC-AUC"
)

plt.title(
    "BEHSOF Validation ROC-AUC"
)

plt.ylim(
    0.0,
    1.05
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "auc_curve.png",
    dpi=300
)

plt.close()


# ============================================================
# GRAPH 3 — LEARNING RATE
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history["epoch"],
    history["learning_rate"],
    marker="o"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Learning Rate"
)

plt.title(
    "BEHSOF Learning Rate Schedule"
)

plt.yscale(
    "log"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "learning_rate.png",
    dpi=300
)

plt.close()


# ============================================================
# Finish
# ============================================================

print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best Validation AUC: "
    f"{best_auc:.4f}"
)

print(
    f"Checkpoint: "
    f"{CHECKPOINT_PATH}"
)

print(
    f"Training history: "
    f"{TRAINING_HISTORY_PATH}"
)

print(
    f"Graphs: "
    f"{GRAPH_DIR}"
)

print()
print("Generated graphs:")

print(
    "  ✓ loss_curve.png"
)

print(
    "  ✓ auc_curve.png"
)

print(
    "  ✓ learning_rate.png"
)

print("=" * 60)