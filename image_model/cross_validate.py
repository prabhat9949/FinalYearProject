import os
import sys
import random
import copy
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

from dataset import (
    load_patients,
    LiverUltrasoundDataset,
    create_samples
)

from model import (
    create_model,
    get_device
)


# ============================================================
# CONFIG
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

SEED = 42

IMAGE_SIZE = 224
BATCH_SIZE = 8

# Maximum number of epochs
EPOCHS = 15

# Discriminative learning rates
BACKBONE_LR = 5e-6
HEAD_LR = 5e-5

WEIGHT_DECAY = 0.05

NUM_FOLDS = 5

# Stop if validation AUC does not improve
PATIENCE = 4


# ============================================================
# DEVICE
# ============================================================

device = get_device()

print("=" * 70)
print("DEVICE")
print("=" * 70)

print("Device:", device)

if device.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        7
    ),

    transforms.ColorJitter(
        brightness=0.10,
        contrast=0.10
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


val_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# LOAD DATASET
# ============================================================

patients = load_patients(
    DATASET_PATH
)


print("\n" + "=" * 70)
print("DATASET")
print("=" * 70)

print(
    "Patients:",
    len(patients)
)


patient_ids = np.array([

    patient["patient_id"]

    for patient in patients

])


patient_labels = np.array([

    patient["label"]

    for patient in patients

])


print(
    "Normal:",
    np.sum(
        patient_labels == 0
    )
)


print(
    "Steatosis:",
    np.sum(
        patient_labels == 1
    )
)


patient_lookup = {

    patient["patient_id"]: patient

    for patient in patients

}


# ============================================================
# CROSS VALIDATION
# ============================================================

skf = StratifiedKFold(

    n_splits=NUM_FOLDS,

    shuffle=True,

    random_state=SEED

)


fold_results = []


# ============================================================
# FOLDS
# ============================================================

for fold, (
    train_indices,
    val_indices
) in enumerate(

    skf.split(
        patient_ids,
        patient_labels
    ),

    start=1

):

    print("\n\n")

    print("=" * 70)

    print(
        f"PARTIAL FINE-TUNING ViT — "
        f"FOLD {fold}/{NUM_FOLDS}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # PATIENT SPLIT
    # --------------------------------------------------------

    train_patient_ids = patient_ids[
        train_indices
    ]

    val_patient_ids = patient_ids[
        val_indices
    ]


    print(
        "Training patients:",
        len(train_patient_ids)
    )

    print(
        "Validation patients:",
        len(val_patient_ids)
    )


    # --------------------------------------------------------
    # PATIENT RECORDS
    # --------------------------------------------------------

    train_patients = [

        patient_lookup[
            int(pid)
        ]

        for pid in train_patient_ids

    ]


    val_patients = [

        patient_lookup[
            int(pid)
        ]

        for pid in val_patient_ids

    ]


    # --------------------------------------------------------
    # IMAGE SAMPLES
    # --------------------------------------------------------

    train_samples = create_samples(
        train_patients
    )

    val_samples = create_samples(
        val_patients
    )


    print(
        "Training images:",
        len(train_samples)
    )

    print(
        "Validation images:",
        len(val_samples)
    )


    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = LiverUltrasoundDataset(

        train_samples,

        transform=train_transform

    )


    val_dataset = LiverUltrasoundDataset(

        val_samples,

        transform=val_transform

    )


    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=0

    )


    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0

    )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = create_model(

        num_classes=2,

        freeze_backbone=True

    )


    model.to(device)


    # --------------------------------------------------------
    # TRAINABLE PARAMETERS
    # --------------------------------------------------------

    trainable_parameters = [

        parameter

        for parameter in model.parameters()

        if parameter.requires_grad

    ]


    trainable_count = sum(

        parameter.numel()

        for parameter in trainable_parameters

    )


    total_count = sum(

        parameter.numel()

        for parameter in model.parameters()

    )


    print(
        f"Trainable parameters: "
        f"{trainable_count:,}"
    )

    print(
        f"Total parameters: "
        f"{total_count:,}"
    )


    print(
        f"Trainable percentage: "
        f"{100.0 * trainable_count / total_count:.2f}%"
    )


    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------

    train_labels = np.array([

        sample[2]

        for sample in train_samples

    ])


    class_counts = np.bincount(

        train_labels,

        minlength=2

    )


    class_weights = (

        len(train_labels)

        /

        (
            2 *
            class_counts
        )

    )


    class_weights = torch.tensor(

        class_weights,

        dtype=torch.float32,

        device=device

    )


    print(
        "Class counts:",
        class_counts.tolist()
    )


    print(
        "Class weights:",
        class_weights
    )


    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(

        weight=class_weights

    )


    # --------------------------------------------------------
    # DISCRIMINATIVE LEARNING RATES
    # --------------------------------------------------------

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


    print(
        "Trainable backbone tensors:",
        len(backbone_parameters)
    )

    print(
        "Trainable head tensors:",
        len(head_parameters)
    )


    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

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


    print(
        f"Backbone LR: {BACKBONE_LR}"
    )

    print(
        f"Head LR: {HEAD_LR}"
    )

    print(
        f"Weight decay: {WEIGHT_DECAY}"
    )


    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

        optimizer,

        T_max=EPOCHS

    )


    # --------------------------------------------------------
    # BEST CHECKPOINT
    # --------------------------------------------------------

    best_auc = -np.inf

    best_state = None

    patience_counter = 0


    # ========================================================
    # EPOCHS
    # ========================================================

    for epoch in range(

        1,

        EPOCHS + 1

    ):

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.train()

        running_loss = 0.0


        for images, labels in train_loader:

            images = images.to(device)

            labels = labels.to(device)


            optimizer.zero_grad()


            outputs = model(
                images
            )


            loss = criterion(

                outputs,

                labels

            )


            loss.backward()


            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(

                trainable_parameters,

                max_norm=1.0

            )


            optimizer.step()


            running_loss += (

                loss.item()
                *
                images.size(0)

            )


        scheduler.step()


        train_loss = (

            running_loss
            /
            len(train_dataset)

        )


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()


        val_labels = []

        val_probabilities = []

        val_patient_ids = []


        sample_index = 0


        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)


                outputs = model(
                    images
                )


                probabilities = torch.softmax(

                    outputs,

                    dim=1

                )[:, 1]


                batch_size = images.size(0)


                for i in range(
                    batch_size
                ):

                    val_labels.append(

                        int(
                            labels[i]
                        )

                    )


                    val_probabilities.append(

                        float(
                            probabilities[i]
                        )

                    )


                    val_patient_ids.append(

                        val_samples[
                            sample_index
                        ][0]

                    )


                    sample_index += 1


        # ----------------------------------------------------
        # PATIENT AGGREGATION
        # ----------------------------------------------------

        patient_results = {}


        for (

            patient_id,

            label,

            probability

        ) in zip(

            val_patient_ids,

            val_labels,

            val_probabilities

        ):

            if patient_id not in patient_results:

                patient_results[
                    patient_id
                ] = {

                    "label": label,

                    "probabilities": []

                }


            patient_results[
                patient_id
            ][
                "probabilities"
            ].append(

                probability

            )


        patient_true = []

        patient_probabilities = []


        for patient_id in sorted(

            patient_results.keys()

        ):

            result = patient_results[
                patient_id
            ]


            patient_true.append(

                result[
                    "label"
                ]

            )


            patient_probabilities.append(

                np.mean(

                    result[
                        "probabilities"
                    ]

                )

            )


        patient_predictions = [

            1 if probability >= 0.5
            else 0

            for probability
            in patient_probabilities

        ]


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        accuracy = accuracy_score(

            patient_true,

            patient_predictions

        )


        precision = precision_score(

            patient_true,

            patient_predictions,

            zero_division=0

        )


        recall = recall_score(

            patient_true,

            patient_predictions,

            zero_division=0

        )


        f1 = f1_score(

            patient_true,

            patient_predictions,

            zero_division=0

        )


        auc = roc_auc_score(

            patient_true,

            patient_probabilities

        )


        print(

            f"Epoch {epoch:02d}/{EPOCHS} | "

            f"Loss {train_loss:.4f} | "

            f"AUC {auc:.4f} | "

            f"Accuracy {accuracy:.4f} | "

            f"Precision {precision:.4f} | "

            f"Recall {recall:.4f} | "

            f"F1 {f1:.4f}"

        )


        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if auc > best_auc:

            best_auc = auc


            best_state = copy.deepcopy(

                model.state_dict()

            )


            patience_counter = 0


            print(
                f"  ✓ New best AUC: "
                f"{best_auc:.4f}"
            )


        else:

            patience_counter += 1


            print(

                f"  No improvement "
                f"({patience_counter}/{PATIENCE})"

            )


        if patience_counter >= PATIENCE:

            print(

                f"  Early stopping at "
                f"epoch {epoch}"

            )

            break


    # ========================================================
    # BEST MODEL
    # ========================================================

    if best_state is None:

        print(
            "ERROR: No best model found."
        )

        continue


    model.load_state_dict(
        best_state
    )


    model.eval()


    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    val_labels = []

    val_probabilities = []

    val_patient_ids = []


    sample_index = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)


            outputs = model(
                images
            )


            probabilities = torch.softmax(

                outputs,

                dim=1

            )[:, 1]


            for i in range(
                images.size(0)
            ):

                val_labels.append(

                    int(
                        labels[i]
                    )

                )


                val_probabilities.append(

                    float(
                        probabilities[i]
                    )

                )


                val_patient_ids.append(

                    val_samples[
                        sample_index
                    ][0]

                )


                sample_index += 1


    # --------------------------------------------------------
    # PATIENT AGGREGATION
    # --------------------------------------------------------

    patient_results = {}


    for (

        patient_id,

        label,

        probability

    ) in zip(

        val_patient_ids,

        val_labels,

        val_probabilities

    ):

        if patient_id not in patient_results:

            patient_results[
                patient_id
            ] = {

                "label": label,

                "probabilities": []

            }


        patient_results[
            patient_id
        ][
            "probabilities"
        ].append(

            probability

        )


    patient_true = []

    patient_probabilities = []


    for patient_id in sorted(

        patient_results.keys()

    ):

        result = patient_results[
            patient_id
        ]


        patient_true.append(

            result[
                "label"
            ]

        )


        patient_probabilities.append(

            np.mean(

                result[
                    "probabilities"
                ]

            )

        )


    patient_predictions = [

        1 if probability >= 0.5
        else 0

        for probability
        in patient_probabilities

    ]


    # --------------------------------------------------------
    # FINAL METRICS
    # --------------------------------------------------------

    final_accuracy = accuracy_score(

        patient_true,

        patient_predictions

    )


    final_precision = precision_score(

        patient_true,

        patient_predictions,

        zero_division=0

    )


    final_recall = recall_score(

        patient_true,

        patient_predictions,

        zero_division=0

    )


    final_f1 = f1_score(

        patient_true,

        patient_predictions,

        zero_division=0

    )


    final_auc = roc_auc_score(

        patient_true,

        patient_probabilities

    )


    fold_results.append({

        "fold": fold,

        "accuracy": final_accuracy,

        "precision": final_precision,

        "recall": final_recall,

        "f1": final_f1,

        "auc": final_auc

    })


    # --------------------------------------------------------
    # FOLD RESULT
    # --------------------------------------------------------

    print("\n" + "-" * 70)

    print(
        f"BEST RESULT — FOLD {fold}"
    )

    print("-" * 70)


    print(
        f"Accuracy:  {final_accuracy:.4f}"
    )

    print(
        f"Precision: {final_precision:.4f}"
    )

    print(
        f"Recall:    {final_recall:.4f}"
    )

    print(
        f"F1:        {final_f1:.4f}"
    )

    print(
        f"ROC-AUC:   {final_auc:.4f}"
    )


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n\n")

print("=" * 70)

print(
    "PARTIAL FINE-TUNING ViT — "
    "5-FOLD PATIENT-LEVEL CV"
)

print("=" * 70)


for result in fold_results:

    print(

        f"Fold {result['fold']} | "

        f"Accuracy={result['accuracy']:.4f} | "

        f"Precision={result['precision']:.4f} | "

        f"Recall={result['recall']:.4f} | "

        f"F1={result['f1']:.4f} | "

        f"AUC={result['auc']:.4f}"

    )


# ============================================================
# MEAN / STD
# ============================================================

metrics = {

    "Accuracy": [
        r["accuracy"]
        for r in fold_results
    ],

    "Precision": [
        r["precision"]
        for r in fold_results
    ],

    "Recall": [
        r["recall"]
        for r in fold_results
    ],

    "F1": [
        r["f1"]
        for r in fold_results
    ],

    "ROC-AUC": [
        r["auc"]
        for r in fold_results
    ]

}


print("\n")

print("=" * 70)

print(
    "MEAN ± STANDARD DEVIATION"
)

print("=" * 70)


for name, values in metrics.items():

    print(

        f"{name:<10} "

        f"{np.mean(values):.4f} ± "

        f"{np.std(values):.4f}"

    )


# ============================================================
# SUMMARY
# ============================================================

print("\n")

print("=" * 70)

print(
    "CROSS-VALIDATION COMPLETE"
)

print("=" * 70)

print(
    "Architecture:"
)

print(
    "ViT-B/16"
)

print(
    "Frozen early blocks + "
    "last 3 trainable blocks + "
    "normalization + classifier"
)

print(
    f"Backbone LR: {BACKBONE_LR}"
)

print(
    f"Head LR: {HEAD_LR}"
)

print(
    f"Weight Decay: {WEIGHT_DECAY}"
)

print(
    f"Early Stopping Patience: {PATIENCE}"
)

print(
    "Evaluation: Patient-level "
    "mean probability across 10 images"
)

print("=" * 70)