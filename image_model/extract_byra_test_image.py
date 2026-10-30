import os
import sys

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


# ============================================================
# IMPORT EXISTING DATASET FUNCTIONS
# ============================================================

sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

from dataset import load_patients


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

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "byra"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CONFIG
# ============================================================

SEED = 42


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


print("=" * 70)
print("BYRA TEST IMAGE EXTRACTION")
print("=" * 70)

print(
    f"Total patients: {len(patients)}"
)


# ============================================================
# EXACT SAME SPLIT AS evaluate.py
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


# ============================================================
# PRINT TEST PATIENTS
# ============================================================

print()
print("=" * 70)
print("TEST PATIENTS")
print("=" * 70)

print(
    "Test patient IDs:",
    test_ids
)


# ============================================================
# FIND FIRST TEST PATIENT
# ============================================================

patient_lookup = {
    patient["patient_id"]: patient
    for patient in patients
}


test_patient_id = test_ids[0]

test_patient = patient_lookup[
    test_patient_id
]


true_label = test_patient[
    "label"
]

images = test_patient[
    "images"
]


# ============================================================
# CLASS NAME
# ============================================================

if true_label == 0:

    class_name = "Normal"

else:

    class_name = "Steatosis"


print()
print(
    f"Selected patient : {test_patient_id}"
)

print(
    f"Actual class     : {class_name}"
)

print(
    f"Class number     : {true_label}"
)

print(
    f"Images available : {images.shape[0]}"
)


# ============================================================
# SELECT FIRST IMAGE
# ============================================================

image_index = 0

image_array = images[
    image_index
]


# ============================================================
# CONVERT TO UINT8
# ============================================================

image_array = np.asarray(
    image_array,
    dtype=np.uint8
)


# ============================================================
# SAVE IMAGE
# ============================================================

output_path = os.path.join(
    OUTPUT_DIR,
    f"test_patient_{test_patient_id}_image_{image_index}.png"
)


image = Image.fromarray(
    image_array
).convert("RGB")


image.save(
    output_path
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata_path = os.path.join(
    OUTPUT_DIR,
    "test_image_info.txt"
)


with open(
    metadata_path,
    "w"
) as f:

    f.write(
        f"Patient ID: {test_patient_id}\n"
    )

    f.write(
        f"Image index: {image_index}\n"
    )

    f.write(
        f"Actual class: {class_name}\n"
    )

    f.write(
        f"Class number: {true_label}\n"
    )

    f.write(
        f"Image shape: {image_array.shape}\n"
    )


# ============================================================
# DONE
# ============================================================

print()
print("=" * 70)
print("IMAGE EXTRACTED")
print("=" * 70)

print(
    f"Saved image: {output_path}"
)

print(
    f"Actual label: {class_name}"
)

print(
    f"Metadata: {metadata_path}"
)

print("=" * 70)