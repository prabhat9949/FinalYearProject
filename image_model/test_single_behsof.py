# import sys
# from pathlib import Path

# import pandas as pd
# import torch
# from PIL import Image
# from torchvision import transforms
# import matplotlib.pyplot as plt

# # ============================================================
# # Paths
# # ============================================================

# PROJECT_ROOT = Path(__file__).resolve().parent.parent

# MODEL_PATH = (
#     PROJECT_ROOT
#     / "saved_models"
#     / "behsof_vit.pt"
# )

# PREDICTIONS_PATH = (
#     PROJECT_ROOT
#     / "outputs"
#     / "behsof"
#     / "test_predictions.csv"
# )

# # ============================================================
# # Import model
# # ============================================================

# sys.path.append(
#     str(Path(__file__).resolve().parent)
# )

# from behsof_model import (
#     create_behsof_model,
#     get_device
# )

# # ============================================================
# # Device
# # ============================================================

# device = get_device()

# print("=" * 60)
# print("BEHSOF — SINGLE TEST IMAGE")
# print("=" * 60)

# print("Device:", device)

# # ============================================================
# # Load test predictions
# # ============================================================

# df = pd.read_csv(
#     PREDICTIONS_PATH
# )

# # Pick the first test image
# row = df.iloc[0]

# image_path = Path(
#     row["image_path"]
# )

# true_label = int(
#     row["true_label"]
# )

# # ============================================================
# # Create model
# # ============================================================

# model = create_behsof_model(
#     num_classes=2
# )

# model = model.to(device)

# checkpoint = torch.load(
#     MODEL_PATH,
#     map_location=device
# )

# model.load_state_dict(
#     checkpoint["model_state_dict"]
# )

# model.eval()

# # ============================================================
# # Transform
# # ============================================================

# transform = transforms.Compose([

#     transforms.Resize(
#         (224, 224)
#     ),

#     transforms.ToTensor(),

#     transforms.Normalize(
#         [0.485, 0.456, 0.406],
#         [0.229, 0.224, 0.225]
#     )
# ])

# # ============================================================
# # Load image
# # ============================================================

# image = Image.open(
#     image_path
# ).convert("RGB")

# input_tensor = transform(
#     image
# ).unsqueeze(0).to(device)

# # ============================================================
# # Prediction
# # ============================================================

# with torch.no_grad():

#     with torch.autocast(
#         device_type=device.type,
#         dtype=torch.float16,
#         enabled=(device.type == "cuda")
#     ):

#         output = model(
#             input_tensor
#         )

#     probabilities = torch.softmax(
#         output,
#         dim=1
#     )[0]

#     predicted_label = (
#         torch.argmax(
#             probabilities
#         ).item()
#     )

# # ============================================================
# # Results
# # ============================================================

# non_nafld_probability = (
#     probabilities[0].item()
# )

# nafld_probability = (
#     probabilities[1].item()
# )

# confidence = (
#     probabilities[predicted_label].item()
# )

# label_names = {
#     0: "Non-NAFLD",
#     1: "NAFLD"
# }

# print()
# print("=" * 60)
# print("IMAGE INFORMATION")
# print("=" * 60)

# print(
#     "Image:",
#     image_path.name
# )

# print(
#     "Path:",
#     image_path
# )

# print()
# print("=" * 60)
# print("PREDICTION")
# print("=" * 60)

# print(
#     "Actual label    :",
#     label_names[true_label]
# )

# print(
#     "Predicted label  :",
#     label_names[predicted_label]
# )

# print(
#     f"Non-NAFLD prob.  : {non_nafld_probability:.4f}"
# )

# print(
#     f"NAFLD prob.      : {nafld_probability:.4f}"
# )

# print(
#     f"Confidence       : {confidence:.4f}"
# )

# print(
#     "Correct          :",
#     predicted_label == true_label
# )

# print("=" * 60)

# # ============================================================
# # Display image
# # ============================================================

# plt.figure(
#     figsize=(8, 6)
# )

# plt.imshow(image)

# plt.title(
#     f"Actual: {label_names[true_label]} | "
#     f"Predicted: {label_names[predicted_label]}\n"
#     f"NAFLD Probability: {nafld_probability:.2%}"
# )

# plt.axis("off")

# plt.tight_layout()

# plt.show()


import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "behsof_vit.pt"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "behsof"
    / "test_predictions.csv"
)


# ============================================================
# Import model
# ============================================================

sys.path.append(
    str(Path(__file__).resolve().parent)
)

from behsof_model import (
    create_behsof_model,
    get_device
)


# ============================================================
# Device
# ============================================================

device = get_device()

print("=" * 60)
print("BEHSOF — NON-NAFLD TEST IMAGE")
print("=" * 60)

print("Device:", device)


# ============================================================
# Load test predictions
# ============================================================

df = pd.read_csv(
    PREDICTIONS_PATH
)


# ============================================================
# Select a TRUE Non-NAFLD image
# ============================================================

non_nafld_images = df[
    df["true_label"] == 0
]

if len(non_nafld_images) == 0:

    raise RuntimeError(
        "No Non-NAFLD images found in test set."
    )


row = non_nafld_images.iloc[0]

image_path = Path(
    row["image_path"]
)

true_label = int(
    row["true_label"]
)


# ============================================================
# Create model
# ============================================================

model = create_behsof_model(
    num_classes=2
)

model = model.to(device)


# ============================================================
# Load trained checkpoint
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# Transform
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ============================================================
# Load image
# ============================================================

image = Image.open(
    image_path
).convert("RGB")


input_tensor = transform(
    image
).unsqueeze(0).to(device)


# ============================================================
# Prediction
# ============================================================

with torch.no_grad():

    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=(device.type == "cuda")
    ):

        output = model(
            input_tensor
        )

    probabilities = torch.softmax(
        output,
        dim=1
    )[0]

    predicted_label = torch.argmax(
        probabilities
    ).item()


# ============================================================
# Probabilities
# ============================================================

non_nafld_probability = (
    probabilities[0].item()
)

nafld_probability = (
    probabilities[1].item()
)

confidence = (
    probabilities[predicted_label].item()
)


label_names = {
    0: "Non-NAFLD",
    1: "NAFLD"
}


# ============================================================
# Results
# ============================================================

print()
print("=" * 60)
print("IMAGE INFORMATION")
print("=" * 60)

print(
    "Image:",
    image_path.name
)

print(
    "Path:",
    image_path
)

print()
print("=" * 60)
print("PREDICTION")
print("=" * 60)

print(
    "Actual label    :",
    label_names[true_label]
)

print(
    "Predicted label :",
    label_names[predicted_label]
)

print(
    f"Non-NAFLD prob. : "
    f"{non_nafld_probability:.4f}"
)

print(
    f"NAFLD prob.     : "
    f"{nafld_probability:.4f}"
)

print(
    f"Confidence      : "
    f"{confidence:.4f}"
)

print(
    "Correct         :",
    predicted_label == true_label
)

print("=" * 60)


# ============================================================
# Display image
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.imshow(
    image
)

plt.title(
    f"Actual: {label_names[true_label]} | "
    f"Predicted: {label_names[predicted_label]}\n"
    f"NAFLD Probability: "
    f"{nafld_probability:.2%}"
)

plt.axis("off")

plt.tight_layout()

plt.show()