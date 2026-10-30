from pathlib import Path
import sys

from sklearn.model_selection import train_test_split

# Allow importing from image-model
sys.path.append(str(Path(__file__).resolve().parent))

from behsof_dataset import get_samples


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


# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# Load dataset
# ============================================================

samples = get_samples(
    DATASET_ROOT,
    remove_duplicates=True
)

print("\nDataset loaded.")


# ============================================================
# Separate paths and labels
# ============================================================

paths = [sample[0] for sample in samples]
labels = [sample[1] for sample in samples]


print(f"Total unique images: {len(samples)}")


# ============================================================
# Train / temporary split
# ============================================================

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    paths,
    labels,
    test_size=(VAL_RATIO + TEST_RATIO),
    stratify=labels,
    random_state=RANDOM_SEED
)


# ============================================================
# Validation / Test split
# ============================================================

relative_test_size = TEST_RATIO / (VAL_RATIO + TEST_RATIO)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths,
    temp_labels,
    test_size=relative_test_size,
    stratify=temp_labels,
    random_state=RANDOM_SEED
)


# ============================================================
# Reconstruct samples
# ============================================================

train_samples = list(zip(train_paths, train_labels))
val_samples = list(zip(val_paths, val_labels))
test_samples = list(zip(test_paths, test_labels))


# ============================================================
# Helper
# ============================================================

def print_distribution(name, samples):

    class_0 = sum(label == 0 for _, label in samples)
    class_1 = sum(label == 1 for _, label in samples)

    print(
        f"{name}: "
        f"{len(samples)} images | "
        f"Non-NAFLD={class_0} | "
        f"NAFLD={class_1}"
    )


# ============================================================
# Print results
# ============================================================

print("\n========================================")
print("BEHSOF DATASET SPLIT")
print("========================================")

print_distribution("Train", train_samples)
print_distribution("Validation", val_samples)
print_distribution("Test", test_samples)

print("========================================")

print(f"\nRandom seed: {RANDOM_SEED}")
print(f"Train ratio: {TRAIN_RATIO}")
print(f"Validation ratio: {VAL_RATIO}")
print(f"Test ratio: {TEST_RATIO}")