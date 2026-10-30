import scipy.io as sio
import os
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

path = os.path.join(
    BASE_DIR,
    "data",
    "ultrasound",
    "dataset_liver_bmodes_steatosis_assessment_IJCARS.mat"
)

data = sio.loadmat(path)
patients = data["data"]

print("=" * 70)
print("LABEL DISTRIBUTION")
print("=" * 70)

classes = []
fat_levels = []
image_counts = []

for i in range(patients.shape[1]):

    patient = patients[0, i]

    patient_id = int(patient["id"][0, 0])
    patient_class = int(patient["class"][0, 0])
    patient_fat = int(patient["fat"][0, 0])

    images = patient["images"]

    classes.append(patient_class)
    fat_levels.append(patient_fat)
    image_counts.append(images.shape[0])

    print(
        f"Patient {patient_id:02d} | "
        f"class={patient_class} | "
        f"fat={patient_fat} | "
        f"images={images.shape[0]}"
    )

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("\nClass distribution:")
for label, count in sorted(Counter(classes).items()):
    print(f"  class {label}: {count} patients")

print("\nFat distribution:")
for label, count in sorted(Counter(fat_levels).items()):
    print(f"  fat {label}: {count} patients")

print("\nImage count:")
print("  Total patients:", len(image_counts))
print("  Total images:", sum(image_counts))
print("  Images per patient:", sorted(set(image_counts)))