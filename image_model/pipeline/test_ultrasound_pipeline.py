from pathlib import Path

from image_model.pipeline.ultrasound_pipeline import UltrasoundPipeline



# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent.parent
)


# ============================================================
# TEST IMAGE
# ============================================================

IMAGE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "byra"
    / "test_patient_11_image_0.png"
)


# ============================================================
# START PIPELINE
# ============================================================

print()
print("=" * 70)
print("ULTRASOUND PIPELINE TEST")
print("=" * 70)

print(
    f"Image: {IMAGE_PATH}"
)


pipeline = UltrasoundPipeline()


# ============================================================
# ANALYZE IMAGE
# ============================================================

result = pipeline.analyze(
    IMAGE_PATH
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("=" * 70)
print("FINAL ULTRASOUND RESULT")
print("=" * 70)


print()
print("BYRA / IJCARS")
print("-" * 40)

print(
    "Prediction:",
    result["ultrasound_analysis"]
          ["byra"]["prediction"]
)

print(
    "Normal probability:",
    result["ultrasound_analysis"]
          ["byra"]["normal_probability"]
)

print(
    "Steatosis probability:",
    result["ultrasound_analysis"]
          ["byra"]["steatosis_probability"]
)

print(
    "Confidence:",
    result["ultrasound_analysis"]
          ["byra"]["confidence"]
)


print()
print("BEHSOF")
print("-" * 40)

print(
    "Prediction:",
    result["ultrasound_analysis"]
          ["behsof"]["prediction"]
)

print(
    "Non-NAFLD probability:",
    result["ultrasound_analysis"]
          ["behsof"]["non_nafld_probability"]
)

print(
    "NAFLD probability:",
    result["ultrasound_analysis"]
          ["behsof"]["nafld_probability"]
)

print(
    "Confidence:",
    result["ultrasound_analysis"]
          ["behsof"]["confidence"]
)


print()
print("SUMMARY")
print("-" * 40)

print(
    result["ultrasound_analysis"]
          ["summary"]
)


print()
print("=" * 70)