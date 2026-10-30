from pathlib import Path

from image_model.pipeline.byra_predictor import ByraPredictor



PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent.parent
)


IMAGE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "byra"
    / "test_patient_11_image_0.png"
)


print("=" * 60)
print("BYRA PIPELINE TEST")
print("=" * 60)


predictor = ByraPredictor()


result = predictor.predict(
    IMAGE_PATH
)


print()
print("Image:")
print(IMAGE_PATH)

print()
print("Prediction:")
print(result)

print("=" * 60)