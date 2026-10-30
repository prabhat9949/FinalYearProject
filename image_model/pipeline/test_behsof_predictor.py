from pathlib import Path

from image_model.pipeline.behsof_predictor import BEHSOFPredictor



PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent.parent
)

TEST_PREDICTIONS = (
    PROJECT_ROOT
    / "outputs"
    / "behsof"
    / "test_predictions.csv"
)


import pandas as pd


df = pd.read_csv(
    TEST_PREDICTIONS
)

# Take one known test image
row = df.iloc[0]

image_path = Path(
    row["image_path"]
)

print("=" * 60)
print("BEHSOF PIPELINE TEST")
print("=" * 60)

print("Image:", image_path)

predictor = BEHSOFPredictor()

result = predictor.predict(
    image_path
)

print()
print(result)

print("=" * 60)