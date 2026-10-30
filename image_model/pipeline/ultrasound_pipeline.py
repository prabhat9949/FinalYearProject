from pathlib import Path

from .byra_predictor import ByraPredictor

from .behsof_predictor import BEHSOFPredictor


class UltrasoundPipeline:

    def __init__(self):

        print("=" * 60)
        print("INITIALIZING ULTRASOUND PIPELINE")
        print("=" * 60)

        # -----------------------------------------------
        # Initialize both models
        # -----------------------------------------------

        self.byra = ByraPredictor()

        self.behsof = BEHSOFPredictor()

        print()
        print("Both ultrasound models loaded successfully.")

    def analyze(self, image_path):

        image_path = Path(image_path)

        if not image_path.exists():

            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        # -----------------------------------------------
        # Byra prediction
        # -----------------------------------------------

        byra_result = self.byra.predict(
            image_path
        )

        # -----------------------------------------------
        # BEHSOF prediction
        # -----------------------------------------------

        behsof_result = self.behsof.predict(
            image_path
        )

        # -----------------------------------------------
        # Evidence summary
        # -----------------------------------------------

        byra_prediction = (
            byra_result["prediction"]
        )

        behsof_prediction = (
            behsof_result["prediction"]
        )

        if (
            byra_prediction == "Steatosis"
            and
            behsof_prediction == "NAFLD"
        ):

            summary = (
                "Both ultrasound models detected "
                "findings consistent with fatty liver / NAFLD."
            )

        elif (
            byra_prediction == "Normal"
            and
            behsof_prediction == "Non-NAFLD"
        ):

            summary = (
                "Both ultrasound models produced "
                "findings not indicating steatosis / NAFLD."
            )

        else:

            summary = (
                "The ultrasound models produced "
                "different task-specific findings. "
                "Review both results separately."
            )

        # -----------------------------------------------
        # Final structured response
        # -----------------------------------------------

        return {

            "ultrasound_analysis": {

                "byra": byra_result,

                "behsof": behsof_result,

                "summary": summary
            }
        }


def analyze_ultrasound(image_path):

    pipeline = UltrasoundPipeline()

    return pipeline.analyze(
        image_path
    )