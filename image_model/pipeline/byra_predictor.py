from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from ..model import create_model, get_device



class ByraPredictor:

    def __init__(self):

        # ==================================================
        # PROJECT ROOT
        # ==================================================

        self.project_root = (
            Path(__file__).resolve().parent.parent.parent
        )

        # ==================================================
        # MODEL PATH
        # ==================================================

        self.model_path = (
            self.project_root
            / "saved_models"
            / "liver_vit.pt"
        )

        # ==================================================
        # DEVICE
        # ==================================================

        self.device = get_device()

        print(
            f"Byra model device: {self.device}"
        )

        # ==================================================
        # CREATE MODEL
        # ==================================================

        self.model = create_model(
            num_classes=2,
            freeze_backbone=True
        )

        # ==================================================
        # LOAD CHECKPOINT
        # ==================================================

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model = self.model.to(
            self.device
        )

        self.model.eval()

        print(
            "Byra checkpoint loaded successfully."
        )

        # ==================================================
        # SAME PREPROCESSING AS evaluate.py
        # ==================================================

        self.transform = transforms.Compose([

            transforms.Resize(
                (224, 224)
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

    def predict(self, image_path):

        # ==================================================
        # LOAD IMAGE
        # ==================================================

        image = Image.open(
            image_path
        ).convert("RGB")

        # ==================================================
        # PREPROCESS
        # ==================================================

        input_tensor = (
            self.transform(image)
            .unsqueeze(0)
            .to(self.device)
        )

        # ==================================================
        # PREDICTION
        # ==================================================

        with torch.no_grad():

            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=(
                    self.device.type == "cuda"
                )
            ):

                output = self.model(
                    input_tensor
                )

            probabilities = torch.softmax(
                output,
                dim=1
            )[0]

        # ==================================================
        # CLASS PROBABILITIES
        #
        # 0 = Normal
        # 1 = Steatosis
        # ==================================================

        normal_probability = (
            probabilities[0].item()
        )

        steatosis_probability = (
            probabilities[1].item()
        )

        # ==================================================
        # PREDICTION
        # ==================================================

        predicted_class = torch.argmax(
            probabilities
        ).item()

        if predicted_class == 0:

            prediction = "Normal"

            confidence = normal_probability

        else:

            prediction = "Steatosis"

            confidence = steatosis_probability

        # ==================================================
        # RESULT
        # ==================================================

        return {

            "model":
                "Byra/IJCARS ViT-B/16",

            "task":
                "Steatosis assessment",

            "prediction":
                prediction,

            "normal_probability":
                normal_probability,

            "steatosis_probability":
                steatosis_probability,

            "confidence":
                confidence
        }