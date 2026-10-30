from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from ..behsof_model import create_behsof_model, get_device



class BEHSOFPredictor:

    def __init__(self):

        self.project_root = (
            Path(__file__).resolve().parent.parent.parent
        )

        self.model_path = (
            self.project_root
            / "saved_models"
            / "behsof_vit.pt"
        )

        self.device = get_device()

        # Create model
        self.model = create_behsof_model(
            num_classes=2
        )

        # Load checkpoint
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

        # Same preprocessing used during evaluation
        self.transform = transforms.Compose([

            transforms.Resize(
                (224, 224)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

    def predict(self, image_path):

        image = Image.open(
            image_path
        ).convert("RGB")

        input_tensor = (
            self.transform(image)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():

            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=(self.device.type == "cuda")
            ):

                output = self.model(
                    input_tensor
                )

            probabilities = torch.softmax(
                output,
                dim=1
            )[0]

        non_nafld_probability = (
            probabilities[0].item()
        )

        nafld_probability = (
            probabilities[1].item()
        )

        predicted_class = torch.argmax(
            probabilities
        ).item()

        if predicted_class == 0:
            prediction = "Non-NAFLD"
            confidence = non_nafld_probability
        else:
            prediction = "NAFLD"
            confidence = nafld_probability

        return {

            "model": "BEHSOF ViT-B/16",

            "task": "NAFLD assessment",

            "prediction": prediction,

            "non_nafld_probability":
                non_nafld_probability,

            "nafld_probability":
                nafld_probability,

            "confidence":
                confidence
        }