import os
import scipy.io as sio
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


class LiverUltrasoundDataset(Dataset):

    def __init__(self, samples, transform=None):
        """
        samples:
            List of tuples:
            (patient_id, image_index, label, image_array)

        transform:
            torchvision transforms
        """

        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        patient_id, image_index, label, image_array = self.samples[index]

        # uint8 grayscale → PIL
        image = Image.fromarray(image_array).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def load_patients(mat_path):
    """
    Load patient-level records from the MATLAB dataset.
    """

    data = sio.loadmat(mat_path)

    patients = data["data"]

    patient_records = []

    for i in range(patients.shape[1]):

        patient = patients[0, i]

        patient_id = int(patient["id"][0, 0])
        label = int(patient["class"][0, 0])

        images = patient["images"]

        patient_records.append({
            "patient_id": patient_id,
            "label": label,
            "images": images
        })

    return patient_records


def create_samples(patient_records):

    """
    Convert patient records into image samples.

    IMPORTANT:
    We do this AFTER splitting patients,
    so images from the same patient never
    cross train/validation/test.
    """

    samples = []

    for patient in patient_records:

        patient_id = patient["patient_id"]
        label = patient["label"]
        images = patient["images"]

        for image_index in range(images.shape[0]):

            image = images[image_index]

            samples.append(
                (
                    patient_id,
                    image_index,
                    label,
                    image
                )
            )

    return samples