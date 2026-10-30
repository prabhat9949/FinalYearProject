import torch
import timm


def create_model(num_classes=2, freeze_backbone=True):

    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=True,
        num_classes=num_classes,
        drop_rate=0.1,
        drop_path_rate=0.1
    )

    if freeze_backbone:

        for param in model.parameters():
            param.requires_grad = False

        # Last 3 Transformer blocks
        for block in model.blocks[-3:]:
            for param in block.parameters():
                param.requires_grad = True

        # Final normalization
        if hasattr(model, "norm") and model.norm is not None:
            for param in model.norm.parameters():
                param.requires_grad = True

        if hasattr(model, "fc_norm") and model.fc_norm is not None:
            for param in model.fc_norm.parameters():
                param.requires_grad = True

        # Classification head
        for param in model.head.parameters():
            param.requires_grad = True

    return model


def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")