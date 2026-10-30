import torch
import timm


def create_behsof_model(num_classes=2):
    """
    ViT-B/16 for BEHSOF ultrasound NAFLD classification.

    Classes:
        0 -> Non-NAFLD
        1 -> NAFLD

    Fine-tuning strategy:
        - First transformer blocks: frozen
        - Last 3 transformer blocks: trainable
        - Final normalization: trainable
        - Classification head: trainable
    """

    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=True,
        num_classes=num_classes,
        drop_rate=0.1,
        drop_path_rate=0.1
    )

    # Freeze everything initially
    for param in model.parameters():
        param.requires_grad = False

    # Fine-tune last 3 transformer blocks
    for block in model.blocks[-3:]:
        for param in block.parameters():
            param.requires_grad = True

    # Final LayerNorm
    if hasattr(model, "norm") and model.norm is not None:
        for param in model.norm.parameters():
            param.requires_grad = True

    # Some timm ViT versions have fc_norm
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


def count_parameters(model):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


if __name__ == "__main__":

    device = get_device()

    print("Creating BEHSOF ViT model...")
    model = create_behsof_model()

    total, trainable = count_parameters(model)

    print()
    print("=" * 45)
    print("BEHSOF ViT-B/16")
    print("=" * 45)
    print(f"Device:              {device}")
    print(f"Total parameters:    {total:,}")
    print(f"Trainable parameters:{trainable:,}")
    print(
        f"Trainable percentage: "
        f"{trainable / total * 100:.2f}%"
    )
    print("=" * 45)