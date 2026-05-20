import torch
from backend.train import build_binary_resnet, DeepfakeDetector


def convert_ckpt_to_pth(ckpt_path: str, output_path: str):
    print("Extracting weights from checkpoint...")


    base_model = build_binary_resnet()
    lightning_model = DeepfakeDetector.load_from_checkpoint(
        ckpt_path, base_model=base_model
    )

    pure_weights = lightning_model.model.state_dict()

    torch.save(pure_weights, output_path)
    print(f"Succes, file saved as: {output_path}")


if __name__ == "__main__":
    # To extract your weights from your very own ckpt change CKPT_PATH
    CKPT_PATH = "../models/resnet50-epoch=23-val_loss=0.0100-val_f1=0.9973.ckpt"
    OUTPUT_PATH = "resnet50_deepfake_weights.pth"

    convert_ckpt_to_pth(CKPT_PATH, OUTPUT_PATH)