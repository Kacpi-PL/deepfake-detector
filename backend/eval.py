import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.transforms import v2
from torch.utils.data import DataLoader

from backend.train import DeepfakeDetector, build_binary_resnet, CustomImageDataset

# GLOBAL CONSTANTS
CLASS_NAMES = {0: "StyleGAN3 (Fake)", 1: "Real"}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denormalize_tensor_to_image(tensor):

    img = tensor.cpu().permute(1, 2, 0).numpy()
    img = IMAGENET_STD * img + IMAGENET_MEAN
    return np.clip(img, 0, 1)



def explain_random_test_image(dataloader, model, device="cuda"):

    batch_features, batch_labels = next(iter(dataloader))
    random_idx = random.randint(0, batch_features.size(0) - 1)

    input_tensor = batch_features[random_idx].unsqueeze(0).to(device)
    true_label_idx = batch_labels[random_idx].item()

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        pred_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][pred_idx].item() * 100

    target_layers = [model.model.layer4[-1]]
    with GradCAM(model=model, target_layers=target_layers) as cam:
        targets = [ClassifierOutputTarget(pred_idx)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

    img_for_display = denormalize_tensor_to_image(input_tensor[0])
    visualization = show_cam_on_image(img_for_display, grayscale_cam, use_rgb=True)

    is_correct = true_label_idx == pred_idx
    color = "green" if is_correct else "red"

    title = (
        f"True: {CLASS_NAMES[true_label_idx]} | Pred: {CLASS_NAMES[pred_idx]} ({confidence:.1f}%)\n"
        f"{'Model prediction correct' if is_correct else 'Model prediction not correct'}"
    )

    plt.figure(figsize=(7, 7))
    plt.imshow(visualization)
    plt.title(title, fontsize=12, fontweight="bold", color=color, pad=15)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def evaluate_and_plot_matrix(dataloader, model, device="cuda"):

    all_preds, all_true_labels = [], []

    with torch.no_grad():
        for batch_features, batch_labels in tqdm(dataloader, desc="Eval"):
            batch_features = batch_features.to(device)
            outputs = model(batch_features)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_true_labels.extend(batch_labels.numpy())

    # Rysowanie macierzy
    cm = confusion_matrix(all_true_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=[CLASS_NAMES[0], CLASS_NAMES[1]],
        yticklabels=[CLASS_NAMES[0], CLASS_NAMES[1]],
        annot_kws={"size": 16, "weight": "bold"},
    )
    plt.title("Confusion Matrix", fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("Model Prediction", fontsize=14, labelpad=10)
    plt.ylabel("Real Label", fontsize=14, labelpad=10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12, rotation=0)
    plt.tight_layout()
    plt.savefig("confusion_matrix_portfolio.png", dpi=300)
    plt.show()

    print("\n--- Classification Metrics Report ---")
    print(
        classification_report(
            all_true_labels,
            all_preds,
            target_names=[CLASS_NAMES[0], CLASS_NAMES[1]],
            digits=3,
        )
    )


def show_misclassified_images_with_gradcam(
    dataloader, model, device="cuda", max_images=9
):


    misclassified = []
    target_layers = [model.model.layer4[-1]]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        for batch_features, batch_labels in dataloader:
            batch_features, batch_labels = (
                batch_features.to(device),
                batch_labels.to(device),
            )

            with torch.no_grad():
                preds = torch.argmax(model(batch_features), dim=1)

            errors = preds != batch_labels

            for i in range(len(errors)):
                if errors[i]:
                    input_tensor = batch_features[i].unsqueeze(0)
                    true_label, pred_label = batch_labels[i].item(), preds[i].item()

                    targets = [ClassifierOutputTarget(pred_label)]
                    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[
                        0, :
                    ]

                    img_rgb = denormalize_tensor_to_image(batch_features[i])
                    visualization = show_cam_on_image(
                        img_rgb, grayscale_cam, use_rgb=True
                    )

                    misclassified.append(
                        {
                            "visual": visualization,
                            "true_label": true_label,
                            "pred_label": pred_label,
                        }
                    )

                    if len(misclassified) >= max_images:
                        break
            if len(misclassified) >= max_images:
                break

    if not misclassified:
        print("No incorrect preds found.")
        return

    rows = int(np.ceil(np.sqrt(len(misclassified))))
    cols = int(np.ceil(len(misclassified) / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 14))
    axes = axes.flatten()

    for idx, item in enumerate(misclassified):
        axes[idx].imshow(item["visual"])
        axes[idx].axis("off")
        axes[idx].set_title(
            f"True: {CLASS_NAMES[item['true_label']]}\nPred: {CLASS_NAMES[item['pred_label']]}",
            fontsize=11,
            color="darkred",
            fontweight="bold",
            pad=10,
        )

    for i in range(len(misclassified), len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle(
        "XAI Error Analysis: Visualizing False Predictions",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig("xai_error_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()



if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PATH = "../data/real-vs-fake-faces-stylegan3"
    BEST_MODEL_PATH = "../models/resnet50-epoch=23-val_loss=0.0100-val_f1=0.9973.ckpt"

    # Setup Dataset & DataLoader
    test_df = pd.read_csv(f"{PATH}/test_split.csv")
    test_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(256, antialias=True),
            v2.CenterCrop(224),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    test_dataloader = DataLoader(
        CustomImageDataset(test_df, PATH, transform=test_transform),
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # Load Model
    base_model = build_binary_resnet()
    trained_model = DeepfakeDetector.load_from_checkpoint(
        BEST_MODEL_PATH, base_model=base_model
    )
    trained_model.to(DEVICE)
    trained_model.eval()

    # Execute
    explain_random_test_image(test_dataloader, trained_model, device=DEVICE)
    evaluate_and_plot_matrix(test_dataloader, trained_model, device=DEVICE)
    show_misclassified_images_with_gradcam(test_dataloader, trained_model, device=DEVICE, max_images=9)