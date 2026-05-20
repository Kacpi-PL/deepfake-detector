import os
import pandas as pd
import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import pytorch_lightning as pl
import torch.optim as optim
from torchmetrics import Accuracy, F1Score
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger

def build_binary_resnet() -> nn.Module:

    # function for initializing resnet50 model with no head returning model

    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    num_features = model.fc.in_features
    # setting dropout to potentially avoid domain shift
    model.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(num_features, 2))
    return model

# Standard custom dataset class
class CustomImageDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, target_transform=None):
        self.img_labels = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = int(self.img_labels.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

# Using pytorch_lightning.LightningModule to wrap whole process of training, get epochs metrics and set optimizer
class DeepfakeDetector(pl.LightningModule):
    def __init__(self, base_model, learning_rate=0.0001):
        super().__init__()
        self.model = base_model
        # Multi-class CrossEntropyLoss utilized for binary classification (N=2) to maintain
        # output shape flexibility, enabling seamless integration with Grad-CAM targeting
        # and easy scalability to multi-generator deepfake detection in the future.
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

        self.train_acc = Accuracy(task="multiclass", num_classes=2)
        self.val_acc = Accuracy(task="multiclass", num_classes=2)

        self.train_f1 = F1Score(task="multiclass", num_classes=2)
        self.val_f1 = F1Score(task="multiclass", num_classes=2)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        preds = torch.argmax(outputs, dim=1)
        train_acc = self.train_acc(preds, labels)
        train_f1 = self.train_f1(preds, labels)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_acc", train_acc, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_f1", train_f1, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        preds = torch.argmax(outputs, dim=1)
        val_acc = self.val_acc(preds, labels)
        val_f1 = self.val_f1(preds, labels)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_acc", val_acc, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_f1", val_f1, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    base_model = build_binary_resnet()

    PATH = "../data/real-vs-fake-faces-stylegan3"

    # Dict for mapping label column
    LABEL_MAP = {"fake": 0, "real": 1}

    # Splitting df to two sets with sklearn. Changed it to fixed df(exported csv) to maintain reproducibility.
    # train_df, temp_df = train_test_split(
    #     df, test_size=0.30, random_state=42, stratify=df["label"]
    # )
    # val_df, test_df = train_test_split(
    #     temp_df, test_size=0.50, random_state=42, stratify=temp_df["label"]
    # )

    train_df = pd.read_csv(os.path.join(PATH, "train_split.csv"))
    val_df = pd.read_csv(os.path.join(PATH, "val_split.csv"))
    #test_df = pd.read_csv(os.path.join(PATH, "test_split.csv"))

    # setting transforms for both test and train
    train_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.RandomApply([v2.JPEG(quality=(80, 100))], p=0.2),
            v2.Resize(256, antialias=True),
            v2.RandomCrop(224),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(imagenet_mean, imagenet_std),
        ]
    )

    test_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(256, antialias=True),
            v2.CenterCrop(224),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(imagenet_mean, imagenet_std),
        ]
    )

    # Creation dataset objects
    train_dataset = CustomImageDataset(train_df, PATH, transform=train_transform)
    val_dataset = CustomImageDataset(val_df, PATH, transform=test_transform)
    #test_dataset = CustomImageDataset(test_df, PATH, transform=test_transform)

    # Wrapping datasets with dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    logger = CSVLogger("../logs", name="resnet50_deepfake")

    lightning_model = DeepfakeDetector(base_model=base_model)

    checkpoint_callback = ModelCheckpoint(
        dirpath="models",
        filename='resnet50-{epoch:02d}-{val_loss:.4f}-{val_f1:.4f}',
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        min_delta=0.001,
        mode="min",
        verbose=True,
    )

    trainer = Trainer(
        max_epochs=30,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator="auto",
        log_every_n_steps=1,
        logger=logger,
        enable_progress_bar=True,
    )
    trainer.fit(lightning_model, train_dataloader, val_dataloader)
