"""
------------------------------------------------------------------------------
Filename: train_and_test_models.py

Purpose:
    Train and evaluate two ResNet18 models (pretrained and untrained) on an
    image classification dataset with JSON annotations of the form:
        { "name": "<class_string>" }

    Script features:
      - Converts string labels ("name") -> integer IDs via a label_map.
      - Uses Apple M1/M2/M3 MPS backend when available, otherwise CUDA or CPU.
      - Trains both a pretrained ResNet18 (ImageNet weights) and an untrained
        randomly initialized ResNet18.
      - Computes and saves confusion matrices (PNG) for each trained model.
      - Saves trained model weights (.pth) for each model.
      - Writes a JSON metrics summary into OUTPUT/FINAL/metrics.json.
      - Produces detailed logging and is heavily commented for clarity.

Directory structure expected (relative to where this script is run):
    train/img/     -> images (png/jpg/jpeg)
    train/ann/     -> json annotations with same name as images (but .json)
    val/img/, val/ann/
    test/img/, test/ann/

Outputs (all written into OUTPUT/FINAL/):
    - {model_name}.pth
    - {model_name}_confusion_matrix.png
    - metrics.json

Notes:
    - Update hyperparameters (epochs, batch size, learning rate) below.
------------------------------------------------------------------------------
"""

import os
import json
import errno
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ---------------------------
# DATA-SPECIFIC VALUES
# ---------------------------
# The order of this list determines the integer mapping (0..22).
LABEL_LIST: List[str] = [
    "cinnamomum_camphora_(linn)_presl",
    "ginkgo_biloba",
    "sapindus_saponaria",
    "koelreuteria_paniculata",
    "magnolia_grandiflora_l",
    "platanus",
    "magnolia_liliflora_desr",
    "celtis_sinensis",
    "zelkova_serrata",
    "liriodendron_chinense",
    "elaeocarpus_decipiens",
    "osmanthus_fragrans",
    "liquidambar_formosana",
    "acer_palmatum",
    "styphnolobium_japonicum",
    "michelia_chapensis",
    "triadica_sebifera",
    "salix_babylonica",
    "cedrus_deodara",
    "photinia_serratifolia",
    "flowering_cherry",
    "prunus_cerasifera_f._atropurpurea",
    "lagerstroemia_indica"
]

# Build a label -> integer index map
LABEL_MAP: Dict[str, int] = {name: idx for idx, name in enumerate(LABEL_LIST)}
NUM_CLASSES = len(LABEL_LIST)  # should be 23

# Data paths (relative)
TRAIN_IMG_DIR = "train/img"
TRAIN_ANN_DIR = "train/ann"
VAL_IMG_DIR = "val/img"
VAL_ANN_DIR = "val/ann"
TEST_IMG_DIR = "test/img"
TEST_ANN_DIR = "test/ann"

# Output directory where we will save models, confusion matrices, metrics
OUTPUT_DIR = Path("OUTPUT/FINAL")

# Hyperparameters (simple defaults; change if desired)
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-4
SEED = 42  # for repeatability where possible


# ---------------------------
# Utility: ensure output directories exist
# ---------------------------
def make_output_dirs():
    """Create the OUTPUT/FINAL directory if it does not already exist."""
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


# ---------------------------
# Device selection
# ---------------------------
def get_device() -> torch.device:
    """
    Return an appropriate device string for this machine:
      - 'mps' if Apple Silicon (M1/M2/M3) supports MPS backend
      - 'cuda' if NVIDIA CUDA is available
      - 'cpu' otherwise

    Using MPS on Apple Silicon allows GPU-accelerated training where supported.
    """
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------
# Dataset class (reads image + json "name")
# ---------------------------
class ImageJsonDataset(Dataset):
    """
    A dataset that returns (image_tensor, label_index) pairs.

    Expects:
      - image files in img_dir (png/jpg/jpeg)
      - JSON annotation files with the same base filename in ann_dir
        that include at least one of the keys: "name" (string) or "label" (int).
    """

    def __init__(self, img_dir: str, ann_dir: str, transform=None, label_map: Dict[str, int] = None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transform = transform
        # label_map maps string class names -> integer indices
        self.label_map = label_map or {}

        # Collect image filenames (sorted for deterministic order)
        self.files = sorted([
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith(("png", "jpg", "jpeg"))
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Returns:
          - img_tensor (C,H,W)
          - label_index (int, 0..NUM_CLASSES-1)
        """
        img_name = self.files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # Open the image and convert to RGB (ensures 3 channels)
        img = Image.open(img_path).convert("RGB")

        # Corresponding JSON annotation
        ann_name = img_name.rsplit(".", 1)[0] + ".json"
        ann_path = os.path.join(self.ann_dir, ann_name)

        # Read JSON and extract either "name" or "label"
        with open(ann_path, "r") as f:
            ann = json.load(f)

        # Prefer "name" (string) annotations; fall back to integer "label"
        if "name" in ann:
            class_name = ann["name"]
            if class_name not in self.label_map:
                # If a name is encountered that's not in the provided label_map,
                # raise a clear error so the user can fix their mapping or dataset.
                raise KeyError(f"Annotation name '{class_name}' not found in LABEL_MAP.")
            label_idx = int(self.label_map[class_name])
        elif "label" in ann:
            # Allow integer label if present
            label_idx = int(ann["label"])
        else:
            # Neither key found — dataset annotation format mismatch
            raise KeyError(f"Annotation JSON {ann_path} missing 'name' or 'label' key.")

        # Apply transforms if provided (resize, to-tensor, normalization, etc.)
        if self.transform:
            img = self.transform(img)

        return img, label_idx


# ---------------------------
# Training / evaluation helpers
# ---------------------------
def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, criterion, device: torch.device):
    """Run one epoch of training and return training accuracy (float)."""
    model.train()
    total = 0
    correct = 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total if total > 0 else 0.0


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, collect=False):
    """
    Evaluate model on loader. If collect=True, also return (preds_array, labels_array).
    Returns:
      - accuracy (float)
      - (optional) preds, labels as numpy arrays
    """
    model.eval()
    total = 0
    correct = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if collect:
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

    acc = correct / total if total > 0 else 0.0
    if collect:
        return acc, np.array(all_preds, dtype=int), np.array(all_labels, dtype=int)
    else:
        return acc


# ---------------------------
# Confusion matrix plotting / saving
# ---------------------------
def save_confusion_matrix(labels: np.ndarray, preds: np.ndarray, class_names: List[str], out_path: Path, title: str = ""):
    """
    Save a confusion matrix plot to out_path (PNG). Uses sklearn ConfusionMatrixDisplay.
    """
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    # Plot with matplotlib; tighten layout and save to file (no interactive show)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(ax=ax, xticks_rotation="vertical", values_format='d', cmap="Blues")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------------------------
# Main execution
# ---------------------------
def main():
    # Set random seeds for reproducibility where possible
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Ensure output folders exist
    make_output_dirs()

    # Device selection and logging
    device = get_device()
    print(f"[INFO] Using device: {device}")

    # Transforms:
    # We include ImageNet normalization because we use pretrained ResNet weights.
    # Normalizing untrained model inputs is still fine.
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = T.Compose([
        T.Resize((224, 224)),  # ResNet expects 224x224 default
        T.ToTensor(),
        T.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    test_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    # Construct datasets (pass label_map so string names -> ints are handled)
    train_ds = ImageJsonDataset(TRAIN_IMG_DIR, TRAIN_ANN_DIR, transform=train_transform, label_map=LABEL_MAP)
    val_ds = ImageJsonDataset(VAL_IMG_DIR, VAL_ANN_DIR, transform=test_transform, label_map=LABEL_MAP)
    test_ds = ImageJsonDataset(TEST_IMG_DIR, TEST_ANN_DIR, transform=test_transform, label_map=LABEL_MAP)

    # DataLoaders: shuffle train, do not shuffle validation/test
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # We'll store results in a dictionary and write to OUTPUT/FINAL/metrics.json at end
    results = {}

    # Loop through two configurations: pretrained and untrained
    models_to_run = [
        ("ResNet18_Pretrained", True),
        ("ResNet18_Untrained", False)
    ]

    for model_name, use_pretrained in models_to_run:
        print(f"\n[INFO] ======== Starting run: {model_name} (pretrained={use_pretrained}) ========")

        # Instantiate model: either with ImageNet weights or random init
        if use_pretrained:
            # When using torchvision weights enums: this should be compatible with torchvision 0.17.x
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet18(weights=None)

        # Replace the final fully-connected layer to match NUM_CLASSES
        model.fc = nn.Linear(in_features=512, out_features=NUM_CLASSES)

        # Move model to chosen device
        model = model.to(device)

        # Loss & optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Training loop (simple; prints train and val accuracy per epoch)
        for epoch in range(1, EPOCHS + 1):
            train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_acc = evaluate_model(model, val_loader, device, collect=False)
            print(f"[{model_name}] Epoch {epoch}/{EPOCHS}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

        # Final evaluation on test set; collect predictions for confusion matrix
        test_acc, preds, labels = evaluate_model(model, test_loader, device, collect=True)
        print(f"[{model_name}] Final Test Accuracy: {test_acc:.4f}")

        # Save model weights to OUTPUT/FINAL/{model_name}.pth
        model_out_path = OUTPUT_DIR / f"{model_name}.pth"
        # Use CPU save (recommended) so the saved state_dict can be loaded on different devices
        torch.save(model.state_dict(), str(model_out_path))
        print(f"[INFO] Saved model weights to: {model_out_path}")

        # Save confusion matrix PNG to OUTPUT/FINAL/{model_name}_confusion_matrix.png
        cm_out_path = OUTPUT_DIR / f"{model_name}_confusion_matrix.png"
        save_confusion_matrix(labels=labels, preds=preds, class_names=LABEL_LIST,
                              out_path=cm_out_path, title=f"Confusion Matrix - {model_name}")
        print(f"[INFO] Saved confusion matrix to: {cm_out_path}")

        # Record results for summary
        results[model_name] = {
            "test_accuracy": float(test_acc),
            "model_path": str(model_out_path),
            "confusion_matrix_path": str(cm_out_path)
        }

    # Save overall metrics summary JSON
    metrics_out_path = OUTPUT_DIR / "metrics.json"
    with open(metrics_out_path, "w") as fh:
        json.dump({
            "num_classes": NUM_CLASSES,
            "label_list": LABEL_LIST,
            "results": results
        }, fh, indent=2)
    print(f"\n[INFO] Saved metrics summary to: {metrics_out_path}")

    # Also print a short summary to console
    print("\n[INFO] ======= Final summary =======")
    for name, info in results.items():
        print(f" - {name}: test_accuracy={info['test_accuracy']:.4f}  model={info['model_path']}  cm={info['confusion_matrix_path']}")

    print("\n[INFO] FINISHED.")


# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    main()
