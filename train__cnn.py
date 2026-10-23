
from pathlib import Path
import copy
import random

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

# ---------------- CONFIGURATION ----------------
SEED = 42
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 0.001
VALIDATION_FRACTION = 0.15

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Training device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ---------------- DATA ----------------
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

evaluation_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

train_path = DATA_DIR / "train"
test_path = DATA_DIR / "test"

if not train_path.is_dir() or not test_path.is_dir():
    raise FileNotFoundError(
        f"Expected dataset folders at {train_path} and {test_path}"
    )

# Load training data twice so validation has no random augmentation.
full_train_augmented = datasets.ImageFolder(
    train_path, transform=train_transform
)
full_train_evaluation = datasets.ImageFolder(
    train_path, transform=evaluation_transform
)
test_dataset = datasets.ImageFolder(
    test_path, transform=evaluation_transform
)

class_names = full_train_augmented.classes

if class_names != full_train_evaluation.classes:
    raise ValueError("Training dataset class mappings do not match.")

if class_names != test_dataset.classes:
    raise ValueError(
        "Train and test class mappings differ. Check folder names."
    )

# Split by sample index, preserving the supplied test set.
generator = torch.Generator().manual_seed(SEED)
total_size = len(full_train_augmented)
validation_size = int(total_size * VALIDATION_FRACTION)
training_size = total_size - validation_size

train_subset, validation_subset = random_split(
    full_train_augmented,
    [training_size, validation_size],
    generator=generator,
)

# Validation uses the same indices but no augmentation.
validation_indices = validation_subset.indices
validation_subset = torch.utils.data.Subset(
    full_train_evaluation, validation_indices
)

train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)
validation_loader = DataLoader(
    validation_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)

print(f"\nClasses: {class_names}")
print(f"Training samples: {training_size}")
print(f"Validation samples: {validation_size}")
print(f"Test samples: {len(test_dataset)}")

# ---------------- CNN MODEL ----------------
class FacialExpressionCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


model = FacialExpressionCNN(len(class_names)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE
)

# ---------------- TRAIN / EVALUATE ----------------
def run_epoch(loader, training):
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(
            device, non_blocking=(device.type == "cuda")
        )
        labels = labels.to(
            device, non_blocking=(device.type == "cuda")
        )

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    return (
        total_loss / total_samples,
        total_correct / total_samples,
    )


best_validation_loss = float("inf")
best_state = None
history = []

for epoch in range(1, EPOCHS + 1):
    train_loss, train_accuracy = run_epoch(train_loader, training=True)

    with torch.no_grad():
        validation_loss, validation_accuracy = run_epoch(
            validation_loader, training=False
        )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "validation_loss": validation_loss,
        "validation_accuracy": validation_accuracy,
    })

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train loss: {train_loss:.4f} | "
        f"Train acc: {train_accuracy:.4f} | "
        f"Val loss: {validation_loss:.4f} | "
        f"Val acc: {validation_accuracy:.4f}"
    )

    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        best_state = copy.deepcopy(model.state_dict())

        torch.save({
            "model_state_dict": best_state,
            "class_names": class_names,
            "input_size": [48, 48],
            "model_name": "FacialExpressionCNN",
        }, MODEL_DIR / "best_cnn.pth")

# ---------------- TEST EVALUATION ----------------
if best_state is None:
    raise RuntimeError("Training did not produce a valid model.")

model.load_state_dict(best_state)
model.eval()

all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.numpy().tolist())
        all_predictions.extend(predictions.cpu().numpy().tolist())

accuracy = np.mean(
    np.array(all_labels) == np.array(all_predictions)
)
macro_f1 = f1_score(
    all_labels, all_predictions, average="macro", zero_division=0
)

print("\n--- FINAL TEST RESULTS ---")
print(f"Test accuracy: {accuracy:.4f}")
print(f"Test macro-F1: {macro_f1:.4f}")
print("\nClassification report:")
print(classification_report(
    all_labels,
    all_predictions,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0,
))

matrix = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(class_names))),
)

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=class_names,
)
display.plot(xticks_rotation=45, cmap="Blues")
plt.title("CNN Test Confusion Matrix")
plt.tight_layout()
plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=150)
plt.close()

history_df = __import__("pandas").DataFrame(history)
history_df.to_csv(REPORT_DIR / "training_history.csv", index=False)

print(f"\nBest model saved to: {MODEL_DIR / 'best_cnn.pth'}")
print(f"Reports saved to: {REPORT_DIR}")
