# Input states for Agent step function
from grid_adventure.grid import GridState
from grid_adventure.env import ImageObservation

# State steppers
from grid_adventure.step import Action
from grid_adventure.grid import step as grid_step

# Movements and Objectives are gridstate parameters. For Grid Adventure V1, we will be using the default ones.
from grid_adventure.movements import MOVEMENTS
from grid_adventure.objectives import OBJECTIVES
from grid_adventure.rendering import DEFAULT_ASSET_ROOT

#Next, we import the methods to create the various entities in the game.
from grid_adventure.entities import (
    AgentEntity,
    FloorEntity,
    WallEntity,
    ExitEntity,
    CoinEntity,
    GemEntity,
    KeyEntity,
    LockedDoorEntity,
    UnlockedDoorEntity,
    LavaEntity,
    BoxEntity,
    SpeedPowerUpEntity,
    ShieldPowerUpEntity,
    PhasingPowerUpEntity,
)

# Utility helpers
from dataclasses import dataclass
from typing import Callable
import os, glob
import random

# PyTorch
# References: https://docs.pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_illustrations.html#randomperspective
# Adapted from Problem Set 4
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision import datasets, transforms
from torchvision.transforms import v2, InterpolationMode
from torchvision.transforms.v2 import functional as Fv2
import torch.nn.functional as F
import math
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np

EPOCHS = 50
VERSION = "v4"
INPUT_SIZE = 32 # Standardised input image size (32x32)
# Since we only need transform once into grid representation at the START
# At START, 
# only one agent, one exit
# all doors are locked, so we can ignore unlocked door entity
# number of keys == number of locked doors
# at most 5 gems, 5 key-door pairs, 5 boxes
# at most 3 for each powerups (boots, shield, ghost)
# each cell either contains a single floor, or floor + 1 single other entity (if background entity will overlap the entire floor, we can ignore the floor entity in that case)
ENTITY_NAMES = [
    'floor', # background with possible entity on top
    'wall', # background
    'agent', # on floor, only 1
    'exit', # on floor, only 1
    'coin', # on floor
    'gem', # on floor, <= 5
    'key', # on floor, <= 5
    'locked_door', # on floor, <= 5
    'lava', # background
    'box', # on floor, <= 5
    'boots', # on floor, <= 3
    'shield', # on floor, <= 3
    'ghost', # on floor, <= 3
]
NUM_ENTITIES = len(ENTITY_NAMES)

ENTITY_FOLDERS = {
    'floor': 0,
    'wall': 1,
    'human': 2,
    'exit': 3,
    'coin': 4,
    'gem': 5,
    'key': 6,
    'locked': 7,
    'lava': 8,
    'box': 9,
    'boots': 10,
    'shield': 11,
    'ghost': 12,
}

ON_FLOOR_ENTITIES = ['human', 'exit', 'coin', 'gem', 'key', 'locked', 'box', 'boots', 'shield', 'ghost']
BACKGROUND_ENTITIES = ['floor', 'wall', 'lava']


## DATASET PREPARATION
# References: https://share.google/aimode/iR2xtUwyp0CGS1to5
assets = DEFAULT_ASSET_ROOT
# Load entity images from entity folder
def load_entity_images_from_folders(folder_name: str) -> Image:
    images = []
    folder_path = os.path.join(assets, folder_name, '*.png')
    for filename in glob.glob(folder_path):
        image = Image.open(filename).convert('RGBA')
        # print(f"Loaded image: {filename} with size {image.size}") # Check size
        images.append(image)
    return images

# References: https://share.google/aimode/8svZuyx7rPlKgIUv6
# https://share.google/aimode/VCUvjt03r7j0fcmH6
# Overlay entity image on top of floor image (Alpha is dropped as it is the transparent part)
# after overlaying, no more transparency, so convert to RGB for efficiency
def overlay_entity_on_floor(floor_image: Image.Image, entity_image: Image.Image, cell_size: int) -> Image.Image:
    new_floor_img = floor_image.resize((cell_size, cell_size), resample=Image.Resampling.NEAREST)
    new_entity_img = entity_image.resize((cell_size, cell_size), resample=Image.Resampling.NEAREST)
    overlay_img = new_floor_img.copy()
    overlay_img.paste(new_entity_img, (0, 0), new_entity_img)
    return overlay_img.convert('RGB')

# Resize and convert to RGB since no need transparency
def create_background_entity_image(background_entity_image: Image.Image, cell_size: int) -> Image.Image:
    return background_entity_image.resize((cell_size, cell_size), resample=Image.Resampling.NEAREST).convert('RGB')

# Generate labelled dataset, all loaded images are initially 128x128x4 (RGBA)
def generate_dataset() -> list[Image.Image]:
    entities_images = {}
    for folder in ENTITY_FOLDERS:
        entities_images[folder] = load_entity_images_from_folders(folder)

    dataset = []
    floor_images = entities_images['floor']
    
    for _ in range(1): # Generate 1 samples for each entity type
        # Generate floor-only images
        for floor_image in floor_images:
            floor = floor_image.resize((128, 128), Image.NEAREST).convert('RGB')
            dataset.append((floor, ENTITY_FOLDERS['floor']))

        # Background entities
        for entity_name in BACKGROUND_ENTITIES:
            class_id = ENTITY_FOLDERS[entity_name]
            for image in entities_images[entity_name]:
                entity = create_background_entity_image(image, 128)
                dataset.append((entity, class_id))
        
        # On-floor entities
        for entity_name in ON_FLOOR_ENTITIES:
            class_id = ENTITY_FOLDERS[entity_name]
            for floor_image in floor_images:
                for entity_image in entities_images[entity_name]:
                    overlay_img = overlay_entity_on_floor(floor_image, entity_image, 128)
                    dataset.append((overlay_img, class_id))
    
    return dataset

# References: https://share.google/aimode/8M2inNY6l92aWBWgw
# Convert to normalised tensor with label at native resolution (128x128)
def dataset_to_tensors(images):
    X = []
    y = []
    for image, label in images:
        arr = np.array(image, dtype=np.float32) / 255.0 # Normalise to [0, 1]
        arr = arr.transpose(2, 0, 1) # CHW
        X.append(arr)
        y.append(label)
    return np.array(X), np.array(y)


## CLASSIFIER CNN MODEL
# References: https://share.google/aimode/K2svUTTodhOw57I0g
# https://docs.pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html
class EntityClassifier(nn.Module):
    def __init__(self, num_classes=NUM_ENTITIES):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 16x16
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 8x8
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4), # 4x4
            )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
            )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


## TRAINING
class RandomPixelation:
    def __init__(self, min_size: int = 32, max_size: int = 128):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image: Image.Image) -> Image.Image:
        # Simulating resizing of main grid image, extract entity image, and resize entity image in Agent class
        # Random sizing between 32 (min) and 128 (initial)
        temp_size = random.randint(self.min_size, self.max_size)
        image = Fv2.resize(image, [temp_size, temp_size], interpolation=InterpolationMode.NEAREST, antialias=False)
        # Unsample back to 32x32 for the CNN input
        image = Fv2.resize(image, [32, 32], interpolation=InterpolationMode.BILINEAR, antialias=True)
        return image

# Augmentations
# Transform pixelation, colour, and random affine
def get_train_augmentations() -> v2.Compose:
    T = v2.Compose([
        RandomPixelation(min_size=32, max_size=128),
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        v2.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        # For v4
        v2.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])
    return T
# For test, only resizing for consistency
def get_test_augmentations() -> v2.Compose:
    T = v2.Compose([
        v2.Resize((32, 32), interpolation=InterpolationMode.BILINEAR, antialias=True)
    ])
    return T

# Training loop adopted from PS4
def train_model(X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor, y_test: torch.Tensor, epochs=50, lr=1e-3) -> tuple[EntityClassifier, list[float]]:
    """
    Trains the model for a specified number of epochs/iterations
    
    Parameters
    ---------- 
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        epochs: Number of epochs, default of 50
        lr: Learning rate, default of 1e-3

    Returns
    -------
        The final model and the loss curve (per epoch)
    """

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    model = EntityClassifier()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

    # References: https://gemini.google.com/share/c60d5006616a
    # Weighted loss to handle class imbalance
    entity_counts = torch.bincount(y_train_tensor)
    num_entities = len(entity_counts)
    total_samples = len(y_train_tensor)
    entity_weights = total_samples / (num_entities * entity_counts.float() + 1e-6)
    loss_function = torch.nn.CrossEntropyLoss(weight=entity_weights)
    # References: https://share.google/aimode/vELJvYxL7cXzWFOIU
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.5, patience=5)
    
    best_test_accuracy = 0
    best_state = None

    train_transform = get_train_augmentations()
    test_transform = get_test_augmentations()

    # Set model to training mode. 
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for X_batch, y_batch in train_loader:
            # Apply transformation to each image to create diversity
            X_batch = torch.stack([train_transform(x) for x in X_batch])
            
            optimiser.zero_grad()
            logits = model(X_batch)
            loss = loss_function(logits, y_batch)
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * len(y_batch)
            correct += (logits.argmax(1) == y_batch).sum().item()
            total += len(y_batch)

        train_accuracy = correct / total

        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = torch.stack([test_transform(x) for x in X_batch])
                logits = model(X_batch)
                test_correct += (logits.argmax(1) == y_batch).sum().item()
                test_total += len(y_batch)
        test_accuracy = test_correct / test_total
        scheduler.step(1 - test_accuracy)

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Progress printing every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/total:.4f} | "
                  f"Train Accuracy: {train_accuracy:.4f} | Test Accuracy: {test_accuracy:.4f} | Best Accuracy: {best_test_accuracy:.4f}")

    # References: https://share.google/aimode/o6eUjmUwcAHKs3evL
    model.load_state_dict(best_state)
    return model


## MAIN
if __name__ == '__main__':
    images = generate_dataset()

    random.seed(42)
    random.shuffle(images)
    split = int(len(images) * 0.8) # 80% train, 20% test
    train_images = images[:split]
    test_images = images[split:]

    print(f"Train: {len(train_images)}, Test: {len(test_images)}")

    X_train, y_train = dataset_to_tensors(train_images)
    X_test, y_test = dataset_to_tensors(test_images)

    all_info1 = ""
    for i, name in enumerate(ENTITY_NAMES):
        n_train = (y_train == i).sum()
        n_test = (y_test == i).sum()
        info1 = f"{name}: train={n_train}, test={n_test}\n"
        print(info1)
        all_info1 += info1

    # Train model
    model = train_model(X_train, y_train, X_test, y_test, epochs=EPOCHS)

    # Final evaluation
    with torch.no_grad():
        model.eval()
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        X_test_t = Fv2.resize(X_test_t, [32, 32], interpolation=InterpolationMode.BILINEAR, antialias=True)
        logits = model(X_test_t)
        preds = logits.argmax(1).numpy()
        acc = (preds == y_test).mean()
        info2 = f"\nFinal Accuracy: {acc:.4f}\n"
        print(info2)

        # Per-entity accuracy
        all_info3 = ""
        for i, name in enumerate(ENTITY_NAMES):
            filter = y_test == i
            if filter.sum() > 0:
                entity_accuracy = (preds[filter] == i).mean()
                info3 = f"{name}: {entity_accuracy:.4f} ({filter.sum()} samples)\n"
                print(info3)
                all_info3 += info3

    # Save model
    model_name = f"gridadv-entity-classifier-e{EPOCHS}-{VERSION}.pth"
    torch.save(model.state_dict(), os.path.join("models", f"{model_name}"))
    print("\nModel saved: ", model_name)

    # Loader snipper
    from utils import generate_torch_loader_snippet
    example_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    snippet = generate_torch_loader_snippet(model, example_inputs=example_input, prefer='auto', compression='zlib')
    with open(os.path.join("model_snippets", f"{model_name}-snippet.txt"), "w", encoding="utf-8") as file:
        file.write(snippet)

    # Save training info
    with open(os.path.join("training_info", f"{model_name}-info.txt"), "w", encoding="utf-8") as file:
        file.write("Model saved: " + model_name + "\n")
        file.write(all_info1)
        file.write(info2)
        file.write(all_info3)