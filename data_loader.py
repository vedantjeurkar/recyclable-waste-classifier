import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DATA_DIR = "data/dataset-resized"
BATCH_SIZE = 32
SEED = 42

# MobileNetV2 was pretrained on ImageNet, so we normalize with ImageNet's stats
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Training images: resize + random changes so the model sees varied versions
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# Validation/test images: only resize, no random changes
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def get_dataloaders():
    # Two copies of the dataset: one with augmentation, one without
    train_full = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    eval_full = datasets.ImageFolder(DATA_DIR, transform=eval_transform)

    # Shuffle all image indices with a fixed seed, then split 70/15/15
    n = len(train_full)
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(n, generator=generator).tolist()

    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(eval_full, val_idx)
    test_ds = Subset(eval_full, test_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, train_full.classes


if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = get_dataloaders()
    print("Classes:", classes)
    print("Train images:", len(train_loader.dataset))
    print("Val images:", len(val_loader.dataset))
    print("Test images:", len(test_loader.dataset))

    images, labels = next(iter(train_loader))
    print("One batch shape:", images.shape)