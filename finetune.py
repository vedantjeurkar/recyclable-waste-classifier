import torch
import torch.nn as nn
from data_loader import get_dataloaders
from model import build_model

UNFREEZE_LAST = 4  # how many of MobileNetV2's last blocks to train
EPOCHS = 6
LR = 1e-4          # much smaller than before, so we don't wreck what it learned


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.eval()
    if training:
        model.classifier.train()
        for block in model.features[-UNFREEZE_LAST:]:
            block.train()

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _, classes = get_dataloaders()

    # Start from the model we already trained
    checkpoint = torch.load("model.pth", map_location=device)
    model = build_model(num_classes=len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    # Unfreeze the last few blocks
    for block in model.features[-UNFREEZE_LAST:]:
        for p in block.parameters():
            p.requires_grad = True

    # Same class weights as before
    subset = train_loader.dataset
    targets = [subset.dataset.targets[i] for i in subset.indices]
    counts = torch.bincount(torch.tensor(targets), minlength=len(classes)).float()
    weights = counts.sum() / (len(classes) * counts)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )

    # Only overwrite model.pth if we beat the current model
    _, best_val_acc = run_epoch(model, val_loader, criterion, device)
    print(f"Starting validation accuracy: {best_val_acc:.3f}")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train loss {train_loss:.3f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.3f} acc {val_acc:.3f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict(), "classes": classes}, "model.pth")
            print("  -> saved improved model")

    print(f"Done. Best validation accuracy: {best_val_acc:.3f}")


if __name__ == "__main__":
    main()