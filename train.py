import torch
import torch.nn as nn
from data_loader import get_dataloaders
from model import build_model

EPOCHS = 8
LR = 0.001


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    if training:
        model.train()
        model.features.eval()  # keep frozen backbone's batch-norm stats fixed
    else:
        model.eval()

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
    print("Using device:", device)

    train_loader, val_loader, test_loader, classes = get_dataloaders()
    model = build_model(num_classes=len(classes)).to(device)

    # Class weights: rarer classes (like "trash") count more in the loss
    subset = train_loader.dataset
    targets = [subset.dataset.targets[i] for i in subset.indices]
    counts = torch.bincount(torch.tensor(targets), minlength=len(classes)).float()
    weights = counts.sum() / (len(classes) * counts)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    # Only the new final layer is trained
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LR)

    best_val_acc = 0.0
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
            print("  -> saved new best model")

    print(f"Done. Best validation accuracy: {best_val_acc:.3f}")


if __name__ == "__main__":
    main()