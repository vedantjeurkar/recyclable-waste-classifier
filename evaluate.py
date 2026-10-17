import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import get_dataloaders
from model import build_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Same seed in data_loader.py means the same test split every time
    _, _, test_loader, classes = get_dataloaders()

    checkpoint = torch.load("model.pth", map_location=device)
    model = build_model(num_classes=len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            all_preds.extend(outputs.argmax(1).cpu().tolist())
            all_labels.extend(labels.tolist())

    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f"Test accuracy: {accuracy:.3f}\n")
    print(classification_report(all_labels, all_preds, target_names=classes))

    # Confusion matrix: rows = true class, columns = predicted class
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Test Set)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Saved confusion_matrix.png")


if __name__ == "__main__":
    main()