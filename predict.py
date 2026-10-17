import torch
from PIL import Image
from data_loader import eval_transform
from model import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model once when the file is imported, not on every request
checkpoint = torch.load("model.pth", map_location=device)
CLASSES = checkpoint["classes"]
model = build_model(num_classes=len(CLASSES), pretrained=False)
model.load_state_dict(checkpoint["model_state"])
model.to(device).eval()


def predict_image(image: Image.Image) -> dict:
    image = image.convert("RGB")
    tensor = eval_transform(image).unsqueeze(0).to(device)  # add a batch dimension

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]

    top_prob, top_idx = probs.max(0)
    return {
        "class": CLASSES[top_idx.item()],
        "confidence": round(top_prob.item(), 4),
        "probabilities": {c: round(p, 4) for c, p in zip(CLASSES, probs.tolist())},
    }


if __name__ == "__main__":
    import sys
    print(predict_image(Image.open(sys.argv[1])))