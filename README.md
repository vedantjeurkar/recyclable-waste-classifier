# Recyclable Waste Classifier API

A computer vision project that classifies waste photos into 6 categories
(cardboard, glass, metal, paper, plastic, trash) and serves the model
through a FastAPI REST endpoint.

## Results (held-out test set, 380 images)

| Stage | Test accuracy |
|---|---|
| Frozen MobileNetV2 + new classifier head | 83% |
| After fine-tuning last 4 blocks | **89%** |

Per-class F1 after fine-tuning: cardboard 0.95, paper 0.92, glass 0.89,
metal 0.89, plastic 0.81, trash 0.77.

See `confusion_matrix.png` for the full breakdown.

## Approach

- **Dataset:** TrashNet (~2,500 images, 6 classes), split 70/15/15 with a fixed seed
- **Model:** MobileNetV2 pretrained on ImageNet (transfer learning)
- **Training:** train the new final layer first, then fine-tune the last 4 blocks at a low learning rate
- **Class imbalance:** class-weighted cross-entropy loss (the "trash" class is the smallest)
- **Augmentation:** random flip, rotation, and color jitter on training images only

## Limitations

- Plastic is the weakest major class; it is often confused with glass and metal (shiny, transparent items)
- The "trash" class has only 24 test images, so its score is noisy
- TrashNet photos have clean backgrounds, so accuracy on messy real-world photos will be lower

## Run it

```bash
pip install -r requirements.txt
python download_data.py      # get the dataset
python train.py              # train the classifier head
python finetune.py           # fine-tune the last blocks
python evaluate.py           # test-set report + confusion matrix
uvicorn main:app --reload    # start the API
```

Open http://127.0.0.1:8000/docs and use `POST /predict` to upload an image.

Example response:

```json
{"class": "glass", "confidence": 0.7474, "probabilities": {"cardboard": 0.0037, "glass": 0.7474, "...": "..."}}
```

## Tech stack

Python, PyTorch, torchvision, scikit-learn, FastAPI, Uvicorn