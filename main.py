import io
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from predict import predict_image, CLASSES

app = FastAPI(
    title="Recyclable Waste Classifier API",
    description="Upload a photo of waste and get its predicted category.",
)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


@app.get("/")
def health():
    return {"status": "ok", "classes": CLASSES}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Please upload a JPEG, PNG, or WebP image.")

    try:
        image = Image.open(io.BytesIO(file.file.read()))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Could not read the uploaded file as an image.")

    return predict_image(image)