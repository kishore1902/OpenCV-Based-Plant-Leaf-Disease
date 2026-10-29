
import json
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError
from starlette.requests import Request
from tensorflow.keras.applications.densenet import preprocess_input

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_densenet121_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"
IMAGE_SIZE = (224, 224)
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB

app = FastAPI(
    title="Plant Disease Detection API",
    description="Plant leaf classification using DenseNet121",
    version="1.0.0",
)

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@lru_cache(maxsize=1)
def load_model():
    """Load the trained model once and reuse it."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH.name}")

    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False,
    )

    return model


@lru_cache(maxsize=1)
def load_class_names():
    """Load class names in the same order used during training."""
    if not CLASS_NAMES_PATH.is_file():
        raise FileNotFoundError(
            f"Missing file: {CLASS_NAMES_PATH.name}"
        )

    with CLASS_NAMES_PATH.open("r", encoding="utf-8") as file:
        names = json.load(file)

    if (
        not isinstance(names, list)
        or not names
        or not all(isinstance(name, str) for name in names)
    ):
        raise ValueError(
            "class_names.json must contain a non-empty list of strings."
        )

    return names


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Apply the same preprocessing used by the DenseNet121 model."""
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    # Do not replace this with image_array / 255.0.
    image_array = preprocess_input(image_array)

    return image_array


def predict_disease(image: Image.Image):
    model = load_model()
    class_names = load_class_names()

    if model.output_shape[-1] != len(class_names):
        raise ValueError(
            "The model output count does not match class_names.json. "
            "Check the class count and class order."
        )

    image_array = preprocess_image(image)
    probabilities = model.predict(image_array, verbose=0)[0]

    if (
        len(probabilities) != len(class_names)
        or not np.all(np.isfinite(probabilities))
    ):
        raise ValueError("The model returned invalid predictions.")

    # Return up to three classes, ranked by predicted probability.
    top_indices = np.argsort(probabilities)[::-1][
        : min(3, len(class_names))
    ]

    predicted_index = int(top_indices[0])

    return {
        "predicted_class": class_names[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "top_predictions": [
            {
                "class_name": class_names[int(index)],
                "probability": float(probabilities[index]),
            }
            for index in top_indices
        ],
    }


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Please select an image.",
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image size must be 10 MB or less.",
        )

    try:
        # Verify that the uploaded content is a valid image.
        with Image.open(BytesIO(contents)) as uploaded_image:
            uploaded_image.verify()

        with Image.open(BytesIO(contents)) as uploaded_image:
            image = uploaded_image.convert("RGB")

        result = predict_disease(image)
        return result

    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process the image: {error}",
        ) from error

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}",
        ) from error
