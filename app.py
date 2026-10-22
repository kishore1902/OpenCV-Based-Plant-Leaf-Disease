
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.densenet import preprocess_input


# -------------------------------
# 1. Configuration
# -------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_densenet121_model.keras"
CLASS_NAMES_PATH=BASE_DIR / "class_names.json"

IMAGE_SIZE = (224, 224)


st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)


# -------------------------------
# 2. Load model and class names
# -------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


@st.cache_data
def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Class names file not found: {CLASS_NAMES_PATH}"
        )

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        names = json.load(file)

    if not isinstance(names, list):
        raise ValueError(
            "class_names.json must contain a JSON list."
        )

    return names


# -------------------------------
# 3. Image preprocessing
# -------------------------------

def preprocess_image(image):
    # Convert uploaded image to RGB.
    image = image.convert("RGB")

    # Resize to the training image size.
    image = image.resize(IMAGE_SIZE)

    # Convert image to a NumPy array.
    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # Add batch dimension.
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Apply DenseNet121 preprocessing exactly once.
    image_array = preprocess_input(image_array)

    return image_array


# -------------------------------
# 4. Prediction
# -------------------------------

def predict_disease(model, image, class_names):
    image_array = preprocess_image(image)

    probabilities = model.predict(
        image_array,
        verbose=0
    )[0]

    if len(probabilities) != len(class_names):
        raise ValueError(
            "Model outputs and class names do not match."
        )

    # Sort classes by probability.
    top_indices = np.argsort(probabilities)[::-1][:3]

    predicted_index = int(top_indices[0])

    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    top_predictions = [
        (class_names[int(i)], float(probabilities[i]))
        for i in top_indices
    ]

    return predicted_class, confidence, top_predictions


# -------------------------------
# 5. Streamlit interface
# -------------------------------

st.title("🌿 Plant Disease Detection")

st.write(
    "Upload a plant leaf image to predict its disease "
    "using a trained DenseNet121 model."
)

st.info(
    "The model recognizes 33 supported plant health and disease "
    "classes. Predictions are informational and may be incorrect."
)


# Load the model.
try:
    model = load_model()
    class_names = load_class_names()

    if model.output_shape[-1] != len(class_names):
        st.error(
            "The model output count does not match class_names.json."
        )
        st.stop()

except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()


# Image upload control.
uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png", "webp"]
)


if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        left, right = st.columns(2)

        with left:
            st.subheader("Uploaded image")
            st.image(
                image,
                use_container_width=True
            )

        with right:
            st.subheader("Prediction")

            if st.button(
                "Predict Disease",
                type="primary",
                use_container_width=True
            ):
                with st.spinner("Analyzing image..."):

                    predicted_class, confidence, top_predictions = (
                        predict_disease(
                            model,
                            image,
                            class_names
                        )
                    )

                st.session_state["result"] = {
                    "filename": uploaded_file.name,
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "top_predictions": top_predictions
                }

        # Show results after clicking the button.
        result = st.session_state.get("result")

        if result and result["filename"] == uploaded_file.name:

            st.divider()
            st.subheader("Prediction result")

            st.success(
                f"Predicted class: {result['predicted_class']}"
            )

            st.metric(
                "Model confidence",
                f"{result['confidence'] * 100:.2f}%"
            )

            st.subheader("Top 3 predictions")

            for name, probability in result["top_predictions"]:
                st.write(
                    f"**{name}** — {probability * 100:.2f}%"
                )
                st.progress(
                    min(max(probability, 0.0), 1.0)
                )

            st.caption(
                "Confidence is the model's predicted probability, "
                "not a guarantee of correctness. The model may "
                "misclassify unrelated images."
            )

    except Exception as error:
        st.error(f"Could not process this image: {error}")
