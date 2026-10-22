
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.densenet import preprocess_input


# -----------------------------------
# 1. Configuration
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_densenet121_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"

IMAGE_SIZE = (224, 224)

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
)

st.title("🌿 Plant Disease Detection")
st.write(
    "Upload a plant leaf image to predict its disease "
    "using a trained DenseNet121 model."
)


# -----------------------------------
# 2. Load the trained model
# -----------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. "
            "Check that the model is committed to your GitHub repository."
        )

    return tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False,
    )


# -----------------------------------
# 3. Load class names
# -----------------------------------

@st.cache_data
def load_class_names():
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
            "class_names.json must contain a non-empty list of class names."
        )

    return names


# -----------------------------------
# 4. Preprocess the uploaded image
# -----------------------------------

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    # Match DenseNet121 preprocessing used during training.
    # Do not divide by 255 separately.
    image_array = preprocess_input(image_array)

    return image_array


# -----------------------------------
# 5. Predict the disease
# -----------------------------------

def predict_disease(model, image, class_names):
    image_array = preprocess_image(image)

    probabilities = model.predict(image_array, verbose=0)[0]

    if len(probabilities) != len(class_names):
        raise ValueError(
            f"Model returns {len(probabilities)} predictions, "
            f"but class_names.json contains {len(class_names)} classes."
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("The model returned invalid prediction values.")

    top_indices = np.argsort(probabilities)[::-1][:3]

    predicted_index = int(top_indices[0])
    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    top_predictions = [
        (class_names[int(index)], float(probabilities[index]))
        for index in top_indices
    ]

    return predicted_class, confidence, top_predictions


# -----------------------------------
# 6. Load application resources
# -----------------------------------

try:
    model = load_model()
    class_names = load_class_names()

    if model.output_shape[-1] != len(class_names):
        st.error(
            "The model output count does not match the class names. "
            "Check the class order and model output."
        )
        st.stop()

except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()


# -----------------------------------
# 7. Upload and predict
# -----------------------------------

uploaded_file = st.file_uploader(
    "Upload a plant leaf image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        # Reset the previous prediction when the upload changes.
        current_file_id = (
            f"{uploaded_file.name}_{uploaded_file.size}"
        )

        if st.session_state.get("file_id") != current_file_id:
            st.session_state["file_id"] = current_file_id
            st.session_state.pop("result", None)

        left, right = st.columns(2)

        with left:
            st.subheader("Uploaded image")
            st.image(image, use_container_width=True)

        with right:
            st.subheader("Prediction")

            if st.button(
                "Predict Disease",
                type="primary",
                use_container_width=True,
            ):
                with st.spinner("Analyzing image..."):
                    prediction = predict_disease(
                        model,
                        image,
                        class_names,
                    )

                    st.session_state["result"] = {
                        "file_id": current_file_id,
                        "predicted_class": prediction[0],
                        "confidence": prediction[1],
                        "top_predictions": prediction[2],
                    }

        result = st.session_state.get("result")

        if result and result["file_id"] == current_file_id:
            st.divider()
            st.subheader("Prediction Result")

            st.success(
                f"Predicted class: {result['predicted_class']}"
            )

            st.metric(
                "Model confidence",
                f"{result['confidence'] * 100:.2f}%",
            )

            st.subheader("Top 3 Predictions")

            for name, probability in result["top_predictions"]:
                st.write(
                    f"**{name}** — {probability * 100:.2f}%"
                )
                st.progress(
                    float(np.clip(probability, 0.0, 1.0))
                )

            st.caption(
                "Confidence is the model's predicted probability, "
                "not a guarantee of correctness. Results may be "
                "unreliable for images outside the training classes."
            )

    except Exception as error:
        st.error(f"Could not process this image: {error}")
