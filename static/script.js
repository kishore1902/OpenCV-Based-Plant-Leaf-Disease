
const imageInput = document.getElementById("image-input");
const dropZone = document.getElementById("drop-zone");
const uploadPlaceholder = document.getElementById("upload-placeholder");
const previewContainer = document.getElementById("preview-container");
const imagePreview = document.getElementById("image-preview");
const predictButton = document.getElementById("predict-button");
const buttonText = document.getElementById("button-text");
const errorMessage = document.getElementById("error-message");

const emptyState = document.getElementById("empty-state");
const loadingState = document.getElementById("loading-state");
const resultState = document.getElementById("result-state");
const predictedClass = document.getElementById("predicted-class");
const confidenceValue = document.getElementById("confidence-value");
const topPredictions = document.getElementById("top-predictions");

let selectedFile = null;
let previewUrl = null;
let requestInProgress = false;

function clearResults() {
    emptyState.hidden = false;
    loadingState.hidden = true;
    resultState.hidden = true;
    topPredictions.replaceChildren();
}

function showError(message) {
    errorMessage.textContent = message;
}

function resetResults() {
    clearResults();
    showError("");
}

function selectImage(file) {
    if (!file) return;

    if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
        showError("Please select a JPG, PNG, or WEBP image.");
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        showError("Image size must be 10 MB or less.");
        return;
    }

    selectedFile = file;

    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
    }

    previewUrl = URL.createObjectURL(file);
    imagePreview.src = previewUrl;

    uploadPlaceholder.hidden = true;
    previewContainer.hidden = false;
    predictButton.disabled = false;

    resetResults();
}

imageInput.addEventListener("change", () => {
    const file = imageInput.files?.[0];
    selectImage(file);
});

dropZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropZone.classList.add("dragging");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragging");
});

dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");

    const file = event.dataTransfer?.files?.[0];
    if (file) selectImage(file);
});

function renderPredictions(predictions) {
    topPredictions.replaceChildren();

    predictions.forEach((item) => {
        const probability = Number(item.probability);
        const percentage = Math.max(
            0,
            Math.min(100, probability * 100)
        );

        const predictionItem = document.createElement("div");
        predictionItem.className = "prediction-item";

        const meta = document.createElement("div");
        meta.className = "prediction-meta";

        const name = document.createElement("span");
        name.className = "prediction-name";
        name.textContent = item.class_name;

        const percent = document.createElement("span");
        percent.className = "prediction-percent";
        percent.textContent = `${percentage.toFixed(2)}%`;

        const track = document.createElement("div");
        track.className = "progress-track";

        const fill = document.createElement("div");
        fill.className = "progress-fill";
        fill.style.width = `${percentage}%`;

        track.appendChild(fill);
        meta.append(name, percent);
        predictionItem.append(meta, track);
        topPredictions.appendChild(predictionItem);
    });
}

predictButton.addEventListener("click", async () => {
    if (!selectedFile || requestInProgress) return;

    requestInProgress = true;
    predictButton.disabled = true;
    buttonText.textContent = "Analyzing...";
    showError("");

    emptyState.hidden = true;
    resultState.hidden = true;
    loadingState.hidden = false;

    try {
        const formData = new FormData();
        formData.append("file", selectedFile);

        const response = await fetch("/predict", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Prediction failed.");
        }

        if (
            !data.predicted_class ||
            !Number.isFinite(data.confidence) ||
            !Array.isArray(data.top_predictions)
        ) {
            throw new Error("The server returned an invalid response.");
        }

        predictedClass.textContent = data.predicted_class;
        confidenceValue.textContent =
            `${(data.confidence * 100).toFixed(2)}%`;

        renderPredictions(data.top_predictions);

        loadingState.hidden = true;
        resultState.hidden = false;

    } catch (error) {
        loadingState.hidden = true;
        emptyState.hidden = false;
        showError(
            error.message || "Something went wrong. Please try again."
        );
    } finally {
        requestInProgress = false;
        predictButton.disabled = !selectedFile;
        buttonText.textContent = "Analyze leaf";
    }
});

window.addEventListener("beforeunload", () => {
    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
    }
});
