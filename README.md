
# 🌿 Plant Disease Detection Using DenseNet121

A deep learning application that classifies plant leaf images into 33 supported plant health and disease classes using a trained DenseNet121 convolutional neural network. The application provides predictions and confidence scores through an interactive Streamlit web interface.

## 🚀 Live Demo

**Streamlit App:** Add your deployed Streamlit URL here.(https://opencv-based-plant-leaf-disease.onrender.com/)

## 📌 Project Overview

Plant diseases can affect crop health and agricultural productivity. This project uses deep learning and image classification to identify potential plant diseases from leaf images.

The trained DenseNet121 model analyzes an uploaded image and returns the predicted class along with the top three predictions and their associated probabilities.

## ✨ Features

- 🌱 Classification across 33 supported classes
- 🧠 Transfer learning with DenseNet121
- 🖼️ Upload leaf images in JPG, JPEG, PNG, and WEBP formats
- 📊 Display the predicted class and confidence score
- 🔝 Show the top three model predictions
- ⚡ Interactive web interface built with Streamlit
- 🔄 Image preprocessing consistent with DenseNet121 requirements

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Deep Learning:** TensorFlow, Keras
- **Model Architecture:** DenseNet121
- **Image Processing:** Pillow, NumPy
- **Web Application:** Streamlit
- **Version Control:** Git and GitHub
- **Deployment:** Streamlit Community Cloud

## 🧠 Model Details

| Parameter | Details |
|---|---|
| Model architecture | DenseNet121 |
| Task | Multiclass image classification |
| Number of classes | 33 |
| Input image size | 224 × 224 |
| Input color mode | RGB |
| Preprocessing | `tf.keras.applications.densenet.preprocess_input` |
| Test accuracy | 96.49% |
| Macro F1-score | 96.35% |
| Weighted F1-score | 96.52% |

*Metrics are from the reported evaluation of the trained model on the project's test dataset.*

## 📂 Project Structure

```text
plant-disease-app/
├── app.py
├── best_densenet121_model.keras
├── class_names.json
├── requirements.txt
└── README.md
```

### File Descriptions

- `app.py` — Streamlit application and prediction pipeline.
- `best_densenet121_model.keras` — Trained DenseNet121 model.
- `class_names.json` — Class labels in the exact order used during training.
- `requirements.txt` — Python package dependencies.
- `README.md` — Project documentation.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Replace `YOUR_USERNAME` and `YOUR_REPOSITORY` with your GitHub details.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Use a Python and TensorFlow version compatible with the model and the deployment environment.

### 4. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

## 🔍 How to Use

1. Launch the application.
2. Upload a plant leaf image.
3. Click **Predict Disease**.
4. View the predicted class and confidence score.
5. Review the top three predictions and their probabilities.

## 🔬 Prediction Pipeline

```text
Upload Leaf Image
        ↓
Convert Image to RGB
        ↓
Resize to 224 × 224
        ↓
DenseNet121 Preprocessing
        ↓
Trained DenseNet121 Model
        ↓
Class Probability Predictions
        ↓
Display Top 3 Predictions
```

## 📊 Evaluation Metrics

The model achieved the following reported test results:

- **Test accuracy:** 96.49%
- **Macro F1-score:** 96.35%
- **Weighted F1-score:** 96.52%

Accuracy measures the proportion of correctly classified test images. Macro F1 gives equal weight to each class, while weighted F1 accounts for the number of test examples in each class.

Performance may vary across disease classes and real-world images.

## ☁️ Deployment on Streamlit Community Cloud

1. Push the project files to a GitHub repository.
2. Visit [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create a new app and connect your repository.
4. Select the branch and set the main file path to `app.py`.
5. Configure a compatible Python version.
6. Deploy the application.

Ensure the model file is available to the deployed application. For large model files, consider Git LFS or external model storage, subject to the hosting platform's limits.

## ⚠️ Limitations

- Predictions are limited to the 33 classes supported by the trained model.
- The model may produce incorrect predictions for unfamiliar plants, diseases, backgrounds, or lighting conditions.
- A high confidence score does not guarantee that a prediction is correct.
- The application is intended for informational and educational purposes, not as a replacement for professional agricultural diagnosis.

## 🔮 Future Improvements

- Add Grad-CAM visualizations to highlight influential image regions.
- Improve performance on classes with lower F1-scores.
- Evaluate robustness on images captured under real-world conditions.
- Add support for additional plant species and diseases.
- Optimize model inference for faster deployment.

## 👨‍💻 Author

**Sai Kishore**

- GitHub: https://github.com/kishore1902
- LinkedIn: https://www.linkedin.com/in/kishore-nulu-971411371/

## 📄 License

Add a license such as the MIT License if you intend to distribute this project under those terms. Also verify the dataset and pretrained model licensing requirements before redistribution.
