# DeepFER: Facial Emotion Recognition Using Deep Learning

DeepFER is a computer-vision project that classifies visible facial expressions into seven categories using a convolutional neural network. It supports image prediction, real-time webcam detection, and an interactive Streamlit interface.

## Emotion Classes

* Angry
* Disgust
* Fear
* Happy
* Neutral
* Sad
* Surprise

## Features

* Facial-expression classification using a custom CNN
* Image normalization and data augmentation
* Training, validation, and test evaluation
* Classification report and confusion matrix
* Prediction from individual image files
* Real-time webcam face detection
* Streamlit interface with image upload and camera capture
* Confidence scores and class-probability visualization

## Dataset

The project uses a FER-style facial-expression dataset containing grayscale facial images organized into `train` and `test` directories.

The dataset is excluded from this repository because of its size. Place it locally in:

```text
data/raw/archive-3/
├── train/
└── test/
```

Each split contains directories for the seven emotion classes.

## Model Architecture

The DeepFER model includes:

1. Data-augmentation layers
2. Input rescaling
3. Four convolutional blocks
4. Batch normalization
5. Max-pooling layers
6. Dropout regularization
7. Global average pooling
8. A fully connected classification layer
9. A seven-class softmax output layer

Training uses the Adam optimizer, categorical cross-entropy loss, early stopping, model checkpointing, and adaptive learning-rate reduction.

## Project Structure

```text
DeepFER-Emotion-Recognition/
├── app/
│   └── app.py
├── data/
│   └── raw/
├── models/
│   ├── class_names.json
│   └── deepfer_best.keras
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_model_training.ipynb
├── reports/
├── results/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── training_history.png
│   └── training_log.csv
├── src/
│   ├── predict_image.py
│   └── webcam_detection.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd DeepFER-Emotion-Recognition
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the Streamlit App

```bash
python -m streamlit run app/app.py
```

Open the local address displayed in the terminal, normally:

```text
http://localhost:8501
```

## Run Webcam Detection

```bash
python src/webcam_detection.py
```

Press `Q` while the webcam window is active to close it.

## Predict an Image

```bash
python src/predict_image.py "path/to/image.jpg"
```

The program displays the predicted expression, confidence score, and probability for each class.

## Evaluation

The evaluation workflow produces:

* Test loss and accuracy
* Precision, recall, and F1-score for every class
* Confusion matrix
* Training and validation accuracy curves
* Training and validation loss curves

The generated evaluation artifacts are available in the `results` directory.

## Technologies

* Python
* TensorFlow and Keras
* OpenCV
* Streamlit
* NumPy
* Pandas
* Matplotlib
* scikit-learn
* Pillow
* Jupyter Notebook

## Limitations and Responsible Use

DeepFER classifies visible facial-expression patterns. Facial expressions do not always represent a person’s actual internal emotional state.

The model may be affected by lighting, camera angle, image quality, facial occlusion, dataset imbalance, and demographic bias. It should not be used for medical diagnosis, surveillance, hiring, education assessment, policing, or other important decisions about individuals.

## Author

**Aniket Kapgate**
B.Tech Computer Science and Engineering — Artificial Intelligence and Machine Learning
