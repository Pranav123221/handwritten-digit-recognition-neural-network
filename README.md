# Handwritten Digit Recognition using Neural Network

An end-to-end Deep Learning project that recognizes handwritten digits (0–9) from 28×28 grayscale images using a Neural Network built with TensorFlow/Keras.

The trained model is integrated into a Streamlit web application, allowing users to provide handwritten digits and receive real-time predictions.

## 🚀 Project Overview

Handwritten digit recognition is a fundamental computer vision and Deep Learning problem.

In this project, a Neural Network is trained to learn patterns from handwritten digit images and classify them into one of ten classes:

`0, 1, 2, 3, 4, 5, 6, 7, 8, 9`

The project covers the complete Deep Learning workflow:

**Data → Preprocessing → Neural Network → Training → Evaluation → Prediction → Model Saving → Deployment**

## 🧠 Model Architecture

The project uses a fully connected Neural Network:

```text
Input: 28 × 28 × 1
        ↓
     Flatten
        ↓
Dense Layer — 128 neurons — ReLU
        ↓
Dense Layer — 64 neurons — ReLU
        ↓
Output Layer — 10 neurons — Softmax
```

### Activation Functions

* **ReLU** — used in the hidden layers
* **Softmax** — used in the output layer for multi-class classification

## 📊 Dataset

The model is trained using handwritten digit images represented as 28×28 pixel grayscale images.

Each image contains:

* 784 pixel values
* 28 × 28 image dimensions
* One target class representing a digit from 0–9

Pixel values are normalized from:

```text
0–255 → 0–1
```

## ⚙️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* TensorFlow
* Keras
* Streamlit

## 🔄 Machine Learning Workflow

### 1. Data Loading

The training and testing datasets are loaded from CSV files.

### 2. Data Exploration

The dataset is analyzed to understand:

* Shape and dimensions
* Pixel values
* Target labels
* Missing values
* Class distribution
* Sample handwritten images

### 3. Data Preprocessing

The image data is:

* Normalized
* Reshaped into `28 × 28 × 1`
* Prepared for Neural Network training

Labels are encoded into the required format for multi-class classification.

### 4. Model Development

A fully connected Neural Network is built using TensorFlow/Keras.

### 5. Model Training

The model learns digit patterns from the training data using:

* Adam optimizer
* Categorical cross-entropy loss
* Accuracy as the evaluation metric

### 6. Model Evaluation

Model performance is evaluated using:

* Accuracy
* Loss
* Confusion Matrix
* Classification Report
* Prediction analysis

### 7. Prediction

The trained model predicts handwritten digits from previously unseen test images.

### 8. Model Saving

The trained model is saved in Keras format:

```text
handwritten_digit_recognition.keras
```

### 9. Deployment

The trained model is integrated into a Streamlit application to provide an interactive prediction interface.

## 🌐 Streamlit Application

The deployed application allows users to:

1. Provide a handwritten digit
2. Process the input image
3. Pass it through the trained Neural Network
4. Receive the predicted digit
5. View the model's prediction confidence

**Live Demo:** Coming Soon

## 📁 Project Structure

```text
handwritten-digit-recognition-neural-network/
│
├── app.py
├── handwritten_digit_recognition.keras
├── requirements.txt
├── README.md
│
└── notebook/
    └── handwritten_digit_recognition.ipynb
```

## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/handwritten-digit-recognition-neural-network.git
```

Navigate into the project:

```bash
cd handwritten-digit-recognition-neural-network
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open locally in your browser.

## 📈 Key Learning Outcomes

Through this project, I practiced:

* Understanding image-based datasets
* Data preprocessing
* Pixel normalization
* Label encoding
* Neural Network architecture
* Dense layers
* ReLU and Softmax
* Model training
* Loss and optimization
* Model evaluation
* Confusion matrix analysis
* Prediction and error analysis
* Saving trained Deep Learning models
* Integrating a trained model into a web application
* Deploying a Deep Learning application

## 🔮 Future Improvements

* Implement CNN architecture for improved image recognition
* Add confidence visualization
* Improve the drawing interface
* Perform hyperparameter tuning
* Compare Neural Network and CNN performance
* Add model performance monitoring
* Deploy an API-based version using FastAPI

## 👨‍💻 Author

**Pranav Sharma**

Computer Science Undergraduate | AI/ML & Generative AI Enthusiast

Focused on building practical AI and software engineering solutions.

---

⭐ If you found this project useful, consider giving the repository a star.
