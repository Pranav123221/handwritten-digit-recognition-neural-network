# 🧠 Handwritten Digit Recognition using Neural Network

> An end-to-end Deep Learning application that recognizes handwritten digits (0–9) from 28×28 grayscale images using a fully connected Neural Network built with TensorFlow/Keras and deployed as an interactive Streamlit application.

---

## 📌 Overview

Handwritten Digit Recognition is a fundamental Computer Vision and Deep Learning problem where a machine learning model learns to identify numerical digits from handwritten images.

This project implements the complete Deep Learning lifecycle—from raw pixel data and preprocessing to Neural Network training, evaluation, model serialization, and web deployment.

The system accepts a handwritten digit as input and predicts the corresponding digit class along with the model's confidence.

### Core Pipeline

```text
Raw Image
    ↓
Image Preprocessing
    ↓
Pixel Normalization
    ↓
28 × 28 × 1 Representation
    ↓
Flatten
    ↓
Fully Connected Neural Network
    ↓
Softmax Probability Distribution
    ↓
Predicted Digit
    ↓
Streamlit Application
```

---

## 🎯 Objectives

The primary objectives of this project are:

* Build a Neural Network for multi-class image classification.
* Understand the complete Deep Learning workflow.
* Process and normalize image pixel data.
* Implement a multi-layer fully connected architecture.
* Train and validate the model on handwritten digit data.
* Analyze model performance using multiple evaluation techniques.
* Perform prediction on unseen test images.
* Serialize the trained model for inference.
* Integrate the model into an interactive web application.
* Deploy the application for real-world accessibility.

---

## 📊 Dataset

The model works with handwritten digit images represented as grayscale pixel values.

Each image contains:

```text
Image dimensions: 28 × 28 pixels
Channels: 1 (grayscale)
Total pixels: 784
Classes: 10
Classes: 0–9
```

Each image can therefore be represented as:

```text
28 × 28 × 1
```

For the fully connected Neural Network, the image is flattened into:

```text
28 × 28 × 1 = 784 features
```

### Data Representation

```text
Original Image
     ↓
28 × 28 × 1
     ↓
Flatten
     ↓
784-dimensional vector
```

---

## 🔍 Exploratory Data Analysis

Before training the model, the dataset is analyzed to understand its structure and quality.

The exploration includes:

* Dataset dimensions
* Feature and target identification
* Missing-value analysis
* Pixel-value distribution
* Label/class distribution
* Image visualization
* Data type inspection
* Sample image analysis

Example visualization:

```text
Pixel Matrix
     ↓
28 × 28 values
     ↓
Grayscale Image
     ↓
Human-readable digit
```

---

## ⚙️ Data Preprocessing

### 1. Pixel Normalization

Raw pixel values are scaled from:

```text
0–255
```

to:

```text
0–1
```

using:

```python
X = X / 255.0
```

This provides a more suitable numerical range for Neural Network optimization.

### 2. Reshaping

The input images are represented as:

```text
28 × 28 × 1
```

using:

```python
X = X.reshape(-1, 28, 28, 1)
```

The additional dimension represents the grayscale channel.

### 3. Label Encoding

The digit labels are converted into a representation suitable for multi-class classification.

For example:

```text
7
```

can be represented as:

```text
[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
```

---

# 🧠 Neural Network Architecture

The project uses a fully connected feed-forward Neural Network.

```text
                 Input Image
              28 × 28 × 1
                    │
                    ▼
                 Flatten
                    │
                    ▼
              784 Features
                    │
                    ▼
        ┌──────────────────────┐
        │ Dense Layer           │
        │ 128 Neurons           │
        │ ReLU Activation       │
        └──────────────────────┘
                    │
                    ▼
        ┌──────────────────────┐
        │ Dense Layer           │
        │ 64 Neurons            │
        │ ReLU Activation       │
        └──────────────────────┘
                    │
                    ▼
        ┌──────────────────────┐
        │ Output Layer          │
        │ 10 Neurons            │
        │ Softmax Activation    │
        └──────────────────────┘
                    │
                    ▼
             Digit Prediction
              0 – 9
```

---

## 🔬 Architecture Details

| Layer   | Configuration | Purpose                             |
| ------- | ------------- | ----------------------------------- |
| Input   | 28×28×1       | Receives image                      |
| Flatten | 784 units     | Converts image to vector            |
| Dense   | 128 neurons   | Learns feature representations      |
| Dense   | 64 neurons    | Learns higher-level representations |
| Output  | 10 neurons    | Predicts digit classes              |

### Activation Functions

#### ReLU

The hidden layers use the Rectified Linear Unit activation function:

```text
ReLU(x) = max(0, x)
```

It introduces non-linearity and allows the network to learn complex patterns.

#### Softmax

The output layer uses Softmax to produce a probability distribution across the ten digit classes.

Example:

```text
0 → 0.01
1 → 0.00
2 → 0.02
3 → 0.01
4 → 0.00
5 → 0.01
6 → 0.00
7 → 0.93
8 → 0.01
9 → 0.01
```

Final prediction:

```text
7
```

---

# ⚡ Model Compilation

The model is compiled using:

```text
Optimizer:
Adam

Loss Function:
Categorical Crossentropy

Metric:
Accuracy
```

### Adam Optimizer

Adam is used to efficiently update the network weights during training.

### Categorical Crossentropy

The loss function measures the difference between the true class distribution and the predicted probability distribution.

---

# 🏋️ Model Training

The model learns through multiple training epochs.

The training process follows:

```text
Input Image
     ↓
Forward Propagation
     ↓
Prediction
     ↓
Loss Calculation
     ↓
Backpropagation
     ↓
Weight Updates
     ↓
Improved Model
```

Training performance is monitored using:

* Training loss
* Validation loss
* Training accuracy
* Validation accuracy

Training history is visualized to analyze convergence and identify potential overfitting.

---

# 📈 Model Evaluation

Model performance is evaluated using multiple metrics rather than relying only on accuracy.

### Evaluation techniques

* Accuracy
* Loss
* Confusion Matrix
* Classification Report
* Individual predictions
* Error analysis

### Confusion Matrix

The confusion matrix helps identify which digit classes the model confuses with one another.

For example:

```text
Actual 7 → Predicted 7 ✓
Actual 5 → Predicted 3 ✗
Actual 9 → Predicted 4 ✗
```

This provides a deeper understanding of model behavior.

---

# 🔎 Error Analysis

Incorrect predictions are inspected individually to understand model weaknesses.

The analysis includes:

```text
Actual Label
      ↓
Model Prediction
      ↓
Compare
      ↓
Identify Incorrect Samples
      ↓
Visual Inspection
```

This helps identify difficult handwriting patterns and provides opportunities for future model improvements.

---

# 🔮 Inference Pipeline

Once training is complete, the trained model is used to make predictions on unseen images.

```text
Test Image
    ↓
Normalize Pixel Values
    ↓
Reshape → 28 × 28 × 1
    ↓
Neural Network
    ↓
Softmax Probabilities
    ↓
Argmax
    ↓
Predicted Digit
```

Example:

```text
Input → Handwritten "7"

Model Output:
7 → 0.98

Prediction:
7
```

---

# 💾 Model Serialization

After training, the model is saved in Keras format:

```text
handwritten_digit_recognition.keras
```

The saved model contains the trained network configuration and learned parameters required for inference.

It can later be loaded without retraining:

```python
model = tf.keras.models.load_model(
    "handwritten_digit_recognition.keras"
)
```

---

# 🌐 Streamlit Application

The trained model is integrated into a Streamlit interface to transform the machine learning model into an interactive application.

### Application Workflow

```text
User
 ↓
Draw / Provide Digit
 ↓
Image Processing
 ↓
Normalization
 ↓
28 × 28 × 1
 ↓
Saved Neural Network
 ↓
Prediction
 ↓
Digit + Confidence
```

### Application Features

* Interactive user interface
* Handwritten digit input
* Automatic image preprocessing
* Real-time prediction
* Prediction confidence
* Lightweight deployment

---

# 🚀 Deployment

The application is designed for deployment using:

```text
GitHub
   ↓
Streamlit Community Cloud
   ↓
Live Web Application
```

### Deployment Architecture

```text
                    User
                     │
                     ▼
             Streamlit Web App
                     │
                     ▼
              Image Processing
                     │
                     ▼
          TensorFlow/Keras Model
                     │
                     ▼
              Digit Prediction
```

---

# 📁 Project Structure

```text
handwritten-digit-recognition-neural-network/
│
├── app.py
│
├── handwritten_digit_recognition.keras
│
├── requirements.txt
│
├── README.md
│
└── notebook/
    │
    └── handwritten_digit_recognition.ipynb
```

---

# 🛠️ Technology Stack

### Programming

* Python

### Data Processing

* NumPy
* Pandas

### Visualization

* Matplotlib

### Machine Learning

* Scikit-learn

### Deep Learning

* TensorFlow
* Keras

### Application

* Streamlit

### Development Environment

* Google Colab
* Jupyter Notebook

### Version Control

* Git
* GitHub

### Deployment

* Streamlit Community Cloud

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/handwritten-digit-recognition-neural-network.git
```

Navigate to the project:

```bash
cd handwritten-digit-recognition-neural-network
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Locally

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will become available through the local Streamlit server.

---

# 📊 Results

The project evaluates the trained Neural Network using:

```text
✓ Validation Accuracy
✓ Validation Loss
✓ Confusion Matrix
✓ Classification Report
✓ Prediction Visualization
✓ Error Analysis
```

> **Model performance:** Add the final accuracy, loss, and other evaluation results here after completing training.

Example:

```text
Validation Accuracy: XX.XX%
Validation Loss: X.XXXX
```

---

# 💡 Key Learning Outcomes

This project provided practical experience with:

### Deep Learning Fundamentals

* Neural Networks
* Dense layers
* Forward propagation
* Backpropagation
* Activation functions
* Loss functions
* Optimization
* Model training

### Data Engineering

* CSV data loading
* Feature/target separation
* Image reshaping
* Pixel normalization
* Label encoding

### Model Evaluation

* Accuracy
* Loss curves
* Confusion matrices
* Classification reports
* Error analysis

### Deployment

* Model serialization
* Loading trained models
* Streamlit application development
* ML inference pipelines
* Cloud deployment

---

# 🚧 Limitations

Although the model performs well on MNIST-style handwritten digits, the system may perform poorly on real-world handwriting that differs significantly from the training distribution.

Potential challenges include:

* Different writing styles
* Image rotation
* Different stroke thickness
* Poor contrast
* Background noise
* Incorrect image positioning
* Non-standard image dimensions

The model is primarily designed for images similar to the training data.

---

# 🔮 Future Improvements

The project can be extended in several directions.

### Deep Learning Improvements

* Replace the Dense Neural Network with a CNN
* Add Dropout for regularization
* Perform hyperparameter tuning
* Experiment with different optimizers
* Compare multiple architectures

### Computer Vision Improvements

* Image centering
* Noise removal
* Thresholding
* Stroke normalization
* Automatic resizing

### Application Improvements

* Confidence visualization
* Prediction probability chart
* Clear/reset drawing functionality
* Multiple digit recognition
* Batch image prediction
* Improved UI/UX

### Production Improvements

* FastAPI inference backend
* React frontend
* Docker containerization
* REST API
* Cloud-based model serving
* Model monitoring

---

# 🔬 Next Version: CNN

A natural next step for this project is replacing the fully connected Neural Network with a **Convolutional Neural Network (CNN)**.

Current architecture:

```text
Image
 ↓
Flatten
 ↓
Dense
 ↓
Dense
 ↓
Output
```

Future architecture:

```text
Image
 ↓
Convolution
 ↓
Pooling
 ↓
Convolution
 ↓
Pooling
 ↓
Flatten
 ↓
Dense
 ↓
Output
```

CNNs are generally better suited for image-related tasks because they can learn spatial and local visual features more effectively.

---

# 🎓 Project Significance

This project demonstrates the transition from traditional Machine Learning to Deep Learning by implementing a complete neural-network-based image classification system.

Rather than stopping at model training, the project extends through:

```text
Data
 ↓
Preprocessing
 ↓
Deep Learning
 ↓
Evaluation
 ↓
Inference
 ↓
Model Serialization
 ↓
Web Application
 ↓
Deployment
```

This makes the project an **end-to-end AI application** rather than only a notebook-based experiment.

---

# 👨‍💻 Author

**Pranav Sharma**

Computer Science Undergraduate focused on:

* Artificial Intelligence
* Machine Learning
* Generative AI
* Deep Learning
* Software Engineering

Building practical AI-powered applications and exploring the intersection of Machine Learning and software development.

---

## ⭐ Acknowledgements

This project was developed as part of my Deep Learning learning journey, with the goal of understanding Neural Networks from fundamentals through deployment.

---

## 📜 License

This project is available under the MIT License.
