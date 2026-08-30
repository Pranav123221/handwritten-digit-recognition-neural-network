import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="centered"
)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "handwritten_digit_recognition.keras"
    )


model = load_model()


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🔢 Handwritten Digit Recognition")

st.write(
    "Draw a handwritten digit (0–9) and let the Neural Network predict it."
)

st.divider()


# --------------------------------------------------
# Drawing Canvas
# --------------------------------------------------

st.subheader("✏️ Draw a Digit")

canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",
    stroke_width=15,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔍 Predict Digit", use_container_width=True):

    if canvas_result.image_data is not None:

        # Convert canvas to PIL Image
        image = Image.fromarray(
            canvas_result.image_data.astype("uint8")
        )

        # Convert to grayscale
        image = image.convert("L")

        # Resize to MNIST format
        image = image.resize((28, 28))

        # Convert to NumPy array
        image_array = np.array(image)

        # Normalize pixel values
        image_array = image_array / 255.0

        # Reshape for model
        image_array = image_array.reshape(1, 28, 28, 1)

        # Make prediction
        predictions = model.predict(
            image_array,
            verbose=0
        )

        # Get predicted class
        predicted_digit = np.argmax(
            predictions[0]
        )

        # Get confidence
        confidence = np.max(
            predictions[0]
        ) * 100

        # Display result
        st.divider()

        st.subheader("🎯 Prediction")

        st.metric(
            label="Predicted Digit",
            value=str(predicted_digit)
        )

        st.write(
            f"Confidence: **{confidence:.2f}%**"
        )

        # Probability distribution
        st.subheader("📊 Prediction Probabilities")

        probabilities = predictions[0]

        probability_data = {
            str(i): float(probabilities[i])
            for i in range(10)
        }

        st.bar_chart(probability_data)
