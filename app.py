

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Currency Note Verification",
    page_icon="💵",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f0f5ff, #ffffff);
}

.main-title {
    text-align: center;
    color: #12345a;
    font-size: 38px;
    font-weight: 800;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #52657a;
    font-size: 17px;
    margin-bottom: 25px;
}

.section-title {
    color: #12345a;
    font-size: 23px;
    font-weight: 700;
}

.result-card {
    background-color: white;
    padding: 22px;
    border-radius: 15px;
    border: 1px solid #dce5f0;
    text-align: center;
    margin-top: 15px;
}

.footer {
    text-align: center;
    color: #667085;
    padding: 20px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD CNN MODEL ----------------

@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model("currency_cnn.keras")


try:
    model = load_cnn_model()
except Exception as e:
    st.error(f"Unable to load CNN model: {e}")
    st.stop()


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.title("💵 Currency AI")
    st.divider()

    st.subheader("About Project")

    st.write(
        "This application uses a Convolutional "
        "Neural Network (CNN) to classify Indian "
        "currency note images."
    )

    st.divider()

    st.subheader("Technologies Used")

    st.write("🐍 Python")
    st.write("🧠 TensorFlow & CNN")
    st.write("🖥️ Streamlit")
    st.write("🖼️ Image Processing")

    st.divider()

    st.caption("Mini Project | AI & Data Science")


# ---------------- MAIN HEADING ----------------

st.markdown(
    '<div class="main-title">💵 Currency Note Verification System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Powered Genuine and Counterfeit Currency Detection</div>',
    unsafe_allow_html=True
)

st.divider()

# ---------------- INTRODUCTION ----------------

col1, col2, col3 = st.columns(3)

with col1:
    st.info("🧠 **CNN Technology**\n\nDeep Learning based image classification")

with col2:
    st.info("📷 **Image Analysis**\n\nUpload and analyze currency photographs")

with col3:
    st.info("⚡ **Quick Detection**\n\nGet an instant model prediction")

st.write("")

# ---------------- IMAGE UPLOAD ----------------

st.markdown(
    '<div class="section-title">📤 Upload Currency Image</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload an Indian currency note image in JPG, JPEG or PNG format."
)

uploaded_file = st.file_uploader(
    "Choose your currency image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        left, right = st.columns([1, 1])

        with left:
            st.subheader("📷 Uploaded Image")

            st.image(
                image,
                caption="Currency Note Image",
                use_container_width=True
            )

        with right:
            st.subheader("🔍 Image Details")

            st.write("**File Name:**", uploaded_file.name)
            st.write("**Image Format:**", uploaded_file.format or "Unknown")
            st.write("**Image Dimensions:**", f"{image.width} × {image.height}")

            st.write("")

            detect_button = st.button(
                "🔍 Detect Currency",
                type="primary",
                use_container_width=True
            )

        # ---------------- PREDICTION ----------------

        if detect_button:

            with st.spinner("Analyzing currency image..."):

                img = image.resize((128, 128))

                img_array = np.array(
                    img,
                    dtype=np.float32
                ) / 255.0

                img_array = np.expand_dims(
                    img_array,
                    axis=0
                )

                prediction = float(
                    model.predict(img_array, verbose=0)[0][0]
                )

                p_real = prediction
                p_fake = 1 - prediction

            st.divider()

            st.markdown(
                '<div class="section-title">📊 Currency Detection Result</div>',
                unsafe_allow_html=True
            )

            if p_real > 0.5:

                st.success("✅ GENUINE CURRENCY")

                st.markdown(
                    '<div class="result-card">'
                    '<h2 style="color:#16803c;">GENUINE NOTE</h2>'
                    '<p style="color:#52657a;">The CNN model classified this image as genuine currency.</p>'
                    '</div>',
                    unsafe_allow_html=True
                )

            else:

                st.error("⚠️ COUNTERFEIT CURRENCY")

                st.markdown(
                    '<div class="result-card">'
                    '<h2 style="color:#d62828;">COUNTERFEIT NOTE</h2>'
                    '<p style="color:#52657a;">The CNN model classified this image as counterfeit currency.</p>'
                    '</div>',
                    unsafe_allow_html=True
                )

            st.write("")

            score1, score2 = st.columns(2)

            with score1:
                st.metric(
                    "Genuine Model Score",
                    f"{p_real * 100:.2f}%"
                )
                st.progress(p_real)

            with score2:
                st.metric(
                    "Counterfeit Model Score",
                    f"{p_fake * 100:.2f}%"
                )
                st.progress(p_fake)

            st.caption(
                "Scores represent the model's output, "
                "not independently verified probabilities."
            )

            st.warning(
                "⚠️ This is an experimental CNN-based prediction. "
                "It is not an official currency authentication system. "
                "Do not use this result alone to determine whether a note is genuine."
            )

    except Exception as e:
        st.error(f"Error processing image: {e}")

else:

    st.info(
        "👆 Please upload a currency note image to start detection."
    )


# ---------------- PROJECT INFORMATION ----------------

st.divider()

st.markdown(
    '<div class="section-title">📌 About Our Project</div>',
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs(
    ["🎯 Objective", "⚙️ Working", "🧠 CNN Model"]
)

with tab1:
    st.write(
        "The main objective of this project is to develop "
        "an image classification system that identifies "
        "genuine and counterfeit Indian currency notes "
        "using deep learning."
    )

with tab2:
    st.write(
        "The user uploads a currency image. The image is "
        "resized and normalized before being passed to "
        "the trained CNN model. The model generates a "
        "prediction score and classifies the image."
    )

with tab3:
    st.write(
        "A Convolutional Neural Network (CNN) is a deep "
        "learning algorithm designed for image recognition "
        "and classification. Our trained model is used "
        "to distinguish genuine and counterfeit currency images."
    )


# ---------------- FOOTER ----------------

st.divider()

st.markdown(
    '<div class="footer">'
    '<b>CNN-Based Counterfeit Currency Detection</b><br>'
    'Mini Project | Artificial Intelligence and Data Science<br>'
    'Developed using Python, TensorFlow and Streamlit'
    '</div>',
    unsafe_allow_html=True
)


st.divider()

st.caption("Mini Project | CNN-Based Counterfeit Currency Detection")
