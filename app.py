
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Currency Note Verification",
    page_icon="💵",
    layout="centered"
)

# Beautiful Background Design
st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(180, 190, 255, 0.45),
            transparent 35%
        ),
        radial-gradient(
            circle at 90% 85%,
            rgba(170, 220, 255, 0.45),
            transparent 35%
        ),
        linear-gradient(
            135deg,
            #eef2ff,
            #f8faff,
            #e6f0ff
        );
    background-attachment: fixed;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stMain"] {
    background: transparent;
}
</style>
""", unsafe_allow_html=True)


# Load trained CNN model
@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model("currency_cnn.keras")

model = load_cnn_model()


# Application heading
st.title("💵 Currency Note Verification System")

st.write(
    "A CNN-based system for detecting genuine "
    "and counterfeit Indian currency notes."
)

st.divider()


# Image upload
st.subheader("Upload Currency Image")

uploaded_file = st.file_uploader(
    "Choose a currency note photograph",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Currency Note",
        use_container_width=True
    )

    if st.button("🔍 Detect Currency", type="primary"):

        # Image preprocessing
        img = image.resize((128, 128))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # CNN prediction
        prediction = float(
            model.predict(img_array, verbose=0)[0][0]
        )

        p_real = prediction
        p_fake = 1 - prediction

        st.divider()
        st.subheader("Detection Result")

        if p_real > 0.5:
            st.success("✅ Prediction: GENUINE CURRENCY")
        else:
            st.error("⚠️ Prediction: COUNTERFEIT CURRENCY")

        st.write(f"Genuine model score: {p_real * 100:.2f}%")
        st.write(f"Counterfeit model score: {p_fake * 100:.2f}%")

        st.caption(
            "Note: This is an experimental CNN-based prediction, "
            "not an official currency authentication."
        )

else:
    st.info("Please upload a currency image to begin detection.")

st.divider()

st.caption("Mini Project | CNN-Based Counterfeit Currency Detection")

    
