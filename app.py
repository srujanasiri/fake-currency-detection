
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Currency Note Authentication System",
    page_icon="💵",
    layout="centered"
)

# Custom styling
st.markdown("""
<style>

/* Elegant Indian currency-inspired background */
.stApp {
    background:
        radial-gradient(
            circle at 10% 15%,
            rgba(34, 139, 94, 0.13),
            transparent 35%
        ),
        radial-gradient(
            circle at 90% 85%,
            rgba(218, 184, 105, 0.16),
            transparent 35%
        ),
        repeating-linear-gradient(
            45deg,
            rgba(30, 100, 65, 0.025) 0px,
            rgba(30, 100, 65, 0.025) 2px,
            transparent 2px,
            transparent 12px
        ),
        linear-gradient(
            135deg,
            #edf7ed,
            #fffdf4,
            #e5f1e7
        );

    background-attachment: fixed;
}

/* Subtle Rupee watermark */
.stApp::before {
    content: "₹";
    position: fixed;
    top: 15%;
    right: 5%;
    font-size: 380px;
    font-weight: 800;
    color: rgba(24, 100, 65, 0.035);
    transform: rotate(-15deg);
    pointer-events: none;
    z-index: 0;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 2rem;
    max-width: 850px;
}

/* Main heading */
.main-title {
    text-align: center;
    color: #145c3b;
    font-size: 38px;
    font-weight: 800;
    padding: 15px;
    letter-spacing: 1px;
}

.subtitle {
    text-align: center;
    color: #526b5c;
    font-size: 16px;
    margin-bottom: 25px;
    line-height: 1.8;
}

/* Section headings */
h2, h3 {
    color: #145c3b !important;
}

/* Upload box */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.85);
    padding: 15px;
    border-radius: 15px;
    border: 2px dashed #78a98b;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #218052, #145c3b);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 12px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton > button:hover {
    background: #10472e;
    color: white;
    transform: scale(1.02);
}

/* Result card */
.result-box {
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    margin: 15px 0;
    font-size: 22px;
    font-weight: bold;
}

.genuine {
    background: #dcfce7;
    color: #166534;
    border: 2px solid #86efac;
}

.fake {
    background: #fee2e2;
    color: #991b1b;
    border: 2px solid #fca5a5;
}

/* Footer */
.footer {
    text-align: center;
    color: #526b5c;
    font-size: 13px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)


# Load trained CNN model
@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model("currency_cnn.keras")

model = load_cnn_model()


# Application heading
st.markdown("""
<div class="main-title">
    💵Currency Note Authentication System
</div>

<div class="subtitle">
    Smart Indian Currency Note Verification System
    <br>
    <b>Powered by Artificial Intelligence & CNN</b>
</div>
""", unsafe_allow_html=True)

st.divider()


# Welcome section
st.markdown("### 🛡️ Welcome to Currency Verification")

st.write(
    "Verify Indian currency notes using our "
    "CNN-based deep learning model. Upload an image "
    "and discover the model's prediction instantly."
)

st.divider()


# Image upload
st.markdown("### 📤 Upload Your Currency Image")

uploaded_file = st.file_uploader(
    "Choose a currency note photograph",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="🖼️ Uploaded Currency Note",
        use_container_width=True
    )

    st.write("")

    if st.button(
        "🔍 Detect Currency",
        type="primary",
        use_container_width=True
    ):

        with st.spinner("🤖 AI is analyzing your currency note..."):

            # Original image preprocessing
            img = image.resize((128, 128))
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Original CNN prediction
            prediction = float(
               prediction = float(model.predict(img_array, verbose=0)[0][0])

                p_fake = prediction
                p_real = 1 - prediction
        st.divider()

        st.markdown("### 📊 Detection Results")

        if p_real > 0.5:

            st.markdown("""
            <div class="result-box genuine">
                ✅ COUNTERFIET CURRENCY
                <br>
                <span style="font-size:15px;">
                    CNN model predicts this note as genuine.
                </span>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class="result-box fake">
                ⚠️ GENUINE CURRENCY
                <br>
                <span style="font-size:15px;">
                    CNN model predicts this note as counterfeit.
                </span>
            </div>
            """, unsafe_allow_html=True)

        # Display scores
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "🟢 Genuine Score",
                f"{p_real * 100:.2f}%"
            )

        with col2:
            st.metric(
                "🔴 Counterfeit Score",
                f"{p_fake * 100:.2f}%"
            )

        # Visual score bars
        st.markdown("#### 📈 Prediction Analysis")

        st.progress(p_real, text="Genuine Score")
        st.progress(p_fake, text="Counterfeit Score")

else:
    st.info("👆 Please upload a currency image to begin detection.")


st.divider()


# About section
with st.expander("ℹ️ About This Project"):
    st.write(
        "NoteSure AI is a CNN-based counterfeit currency "
        "detection application developed using Python, "
        "TensorFlow and Streamlit."
    )

with st.expander("📌 How to Use"):
    st.write("""
    1. Upload an Indian currency image.
    2. Preview the uploaded photograph.
    3. Click Detect Currency.
    4. View the CNN prediction and scores.
    """)


# Footer
st.markdown("""
<div class="footer">
    💵 <b>Currency Note Authentication</b>
    <br><br>
    CNN-Based Counterfeit Currency Detection
    <br>
    Mini Project | Artificial Intelligence & Deep Learning
    <br><br>
    © 2026 NoteSure AI
</div>
""", unsafe_allow_html=True)
