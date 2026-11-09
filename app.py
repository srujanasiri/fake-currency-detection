

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Currency Shield | Note Verification",
    page_icon="💵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# CUSTOM CSS - PREMIUM UI DESIGN
# ==========================================

st.markdown("""
<style>

/* Main background */

.stApp {
    background:
        radial-gradient(circle at 5% 5%,
        rgba(59,130,246,0.16), transparent 30%),
        radial-gradient(circle at 95% 90%,
        rgba(37,99,235,0.13), transparent 30%),
        linear-gradient(135deg,#f0f5ff,#f8faff,#edf3ff);

    background-attachment: fixed;
}

/* Main content */

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Header */

[data-testid="stHeader"] {
    background: transparent;
}

/* Hero section */

.hero {
    background: linear-gradient(120deg,#101d3a,#173d78,#2563a8);
    padding: 42px 35px;
    border-radius: 24px;
    color: white;
    box-shadow: 0 12px 35px rgba(20,50,100,0.20);
    margin-bottom: 30px;
}

.hero-tag {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    padding: 7px 15px;
    border-radius: 30px;
    font-size: 13px;
    letter-spacing: 1px;
    color: #dbeafe;
}

.hero h1 {
    color: white;
    font-size: 38px;
    font-weight: 800;
    margin-top: 20px;
    margin-bottom: 10px;
}

.hero p {
    color: #dbeafe;
    font-size: 16px;
    line-height: 1.7;
}

/* Section headings */

.section-title {
    color: #172554;
    font-size: 23px;
    font-weight: 750;
    margin-bottom: 5px;
}

.section-subtitle {
    color: #64748b;
    font-size: 14px;
    margin-bottom: 20px;
}

/* Cards */

.custom-card {
    background: rgba(255,255,255,0.90);
    border: 1px solid #dce7f7;
    border-radius: 20px;
    padding: 25px;
    box-shadow: 0 8px 25px rgba(30,64,120,0.07);
    margin-bottom: 20px;
}

.feature-card {
    background: white;
    border: 1px solid #e1e9f5;
    padding: 20px;
    border-radius: 17px;
    min-height: 120px;
    box-shadow: 0 5px 18px rgba(30,64,120,0.05);
}

.feature-icon {
    font-size: 27px;
}

.feature-title {
    color: #1e3a8a;
    font-weight: 750;
    font-size: 16px;
    margin-top: 8px;
}

.feature-text {
    color: #64748b;
    font-size: 13px;
    margin-top: 5px;
}

/* Buttons */

.stButton > button {
    background: linear-gradient(100deg,#2563eb,#1d4ed8);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 25px;
    font-weight: 700;
    min-height: 48px;
    transition: 0.25s ease;
}

.stButton > button:hover {
    background: linear-gradient(100deg,#1d4ed8,#1e40af);
    color: white;
    box-shadow: 0 7px 18px rgba(37,99,235,0.25);
    transform: translateY(-2px);
}

/* File uploader */

[data-testid="stFileUploader"] {
    background: #f8fbff;
    border: 2px dashed #93b4ed;
    border-radius: 15px;
    padding: 15px;
}

/* Result cards */

.result-card {
    border-radius: 18px;
    padding: 25px;
    text-align: center;
    margin-top: 12px;
    margin-bottom: 20px;
}

.genuine {
    background: linear-gradient(135deg,#dcfce7,#f0fdf4);
    border: 1px solid #86efac;
    color: #166534;
}

.counterfeit {
    background: linear-gradient(135deg,#fee2e2,#fff1f2);
    border: 1px solid #fca5a5;
    color: #991b1b;
}

.result-icon {
    font-size: 42px;
}

.result-heading {
    font-size: 26px;
    font-weight: 800;
    margin-top: 8px;
}

.result-description {
    font-size: 14px;
    margin-top: 8px;
}

/* Score cards */

.score-card {
    background: white;
    border: 1px solid #dce7f7;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 5px 18px rgba(30,64,120,0.05);
}

.score-label {
    color: #64748b;
    font-size: 14px;
    font-weight: 600;
}

.score-value {
    color: #1e3a8a;
    font-size: 28px;
    font-weight: 800;
    margin-top: 8px;
}

/* Footer */

.footer {
    text-align: center;
    padding: 25px 10px;
    color: #64748b;
    font-size: 13px;
}

div[data-testid="stImage"] img {
    border-radius: 16px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD TRAINED CNN MODEL
# ==========================================

@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model("currency_cnn.keras")

try:
    model = load_cnn_model()
except Exception as e:
    st.error("Unable to load the CNN model. Please check currency_cnn.keras.")
    st.exception(e)
    st.stop()


# ==========================================
# HERO SECTION
# ==========================================

st.markdown("""
<div class="hero">

    <div class="hero-tag">
        ✦ AI POWERED CURRENCY SECURITY
    </div>

    <h1>💵 Currency Shield</h1>

    <p>
        Smart Indian Currency Note Verification System
        <br>
        Powered by Deep Learning and Convolutional Neural Networks.
        Upload a currency image and let our CNN model
        analyze it.
    </p>

</div>
""", unsafe_allow_html=True)


# ==========================================
# FEATURES SECTION
# ==========================================

st.markdown("""
<div class="section-title">
    Welcome to Currency Shield
</div>

<div class="section-subtitle">
    Intelligent currency image analysis at your fingertips.
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🤖</div>
        <div class="feature-title">AI Technology</div>
        <div class="feature-text">
            CNN-based deep learning analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">⚡</div>
        <div class="feature-title">Fast Detection</div>
        <div class="feature-text">
            Analyze uploaded currency images.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🛡️</div>
        <div class="feature-title">Smart Verification</div>
        <div class="feature-text">
            Genuine and counterfeit model predictions.
        </div>
    </div>
    """, unsafe_allow_html=True)


st.write("")
st.divider()


# ==========================================
# IMAGE UPLOAD SECTION
# ==========================================

st.markdown("""
<div class="section-title">
    🔍 Currency Note Analysis
</div>

<div class="section-subtitle">
    Upload a clear photograph of an Indian currency note.
</div>
""", unsafe_allow_html=True)


left, right = st.columns([1, 1], gap="large")

uploaded_file = None
image = None

with left:

    st.markdown("""
    <div class="custom-card">
        <h3 style="color:#1e3a8a;">
            📤 Upload Your Image
        </h3>
        <p style="color:#64748b;font-size:14px;">
            Select a currency photograph in JPG,
            JPEG or PNG format.
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose Currency Image",
        type=["jpg", "jpeg", "png"],
        key="currency_upload"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        st.success("Image uploaded successfully!")


with right:

    if image is not None:

        st.markdown("""
        <div class="section-title">
            🖼️ Image Preview
        </div>
        """, unsafe_allow_html=True)

        st.image(
            image,
            caption="Uploaded Currency Note",
            use_container_width=True
        )

    else:

        st.markdown("""
        <div class="custom-card"
             style="text-align:center;padding:55px 15px;">

            <div style="font-size:55px;">💵</div>

            <h3 style="color:#1e3a8a;">
                Your Image Preview
            </h3>

            <p style="color:#64748b;">
                Your uploaded currency image
                will appear here.
            </p>

        </div>
        """, unsafe_allow_html=True)


# ==========================================
# DETECTION SECTION
# ==========================================

if image is not None:

    st.write("")

    detect_col1, detect_col2, detect_col3 = st.columns([1, 2, 1])

    with detect_col2:

        detect_button = st.button(
            "🔍  DETECT CURRENCY",
            type="primary",
            use_container_width=True
        )

    if detect_button:

        with st.spinner("Analyzing currency image using CNN..."):

            # Original image preprocessing
            img = image.resize((128, 128))

            img_array = np.array(
                img,
                dtype=np.float32
            ) / 255.0

            img_array = np.expand_dims(
                img_array,
                axis=0
            )

            # Original CNN prediction logic
            prediction = float(
                model.predict(
                    img_array,
                    verbose=0
                )[0][0]
            )

            p_real = prediction
            p_fake = 1 - prediction

        st.divider()

        st.markdown("""
        <div class="section-title">
            📊 Currency Detection Results
        </div>

        <div class="section-subtitle">
            CNN model prediction analysis.
        </div>
        """, unsafe_allow_html=True)

        # ==================================
        # RESULT DISPLAY
        # ==================================

        if p_real > 0.5:

            st.markdown("""
            <div class="result-card genuine">

                <div class="result-icon">✅</div>

                <div class="result-heading">
                    GENUINE CURRENCY
                </div>

                <div class="result-description">
                    The CNN model predicts that this
                    currency note is genuine.
                </div>

            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class="result-card counterfeit">

                <div class="result-icon">⚠️</div>

                <div class="result-heading">
                    COUNTERFEIT CURRENCY
                </div>

                <div class="result-description">
                    The CNN model predicts that this
                    currency note is counterfeit.
                </div>

            </div>
            """, unsafe_allow_html=True)


        # ==================================
        # SCORE DISPLAY
        # ==================================

        score1, score2 = st.columns(2)

        with score1:

            st.markdown(f"""
            <div class="score-card">

                <div class="score-label">
                    🟢 Genuine Model Score
                </div>

                <div class="score-value">
                    {p_real * 100:.2f}%
                </div>

            </div>
            """, unsafe_allow_html=True)

        with score2:

            st.markdown(f"""
            <div class="score-card">

                <div class="score-label">
                    🔴 Counterfeit Model Score
                </div>

                <div class="score-value">
                    {p_fake * 100:.2f}%
                </div>

            </div>
            """, unsafe_allow_html=True)


        # ==================================
        # MODEL SCORE VISUALIZATION
        # ==================================

        st.write("")

        st.markdown("""
        <div class="section-title">
            Model Score Visualization
        </div>
        """, unsafe_allow_html=True)

        st.progress(p_real, text="Genuine model score")
        st.progress(p_fake, text="Counterfeit model score")


        # ==================================
        # DISCLAIMER
        # ==================================

        st.warning(
            "Important: This is an experimental CNN-based "
            "prediction. Model scores are not independently "
            "verified probabilities and this system is not an "
            "official currency authentication device."
        )


# ==========================================
# INFORMATION SECTION
# ==========================================

st.write("")
st.divider()

with st.expander("ℹ️ About Currency Shield"):

    st.write("""
    Currency Shield is a mini-project that uses
    Convolutional Neural Networks (CNN) to classify
    uploaded Indian currency note images.

    The trained CNN model analyzes the input image
    and produces a prediction score for genuine
    and counterfeit classes.

    This application is intended for educational
    and experimental purposes.
    """)

with st.expander("📌 How to Use This Application"):

    st.markdown("""
    1. Upload an Indian currency note photograph.
    2. Preview the uploaded image.
    3. Click the Detect Currency button.
    4. Wait for the CNN model to analyze the image.
    5. View the prediction and model scores.
    """)


# ==========================================
# FOOTER
# ==========================================

st.markdown("""
<div class="footer">

    <b style="color:#1e3a8a;">
        💵 CURRENCY SHIELD
    </b>

    <br><br>

    CNN-Based Counterfeit Currency Detection System

    <br>

    Mini Project | Artificial Intelligence & Deep Learning

    <br><br>

    © 2026 Currency Shield. Educational Use Only.

</div>
""", unsafe_allow_html=True)

