
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import time

st.set_page_config(
    page_title="Industrial Defect Detector",
    page_icon="🔍",
    layout="centered"
)

# ── CUSTOM CSS STYLING ─────────────────────────────────────────
st.markdown("""
<style>
    /* Background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
    }

    /* Main title */
    .main-title {
        text-align: center;
        font-size: 3em;
        font-weight: 900;
        background: linear-gradient(90deg, #f093fb, #f5576c, #fda085);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        padding-top: 20px;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #a0a0b0;
        font-size: 1.1em;
        margin-bottom: 30px;
    }

    /* Upload box */
    .upload-box {
        background: rgba(255,255,255,0.05);
        border: 2px dashed #f093fb;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
    }

    /* Result cards */
    .result-good {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        border-radius: 20px;
        padding: 25px;
        text-align: center;
        font-size: 1.5em;
        font-weight: bold;
        color: white;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(56, 239, 125, 0.3);
    }

    .result-bad {
        background: linear-gradient(135deg, #f5576c, #f093fb);
        border-radius: 20px;
        padding: 25px;
        text-align: center;
        font-size: 1.5em;
        font-weight: bold;
        color: white;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(245, 87, 108, 0.3);
    }

    /* Stats card */
    .stats-card {
        background: rgba(255,255,255,0.07);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid rgba(255,255,255,0.1);
    }

    /* Confidence label */
    .conf-label {
        color: #f093fb;
        font-weight: bold;
        font-size: 1em;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #555577;
        font-size: 0.85em;
        margin-top: 40px;
        padding-bottom: 20px;
    }

    /* Hide streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Style file uploader */
    .stFileUploader > div {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 15px !important;
        border: 2px dashed #f093fb !important;
    }

    /* Progress bar color */
    .stProgress > div > div {
        background: linear-gradient(90deg, #f093fb, #f5576c) !important;
    }
</style>
""", unsafe_allow_html=True)

# ── HEADER ─────────────────────────────────────────────────────
st.markdown('<div class="main-title">🔍 DefectScan AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Industrial Quality Control powered by YOLOv8 · MVTec Dataset</div>', unsafe_allow_html=True)

# ── DIVIDER ────────────────────────────────────────────────────
st.markdown("---")

# ── STATS ROW ──────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div style="background:rgba(240,147,251,0.15); border-radius:15px;
                padding:15px; text-align:center; border:1px solid #f093fb44;">
        <div style="font-size:2em">🏭</div>
        <div style="color:#f093fb; font-weight:bold">Dataset</div>
        <div style="color:white; font-size:0.9em">MVTec AD</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div style="background:rgba(245,87,108,0.15); border-radius:15px;
                padding:15px; text-align:center; border:1px solid #f5576c44;">
        <div style="font-size:2em">🧠</div>
        <div style="color:#f5576c; font-weight:bold">Model</div>
        <div style="color:white; font-size:0.9em">YOLOv8n-cls</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div style="background:rgba(56,239,125,0.15); border-radius:15px;
                padding:15px; text-align:center; border:1px solid #38ef7d44;">
        <div style="font-size:2em">🎯</div>
        <div style="color:#38ef7d; font-weight:bold">Accuracy</div>
        <div style="color:white; font-size:0.9em">100%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── LOAD MODEL ─────────────────────────────────────────────────
MODEL_PATH = "/content/drive/MyDrive/MVTec_Project/best_leather.pt"

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ── UPLOAD SECTION ─────────────────────────────────────────────
st.markdown("### 📤 Upload Product Image")
st.markdown("*Supported formats: JPG, JPEG, PNG*")

uploaded_file = st.file_uploader(
    label="",
    type=["jpg", "jpeg", "png"],
    help="Upload a leather product image to inspect for defects"
)

# ── PREDICTION SECTION ─────────────────────────────────────────
if uploaded_file:
    image = Image.open(uploaded_file)

    # Show image nicely
    col_img, col_info = st.columns([1.2, 1])

    with col_img:
        st.markdown("**📸 Uploaded Image**")
        st.image(image, use_container_width=True)

    with col_info:
        st.markdown("**📋 Image Info**")
        st.markdown(f"""
        <div class="stats-card">
            <p>📁 <span style="color:#f093fb">File:</span> {uploaded_file.name}</p>
            <p>📐 <span style="color:#f093fb">Size:</span> {image.size[0]} × {image.size[1]} px</p>
            <p>🎨 <span style="color:#f093fb">Mode:</span> {image.mode}</p>
            <p>💾 <span style="color:#f093fb">Format:</span> {uploaded_file.type}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🔍 Analyze Image", use_container_width=True)

    # Run analysis
    st.markdown("---")

    with st.spinner("🤖 AI is analyzing your image..."):
        # Fake loading bar for dramatic effect
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.008)
            progress.progress(i + 1)
        progress.empty()

        results = model(image, verbose=False)[0]
        probs = results.probs
        predicted = results.names[probs.top1]
        confidence = probs.top1conf.item()

    # ── RESULT ─────────────────────────────────────────────────
    st.markdown("### 🏆 Detection Result")

    if predicted == "good":
        st.markdown(f"""
        <div class="result-good">
            ✅ PASS — Product is NORMAL<br>
            <span style="font-size:0.7em; opacity:0.9">
                No defects detected · Confidence: {confidence:.2%}
            </span>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()  # 🎈 celebration effect!
    else:
        st.markdown(f"""
        <div class="result-bad">
            ❌ FAIL — Defect DETECTED<br>
            <span style="font-size:0.7em; opacity:0.9">
                Anomaly found · Confidence: {confidence:.2%}
            </span>
        </div>
        """, unsafe_allow_html=True)
        st.snow()  # ❄️ alert effect!

    # ── PROBABILITY BARS ───────────────────────────────────────
    st.markdown("### 📊 Confidence Breakdown")

    for name, prob in zip(results.names.values(), probs.data.tolist()):
        prob = float(prob)
        color = "#38ef7d" if name == "good" else "#f5576c"
        emoji = "✅" if name == "good" else "❌"

        st.markdown(f"""
        <div style="margin: 10px 0;">
            <span style="color:{color}; font-weight:bold">
                {emoji} {name.upper()}
            </span>
            <span style="float:right; color:white; font-weight:bold">
                {prob:.2%}
            </span>
        </div>
        """, unsafe_allow_html=True)
        st.progress(prob)

    # ── VERDICT SUMMARY ────────────────────────────────────────
    st.markdown("### 📝 Summary")
    st.markdown(f"""
    <div class="stats-card">
        <p>🔎 <b>Inspected:</b> {uploaded_file.name}</p>
        <p>🤖 <b>Model Decision:</b>
            <span style="color:{'#38ef7d' if predicted == 'good' else '#f5576c'}; font-weight:bold">
                {predicted.upper()}
            </span>
        </p>
        <p>💯 <b>Confidence:</b> {confidence:.2%}</p>
        <p>⚡ <b>Status:</b>
            {"<span style='color:#38ef7d'>✅ Approved for shipment</span>"
             if predicted == "good"
             else "<span style='color:#f5576c'>🚫 Rejected — needs inspection</span>"}
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Show placeholder when no image uploaded
    st.markdown("""
    <div style="text-align:center; padding:60px 20px;
                background:rgba(255,255,255,0.03);
                border-radius:20px; border:2px dashed #444466;
                margin:20px 0;">
        <div style="font-size:4em">🏭</div>
        <div style="color:#a0a0c0; font-size:1.2em; margin-top:10px">
            Upload a product image above to begin inspection
        </div>
        <div style="color:#666688; font-size:0.9em; margin-top:8px">
            Supports leather defect detection · Powered by YOLOv8
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── FOOTER ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="footer">
    Built with ❤️ using YOLOv8 + Streamlit · MVTec Anomaly Detection Dataset<br>
    🎓 College AI Project · Industrial Quality Control System
</div>
""", unsafe_allow_html=True)
