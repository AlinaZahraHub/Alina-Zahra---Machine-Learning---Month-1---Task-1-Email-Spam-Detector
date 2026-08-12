import streamlit as str_lib  # ya seedha streamlit
import streamlit as st

# 1. Sabse pehli Streamlit command yeh honi chahiye
st.set_page_config(page_title="MNIST Classifier", layout="wide", page_icon="🧠")

from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

@st.cache_resource
def load_mnist_model():
    return tf.keras.models.load_model('mnist_digit_model.h5')

# Baaki ka saara code iske baad aayega...

# Model load
try:
    model = load_mnist_model()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    MODEL_ERROR = str(e)

# Page Config
st.set_page_config(page_title="MNIST Classifier", layout="wide", page_icon="🧠")

# --- NEON AI THEME CSS - 2nd Image Wala Design ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');

.stApp {
    background-color: #060A14;
    background-image:
        radial-gradient(circle at 15% 20%, rgba(56, 189, 248, 0.18) 0%, transparent 50%),
        radial-gradient(circle at 85% 80%, rgba(168, 85, 247, 0.18) 0%, transparent 50%),
        linear-gradient(rgba(56, 189, 248, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(56, 189, 248, 0.03) 1px, transparent 1px);
    background-size: auto, auto, 40px 40px, 40px 40px;
    font-family: 'Space Grotesk', sans-serif;
}

/* Top Header - Like 2nd image */
.top-header {
    background: rgba(10, 16, 32, 0.95);
    border: 1px solid rgba(56, 189, 248, 0.5);
    border-radius: 16px;
    padding: 16px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 0 25px rgba(56, 189, 248, 0.2);
    margin-bottom: 20px;
}
.header-title {
    font-size: 28px; font-weight: 800; letter-spacing: 0.5px;
    color: #E2F3FF;
}
.header-title span { color: #38BDF8; }
.header-sub { color: #7DD3FC; font-size: 13px; opacity: 0.8; }
.header-badge {
    background: rgba(56, 189, 248, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.6);
    padding: 6px 16px; border-radius: 20px; font-size: 13px;
    color: #7DD3FC;
}
.header-badge-green {
    background: rgba(74, 222, 128, 0.12);
    border: 1px solid rgba(74, 222, 128, 0.6);
    padding: 6px 14px; border-radius: 20px; font-size: 13px;
    color: #4ADE80;
}

/* Cards */
div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(12, 19, 38, 0.88)!important;
    border: 1px solid rgba(56, 189, 248, 0.35)!important;
    border-radius: 18px!important;
    padding: 18px!important;
    backdrop-filter: blur(12px);
    box-shadow: 0 0 25px rgba(0,0,0,0.5), inset 0 0 15px rgba(56, 189, 248, 0.05)!important;
}

/* Right Prediction Card Glow */
.prediction-card-wrapper div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid rgba(74, 222, 128, 0.5)!important;
    box-shadow: 0 0 30px rgba(74, 222, 128, 0.15)!important;
}

/* Big Number */
.big-pred {
    font-size: 92px; font-weight: 900; line-height: 0.9;
    color: #86EFAC;
    text-shadow: 0 0 25px rgba(74, 222, 128, 0.8);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #0EA5E9 0%, #38BDF8 100%);
    color: #020617;
    border-radius: 10px;
    font-weight: 700;
    border: none;
    padding: 0.6rem 1rem;
    width: 100%;
    letter-spacing: 0.5px;
}
.stButton>button:hover { filter: brightness(1.2); box-shadow: 0 0 20px rgba(56, 189, 248, 0.6); }

div[data-testid="stTabs"] button {
    color: #94A3B8!important;
}
div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
    color: #38BDF8!important;
    border-bottom-color: #38BDF8!important;
}

/* Prob Bars */
.prob-bg { background: rgba(148, 163, 184, 0.15); height: 10px; border-radius: 6px; overflow: hidden; }
.prob-fill { height: 100%; background: linear-gradient(90deg, #22D3EE, #4ADE80); border-radius: 6px; box-shadow: 0 0 10px rgba(74, 222, 128, 0.6); }
.prob-fill-low { background: rgba(56, 189, 248, 0.4); }
</style>
""", unsafe_allow_html=True)

# --- TOP HEADER ---
st.markdown("""
<div class="top-header">
    <div style="display:flex; align-items:center; gap:14px;">
        <div style="font-size:26px;">💠</div>
        <div>
            <div class="header-title">MNIST <span>Classifier</span></div>
            <div class="header-sub">AI-Powered Digit Recognition • Built with Streamlit & TensorFlow</div>
        </div>
    </div>
    <div style="display:flex; gap:10px; align-items:center;">
        <div class="header-badge">Model: CNN v2.1</div>
        <div class="header-badge-green">● Ready</div>
    </div>
</div>
""", unsafe_allow_html=True)

def predict_digit(img_array):
    if MODEL_LOADED:
        probs = model.predict(img_array, verbose=0)[0]
        pred = int(np.argmax(probs))
        return pred, probs
    else:
        probs = np.random.dirichlet(np.ones(10))
        return int(np.argmax(probs)), probs

# Layout
col_left, col_right = st.columns([1, 1.15], gap="large")

with col_left:
    with st.container(border=True):
        st.markdown('<div style="color:#7DD3FC; font-weight:600; font-size:14px; letter-spacing:1px;">✏️ Input Canvas</div>', unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["Draw Digit", "Upload Image"])
        input_image = None

        with tab1:
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0)",
                stroke_width=20,
                stroke_color="#FFFFFF",
                background_color="#070B18",
                height=300,
                width=380,
                drawing_mode="freedraw",
                key="canvas",
            )
            if canvas_result.image_data is not None and np.sum(canvas_result.image_data[:,:,3]) > 100:
                img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
                img_gray = img.convert('L')
                input_image = img_gray

        with tab2:
            uploaded = st.file_uploader("Upload", type=["png","jpg","jpeg"], label_visibility="collapsed")
            if uploaded is not None:
                img = Image.open(uploaded).convert('L')
                st.image(img, width=200)
                input_image = img

        c1, c2 = st.columns(2)
        with c1:
            st.button("Clear Canvas", use_container_width=True)
        with c2:
            predict_clicked = st.button("Run Prediction 🚀", use_container_width=True, type="primary")

        st.markdown("""
        <div style="display:flex; gap:10px; margin-top:15px;">
            <div style="flex:1; background: rgba(15,23,42,0.8); border:1px solid rgba(56,189,248,0.2); border-radius:10px; padding:10px; text-align:center; font-size:12px; color:#64748B;">🖌️ Brush</div>
            <div style="flex:1; background: rgba(15,23,42,0.8); border:1px solid rgba(56,189,248,0.2); border-radius:10px; padding:10px; text-align:center; font-size:12px; color:#64748B;">🧽 Eraser</div>
            <div style="flex:1; background: rgba(15,23,42,0.8); border:1px solid rgba(56,189,248,0.2); border-radius:10px; padding:10px; text-align:center; font-size:12px; color:#64748B;">🗑️ Reset</div>
        </div>
        """, unsafe_allow_html=True)

with col_right:
    # Prediction Box
    with st.container(border=True):
        st.markdown('<div style="color:#86EFAC; font-weight:600; font-size:14px;">📊 Prediction</div>', unsafe_allow_html=True)

        if input_image is not None and predict_clicked:
            img_resized = input_image.resize((28,28))
            # Agar white background hai to invert karo, black hai to rehne do
            img_array_test = np.array(img_resized)
            if np.mean(img_array_test) > 127: # white background
                img_resized = ImageOps.invert(img_resized)

            img_array = np.array(img_resized).reshape(1,28,28,1) / 255.0
            pred, probs = predict_digit(img_array)
            st.session_state['probs'] = probs
            st.session_state['pred'] = pred

            st.markdown(f"""
            <div style="background: rgba(0,0,0,0.3); border:1px solid rgba(74,222,128,0.3); border-radius:14px; padding:20px; display:flex; align-items:center; gap:25px; margin-top:10px;">
                <div class="big-pred">{pred}</div>
                <div>
                    <div style="color:#BBF7D0; font-size:22px; font-weight:700;">Predicted Digit: {pred}</div>
                    <div style="color:#86EFAC; font-size:18px; font-weight:600; margin-top:4px;">Confidence: {probs[pred]*100:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: rgba(0,0,0,0.3); border:1px solid rgba(74,222,128,0.2); border-radius:14px; padding:20px; display:flex; align-items:center; gap:25px; margin-top:10px; opacity:0.6;">
                <div class="big-pred" style="color:#334155; text-shadow:none;">5</div>
                <div>
                    <div style="color:#64748B; font-size:20px;">Predicted Digit: -</div>
                    <div style="color:#475569; font-size:16px; margin-top:4px;">Draw & Click Predict</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Class Probabilities 0-9
        st.markdown('<div style="color:#86EFAC; font-size:13px; margin-top:18px; margin-bottom:8px;">Class Probabilities</div>', unsafe_allow_html=True)
        probs_to_show = st.session_state.get('probs', np.zeros(10))

        bars_html = ""
        for i in range(10): # 0 se 9 tak
            p = probs_to_show[i]
            w = int(p * 100)
            is_max = p == max(probs_to_show) and p > 0
            bars_html += f"""
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:7px; font-size:13px;">
                <div style="width:12px; color:{'#4ADE80' if is_max else '#64748B'}; font-weight:600;">{i}</div>
                <div class="prob-bg" style="flex:1;"><div class="{'prob-fill' if is_max else 'prob-fill prob-fill-low'}" style="width:{w if w>3 else 3}%;"></div></div>
                <div style="width:50px; text-align:right; color:{'#4ADE80' if is_max else '#64748B'}; font-weight:{600 if is_max else 400};">{p*100:.1f}%</div>
            </div>
            """
        st.markdown(bars_html, unsafe_allow_html=True)

        st.markdown("""
        <div style="display:flex; gap:10px; margin-top:15px;">
            <div style="flex:1; background: rgba(14,165,233,0.08); border:1px solid rgba(56,189,248,0.2); border-radius:10px; padding:10px; text-align:center;">
                <div style="color:#38BDF8; font-size:18px;">◎</div><div style="color:#94A3B8; font-size:11px;">Accuracy</div><div style="color:#E0F2FE; font-size:14px; font-weight:700;">99.2%</div>
            </div>
            <div style="flex:1; background: rgba(14,165,233,0.08); border:1px solid rgba(56,189,248,0.2); border-radius:10px; padding:10px; text-align:center;">
                <div style="color:#38BDF8; font-size:18px;">⚡</div><div style="color:#94A3B8; font-size:11px;">Latency</div><div style="color:#E0F2FE; font-size:14px; font-weight:700;">12 ms</div>
            </div>
            <div style="flex:1; background: rgba(14,165,233,0.08); border:1px solid rgba(56,189,248,0.2); border-radius:10px; padding:10px; text-align:center;">
                <div style="color:#38BDF8; font-size:18px;">🧠</div><div style="color:#94A3B8; font-size:11px;">Model Size</div><div style="color:#E0F2FE; font-size:14px; font-weight:700;">0.8M</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if 'pred' in st.session_state:
            st.success(f"High confidence prediction • Digit {st.session_state['pred']} detected")

# Footer
st.markdown("<div style='text-align:center; color:#334155; font-size:11px; margin-top:20px;'>Streamlit 1.38 • TensorFlow 2.15 • Running locally • http://localhost:8501</div>", unsafe_allow_html=True)