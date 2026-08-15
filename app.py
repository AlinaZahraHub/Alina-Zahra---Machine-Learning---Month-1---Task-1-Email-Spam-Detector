import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import datetime

# Secure NLTK downloads for Streamlit Cloud
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load saved vectorizer and trained model directly from pickle files
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Page Setup
st.set_page_config(
    page_title="AI Email Spam Detector",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS with fixed alignment for confidence text
st.markdown("""
<style>
.stApp {
    background-color: #061611;
    color: #e2e8f0;
}
.stTextArea textarea {
    background-color: #061611 !important;
    color: #f3f4f6 !important;
    border: 1px solid #11382c !important;
    border-radius: 10px;
    padding: 12px;
    font-size: 14px;
}
div.stButton > button {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 6px 12px;
    font-weight: 600;
    font-size: 13px;
    box-shadow: 0 2px 10px rgba(16, 185, 129, 0.2);
    width: 100%;
    white-space: nowrap;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #059669 0%, #047857 100%);
    color: #ffffff;
}
.custom-card {
    background-color: #081d17;
    border: 1px solid #11382c;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 20px;
    width: 100%;
    box-sizing: border-box;
}
.sub-card {
    background-color: #061611;
    border: 1px solid #11382c;
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 12px;
}
.spam-main-banner {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.25) 0%, rgba(185, 28, 28, 0.2) 100%);
    border: 1px solid #ef4444;
    border-radius: 12px;
    padding: 14px 16px;
    color: #fca5a5;
    margin-bottom: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.safe-main-banner {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.25) 0%, rgba(4, 120, 87, 0.2) 100%);
    border: 1px solid #10b981;
    border-radius: 12px;
    padding: 14px 16px;
    color: #6ee7b7;
    margin-bottom: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.rec-spam-box {
    background-color: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.4);
    border-radius: 10px;
    padding: 12px 14px;
    color: #fca5a5;
}
.rec-safe-box {
    background-color: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.4);
    border-radius: 10px;
    padding: 12px 14px;
    color: #6ee7b7;
}
.badge-high {
    background-color: rgba(239, 68, 68, 0.2);
    color: #f87171;
    border: 1px solid #ef4444;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    float: right;
}
.badge-safe {
    background-color: rgba(16, 185, 129, 0.2);
    color: #34d399;
    border: 1px solid #10b981;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    float: right;
}
</style>
""", unsafe_allow_html=True)

# Session States
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"
if "history" not in st.session_state:
    st.session_state.history = []
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# Top Header Layout
col_logo, col_status, col_nav = st.columns([2.2, 1.3, 2.5])

with col_logo:
    st.markdown("<h3 style='margin:0; font-size:17px; color:#ffffff; padding-top:6px;'>❖ AI Spam Detector</h3>", unsafe_allow_html=True)

with col_status:
    st.markdown("<p style='margin:0; text-align:right; font-size:12px; color:#34d399; padding-top:10px;'>🟢 Online</p>", unsafe_allow_html=True)

with col_nav:
    n1, n2, n3 = st.columns(3)
    with n1:
        if st.button("Home"):
            st.session_state.page = "Dashboard"
    with n2:
        if st.button("Logs"):
            st.session_state.page = "History"
    with n3:
        if st.button("⚙"):
            st.session_state.page = "Settings"

st.markdown("<div style='margin-top: -5px;'><hr style='border-color: #0d2920;'></div>", unsafe_allow_html=True)

# --- DASHBOARD PAGE ---
if st.session_state.page == "Dashboard":
    st.markdown("<h1 style='font-size: 22px; margin-bottom: 0; color: #ffffff;'>Detection Console</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; font-size: 12px; margin-top: 2px;'>Paste any email or message to check spam predictions.</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        char_count = len(st.session_state.input_text)
        st.markdown(f"""
        <div class='custom-card'>
            <p style='font-weight:600; margin-top:0; margin-bottom:2px; font-size:14px; color:#ffffff;'>
                ✏ &nbsp; Input Section 
                <span style='float:right; font-size:11px; color:#64748b; background:#061611; padding:2px 8px; border-radius:6px; border:1px solid #11382c;'>{char_count} chars</span>
            </p>
            <p style='font-size:11px; color:#64748b; margin-top:0; margin-bottom:10px;'>Paste content to analyze below</p>
        """, unsafe_allow_html=True)
        
        user_input = st.text_area("", value=st.session_state.input_text, height=240, placeholder="Paste your message here...")
        st.session_state.input_text = user_input
        
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_clicked = st.button("✦ Analyze Message")
        st.markdown("<br>", unsafe_allow_html=True)
        
        b1, b2, b3 = st.columns([1, 1.2, 1.2])
        with b1:
            if st.button("🗑 Clear"):
                st.session_state.input_text = ""
                st.rerun()
        with b2:
            if st.button("📄 Ham"):
                st.session_state.input_text = "Hey Alina, are we still meeting tomorrow at 3 PM to review the final project report?"
                st.rerun()
        with b3:
            if st.button("📄 Spam"):
                st.session_state.input_text = "Flexible Online Work Opportunities – Earn from Home! Message now to get started and explore online work options."
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        if analyze_clicked:
            if not st.session_state.input_text.strip():
                st.warning("Please enter or load some text to analyze.")
            else:
                processed = transform_text(st.session_state.input_text)
                vector_input = tfidf.transform([processed])
                
                # Direct Model Prediction without Guardrails
                model_result = model.predict(vector_input)[0]
                result = model_result
                
                conf = 95.0
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(vector_input)[0]
                    conf = round(float(max(probs)) * 100, 1)

                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                result_label = "Spam / Risk" if result == 1 else "Safe / Legitimate"
                
                st.session_state.history.insert(0, {
                    "time": timestamp,
                    "text": st.session_state.input_text[:40] + "...",
                    "result": result_label,
                    "conf": f"{conf}%"
                })

                if result == 1:
                    st.markdown(f"""
                    <div class='custom-card'>
                        <p style='font-weight:600; margin-top:0; margin-bottom:10px; font-size:14px; color:#ffffff;'>
                            🛡 &nbsp; Result Display <span class='badge-high'>High Risk</span>
                        </p>
                        <div class='spam-main-banner'>
                            <div>
                                <span style='font-size:15px; font-weight:800; display:block;'>🚨 SPAM DETECTED</span>
                                <span style='font-size:10px; color:#fca5a5;'>Classified as spam by model</span>
                            </div>
                            <div style='text-align: right;'>
                                <span style='font-size:10px; color:#fca5a5;'>Confidence</span><br>
                                <span style='font-size:18px; font-weight:800; color:#f87171;'>{conf}%</span>
                            </div>
                        </div>
                        <div class='sub-card'>
                            <div style='display: flex; justify-content: space-between; font-size:11px; color:#94a3b8; font-weight:600; margin-bottom:6px;'>
                                <span>Confidence Meter</span>
                                <span style='color:#f87171;'>{conf}% Spam</span>
                            </div>
                            <div style='background:#11382c; border-radius:10px; height:6px; width:100%; overflow:hidden;'>
                                <div style='background:linear-gradient(90deg, #ef4444, #f87171); height:100%; width:{conf}%;'></div>
                            </div>
                        </div>
                        <div class='rec-spam-box'>
                            <p style='font-size:11px; font-weight:700; margin:0 0 2px 0;'>🕒 RECOMMENDATION</p>
                            <p style='font-size:11px; margin:0; color:#fca5a5;'>This message is classified as spam. Exercise caution.</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='custom-card'>
                        <p style='font-weight:600; margin-top:0; margin-bottom:10px; font-size:14px; color:#ffffff;'>
                            🛡 &nbsp; Result Display <span class='badge-safe'>Verified Safe</span>
                        </p>
                        <div class='safe-main-banner'>
                            <div>
                                <span style='font-size:15px; font-weight:800; display:block;'>✅ NOT SPAM</span>
                                <span style='font-size:10px; color:#6ee7b7;'>Classified as safe by model</span>
                            </div>
                            <div style='text-align: right;'>
                                <span style='font-size:10px; color:#6ee7b7;'>Confidence</span><br>
                                <span style='font-size:18px; font-weight:800; color:#34d399;'>{conf}%</span>
                            </div>
                        </div>
                        <div class='sub-card'>
                            <div style='display: flex; justify-content: space-between; font-size:11px; color:#94a3b8; font-weight:600; margin-bottom:6px;'>
                                <span>Confidence Meter</span>
                                <span style='color:#34d399;'>{conf}% Safe</span>
                            </div>
                            <div style='background:#11382c; border-radius:10px; height:6px; width:100%; overflow:hidden;'>
                                <div style='background:linear-gradient(90deg, #10b981, #34d399); height:100%; width:{conf}%;'></div>
                            </div>
                        </div>
                        <div class='rec-safe-box'>
                            <p style='font-size:11px; font-weight:700; margin:0 0 2px 0;'>🕒 RECOMMENDATION</p>
                            <p style='font-size:11px; margin:0; color:#6ee7b7;'>This message appears safe and legitimate.</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='custom-card' style='min-height: 380px;'>
                <p style='font-weight:600; margin-top:0; margin-bottom:15px; font-size:14px; color:#ffffff;'>🛡 &nbsp; Result Display</p>
                <div style='text-align: center; color: #64748b; padding: 90px 0;'>
                    <div style='font-size: 30px; margin-bottom: 8px;'>✉</div>
                    <h3 style='color: #94a3b8; font-size: 15px; margin: 0;'>Awaiting analysis</h3>
                    <p style='font-size: 11px; margin-top: 4px; color: #475569;'>Paste content and hit Analyze to see live prediction.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

# --- HISTORY PAGE ---
elif st.session_state.page == "History":
    st.markdown("<h1 style='font-size: 22px; color:#ffffff;'>Analysis Logs</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; font-size: 12px;'>History of scanned messages during session.</p>", unsafe_allow_html=True)
    
    if not st.session_state.history:
        st.info("No logs found yet. Scan a message from the Home tab.")
    else:
        if st.button("Clear Logs"):
            st.session_state.history = []
            st.rerun()
        
        for idx, item in enumerate(st.session_state.history):
            st.markdown(f"""
            <div class='custom-card' style='padding: 12px; margin-bottom: 10px;'>
                <span style='color: #34d399; font-weight: bold;'>{item['result']}</span> &nbsp;|&nbsp; <span style='color: #64748b; font-size: 11px;'>{item['time']}</span>
                <p style='margin-top: 6px; color: #f3f4f6; font-size: 12px;'><b>Snippet:</b> {item['text']}</p>
                <p style='font-size: 11px; color: #64748b; margin:0;'>Score: {item['conf']}</p>
            </div>
            """, unsafe_allow_html=True)

# --- SETTINGS PAGE ---
elif st.session_state.page == "Settings":
    st.markdown("<h1 style='font-size: 22px; color:#ffffff;'>System Settings</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; font-size: 12px;'>Model configuration parameters.</p>", unsafe_allow_html=True)
    
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    st.subheader("Engine Info")
    st.write("Active Model: **Trained Classifier (Pure ML)**")
    st.write("Status: **Synchronized**")
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<br><hr style='border-color: #0d2920;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #475569; font-size: 11px;'>Responsive Spam Detection Dashboard.</p>", unsafe_allow_html=True)
