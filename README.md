# Alina Zahra - Machine Learning - Month 1 - Task 1: Email Spam Detector

> An end-to-end Machine Learning web application built for real-time email spam classification, phishing detection, and modern online scam prevention. Developed as part of the Machine Learning Internship Program at **Arch Technologies**.

---

## 🔗 Live Demo
* **Live Application:** [View Deployed Streamlit App](#) *(Yahan apna Streamlit Cloud URL dal dein)*

---

## 📌 Problem Statement & Motivation
With the rapid increase in digital communication, unsolicited email spam, fraudulent job traps, and phishing scams have become major security threats. Standard machine learning classifiers often miss newly trending scams (such as fake work-from-home offers or financial phishing). This project bridges that gap by combining a supervised machine learning classification model with robust, rule-based keyword security guardrails to protect users from malicious emails in real-time.

---

## ✨ Key Features & Architecture
* **Dual-Layer Detection Engine:** Integrates a trained Scikit-Learn machine learning classifier with comprehensive keyword security guardrails to catch both dataset patterns and trending email scams instantly.
* **Interactive Responsive Dashboard:** Designed with a custom glassmorphism-inspired dark UI using Streamlit and CSS.
* **Live Confidence Scoring:** Calculates and visualizes dynamic probability confidence scores for every scanned email.
* **Session Logs & History:** Tracks scanned emails during the session for quick auditing.
* **Quick Presets:** One-click sample loaders for testing Ham (safe) and Spam (risk) email inputs.

---

## 🛠️ Tech Stack & Tools
* **Programming Language:** Python
* **Machine Learning & NLP:** Scikit-Learn, NLTK (Tokenization, Stopwords Removal, Porter Stemming), Pickle
* **Frontend & UI:** Streamlit, Custom CSS
* **Deployment & Version Control:** Streamlit Community Cloud, GitHub

---

## 🔄 Project Architecture & Workflow
1. **User Input:** User pastes an email text into the Streamlit web console.
2. **Text Preprocessing:** The text undergoes lowercase conversion, tokenization, alphanumeric filtering, stopword removal, and stemming via NLTK.
3. **Feature Extraction:** Processed text is transformed into numerical feature vectors using the saved TF-IDF vectorizer.
4. **Classification & Guardrails:** The machine learning model predicts the probability score, while the custom keyword guardrail scans for latest trending phishing or email scam patterns.
5. **Result Render:** The dashboard outputs the final classification badge, confidence meter, and security recommendation.

---

## ⚙️ Installation & Local Setup Guide

Follow these steps to run the project locally on your machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/email-spam-detector.git](https://github.com/YOUR_USERNAME/email-spam-detector.git)
   cd email-spam-detector

```

2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Run the Streamlit application:**
```bash
streamlit run app.py

```



---

## 📂 Project Structure

```text
email-spam-detector/
│
├── app.py                      # Main Streamlit application interface and logic
├── email-spam-detection.ipynb  # Jupyter Notebook containing exploratory data analysis and model training
├── model.pkl                   # Trained machine learning classification model
├── vectorizer.pkl              # TF-IDF vectorizer for feature extraction
├── spam.csv                    # Dataset used for training and testing
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation

```

---

## 👩‍💻 Author & Acknowledgement

* **Name:** Alina Zahra
* **Internship Program:** Machine Learning Internship & Training Program (August – September 2026)
* **Organization:** Arch Technologies [@Arch Technologies]

```

```
