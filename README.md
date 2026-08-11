# Alina Zahra - Machine Learning - Month 1 - Task 1: Email Spam Detector

> An end-to-end Machine Learning web application built for real-time email spam classification and secure message screening. Developed as part of the Machine Learning Internship Program at **Arch Technologies**.

---

## 🔗 Live Demo
* **Live Application:** https://email-spam-detector-web.streamlit.app/

---

## 📌 Problem Statement & Motivation
With the rapid growth of digital communication, filtering unsolicited email spam and phishing content is crucial for secure messaging. This project delivers an end-to-end supervised machine learning pipeline trained directly on email datasets (`email.csv`) to automatically analyze and classify incoming text messages into spam or safe categories in real-time.

---

## ✨ Key Features & Architecture
* **Trained ML Classification Engine:** Powered by a supervised machine learning classifier and a TF-IDF vectorizer trained on real email text datasets.
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
4. **Model Classification:** The trained machine learning model evaluates the feature vector to predict spam or safe status along with probability confidence.
5. **Result Render:** The dashboard outputs the final classification badge, confidence meter, and recommendation.

---

## ⚙️ Installation & Local Setup Guide

Follow these steps to run the project locally on your machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/AlinaZahraHub/email-spam-detector.git](https://github.com/AlinaZahraHub/email-spam-detector.git)
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
├── email-spam-detection.ipynb  # Jupyter Notebook containing EDA and model training
├── model.pkl                   # Trained machine learning classification model
├── vectorizer.pkl              # TF-IDF vectorizer for feature extraction
├── email.csv                   # Dataset used for training and testing
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation

```

---

## 👩‍💻 Author & Acknowledgement

* **Name:** Alina Zahra
* **Internship Program:** Machine Learning Internship & Training Program (August – September 2026)
* **Organization:** Arch Technologies
* **GitHub:** https://github.com/AlinaZahraHub
* **LinkedIn:** https://www.linkedin.com/in/alina-zahra12/

```

```
