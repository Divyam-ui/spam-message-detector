# 📩 Spam Message Detector

A beginner-friendly NLP machine-learning project that classifies SMS messages as **spam** or **ham** using **TF-IDF + Multinomial Naive Bayes**.

## Features

- UCI SMS Spam Collection dataset
- Duplicate/missing-value cleanup
- Stratified train/test split
- TF-IDF feature extraction
- Unigram + bigram features
- Multinomial Naive Bayes classifier
- Accuracy, precision, recall, F1-score and confusion matrix
- Saved reusable model with Joblib
- Streamlit web interface
- Prediction confidence display

## Project structure

```text
spam-message-detector/
├── data/
├── models/
├── src/
│   └── train.py
├── app.py
├── requirements.txt
└── README.md
```

## 1. Create a virtual environment (recommended)

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Train the model

```bash
python src/train.py
```

The training script automatically downloads the UCI SMS Spam Collection into `data/`, trains the model, prints evaluation metrics, and saves the trained pipeline to `models/spam_detector.joblib`.

## 4. Run the application

```bash
streamlit run app.py
```

A browser window should open with the Spam Message Detector.

## How it works

```text
SMS message
    ↓
TF-IDF vectorization
    ↓
Numerical sparse features
    ↓
Multinomial Naive Bayes
    ↓
Spam / Ham prediction
```

## Important ML design choice

The model uses a Scikit-learn `Pipeline`, so TF-IDF is fitted only on the training data and the same fitted transformation is applied to test/new messages. This helps prevent data leakage.

## Dataset

UCI SMS Spam Collection: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

The dataset contains labeled SMS messages with `ham` and `spam` classes.

## Possible future improvements

- Compare Naive Bayes with Logistic Regression and Linear SVM
- Add a confusion-matrix visualization
- Experiment with preprocessing choices
- Add a message-history section
- Deploy the Streamlit app
- Test the model on newer real-world SMS data
