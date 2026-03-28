# sms-spam-classifier
# 📩 SMS Spam Classifier

## 🚀 Overview
A Machine Learning-based SMS Spam Classifier that classifies messages as **Spam** or **Ham** using NLP techniques.

## 📊 Dataset
- 5,572 SMS messages
- 13.4% spam, 86.6% ham
- Source: Kaggle SMS Spam Collection Dataset

## ⚙️ Techniques Used
- Text Preprocessing (tokenization, stopword removal, stemming)
- TF-IDF Vectorization (3000 features)
- Multinomial Naive Bayes

## 📈 Results
- Accuracy: **97.1%**
- Precision: **100% (Zero False Positives)**

## 🧠 Models Compared
- Naive Bayes
- SVM
- Random Forest
- XGBoost

## ▶️ Run Locally
```bash
pip install -r requirements.txt
python main.py
