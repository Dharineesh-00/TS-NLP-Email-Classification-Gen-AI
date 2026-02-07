"# Email Classification Using GenAI

Automated email classification system that compares traditional TF-IDF with GenAI-powered sentence transformers.

## 📧 What it does

Classifies emails into 4 categories:
- **Spam** - Scams and unwanted messages
- **Promotion** - Marketing and offers
- **Support** - Customer service inquiries
- **Personal** - Private conversations

## 🚀 Quick Start

### Install dependencies
```bash
pip install scikit-learn sentence-transformers numpy
```

### Run the project
```bash
python email_classification.py
```

## 🧠 Models Compared

1. **Baseline**: TF-IDF + Logistic Regression
2. **GenAI**: Sentence Transformers (all-MiniLM-L6-v2) + Logistic Regression

## 📊 Features

- 20 mock emails for demonstration
- Train/test split with evaluation metrics
- Real-time prediction function
- Confidence scores for each prediction

## 📖 Documentation

See [PROJECT_REPORT.md](PROJECT_REPORT.md) for detailed technical documentation.

## 🎯 Example Usage

```python
predict_email_class("Congratulations! You have won a $1000 gift card.")
# Output: Spam (64.21% confidence)
```

## 📦 Project Structure

```
.
├── email_classification.py   # Main implementation
├── PROJECT_REPORT.md         # Technical documentation
└── README.md                 # This file
```

## 🔧 Tech Stack

- Python 3.x
- scikit-learn
- sentence-transformers
- NumPy" 
