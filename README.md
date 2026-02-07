"# Email Classification Using GenAI

Automated email classification system that compares traditional TF-IDF with GenAI-powered sentence transformers.

## 📧 What it does

Classifies emails into 4 categories:
- **Spam** - Scams and unwanted messages
- **Promotions** - Marketing and offers
- **Support** - Customer service inquiries
- **Personal** - Private conversations

## 📊 Dataset

The project uses **dataset.csv** containing 800+ real-world email samples with balanced distribution across all categories.

## 🚀 Quick Start

### Install dependencies
```bash
pip install scikit-learn sentence-transformers numpy pandas
```

### Run the project
```bash
python email_classification.py
```

## 🧠 Models Compared

1. **Baseline**: TF-IDF + Logistic Regression
2. **GenAI**: Sentence Transformers (all-MiniLM-L6-v2) + Logistic Regression

## 📈 Features

- 800+ email samples from CSV dataset
- Train/test split with stratified sampling
- Comprehensive evaluation metrics (Accuracy, F1-Score)
- Real-time prediction function with confidence scores
- Detailed performance comparison

## 📖 Documentation

See [PROJECT_REPORT.md](PROJECT_REPORT.md) for detailed technical documentation.

## 🎯 Example Usage

```python
predict_email_class("Congratulations! You have won a $1000 gift card.")
# Output: Spam (high confidence)
```

## 📦 Project Structure

```
.
├── email_classification.py   # Main implementation
├── dataset.csv               # Email dataset (800+ samples)
├── PROJECT_REPORT.md         # Technical documentation
└── README.md                 # This file
```

## 🔧 Tech Stack

- Python 3.x
- scikit-learn
- sentence-transformers
- pandas
- NumPy" 
