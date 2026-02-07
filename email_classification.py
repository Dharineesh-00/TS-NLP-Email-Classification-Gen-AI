"""
Email Classification Project
Comparing Baseline (TF-IDF) vs GenAI (Sentence Transformers) Models
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sentence_transformers import SentenceTransformer

# ============================================================================
# 1. LOAD DATA FROM CSV
# ============================================================================

print("="*80)
print("EMAIL CLASSIFICATION PROJECT")
print("="*80)

# Load dataset
print("\nLoading dataset from dataset.csv...")
df = pd.read_csv('dataset.csv')

# Extract emails and labels
emails = df['text'].tolist()
labels = df['label'].tolist()

print(f"✓ Dataset loaded successfully")
print(f"\nTotal emails: {len(emails)}")
print(f"Labels: {set(labels)}")

# Display label distribution
label_counts = df['label'].value_counts()
print(f"\nLabel Distribution:")
for label, count in label_counts.items():
    print(f"  {label:<12}: {count} emails ({count/len(emails)*100:.1f}%)")

# ============================================================================
# 2. SPLIT DATA
# ============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    emails, labels, test_size=0.3, random_state=42, stratify=labels
)

print(f"\nTraining set: {len(X_train)} emails")
print(f"Test set: {len(X_test)} emails")

# ============================================================================
# 3. BASELINE MODEL: TF-IDF + Logistic Regression
# ============================================================================

print("\n" + "="*80)
print("BASELINE MODEL: TF-IDF + Logistic Regression")
print("="*80)

# Vectorize text data
tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train baseline model
baseline_model = LogisticRegression(max_iter=1000, random_state=42)
baseline_model.fit(X_train_tfidf, y_train)

# Predictions
y_pred_baseline = baseline_model.predict(X_test_tfidf)

# Evaluation
baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
baseline_f1 = f1_score(y_test, y_pred_baseline, average='weighted')

print(f"\n✓ Baseline Model Trained")
print(f"  Accuracy:  {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
print(f"  F1-Score:  {baseline_f1:.4f}")

# ============================================================================
# 4. GenAI MODEL: Sentence Transformers + Logistic Regression
# ============================================================================

print("\n" + "="*80)
print("GenAI MODEL: Sentence Transformers + Logistic Regression")
print("="*80)

print("\nLoading sentence transformer model (all-MiniLM-L6-v2)...")
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
print("Generating embeddings for training data...")
X_train_embeddings = sentence_model.encode(X_train, show_progress_bar=False)
print("Generating embeddings for test data...")
X_test_embeddings = sentence_model.encode(X_test, show_progress_bar=False)

# Train GenAI model
genai_model = LogisticRegression(max_iter=1000, random_state=42)
genai_model.fit(X_train_embeddings, y_train)

# Predictions
y_pred_genai = genai_model.predict(X_test_embeddings)

# Evaluation
genai_accuracy = accuracy_score(y_test, y_pred_genai)
genai_f1 = f1_score(y_test, y_pred_genai, average='weighted')

print(f"\n✓ GenAI Model Trained")
print(f"  Accuracy:  {genai_accuracy:.4f} ({genai_accuracy*100:.2f}%)")
print(f"  F1-Score:  {genai_f1:.4f}")

# ============================================================================
# 5. COMPARISON
# ============================================================================

print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)
print(f"\n{'Model':<25} {'Accuracy':<15} {'F1-Score':<15}")
print("-" * 55)
print(f"{'Baseline (TF-IDF)':<25} {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)   {baseline_f1:.4f}")
print(f"{'GenAI (Transformers)':<25} {genai_accuracy:.4f} ({genai_accuracy*100:.2f}%)   {genai_f1:.4f}")

improvement = ((genai_accuracy - baseline_accuracy) / baseline_accuracy) * 100
print(f"\nImprovement: {improvement:+.2f}%")

# ============================================================================
# 6. REAL-TIME PREDICTION FUNCTION
# ============================================================================

def predict_email_class(email_text):
    """
    Predicts the class of a new email using the GenAI model.
    
    Args:
        email_text (str): The email text to classify
        
    Returns:
        str: Predicted class label
    """
    # Generate embedding for the new email
    email_embedding = sentence_model.encode([email_text])
    
    # Predict using GenAI model
    prediction = genai_model.predict(email_embedding)
    
    # Get prediction probability
    probabilities = genai_model.predict_proba(email_embedding)[0]
    classes = genai_model.classes_
    
    return prediction[0], dict(zip(classes, probabilities))

# ============================================================================
# 7. TEST WITH EXAMPLE EMAIL
# ============================================================================

print("\n" + "="*80)
print("REAL-TIME PREDICTION TEST")
print("="*80)

test_email = "Congratulations! You have won a $1000 gift card. Click here."

print(f"\nTest Email: \"{test_email}\"")
print("\nRunning prediction...")

predicted_class, probabilities = predict_email_class(test_email)

print(f"\n✓ Prediction Complete!")
print(f"\nPredicted Class: {predicted_class}")
print(f"\nClass Probabilities:")
for cls, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
    print(f"  {cls:<12}: {prob:.4f} ({prob*100:.2f}%)")

# ============================================================================
# ADDITIONAL TEST EXAMPLES
# ============================================================================

print("\n" + "="*80)
print("ADDITIONAL TEST EXAMPLES")
print("="*80)

additional_tests = [
    "Can you help me reset my password? I can't log in.",
    "Hi friend! Want to grab lunch tomorrow?",
    "50% off on all products today only! Don't miss out!"
]

for i, test in enumerate(additional_tests, 1):
    pred_class, probs = predict_email_class(test)
    print(f"\n{i}. \"{test}\"")
    print(f"   → Predicted: {pred_class} (confidence: {max(probs.values())*100:.2f}%)")

print("\n" + "="*80)
print("PROJECT COMPLETED!")
print("="*80)
