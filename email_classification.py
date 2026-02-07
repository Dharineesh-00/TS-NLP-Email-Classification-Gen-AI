"""
Email Classification Project
Comparing Baseline (TF-IDF) vs GenAI (Sentence Transformers) Models
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sentence_transformers import SentenceTransformer

# ============================================================================
# 1. DUMMY DATA: 20 Mock Emails
# ============================================================================

emails = [
    # Spam (5 emails)
    "Congratulations! You've won $1 million. Click here to claim your prize now!",
    "URGENT: Your account will be closed. Verify your password immediately.",
    "Get rich quick! Make $5000 per week working from home. No experience needed!",
    "Free Viagra! Buy now and get 50% discount. Limited time offer!",
    "You have inherited $10 million from a Nigerian prince. Send us your bank details.",
    
    # Promotion (5 emails)
    "Exclusive 30% OFF on all items this weekend. Shop now at our store!",
    "New arrivals! Check out our latest collection. Free shipping on orders over $50.",
    "Flash sale: Up to 70% discount on electronics. Hurry, limited stock!",
    "Subscribe to our premium membership and get access to exclusive content.",
    "Holiday special: Buy one get one free on selected items. Visit our website today!",
    
    # Support (5 emails)
    "Your ticket #12345 has been updated. Our team is working on your issue.",
    "Thank you for contacting our support team. We will respond within 24 hours.",
    "Your subscription will expire in 7 days. Please renew to continue using our service.",
    "We have received your refund request. It will be processed in 5-7 business days.",
    "Your password has been successfully changed. If this wasn't you, contact support immediately.",
    
    # Personal (5 emails)
    "Hey! Are you free for coffee this Saturday? Let me know!",
    "Thanks for the birthday gift! I really loved it. See you soon!",
    "Can you send me the presentation slides from yesterday's meeting? Thanks!",
    "Mom called and asked if you're coming for Thanksgiving dinner.",
    "I found that book you were looking for. Do you still want to borrow it?"
]

labels = [
    'Spam', 'Spam', 'Spam', 'Spam', 'Spam',
    'Promotion', 'Promotion', 'Promotion', 'Promotion', 'Promotion',
    'Support', 'Support', 'Support', 'Support', 'Support',
    'Personal', 'Personal', 'Personal', 'Personal', 'Personal'
]

print("="*80)
print("EMAIL CLASSIFICATION PROJECT")
print("="*80)
print(f"\nTotal emails: {len(emails)}")
print(f"Labels: {set(labels)}")

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
