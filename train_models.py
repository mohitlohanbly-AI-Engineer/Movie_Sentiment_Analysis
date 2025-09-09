# train_models.py

import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re

# ===============================
# 1. Load dataset
# ===============================
print("📂 Loading dataset...")
df = pd.read_csv("data/IMDB.csv")

# ===============================
# 2. Fast text cleaning
# ===============================
def fast_clean(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)  # keep only letters & spaces
    return re.sub(r"\s+", " ", text).strip()

print("🧹 Cleaning text...")
df["cleaned_review"] = df["review"].astype(str).apply(fast_clean)

# Optional: speed up training on large dataset
# df = df.sample(n=10000, random_state=42)

# ===============================
# 3. Train/test split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    df["cleaned_review"],
    df["sentiment"],
    test_size=0.2,
    random_state=42,
    stratify=df["sentiment"]
)

# ===============================
# 4. TF-IDF Vectorizer
# ===============================
print("🔠 Vectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ===============================
# 5. Train models
# ===============================
print("🤖 Training Logistic Regression...")
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_vec, y_train)

print("🤖 Training SVM...")
svm = LinearSVC()
svm.fit(X_train_vec, y_train)

# ===============================
# 6. Evaluate models
# ===============================
print("\n📊 Evaluation Results:")

logreg_preds = logreg.predict(X_test_vec)
svm_preds = svm.predict(X_test_vec)

print("\nLogistic Regression Report:")
print(classification_report(y_test, logreg_preds))

print("\nSVM Report:")
print(classification_report(y_test, svm_preds))

print("\nAccuracy Scores:")
print("LogReg:", accuracy_score(y_test, logreg_preds))
print("SVM   :", accuracy_score(y_test, svm_preds))

# Confusion matrix visualization
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_test, logreg_preds), annot=True, fmt="d", cmap="Blues")
plt.title("Logistic Regression Confusion Matrix")

plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(y_test, svm_preds), annot=True, fmt="d", cmap="Greens")
plt.title("SVM Confusion Matrix")

plt.tight_layout()
plt.show()

# ===============================
# 7. Save models & vectorizer
# ===============================
print("💾 Saving models...")
joblib.dump(logreg, "models/logreg_model.pkl")
joblib.dump(svm, "models/svm_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\n✅ Training complete! Models and vectorizer saved in 'models/'")
