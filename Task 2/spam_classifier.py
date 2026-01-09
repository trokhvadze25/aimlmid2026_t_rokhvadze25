import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# =========================
# CONFIGURATION
# =========================
DATA_PATH = "data/t_rokhvadze25_69428.csv"
TARGET_COLUMN = "is_spam"

# =========================
# LOAD DATA
# =========================
data = pd.read_csv(DATA_PATH)

print("Dataset columns:")
print(data.columns)

# Separate features and target
X = data.drop(columns=[TARGET_COLUMN])
y = data[TARGET_COLUMN]

# =========================
# TRAIN / TEST SPLIT (70/30)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# =========================
# LOGISTIC REGRESSION MODEL
# =========================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# =========================
# MODEL COEFFICIENTS
# =========================
print("\nModel coefficients:")
for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"{feature}: {coef:.4f}")

print("\nModel intercept:")
print(model.intercept_[0])

# =========================
# VALIDATION
# =========================
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nAccuracy:")
print(accuracy)

# =========================
# EMAIL TEXT CLASSIFICATION
# =========================
def extract_features(text):
    """
    Extract features matching the dataset:
    - words
    - links
    - capital_words
    - spam_word_count
    """
    text_lower = text.lower()
    words = len(text.split())
    links = text_lower.count("http")
    capital_words = sum(1 for w in text.split() if w.isupper())
    spam_word_count = (
        text_lower.count("free")
        + text_lower.count("win")
        + text_lower.count("urgent")
        + text_lower.count("money")
    )

    return [words, links, capital_words, spam_word_count]

def classify_email(text):
    features = extract_features(text)
    prediction = model.predict([features])[0]
    return "SPAM" if prediction == 1 else "LEGITIMATE"

# =========================
# TEST EMAILS
# =========================
spam_email = "FREE MONEY URGENT WIN NOW http://spam.link"
legit_email = "Hello team, the meeting agenda is attached."

print("\nSpam email prediction:", classify_email(spam_email))
print("Legitimate email prediction:", classify_email(legit_email))