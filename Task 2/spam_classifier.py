
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

DATA_PATH = "data/t_rokhvadze25_69428.csv"
TARGET_COLUMN = "class"  # change if your CSV uses a different name

data = pd.read_csv(DATA_PATH)

X = data.drop(columns=[TARGET_COLUMN])
y = data[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Model coefficients:")
for name, coef in zip(X.columns, model.coef_[0]):
    print(f"{name}: {coef:.4f}")
print("Intercept:", model.intercept_[0])

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print("Confusion Matrix:\n", cm)
print("Accuracy:", acc)

def extract_features(text):
    text = text.lower()
    return [
        text.count("free"),
        text.count("win"),
        text.count("http"),
        len(text.split())
    ]

def classify_email(text):
    features = extract_features(text)
    prediction = model.predict([features])[0]
    return "SPAM" if prediction == 1 else "LEGITIMATE"

print(classify_email("You won free money! Click http://spam.link now"))
print(classify_email("Hello team, our meeting is scheduled for tomorrow."))
