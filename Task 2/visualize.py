import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

DATA_PATH = "data/t_rokhvadze25_69428.csv"
TARGET_COLUMN = "is_spam"

data = pd.read_csv(DATA_PATH)

X = data.drop(columns=[TARGET_COLUMN])
y = data[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# =========================
# VISUALIZATION 1: CLASS DISTRIBUTION
# =========================
plt.figure()
y.value_counts().plot(kind="bar")
plt.title("Class Distribution (Spam vs Legitimate)")
plt.xlabel("Class (0 = Legitimate, 1 = Spam)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("images/class_distribution.png")
plt.close()

# =========================
# VISUALIZATION 2: CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Legitimate", "Spam"],
    yticklabels=["Legitimate", "Spam"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")
plt.tight_layout()
plt.savefig("images/confusion_matrix.png")
plt.close()