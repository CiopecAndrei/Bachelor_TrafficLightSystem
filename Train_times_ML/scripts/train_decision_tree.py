# train_decision_tree.py

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib
import os

# === Paths ===
BASE_DIR = r"C:\Users\ciope\Desktop\Licenta\prototypes\Train_times_ML"
DATA_PATH = os.path.join(BASE_DIR, "data", "traffic_timing_balanced.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "decision_tree_model.pkl")
PLOT_PATH = os.path.join(BASE_DIR, "outputs", "decision_tree_plot.png")

# === Load Dataset ===
df = pd.read_csv(DATA_PATH)

# === Features and Target ===
X = df[['cars_south', 'cars_north', 'cars_west', 'cars_east']]
y = df['green_time_bucket']

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# === Train Model ===
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# === Evaluate ===
y_pred = clf.predict(X_test)
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# === Cross-validation ===
scores = cross_val_score(clf, X, y, cv=5)
print("=== Cross-validation ===")
print("Scores:", scores)
print("Mean accuracy: {:.3f} ± {:.3f}".format(scores.mean(), scores.std()))

# === Save Model ===
joblib.dump(clf, MODEL_PATH)
print(f"\nModel saved to: {MODEL_PATH}")

# === Save Tree Plot ===
plt.figure(figsize=(10, 6))
plot_tree(clf, feature_names=X.columns, class_names=["Short", "Medium", "Long"], filled=True)
plt.title("Decision Tree for Green Time Buckets")
plt.savefig(PLOT_PATH)
print(f"Tree plot saved to: {PLOT_PATH}")
