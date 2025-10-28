# scripts/plot_confusion_matrix.py
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load model and data
clf = joblib.load(r"C:\Users\ciope\Desktop\Licenta\prototypes\Train_times_ML\models\decision_tree_model.pkl")
df = pd.read_csv(r"C:\Users\ciope\Desktop\Licenta\prototypes\Train_times_ML\data\traffic_timing_balanced.csv")

X = df[['cars_south', 'cars_north', 'cars_west', 'cars_east']]
y = df['green_time_bucket']
y_pred = clf.predict(X)

# Confusion matrix
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Short", "Medium", "Long"], yticklabels=["Short", "Medium", "Long"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

output_path = r"C:\Users\ciope\Desktop\Licenta\prototypes\Train_times_ML\outputs\confusion_matrix.png"
plt.savefig(output_path)
print(f"Saved to: {output_path}")
