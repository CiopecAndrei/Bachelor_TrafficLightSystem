# scripts/plot_crossval_accuracy.py
import joblib
import pandas as pd
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

clf = joblib.load(r"C:\Users\ciope\Desktop\Licenta\prototypes\Train_times_ML\models\decision_tree_model.pkl")
df = pd.read_csv(r"C:\Users\ciope\Desktop\Licenta\prototypes\Train_times_ML\data\traffic_timing_balanced.csv")
X = df[['cars_south', 'cars_north', 'cars_west', 'cars_east']]
y = df['green_time_bucket']

scores = cross_val_score(clf, X, y, cv=5)

plt.figure(figsize=(6, 4))
plt.plot(range(1, 6), scores, marker='o', color='purple')
plt.ylim(0.7, 1.05)
plt.title("Cross-Validation Scores (5-Fold)")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.grid(True)

output_path = r"C:\Users\ciope\Desktop\Licenta\prototypes\Train_times_ML\outputs\crossval_scores.png"
plt.savefig(output_path)
print(f"Saved to: {output_path}")
