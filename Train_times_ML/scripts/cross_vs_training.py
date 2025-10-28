import joblib
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Încarcă modelul și datele
clf = joblib.load(r"C:\Users\ciope\Desktop\Licenta\prototypes\Train_times_ML\models\decision_tree_model.pkl")
df = pd.read_csv(r"C:\Users\ciope\Desktop\Licenta\prototypes\Train_times_ML\data\traffic_timing_balanced.csv")
X = df[['cars_south', 'cars_north', 'cars_west', 'cars_east']]
y = df['green_time_bucket']

# Calculează scorurile de cross-validation
scores = cross_val_score(clf, X, y, cv=5)

# Calculează acuratețea pe datele de antrenament complete
train_preds = clf.predict(X)
train_score = accuracy_score(y, train_preds)

# Plotează comparativ
plt.figure(figsize=(6, 4))
plt.plot(range(1, 6), scores, marker='o', color='purple', label='Cross-Validation Score')
plt.axhline(train_score, color='orange', linestyle='--', label='Training Score')
plt.ylim(0.7, 1.05)
plt.title("Cross-Validation vs Training Accuracy")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

# Salvează figura
output_path = r"C:\Users\ciope\Desktop\Licenta\prototypes\Train_times_ML\outputs\crossvalvstrain_scores.png"
plt.savefig(output_path)
print(f"Saved to: {output_path}")
