# scripts/plot_feature_importance.py
import joblib
import matplotlib.pyplot as plt

model_path = r"C:\Users\ciope\Desktop\Licenta\prototypes\Train_times_ML\models\decision_tree_model.pkl"
clf = joblib.load(model_path)

importances = clf.feature_importances_
features = ['cars_south', 'cars_north', 'cars_west', 'cars_east']

plt.figure(figsize=(8, 6))
plt.bar(features, importances, color='teal')
plt.title("Feature Importance")
plt.ylabel("Importance")
plt.tight_layout()

output_path = r"C:\Users\ciope\Desktop\Licenta\prototypes\Train_times_ML\outputs\feature_importance.png"
plt.savefig(output_path)
print(f"Saved to: {output_path}")
