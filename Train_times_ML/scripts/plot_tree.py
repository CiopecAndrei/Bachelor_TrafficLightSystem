# scripts/export_tree_graphviz.py
import joblib
from sklearn.tree import export_graphviz
import graphviz
import os

# Load model
model_path = r"C:\Users\ciope\Desktop\Licenta\prototypes\Train_times_ML\models\decision_tree_model.pkl"
clf = joblib.load(model_path)

# Export to DOT format
dot_data = export_graphviz(
    clf,
    out_file=None,
    feature_names=['cars_south', 'cars_north', 'cars_west', 'cars_east'],
    class_names=["Short", "Medium", "Long"],
    filled=True,
    rounded=True,
    special_characters=True
)

# Generate graph
graph = graphviz.Source(dot_data)

# Save output
output_dir = r"C:\Users\ciope\Desktop\Licenta\prototypes\Train_times_ML\outputs"
graph.render(filename="decision_tree_graphviz", directory=output_dir, format="pdf", cleanup=True)
print(f"Saved Graphviz tree to: {os.path.join(output_dir, 'decision_tree_graphviz.pdf')}")
