import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
file_path = os.path.join("..", "data", "heart.csv")
data = pd.read_csv(file_path)

# Features and target
X = data.drop("target", axis=1)
y = data["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7 Decision Tree Variants
models = {
    "Gini": DecisionTreeClassifier(criterion="gini"),
    "Entropy": DecisionTreeClassifier(criterion="entropy"),
    "Max Depth (5)": DecisionTreeClassifier(max_depth=5),
    "Min Samples Split (10)": DecisionTreeClassifier(min_samples_split=10),
    "Min Samples Leaf (5)": DecisionTreeClassifier(min_samples_leaf=5),
    "Cost Complexity Pruning": DecisionTreeClassifier(ccp_alpha=0.01),
    "Random Splitter": DecisionTreeClassifier(splitter="random")
}

results = {}

print("\n=== Model Performance ===\n")

# Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name}: {acc:.4f}")

# Plot one tree (Gini)
plt.figure(figsize=(12, 7))
plot_tree(models["Gini"], filled=True)
plt.title("Decision Tree (Gini)")
plt.show()

# Save results
results_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
results_df.to_csv("../results.csv", index=False)

print("\nResults saved to results.csv")