import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Wine dataset
data = load_wine()
X = data.data  # Features
y = data.target  # Labels

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Decision Tree model
model = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Export and display tree rules
tree_rules = export_text(model, feature_names=data.feature_names)
print("Decision Tree Rules:")
print(tree_rules)

# Display feature importances
feature_importances = model.feature_importances_
print("Feature Importances:")
for name, importance in zip(data.feature_names, feature_importances):
    print(f"{name}: {importance:.3f}")

# Visualize the decision tree
plt.figure(figsize=(15, 10))
plot_tree(
    model, 
    feature_names=data.feature_names, 
    class_names=data.target_names, 
    filled=True, 
    rounded=True, 
    fontsize=10
)
plt.title("Decision Tree Visualization")
plt.show()