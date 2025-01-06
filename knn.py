# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# Load the Iris dataset
iris = datasets.load_iris()

# Features (X): sepal length, sepal width, petal length, petal width
X = iris.data  

# Labels (y): target classes (0 = setosa, 1 = versicolor, 2 = virginica)
y = iris.target

# Split the dataset into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Test different values of k
k_values = range(1, 11)
scores = []

# Cross-validation to find the best k
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X, y, cv=5, scoring='accuracy')  # 5-fold cross-validation
    scores.append(score.mean())

# Find the best k value
best_k = k_values[np.argmax(scores)]
print(f"Best k value: {best_k} : {scores}")

# Create and train the KNN classifier with the best k
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)  # Train the model

# Test the model on the testing set
y_pred = knn.predict(X_test)  # Make predictions on the test data

# Calculate accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy with KNN (k={best_k}): {acc:.2f}\n")

# Print a detailed classification report
print("Classification Report for KNN:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Plot the confusion matrix for the KNN model
ConfusionMatrixDisplay.from_estimator(
    knn, X_test, y_test, 
    display_labels=iris.target_names, cmap=plt.cm.Blues  # Display options
)
plt.title(f"Confusion Matrix (KNN, k={best_k})")  # Add a title with the best k
plt.show()  # Display the plot
