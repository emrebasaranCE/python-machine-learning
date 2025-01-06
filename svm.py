# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# Load the Iris dataset
# The dataset contains features of iris flowers (sepal/petal length and width)
# and their corresponding class labels (setosa, versicolor, virginica).
iris = datasets.load_iris()

# Features (X): sepal length, sepal width, petal length, petal width
X = iris.data  

# Labels (y): target classes (0 = setosa, 1 = versicolor, 2 = virginica)
y = iris.target

# Split the dataset into training and testing sets (70% training, 30% testing)
# random_state ensures reproducibility of the split.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a dictionary to store SVM models for each kernel type
kernels = ['linear', 'poly', 'rbf', 'sigmoid']  # Different kernel options for SVM
models = {}

# Loop through each kernel type and train an SVM model
for kernel in kernels:
    print(f"Training SVM with {kernel} kernel...")
    clf = SVC(kernel=kernel, gamma='scale', C=1.0)  # Initialize SVM with the specified kernel
    clf.fit(X_train, y_train)  # Train the model on the training data
    models[kernel] = clf  # Store the trained model in the dictionary
    
    # Test the model on the testing set
    y_pred = clf.predict(X_test)  # Make predictions on the test data
    acc = accuracy_score(y_test, y_pred)  # Calculate accuracy
    print(f"Accuracy with {kernel} kernel: {acc:.2f}\n")  # Print accuracy for this kernel

# Example: Evaluate the model with the RBF kernel
best_model = models['rbf']  # Retrieve the trained model with the RBF kernel
y_pred = best_model.predict(X_test)  # Predict using the RBF model

# Print a detailed classification report
# Shows precision, recall, f1-score, and support for each class
print("Classification Report for RBF Kernel:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Plot the confusion matrix for the RBF kernel model
ConfusionMatrixDisplay.from_estimator(
    best_model, X_test, y_test, 
    display_labels=iris.target_names, cmap=plt.cm.Blues  # Display options
)
plt.title("Confusion Matrix (RBF Kernel)")  # Add a title
plt.show()  # Display the plot
