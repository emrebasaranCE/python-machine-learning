import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# Generate a sample dataset
X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

# Visualize the dataset
plt.scatter(X[:, 0], X[:, 1], s=50, c='gray')
plt.title("Sample Dataset")
plt.show()

# Apply DBSCAN
epsilon = 0.2  # Radius of the neighborhood
min_samples = 5  # Minimum points to form a dense region
dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
labels = dbscan.fit_predict(X)

# Visualize the DBSCAN clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
plt.title("DBSCAN Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar(label='Cluster')
plt.show()

# Evaluate the clustering (excluding noise points)
if len(set(labels)) > 1:
    silhouette_avg = silhouette_score(X[labels != -1], labels[labels != -1])
    print(f"Silhouette Score (excluding noise): {silhouette_avg:.2f}")
else:
    print("All points are noise or a single cluster, Silhouette Score cannot be computed.")
