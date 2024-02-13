import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Canvas

class KMeans:
    def __init__(self, k=2, max_iter=100, n_init=10):
        """
        Initialize KMeans clustering algorithm.

        Parameters:
        - k: Number of clusters (default is 2).
        - max_iter: Maximum number of iterations for each initialization (default is 100).
        - n_init: Number of initializations to perform (default is 10).

        
        Iterations in the K-means algorithm are necessary to refine the cluster assignments and update the cluster centroids until convergence. Here's why iterations are needed:

        Assignment Step: In each iteration, every data point is assigned to the nearest centroid. This step is crucial because it determines which cluster each data point belongs to.

        Update Step: After all data points are assigned to clusters, the centroids of the clusters are updated based on the mean of the data points assigned to each cluster. This step ensures that the centroids move towards the center of their respective clusters.

        Convergence: The iterations continue until either the centroids stop changing significantly between iterations or a maximum number of iterations is reached. This convergence criterion ensures that the algorithm stops when the clusters have stabilized.
        
        Convergence in the context of the K-means algorithm means that the algorithm has reached a stable and consistent solution. In simple terms, convergence occurs when the cluster centroids stop changing significantly between iterations, indicating that the algorithm has found a good clustering arrangement. It means that the algorithm has done its best to group the data points into clusters, and further iterations won't lead to significant improvements in the clustering result.

        Without iterations, the algorithm would not be able to accurately determine the cluster assignments and centroids, and the clustering result may not be optimal. Iterations allow the algorithm to gradually improve the clustering by iteratively refining the cluster assignments and centroids until convergence.
        """

        self.k = k
        self.max_iter = max_iter
        self.n_init = n_init

    def fit(self, data):
        """
        Fit the KMeans model to the data.

        Parameters:
        - data: Input data, numpy array of shape (n_samples, n_features).
        """
        best_inertia = float('inf')
        # Perform multiple initializations
        for _ in range(self.n_init):
            # Initialize centroids randomly
            centroids = data[np.random.choice(len(data), self.k, replace=False)]
            # Perform KMeans clustering
            for _ in range(self.max_iter):
                # Assign each data point to the closest centroid
                clusters = [[] for _ in range(self.k)]
                for point in data:
                    distances = [np.linalg.norm(point - centroid) for centroid in centroids]
                    cluster_idx = np.argmin(distances)
                    clusters[cluster_idx].append(point)
                # Update centroids
                for i in range(self.k):
                    centroids[i] = np.mean(clusters[i], axis=0)
            # Calculate total within-cluster variance (inertia)
            inertia = sum(np.sum((data - centroids[label])**2) for label, cluster in enumerate(clusters))
            # Update best clustering if inertia is lower
            if inertia < best_inertia:
                best_inertia = inertia
                self.centroids = centroids
                self.labels = np.zeros(len(data))
                for i, cluster in enumerate(clusters):
                    self.labels[[data.tolist().index(point.tolist()) for point in cluster]] = i

    def visualize(self, data):
        """
        Visualize the clustering result using matplotlib.

        Parameters:
        - data: Input data, numpy array of shape (n_samples, n_features).
        """
        plt.scatter(data[:, 0], data[:, 1], c=self.labels, cmap='viridis')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', marker='x')
        plt.show()

def display_gui(data, labels, centroids):
    """
    Display the clustering result in a Tkinter GUI.

    Parameters:
    - data: Input data, numpy array of shape (n_samples, n_features).
    - labels: Cluster labels assigned to each data point.
    - centroids: Centroids of the clusters.
    """
    root = Tk()
    root.title("K-Means Clustering")

    canvas = Canvas(root, width=400, height=400)
    canvas.pack()

    # Scale data points to fit canvas
    data_scaled = data * 300 + 200

    # Plot data points
    for i, (x, y) in enumerate(data_scaled):
        color = ('#FF5733' if labels[i] == 0 else '#33FF57')
        canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill=color)

    # Plot centroids
    centroids_scaled = centroids * 300 + 200
    for (x, y) in centroids_scaled:
        canvas.create_rectangle(x - 5, y - 5, x + 5, y + 5, fill='red')

    root.mainloop()

# Generate random data
np.random.seed(0)
data = np.random.randn(100, 2)

# Perform K-means clustering
kmeans = KMeans(k=3, n_init=10)
kmeans.fit(data)

# Visualize clustering
kmeans.visualize(data)

# Display result in Tkinter GUI
display_gui(data, kmeans.labels, kmeans.centroids)
