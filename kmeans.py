import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Canvas, Button, TOP, Label
import pandas as pd
import seaborn as sns

class KMeans:
    
    def __init__(self, k=2, max_iter=100, n_init=10):
        self.k = k
        self.max_iter = max_iter
        self.n_init = n_init
        """
        Initialize KMeans clustering algorithm.

        Parameters:
        - k: Number of clusters (default is 2).
        - max_iter: Maximum number of iterations for each initialization (default is 100).
        - n_init: Number of times the algorithm will be initialized with different centroids (default is 10).
        """

    def fit(self, data):
        """
        Fit the KMeans model to the data.

        Parameters:
        - data: Input data to be clustered.
        """

        """In the context of K-means clustering, inertia is a metric used to evaluate the quality of a clustering solution. It represents the sum of squared distances between each data point and its nearest centroid. In other words, inertia measures how compact the clusters are; lower inertia indicates tighter clusters."""

        """Convergence in the context of K-means clustering refers to the state where the algorithm has reached a stable solution, and further iterations do not significantly change the cluster assignments or the positions of the centroids. In other words, convergence occurs when the algorithm has found centroids that minimize the inertia or the sum of squared distances between data points and their assigned centroids.

        In practical terms, convergence is typically achieved when either:

        The centroids do not change significantly between iterations (i.e., they stabilize).
        The cluster assignments of data points do not change between iterations.
        When the algorithm converges, it means that it has likely found a satisfactory clustering solution, and further iterations are unlikely to improve the clustering quality significantly. However, it's important to note that K-means clustering is sensitive to the initial centroid positions, and it may converge to local optima rather than the global optimum. Therefore, it's common to run the algorithm multiple times with different initializations to mitigate this issue."""

        
        best_inertia = float('inf') # Initialize best inertia to positive infinity
        for _ in range(self.n_init): # Iterate over the number of initializations
            centroids = data[np.random.choice(len(data), self.k, replace=False)] # Randomly initialize centroids
            for _ in range(self.max_iter): # Iterate over the maximum number of iterations
                clusters = [[] for _ in range(self.k)] # Initialize clusters
                for point in data: # Assign each data point to the nearest centroid
                    distances = [np.linalg.norm(point - centroid) for centroid in centroids]
                    cluster_idx = np.argmin(distances)
                    clusters[cluster_idx].append(point)
                for i in range(self.k): # Update centroids based on the mean of data points in each cluster
                    centroids[i] = np.mean(clusters[i], axis=0)
            inertia = sum(np.sum((data - centroids[label])**2) for label, cluster in enumerate(clusters))
            if inertia < best_inertia: # Update centroids and labels if inertia is smaller
                best_inertia = inertia
                self.centroids = centroids
                self.labels = np.zeros(len(data))
                for i, cluster in enumerate(clusters):
                    self.labels[[data.tolist().index(point.tolist()) for point in cluster]] = i

    def visualize(self, data):
        """
        Visualize the clustered data and centroids.

        Parameters:
        - data: Input data to be visualized.
        """
        plt.scatter(data[:, 0], data[:, 1], c=self.labels, cmap='viridis')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', marker='x')
        plt.show()

def display_gui(data, labels, centroids):

    # membuat window ukuran 300x300 pixels
    root = Tk()
    root.title("K-Means Clustering") # judul winndow

    canvas = Canvas(root, width=300, height=100)
    canvas.pack()

    # membuat heading / judul pada posisi x = 100, y = 10
    canvas.create_text(100, 10, text="K-Means Clustering", font=("Arial", 14), fill="black")

    description_label = Label(root, text="Data Description", font=("Arial", 14, "bold"))
    description_label.pack(pady=5)

    # Calculate data description statistics
    avg = np.mean(data, axis=0) #average
    std_dev = np.std(data, axis=0) #standar deviasi
    min_val = np.min(data, axis=0) #nilai minimal
    max_val = np.max(data, axis=0) #nilai maksimal

    # Convert numpy arrays to strings
    avg_str = ", ".join(map(str, avg))
    std_dev_str = ", ".join(map(str, std_dev))
    min_val_str = ", ".join(map(str, min_val))
    max_val_str = ", ".join(map(str, max_val))

    avg_label = Label(root, text=f"Average: {avg}", font=("Arial", 12))
    avg_label.pack()

    std_dev_label = Label(root, text=f"Standard Deviation: {std_dev}", font=("Arial", 12))
    std_dev_label.pack()

    min_label = Label(root, text=f"Minimum Value: {min_val}", font=("Arial", 12))
    min_label.pack()

    max_label = Label(root, text=f"Maximum Value: {max_val}", font=("Arial", 12))
    max_label.pack()

    # fungsi untuk membuat histogram
    def visualize_histogram():
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(data[:, 0], bins=20, color='skyblue', edgecolor='black')
        plt.xlabel('Feature 1')
        plt.ylabel('Frequency')
        plt.title('Histogram of Feature 1')

        plt.subplot(1, 2, 2)
        plt.hist(data[:, 1], bins=20, color='salmon', edgecolor='black')
        plt.xlabel('Feature 2')
        plt.ylabel('Frequency')
        plt.title('Histogram of Feature 2')
        plt.tight_layout()
        plt.show()

    # fungsi untuk membuat scatter matrix
    def visualize_scatter_matrix():
        df = pd.DataFrame(data, columns=['Feature 1', 'Feature 2'])
        sns.pairplot(df)
        plt.show()


    # fungsi untuk menampilkan scatterplot menggunakan button pada Tkinter
    def show_matplotlib_figure():
        kmeans.visualize(data)
    
    # membuat button untuk menampilkan scatter plot hasil clustering
    button = Button(root, text="Click to Show Scatter Plot", command=show_matplotlib_figure)
    button.pack()

    # button untuk menampilkan histogram
    visualize_histogram_button = Button(root, text="Visualize Histogram", command=visualize_histogram)
    visualize_histogram_button.pack(pady=5)

    # button untuk menampilkan scatter matrix
    visualize_scatter_matrix_button = Button(root, text="Visualize Scatter Matrix", command=visualize_scatter_matrix)
    visualize_scatter_matrix_button.pack(pady=5)

    root.mainloop()

# Generate random data
np.random.seed(0)
data = np.random.randn(100, 2)

# Perform K-means clustering
kmeans = KMeans(k=2, n_init=10)
kmeans.fit(data)

# Display result in Tkinter GUI
display_gui(data, kmeans.labels, kmeans.centroids)
