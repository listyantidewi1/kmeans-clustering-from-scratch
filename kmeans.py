import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Canvas, Button, TOP

class KMeans:
    def __init__(self, k=2, max_iter=100, n_init=10):
        self.k = k
        self.max_iter = max_iter
        self.n_init = n_init

    def fit(self, data):
        best_inertia = float('inf')
        for _ in range(self.n_init):
            centroids = data[np.random.choice(len(data), self.k, replace=False)]
            for _ in range(self.max_iter):
                clusters = [[] for _ in range(self.k)]
                for point in data:
                    distances = [np.linalg.norm(point - centroid) for centroid in centroids]
                    cluster_idx = np.argmin(distances)
                    clusters[cluster_idx].append(point)
                for i in range(self.k):
                    centroids[i] = np.mean(clusters[i], axis=0)
            inertia = sum(np.sum((data - centroids[label])**2) for label, cluster in enumerate(clusters))
            if inertia < best_inertia:
                best_inertia = inertia
                self.centroids = centroids
                self.labels = np.zeros(len(data))
                for i, cluster in enumerate(clusters):
                    self.labels[[data.tolist().index(point.tolist()) for point in cluster]] = i

    def visualize(self, data):
        plt.scatter(data[:, 0], data[:, 1], c=self.labels, cmap='viridis')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', marker='x')
        plt.show()

def display_gui(data, labels, centroids):
    root = Tk()
    root.title("K-Means Clustering")

    canvas = Canvas(root, width=300, height=300)
    canvas.pack()

    canvas.create_text(100, 10, text="K-Means Clustering", font=("Arial", 14), fill="black")

    def show_matplotlib_figure():
        kmeans.visualize(data)

    button = Button(root, text="Show Matplotlib Figure", command=show_matplotlib_figure)
    button.pack()

    root.mainloop()

# Generate random data
np.random.seed(0)
data = np.random.randn(100, 2)

# Perform K-means clustering
kmeans = KMeans(k=2, n_init=10)
kmeans.fit(data)

# Display result in Tkinter GUI
display_gui(data, kmeans.labels, kmeans.centroids)
