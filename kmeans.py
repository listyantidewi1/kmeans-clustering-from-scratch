import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Canvas, Button, TOP, Label

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
    avg = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)

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

    
    def show_matplotlib_figure():
        kmeans.visualize(data)

    button = Button(root, text="Click to Show Scatter Plot", command=show_matplotlib_figure)
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
