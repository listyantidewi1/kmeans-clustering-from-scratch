import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from tkinter import Tk, Canvas, Button, TOP, Label
import pandas as pd
import seaborn as sns

def fit_kmeans(data, k=2, max_iter=100, n_init=10):
    best_inertia = float('inf')
    best_centroids = None
    best_labels = None
    
    for _ in range(n_init):
        centroids = data[np.random.choice(len(data), k, replace=False)]
        for _ in range(max_iter):
            clusters = [[] for _ in range(k)]
            for point in data:
                distances = [((point - centroid) ** 2).sum() for centroid in centroids]
                cluster_idx = min(range(k), key=lambda i: distances[i])
                clusters[cluster_idx].append(point)
            for i in range(k):
                centroids[i] = np.mean(clusters[i], axis=0)
        inertia = sum(np.sum((data - centroids[label])**2) for label, cluster in enumerate(clusters))
        if inertia < best_inertia:
            best_inertia = inertia
            best_centroids = centroids
            best_labels = np.zeros(len(data))
            for i, cluster in enumerate(clusters):
                best_labels[[data.tolist().index(point.tolist()) for point in cluster]] = i
    
    return best_centroids, best_labels

def visualize_clusters(data, centroids, labels):
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
    plt.show()

def silhouette_score(data, labels, k):
    n = len(data)
    a = np.zeros(n)
    b = np.zeros(n)
    for i in range(n):
        cluster_i = labels[i]
        cluster_i_points = data[labels == cluster_i]
        a[i] = np.mean([sum((data[i] - point) ** 2) ** 0.5 for point in cluster_i_points])
        b[i] = min([np.mean([sum((data[i] - point) ** 2) ** 0.5 for point in data[labels == j]]) for j in range(k) if j != cluster_i])
    sil_scores = (b - a) / np.maximum(a, b)
    return np.mean(sil_scores)

def display_gui(data, labels, centroids):
    root = Tk()
    root.title("K-Means Clustering")
    canvas = Canvas(root, width=300, height=100)
    canvas.pack()
    canvas.create_text(100, 10, text="K-Means Clustering", font=("Arial", 14), fill="black")

    description_label = Label(root, text="Data Description", font=("Arial", 14, "bold"))
    description_label.pack(pady=5)

    avg = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)

    avg_label = Label(root, text=f"Average: {avg}", font=("Arial", 12))
    avg_label.pack()

    std_dev_label = Label(root, text=f"Standard Deviation: {std_dev}", font=("Arial", 12))
    std_dev_label.pack()

    min_label = Label(root, text=f"Minimum Value: {min_val}", font=("Arial", 12))
    min_label.pack()

    max_label = Label(root, text=f"Maximum Value: {max_val}", font=("Arial", 12))
    max_label.pack()

    silhouette_avg = silhouette_score(data, labels, len(centroids))
    silhouette_score_value_label = Label(root, text="Silhouette score: "+str(silhouette_avg), font=("Arial", 12))
    silhouette_score_value_label.pack()

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

    def visualize_scatter_matrix():
        df = pd.DataFrame(data, columns=['Feature 1', 'Feature 2'])
        sns.pairplot(df)
        plt.show()

    def show_matplotlib_figure():
        visualize_clusters(data, centroids, labels)
    
    button = Button(root, text="Click to Show Scatter Plot", command=show_matplotlib_figure)
    button.pack()

    visualize_histogram_button = Button(root, text="Visualize Histogram", command=visualize_histogram)
    visualize_histogram_button.pack(pady=5)

    visualize_scatter_matrix_button = Button(root, text="Visualize Scatter Matrix", command=visualize_scatter_matrix)
    visualize_scatter_matrix_button.pack(pady=5)

    root.mainloop()

np.random.seed(0)
data = np.random.randn(100, 2)

centroids, labels = fit_kmeans(data, k=3, n_init=100)
display_gui(data, labels, centroids)
