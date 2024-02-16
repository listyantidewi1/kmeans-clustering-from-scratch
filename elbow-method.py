import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from tkinter import Tk, Canvas, Button, Label
import seaborn as sns

# Ignore warnings
warnings.filterwarnings('ignore', category=Warning)

# Read the dataset
data = pd.read_csv('clustering.csv')
X = data[["LoanAmount","ApplicantIncome"]]

# Define the number of clusters (K)
K = 4

# Step 1 and 2 - Select random centroids for each cluster
Centroids = X.sample(n=K, random_state=42)

# Define the maximum number of clusters to consider
MAX_CLUSTERS = 10

# Function to determine the optimal number of clusters using the elbow method
def find_optimal_clusters():
    distortions = []
    for k in range(1, MAX_CLUSTERS + 1):
        centroids = X.sample(n=k, random_state=42).values
        cluster_assignments = assign_clusters(X.values, centroids)
        distortion = calculate_distortion(X.values, centroids, cluster_assignments)
        distortions.append(distortion)
    return distortions

# Function to assign each point to the nearest centroid
def assign_clusters(X_values, centroids):
    cluster_assignments = []
    for point in X_values:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster_assignments.append(np.argmin(distances))
    return cluster_assignments

# Function to calculate the total distortion (sum of squared distances from each point to its centroid)
def calculate_distortion(X_values, centroids, cluster_assignments):
    total_distortion = 0
    for i, point in enumerate(X_values):
        cluster = cluster_assignments[i]
        total_distortion += euclidean_distance(point, centroids[cluster]) ** 2
    return total_distortion

# Function to visualize the elbow plot
def visualize_elbow_plot(distortions):
    plt.plot(range(1, MAX_CLUSTERS + 1), distortions, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion')
    plt.title('Elbow Method')
    plt.show()

# Function to visualize scatter plot of raw data
def visualize_scatter_plot_raw_data():
    plt.scatter(X["ApplicantIncome"], X["LoanAmount"], c='black')
    plt.xlabel('AnnualIncome')
    plt.ylabel('Loan Amount (In Thousands)')
    plt.show()

# Function to visualize scatter matrix
def visualize_scatter_matrix():
    df = pd.DataFrame(X, columns=['ApplicantIncome', 'LoanAmount'])
    sns.pairplot(df)
    plt.show()

# Function to visualize initial centroids
def initial_centroids():
    plt.scatter(X["ApplicantIncome"], X["LoanAmount"], c='black') 
    plt.scatter(Centroids["ApplicantIncome"], Centroids["LoanAmount"], c='red')
    plt.xlabel('AnnualIncome')
    plt.ylabel('Loan Amount (In Thousands)')
    plt.show()

# Function to calculate distance between two points
def euclidean_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

# Function to perform K-means clustering
def kmeans():
    centroids = Centroids.values
    X_values = X.values
    cluster_assignments = assign_clusters(X_values, centroids)
    distortion = calculate_distortion(X_values, centroids, cluster_assignments)
    silhouette_score = calculate_silhouette_score(X_values, cluster_assignments)
    print(f"Distortion: {distortion}")
    print(f"Silhouette Score: {silhouette_score}")
    visualize_clusters(centroids, cluster_assignments)

# Function to calculate silhouette score
def calculate_silhouette_score(X_values, cluster_assignments):
    silhouette_values = []
    for i, point in enumerate(X_values):
        cluster = cluster_assignments[i]
        cluster_points = X_values[np.array(cluster_assignments) == cluster]
        a_i = np.mean([euclidean_distance(point, other_point) for other_point in cluster_points])
        
        b_i = np.inf
        for j in range(K):
            if j != cluster:
                other_cluster_points = X_values[np.array(cluster_assignments) == j]
                b_ij = np.mean([euclidean_distance(point, other_point) for other_point in other_cluster_points])
                b_i = min(b_i, b_ij)
        
        silhouette_values.append((b_i - a_i) / max(a_i, b_i))
    
    silhouette_avg = np.mean(silhouette_values)
    return silhouette_avg

# Function to visualize clusters
def visualize_clusters(centroids, cluster_assignments):
    for i in range(K):
        cluster_points = X.values[np.array(cluster_assignments) == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], cmap='viridis')
        plt.scatter(centroids[i][0], centroids[i][1], c='red', marker='x')
        plt.text(centroids[i][0], centroids[i][1], f'Cluster {i+1}', fontsize=12, color='red')
    
    plt.xlabel('Income')
    plt.ylabel('Loan Amount (In Thousands)')
    plt.show()

# Display GUI
root = Tk()
root.title("K-Means Clustering")
canvas = Canvas(root, width=300, height=100)
canvas.pack()
canvas.create_text(100, 10, text="K-Means Clustering", font=("Arial", 14), fill="black")

description_label = Label(root, text="Data Description", font=("Arial", 14, "bold"))
description_label.pack(pady=5)

# Calculate statistics about the data
avgIncome = np.mean(X["ApplicantIncome"])
avgLoan = np.mean(X["LoanAmount"])
stdDevIncome = np.std(X['ApplicantIncome'])
stdDevLoan = np.std(X["LoanAmount"])
maxIncome = np.max(X["ApplicantIncome"])
minIncome = np.min(X["ApplicantIncome"])
maxLoan = np.max(X['LoanAmount'])
minLoan = np.min(X['LoanAmount'])

# Display statistics in the GUI
avgIncomeLabel = Label(root, text=f"Average Income : {avgIncome}", font=("Arial", 12))
avgIncomeLabel.pack()

avgLoanLabel = Label(root, text=f"Average Loan : {avgLoan}", font=("Arial", 12))
avgLoanLabel.pack()

stdDevIncomeLabel = Label(root, text=f"Income Standard Deviation : {stdDevIncome}", font=("Arial", 12))
stdDevIncomeLabel.pack()

stdDevLoanLabel = Label(root, text=f"Income Standard Deviation : {stdDevLoan}", font=("Arial", 12))
stdDevLoanLabel.pack()

minIncomeLabel = Label(root, text=f"Minimum Income : {minIncome}", font=("Arial", 12))
maxIncomeLabel = Label(root, text=f"Maximum Income : {maxIncome}", font=("Arial", 12))

minLoanLabel = Label(root, text=f"Minimum Loan Amount : {minLoan}", font=("Arial", 12))
maxLoanLabel = Label(root, text=f"Maximum Loan Amount : {maxLoan}", font=("Arial", 12))

# Buttons to trigger different actions
visualizeScatterMatrixButton = Button(root, text="Show Scatter Matrix", command=visualize_scatter_matrix)
visualizeScatterMatrixButton.pack()

visualizeScatterPlotRawButton = Button(root, text="Show Raw Data Plot", command=visualize_scatter_plot_raw_data)
visualizeScatterPlotRawButton.pack()

displayInitialCentroidsButton = Button(root, text="Show Initial Centroids", command=initial_centroids)
displayInitialCentroidsButton.pack()

buttonClustering = Button(root, text="Analyze Cluster", command=kmeans)
buttonClustering.pack()

# Find and visualize the optimal number of clusters using the elbow method
distortions = find_optimal_clusters()
visualize_elbow_plot(distortions)

root.mainloop()
