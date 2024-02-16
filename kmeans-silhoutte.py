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

# Define functions for visualization and K-means clustering

# Function to visualize scatter plot of raw data
def visualize_scatter_plot_raw_data():
    plt.scatter(X["LoanAmount"], X["ApplicantIncome"], c='black')
    plt.xlabel('Loan Amount')
    plt.ylabel('Applicant Income')
    plt.show()

# Function to visualize scatter matrix
def visualize_scatter_matrix():
    df = pd.DataFrame(X, columns=['ApplicantIncome', 'LoanAmount'])
    sns.pairplot(df)
    plt.show()

# Function to visualize initial centroids
def initial_centroids():
    plt.scatter(X["LoanAmount"], X["ApplicantIncome"], c='black') 
    plt.scatter(Centroids["LoanAmount"], Centroids["ApplicantIncome"], c='red')
    plt.xlabel('LoanAmount')
    plt.ylabel('ApplicantIncome')
    plt.show()

# Function to calculate distance between two points
def euclidean_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

# Function to perform K-means clustering
def kmeans():
    centroids = Centroids.values
    X_values = X.values
    
    while True:
        # Assign each point to the nearest centroid
        cluster_assignments = []
        for point in X_values:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            cluster_assignments.append(np.argmin(distances))
        
        # Update centroids
        new_centroids = []
        for i in range(K):
            cluster_points = X_values[np.array(cluster_assignments) == i]
            new_centroid = np.mean(cluster_points, axis=0)
            new_centroids.append(new_centroid)
        
        # Check convergence
        if np.allclose(centroids, new_centroids):
            break
        
        centroids = np.array(new_centroids)
    
    # Calculate silhouette score
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
    
    # Plot clusters
    for i in range(K):
        cluster_points = X_values[np.array(cluster_assignments) == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], cmap='viridis')
        plt.scatter(centroids[i][0], centroids[i][1], c='red', marker='x')
        plt.text(centroids[i][0], centroids[i][1], f'Cluster {i+1}', fontsize=12, color='red')
    
    plt.xlabel('Loan Amount (In Thousands)')
    plt.ylabel('Income ')
    plt.show()
    
    # Display silhouette score
    silhouette_label = Label(root, text=f"Silhouette Score: {silhouette_avg}", font=("Arial", 12))
    silhouette_label.pack()

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

root.mainloop()
