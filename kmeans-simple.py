import pandas as pd  # Importing pandas library for data manipulation
import numpy as np  # Importing numpy library for numerical computations
import matplotlib.pyplot as plt  # Importing matplotlib for data visualization
import warnings  # Importing warnings to suppress potential warnings
from tkinter import Tk, Canvas, Button, Label  # Importing necessary components from tkinter for GUI
import seaborn as sns  # Importing seaborn for enhanced data visualization capabilities

# Ignore warnings
warnings.filterwarnings('ignore', category=Warning)

# Read the dataset
data = pd.read_csv('clustering.csv')  # Loading data from 'clustering.csv' into a pandas DataFrame
X = data[["LoanAmount","ApplicantIncome"]]  # Extracting features for clustering

# Define the number of clusters (K)
K = 2  # Setting the number of clusters to 4

# Step 1 and 2 - Select random centroids for each cluster
Centroids = X.sample(n=K, random_state=42)  # Randomly selecting K centroids from the data

# Define functions for visualization and K-means clustering

# Function to visualize scatter plot of raw data
def visualize_scatter_plot_raw_data():
    plt.scatter(X["LoanAmount"], X["ApplicantIncome"], c='black')  # Plotting raw data points
    plt.xlabel('Loan Amount')  # Labeling x-axis
    plt.ylabel('Applicant Income')  # Labeling y-axis
    plt.show()  # Displaying the plot

# Function to visualize scatter matrix
def visualize_scatter_matrix():
    df = pd.DataFrame(X, columns=['ApplicantIncome', 'LoanAmount'])
    sns.pairplot(df)  # Creating a pairplot for visualizing relationships between variables
    plt.show()  # Displaying the plot

# Function to visualize initial centroids
def initial_centroids():
    plt.scatter(X["LoanAmount"], X["ApplicantIncome"], c='black')  # Plotting raw data points
    plt.scatter(Centroids["LoanAmount"], Centroids["ApplicantIncome"], c='red')  # Plotting initial centroids
    plt.xlabel('LoanAmount')  # Labeling x-axis
    plt.ylabel('ApplicantIncome')  # Labeling y-axis
    plt.show()  # Displaying the plot

# Function to calculate distance between two points
def euclidean_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5  # Calculating Euclidean distance

# Function to perform K-means clustering
def kmeans():
    centroids = Centroids.values  # Extracting centroid values
    X_values = X.values  # Extracting feature values
    
    while True:
        # Assign each point to the nearest centroid
        cluster_assignments = []  # Initializing list to store cluster assignments
        for point in X_values:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]  # Calculating distances to centroids
            cluster_assignments.append(np.argmin(distances))  # Assigning point to the nearest centroid. np.argmin -> Returns the indexes of the minimum values along an axis.
        
        # Update centroids
        new_centroids = []  # Initializing list to store updated centroids
        for i in range(K):
            cluster_points = X_values[np.array(cluster_assignments) == i]  # Extracting points belonging to cluster i
            new_centroid = np.mean(cluster_points, axis=0)  # Calculating new centroid as the mean of cluster points
            new_centroids.append(new_centroid)  # Appending new centroid
        
        # Check convergence
        # np.allclose Returns True if two arrays are element-wise equal within a tolerance. The tolerance values are positive, typically very small numbers.
            #In the context of k-means clustering, np.allclose is used to check whether the centroids have converged. If the centroids have converged, it means that the algorithm has reached a stable solution and can terminate
        if np.allclose(centroids, new_centroids):  # Checking if centroids have converged
            break
        
        centroids = np.array(new_centroids)  # Updating centroids
    
    # Calculate silhouette score, rumus: https://www.analyticsvidhya.com/blog/2021/05/k-mean-getting-the-optimal-number-of-clusters/#:~:text=The%20silhouette%20coefficient%20or%20silhouette,scikit%2Dlearn%2Fsklearn%20library.
    silhouette_values = []  # Initializing list to store silhouette values

    # Iterate over X_values directly
    for point_idx in range(len(X_values)):
        point = X_values[point_idx]  # Get the current point
        cluster = cluster_assignments[point_idx]  # Getting cluster assignment for point
        cluster_points = X_values[np.array(cluster_assignments) == cluster]  # Extracting points in the same cluster
    
        # Calculate average intra-cluster distance
        a_i = np.mean([euclidean_distance(point, other_point) for other_point in cluster_points])
    
        b_i = np.inf
        # Iterate over range(K) to find inter-cluster distances
        for j in range(K):
            if j != cluster:
                other_cluster_points = X_values[np.array(cluster_assignments) == j]  # Extracting points from other clusters
                b_ij = np.mean([euclidean_distance(point, other_point) for other_point in other_cluster_points])  # Calculating average inter-cluster distance
                b_i = min(b_i, b_ij)  # Finding minimum inter-cluster distance
    
        silhouette_values.append((b_i - a_i) / max(a_i, b_i))  # Computing silhouette value

    silhouette_avg = np.mean(silhouette_values)  # Calculating average silhouette score

    
    # Plot clusters
    for i in range(K):
        cluster_points = X_values[np.array(cluster_assignments) == i]  # Extracting points belonging to cluster i
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], cmap='viridis')  # Plotting cluster points
        plt.scatter(centroids[i][0], centroids[i][1], c='red', marker='x')  # Plotting centroid
        plt.text(centroids[i][0], centroids[i][1], f'Cluster {i+1}', fontsize=12, color='red')  # Adding cluster label
    
    plt.xlabel('Loan Amount (In Thousands)')  # Labeling x-axis
    plt.ylabel('Income ')  # Labeling y-axis
    plt.show()  # Displaying the plot
    
    # Display silhouette score
    silhouette_label = Label(root, text=f"Silhouette Score: {silhouette_avg}", font=("Arial", 12))
    silhouette_label.pack()

# Function to visualize the Elbow Method for determining the optimal number of clusters
    # rumus: https://www.analyticsvidhya.com/blog/2021/01/in-depth-intuition-of-k-means-clustering-algorithm-in-machine-learning/
def elbow_method():
    distortions = []  # Initializing list to store distortion values
    
    # Calculating distortion for different values of K
    for k in range(1, 11):  # Trying K values from 1 to 10
        Centroids = X.sample(n=k, random_state=42)  # Randomly selecting K centroids from the data
        centroids = Centroids.values  # Extracting centroid values
        X_values = X.values  # Extracting feature values
        
        cluster_assignments = []  # Initializing list to store cluster assignments
        for point in X_values:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]  # Calculating distances to centroids
            cluster_assignments.append(np.argmin(distances))  # Assigning point to the nearest centroid
        
        distortion = 0  # Initializing distortion for current K
        for i in range(k):
            cluster_points = X_values[np.array(cluster_assignments) == i]  # Extracting points belonging to cluster i
            centroid = np.mean(cluster_points, axis=0)  # Calculating centroid of cluster
            distortion += np.sum((cluster_points - centroid) ** 2)  # Calculating sum of squared distances from centroid
        
        distortions.append(distortion)  # Appending distortion to list
    
    # Plotting the Elbow Method
    plt.plot(range(1, 11), distortions, marker='o')  # Plotting K values against distortions
    plt.xlabel('Number of Clusters')  # Labeling x-axis
    plt.ylabel('Distortion')  # Labeling y-axis
    plt.title('Elbow Method')  # Adding title
    plt.show()  # Displaying the plot

# Display GUI
root = Tk()  # Creating Tkinter window
root.title("K-Means Clustering")  # Setting window title
canvas = Canvas(root, width=300, height=100)  # Creating canvas for GUI components
canvas.pack()  # Packing canvas into window
canvas.create_text(100, 10, text="K-Means Clustering", font=("Arial", 14), fill="black")  # Adding title text

description_label = Label(root, text="Data Description", font=("Arial", 14, "bold"))  # Creating label for data description
description_label.pack(pady=5)  # Packing label into window

# Calculate statistics about the data
avgIncome = np.mean(X["ApplicantIncome"])  # Calculating average income
avgLoan = np.mean(X["LoanAmount"])  # Calculating average loan amount
stdDevIncome = np.std(X['ApplicantIncome'])  # Calculating standard deviation of income
stdDevLoan = np.std(X["LoanAmount"])  # Calculating standard deviation of loan amount
maxIncome = np.max(X["ApplicantIncome"])  # Finding maximum income
minIncome = np.min(X["ApplicantIncome"])  # Finding minimum income
maxLoan = np.max(X['LoanAmount'])  # Finding maximum loan amount
minLoan = np.min(X['LoanAmount'])  # Finding minimum loan amount

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

# Button to trigger the Elbow Method visualization
elbowMethodButton = Button(root, text="Visualize Elbow Method", command=elbow_method)
elbowMethodButton.pack()

visualizeScatterMatrixButton = Button(root, text="Show Scatter Matrix", command=visualize_scatter_matrix)
visualizeScatterMatrixButton.pack()

visualizeScatterPlotRawButton = Button(root, text="Show Raw Data Plot", command=visualize_scatter_plot_raw_data)
visualizeScatterPlotRawButton.pack()

displayInitialCentroidsButton = Button(root, text="Show Initial Centroids", command=initial_centroids)
displayInitialCentroidsButton.pack()

buttonClustering = Button(root, text="Analyze Cluster", command=kmeans)
buttonClustering.pack()

root.mainloop()  # Running the GUI event loop
