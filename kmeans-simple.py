# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from tkinter import Tk, Canvas, Button, TOP, Label
import seaborn as sns
from matplotlib import colormaps

# Ignore warnings
warnings.filterwarnings('ignore', category=Warning)

# Read the dataset
data = pd.read_csv('clustering.csv')
X = data[["LoanAmount","ApplicantIncome"]]

# Define the number of clusters (K)
K=4

# Step 1 and 2 - Select random centroids for each cluster
Centroids = (X.sample(n=K))

# Define functions for visualization and K-means clustering

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

# Function to perform K-means clustering
def kmeans(Centroids=Centroids):
    diff = 1  # Initialize diff variable to 1 to enter the while loop
    j = 0     # Initialize iteration counter

    # Continue loop until centroids stop changing
    while diff != 0:
        XD = X.copy()  # Store current centroids in XD
        i = 1          # Initialize index counter
        
        # Loop over centroids
        for index1, row_c in Centroids.iterrows():
            ED = []  # Initialize list to store Euclidean distances
            
            # Loop over data points
            for index2, row_d in XD.iterrows():
                d1 = (row_c["ApplicantIncome"] - row_d["ApplicantIncome"])**2
                d2 = (row_c["LoanAmount"] - row_d["LoanAmount"])**2
                d = np.sqrt(d1 + d2)  # Calculate Euclidean distance between centroid and data point
                ED.append(d)          # Append distance to list
                
            X.loc[:, i] = ED  # Update corresponding row in X with distances
            i += 1            # Increment index counter
        
        C = []  # Initialize list to store cluster assignments
        
        # Loop over data points
        for index, row in X.iterrows():
            min_dist = row[1]  # Initialize minimum distance and cluster position
            pos = 1
            
            # Find nearest centroid and assign cluster
            for k in range(K):
                if row[k+1] < min_dist:
                    min_dist = row[k+1]
                    pos = k+1
            C.append(pos)  # Append cluster assignment to list
        
        X["Cluster"] = C  # Assign cluster assignments to X
        
        # Calculate new centroids based on cluster assignments
        Centroids_new = X.groupby(["Cluster"]).mean()[["LoanAmount", "ApplicantIncome"]]
        diff_sum = []
        
        if j == 0:  # Check if it's the first iteration
            diff = 1  # Set diff to 1 to continue loop
            j += 1    # Increment iteration counter
        else:
            # Calculate difference between new and old centroids
            diff = (Centroids_new['LoanAmount'] - Centroids['LoanAmount']).sum() + \
                   (Centroids_new['ApplicantIncome'] - Centroids['ApplicantIncome']).sum()
            print(diff.sum())  # Print sum of differences
            diff_sum.append(diff.sum())
        
        Centroids = X.groupby(["Cluster"]).mean()[["LoanAmount", "ApplicantIncome"]]  # Update centroids for next iteration
    
    # Plot clusters
    for k in range(K):
        data = X[X["Cluster"] == k+1]
        plt.scatter(data["ApplicantIncome"], data["LoanAmount"], cmap='viridis')
    plt.scatter(Centroids["ApplicantIncome"], Centroids["LoanAmount"], c='red')
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

root.mainloop()
