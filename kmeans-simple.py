#import libraries
import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import warnings
from tkinter import Tk, Canvas, Button, TOP, Label
warnings.filterwarnings('ignore', category=Warning)
import seaborn as sns

data = pd.read_csv('clustering.csv')
data.head()
X = data[["LoanAmount","ApplicantIncome"]]

# Step 1 and 2 - Choose the number of clusters (k) and select random centroid for each cluster

#number of clusters
K=3

# Select random observation as centroids
Centroids = (X.sample(n=K))

def visualize_scatter_plot_raw_data():
    #Visualise data points
    plt.scatter(X["ApplicantIncome"],X["LoanAmount"],c='black')
    plt.xlabel('AnnualIncome')
    plt.ylabel('Loan Amount (In Thousands)')
    plt.show()

def visualize_scatter_matrix():
    df = pd.DataFrame(X, columns=['ApplicantIncome', 'LoanAmount'])
    sns.pairplot(df)
    plt.show()

def initial_centroids():
    plt.scatter(X["ApplicantIncome"],X["LoanAmount"],c='black') 
    plt.scatter(Centroids["ApplicantIncome"],Centroids["LoanAmount"],c='red')
    plt.xlabel('AnnualIncome')
    plt.ylabel('Loan Amount (In Thousands)')
    plt.show()

def kmeans(Centroids=Centroids):
    #Centroids = (X.sample(n=K))
    # Initialize diff variable to 1 to enter the while loop
    diff = 1
    # Initialize iteration counter
    j = 0

    # Continue loop until centroids stop changing
    while diff != 0:
        # Store current centroids in XD
        XD = X.copy()
        # Initialize index counter
        i = 1
        # Loop over centroids
        for index1, row_c in Centroids.iterrows():
            # Initialize list to store Euclidean distances
            ED = []
            # Loop over data points
            for index2, row_d in XD.iterrows():
                # Calculate Euclidean distance between centroid and data point
                d1 = (row_c["ApplicantIncome"] - row_d["ApplicantIncome"])**2
                d2 = (row_c["LoanAmount"] - row_d["LoanAmount"])**2
                d = np.sqrt(d1 + d2)
                # Append distance to list
                ED.append(d)
            # Update corresponding row in X with distances
            X.loc[:, i] = ED
            # Increment index counter
            i += 1

        # Initialize list to store cluster assignments
        C = []
        # Loop over data points
        for index, row in X.iterrows():
            # Initialize minimum distance and cluster position
            min_dist = row[1]
            pos = 1
            # Find nearest centroid and assign cluster
            for k in range(K):
                if row[k+1] < min_dist:
                    min_dist = row[k+1]
                    pos = k+1
            # Append cluster assignment to list
            C.append(pos)
        # Assign cluster assignments to X
        X["Cluster"] = C

        # Calculate new centroids based on cluster assignments
        Centroids_new = X.groupby(["Cluster"]).mean()[["LoanAmount", "ApplicantIncome"]]
        diff_sum = []
        # Check if it's the first iteration
        if j == 0:
            # Set diff to 1 to continue loop
            diff = 1
            # Increment iteration counter
            j += 1
        else:
            # Calculate difference between new and old centroids
            diff = (Centroids_new['LoanAmount'] - Centroids['LoanAmount']).sum() + \
               (Centroids_new['ApplicantIncome'] - Centroids['ApplicantIncome']).sum()
            # Print sum of differences
            print(diff.sum())
            diff_sum.append(diff.sum())

        # Update centroids for next iteration
        Centroids = X.groupby(["Cluster"]).mean()[["LoanAmount", "ApplicantIncome"]]

    color=['blue','green','cyan']
    for k in range(K):
        data=X[X["Cluster"]==k+1]
        plt.scatter(data["ApplicantIncome"],data["LoanAmount"],c=color[k])
    plt.scatter(Centroids["ApplicantIncome"],Centroids["LoanAmount"],c='red')
    plt.xlabel('Income')
    plt.ylabel('Loan Amount (In Thousands)')
    plt.show()


# display GUI
root = Tk()
root.title("K-Means Clustering")
canvas = Canvas(root, width=300, height=100)
canvas.pack()
canvas.create_text(100, 10, text="K-Means Clustering", font=("Arial", 14), fill="black")

description_label = Label(root, text="Data Description", font=("Arial", 14, "bold"))
description_label.pack(pady=5)

avgIncome = np.mean(X["ApplicantIncome"])
avgLoan = np.mean(X["LoanAmount"])
stdDevIncome = np.std(X['ApplicantIncome'])
stdDevLoan = np.std(X["LoanAmount"])
maxIncome = np.max(X["ApplicantIncome"])
minIncome = np.min(X["ApplicantIncome"])
maxLoan = np.max(X['LoanAmount'])
minLoan = np.min(X['LoanAmount'])

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

visualizeScatterMatrixButton = Button(root, text="Show Scatter Matrix", command=visualize_scatter_matrix)
visualizeScatterMatrixButton.pack()

visualizeScatterPlotRawButton = Button(root, text="Show Raw Data Plot", command=visualize_scatter_plot_raw_data)
visualizeScatterPlotRawButton.pack()

displayInitialCentroidsButton = Button(root, text="Show Initial Centroids", command=initial_centroids)
displayInitialCentroidsButton.pack()

buttonClustering = Button(root, text="Analyze Cluster", command=kmeans)
buttonClustering.pack()

root.mainloop()
