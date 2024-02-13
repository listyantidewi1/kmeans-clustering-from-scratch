# K-Means Clustering Algorithm

## Description
This code implements the K-means clustering algorithm from scratch in Python. K-means is a popular unsupervised machine learning algorithm used for clustering data points into groups or clusters based on similarity. The algorithm iteratively assigns data points to the nearest centroid and updates the centroids until convergence. 

## Code Overview
- The code defines a `KMeans` class with methods to fit the model to data (`fit`), visualize the clustering result (`visualize`), and display the result in a Tkinter GUI (`display_gui`).
- The `KMeans` class accepts parameters such as the number of clusters (`k`), maximum iterations per initialization (`max_iter`), and number of initializations (`n_init`).
- In the `fit` method, the algorithm performs multiple initializations and selects the best clustering based on the lowest total within-cluster variance.
- The `visualize` method uses matplotlib to visualize the clustering result by plotting data points and centroids.
- The `display_gui` function creates a Tkinter GUI to display the clustering result with data points and centroids plotted on a canvas.

## K-Means Algorithm Steps
1. **Initialization**: Randomly initialize cluster centroids.
2. **Assignment Step**: Assign each data point to the nearest centroid.
3. **Update Step**: Update the centroids based on the mean of the data points assigned to each cluster.
4. **Convergence**: Iterate steps 2 and 3 until the centroids stop changing significantly between iterations or a maximum number of iterations is reached.

## Convergence
Convergence occurs when the algorithm reaches a stable solution, indicated by the centroids no longer changing significantly between iterations. It means that the algorithm has found a good clustering arrangement, and further iterations won't lead to significant improvements in the clustering result.

## Installation
To run this code, you need to have Python installed on your system. You can download and install Python from the [official Python website](https://www.python.org/downloads/).

Once Python is installed, you can install the required libraries using pip, the Python package manager. Open a terminal or command prompt and run the following commands:


### Install numpy
```bash
pip install numpy 

### Install matplotlib
```bash
pip install matplotlib

### Install TKinter
```bash
pip install tk
