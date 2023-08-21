# Import necessary libraries
import matplotlib.pyplot as plt  # For creating plots
from kmeans import KMeans  # Custom KMeans implementation (assuming this is a file named 'kmeans.py')
import pandas as pd  # For handling data in a tabular format

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('cluster.csv')

# Extract the data from the DataFrame and store it in a NumPy array
X = df.iloc[:, :].values

# Create a KMeans instance with specified settings
km = KMeans(n_clusters=4, max_iter=5000)

# Perform KMeans clustering on the data and obtain cluster assignments
y_means = km.fit_predict(X)

# Create a scatter plot to visualize the clustered data
plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], color='red')  # Scatter plot for cluster 0 points
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], color='blue')  # Scatter plot for cluster 1 points
plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], color='green')  # Scatter plot for cluster 2 points
plt.scatter(X[y_means == 3, 0], X[y_means == 3, 1], color='yellow')  # Scatter plot for cluster 3 points

# Display the plot
plt.show()
