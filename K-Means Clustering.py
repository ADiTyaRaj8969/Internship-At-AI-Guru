from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Sample Data 
X = np.array([[1,2],[1,4],[1,0],[4,2],[4,4],[4,0]])

# Apply K-Means
kmeans = KMeans(n_clusters = 2, random_state = 0).fit(X)

# Get Cluster Labels
labels = kmeans.labels_

# Get Cluster centers
centroids = kmeans.cluster_centers_

# Visualize
plt.scatter(X[:,0],X[:,1], c =labels)
plt.scatter(centroids[:,0],centroids[:,1], marker ='x',color='red')
plt.title("K-Means Clustering" )
plt.show()
