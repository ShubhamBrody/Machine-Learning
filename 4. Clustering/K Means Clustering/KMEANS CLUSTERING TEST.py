import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [3,4]].values

from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show() 


kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
Y_kmeans = kmeans.fit_predict(X)
print(Y_kmeans)


plt.scatter(X[Y_kmeans == 0,0], X[Y_kmeans == 0,1], s = 100, c = 'red', label='CLUSTER 1')
plt.scatter(X[Y_kmeans == 1,0], X[Y_kmeans == 1,1], s = 100, c = 'blue', label='CLUSTER 2')
plt.scatter(X[Y_kmeans == 2,0], X[Y_kmeans == 2,1], s = 100, c = 'green', label='CLUSTER 3')
plt.scatter(X[Y_kmeans == 3,0], X[Y_kmeans == 3,1], s = 100, c = 'cyan', label='CLUSTER 4')
plt.scatter(X[Y_kmeans == 4,0], X[Y_kmeans == 4,1], s = 100, c = 'orange', label='CLUSTER 5')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300, c = 'yellow', label='CENTROID')
plt.title('Clusters of customers')
plt.xlabel('Annual Income(in Thousands)')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()
