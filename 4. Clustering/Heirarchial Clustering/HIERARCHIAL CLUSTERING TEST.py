import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [3,4]].values


import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.xlabel('Customers')
plt.ylabel('Eucledian Distance')
plt.show()


from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
Y_hc = hc.fit_predict(X)



plt.scatter(X[Y_hc == 0,0], X[Y_hc == 0,1], s = 100, c = 'red', label='CLUSTER 1')
plt.scatter(X[Y_hc == 1,0], X[Y_hc == 1,1], s = 100, c = 'blue', label='CLUSTER 2')
plt.scatter(X[Y_hc == 2,0], X[Y_hc == 2,1], s = 100, c = 'green', label='CLUSTER 3')
#plt.scatter(X[Y_hc == 3,0], X[Y_hc == 3,1], s = 100, c = 'cyan', label='CLUSTER 4')
#plt.scatter(X[Y_hc == 4,0], X[Y_hc == 4,1], s = 100, c = 'orange', label='CLUSTER 5')
#plt.scatter(hc.cluster_centers_[:,0], hc.cluster_centers_[:,1], s = 300, c = 'yellow', label='CENTROID')
plt.title('Clusters of customers')
plt.xlabel('Annual Income(in Thousands)')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()
