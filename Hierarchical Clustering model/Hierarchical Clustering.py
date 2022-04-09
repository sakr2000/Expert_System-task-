# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 23:06:08 2022

@author: sakr
"""

# Required imports

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering as AC

#generating random isotropic blobs for clustering 

dataset = make_blobs(n_samples=300,centers=5,n_features=2,cluster_std=1.6,random_state=50)
data = dataset[0]

#ploting the blobs 

print(data)
plt.scatter(dataset[0][:,0], dataset[0][:,1])

# creating a dendrogram 

dendrogram = sch.dendrogram(sch.linkage(data,method='ward'))

#defining the hierarchy clustering algorithm and computing the clustering 

HC = AC(n_clusters=5,affinity='euclidean',linkage='ward')

y_hc = HC.fit_predict(data)
print(y_hc)


#polting the data

plt.scatter(data[y_hc==0,0],data[ y_hc==0,1],color='red',s=30)
plt.scatter(data[y_hc==1,0],data[ y_hc==1,1],color='green',s=30)
plt.scatter(data[y_hc==2,0],data[ y_hc==2,1],color='blue',s=30)
plt.scatter(data[y_hc==3,0],data[ y_hc==3,1],color='cyan',s=30)
plt.scatter(data[y_hc==4,0],data[ y_hc==4,1],color='brown',s=30)


