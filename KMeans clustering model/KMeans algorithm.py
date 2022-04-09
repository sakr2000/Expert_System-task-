# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 22:06:08 2022

@author: sakr
"""

# Required imports

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

#generating random isotropic blobs for clustering 

dataset = make_blobs(n_samples=200,centers=4,n_features=2,cluster_std=1.6,random_state=50)

points = dataset[0]

#ploting the blobs 

print(points)
plt.scatter(dataset[0][:,0], dataset[0][:,1])

#defining the kMeans algorithm and computing the clustering 

kmeans = KMeans(n_clusters=4)
kmeans.fit(points)


clusters = kmeans.cluster_centers_
print(clusters)

y_km = kmeans.fit_predict(points)
print(y_km)


plt.scatter(points[y_km==0,0],points[ y_km==0,1],color='red',s=30)
plt.scatter(points[y_km==1,0],points[ y_km==1,1],color='green',s=30)
plt.scatter(points[y_km==2,0],points[ y_km==2,1],color='blue',s=30)
plt.scatter(points[y_km==3,0],points[ y_km==3,1],color='cyan',s=30)

plt.scatter(clusters[0][0],clusters[0][1],marker='*',s=300,color='black')
plt.scatter(clusters[1][0],clusters[1][1],marker='*',s=300,color='black')
plt.scatter(clusters[2][0],clusters[2][1],marker='*',s=300,color='black')
plt.scatter(clusters[3][0],clusters[3][1],marker='*',s=300,color='black')





