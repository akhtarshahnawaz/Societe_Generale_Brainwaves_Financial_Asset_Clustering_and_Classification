import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
np.random.seed(0)


###########################################
#Loading Data
##########################################
train = pd.read_csv("../../data/train.csv")
train.drop(["Y","Time"], axis=1, inplace=True)

###########################################
#Finding Percentage Changes
##########################################
train = train.diff()/train
train = train.corr(method="kendall")

###########################################
#Clustering
##########################################
print "Clustering started"
clusterer = AffinityPropagation(max_iter=1000)
labels = clusterer.fit_predict(train)
print "Total Clusters", labels.max()

pd.DataFrame({"Asset":train.index,"Cluster":labels}).to_csv("csv/model1.csv", index=False)