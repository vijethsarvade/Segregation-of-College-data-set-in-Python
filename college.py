import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
coll=pd.read_csv("C:\\Users\\Prashanthi\\Desktop\\Python\\sir_working\\Algorithms\\Kmeans Clustering\\College.csv")
coll.columns
coll.shape
coll.shape[0]# no.rows
coll.shape[1]# no.columns
coll.info()
coll.describe()
#scatter plot of Room.board v/s Grad.Rate
sns.set_style('whitegrid')
sns.lmplot('Room.Board','Grad.Rate',data=coll, hue='Private',
           palette='coolwarm',size=6,aspect=1,fit_reg=False)
#scatter plot of F.undergrad v/s Outstate
sns.set_style('whitegrid')
sns.lmplot('F.Undergrad','Outstate',data=coll, hue='Private', #lenier model plot
           palette='coolwarm',size=6,aspect=1,fit_reg=False)
#histogram showing OutState Tuition based on the Private column
sns.set_style('whitegrid')
g = sns.FacetGrid(coll,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Outstate',bins=20,alpha=0.7)
#histogram for Grad.Rate column based on the Private column
sns.set_style('whitegrid')
g = sns.FacetGrid(coll,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Outstate',bins=20,alpha=0.7)
j = sns.FacetGrid(coll,hue="Private",palette='coolwarm',size=6,aspect=2)
j = j.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)

#error grade rate is 100% need to be adjusted
coll.loc[coll['Grad.Rate']>100,'Grad.Rate']=100
coll['Grad.Rate']['Cazenovia College']=100 #or
coll[coll['Grad.Rate']==100]
k = sns.FacetGrid(coll,hue="Private",palette='coolwarm',size=6,aspect=2)
k = j.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)

#Model building
from sklearn.cluster import KMeans
#K-means model with 2 clusters
kmeans = KMeans(n_clusters=2)
#Droping the categorical column
coll2=coll.drop(['Unnamed: 0','Private'],axis=1)
#Fiting the model
kmeans.fit(coll2)
#Gives centralized values
kmeans.cluster_centers_
#Gives the clusters
kmeans.labels_
#We can also visualize the label
coll['newlabel']=kmeans.labels_
coll.newlabel[coll['newlabel']==1]='Private'
coll.newlabel[coll['newlabel']==0]='NonCol Private'
coll3=coll[['newlabel''Private']]
sns.set_style('whitegrid')
sns.lmplot('Room.Board','Grad.Rate',data=coll, hue='newlabel',
           palette='coolwarm',size=6,aspect=1,fit_reg=False)



