# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 15:21:00 2023

@author: supremcourt
"""
'''
Business Objective:

maximize: A composite measure of wine quality, potentially derived from expert ratings or sensory evaluations.

minimize: production cost

constrains: consistency of Brand Image

'''
'''
Data Dictionary  : 
    
    
                              
'Type':Nominal Columns:
Description: Nominal variable representing the type or class of wine.
Categories: e.g., 1, 2, 3, or 'Red', 'White', 'Rose'.


'Alcohol':Ordinal Column:
Description: Ordinal variable representing the alcohol content of the wine.
Order: Increasing alcohol content.

'Malic':Ordinal Column
Description: Ordinal variable representing the malic acid content.
Order: Increasing malic acid content.

'Ash':Ordinal Column
Description: Ordinal variable representing the ash content.
Order: Increasing ash content.

'Alcalinity':Ordinal Column
Description: Ordinal variable representing the alkalinity of the wine.
Order: Increasing alkalinity.

'Magnesium':Ordinal Column
Description: Ordinal variable representing the magnesium content.
Order: Increasing magnesium content.

'Phenols':Ordinal Column
Description: Ordinal variable representing the total phenols content.
Order: Increasing total phenols content.

'Flavanoids':Ordinal Column
Description: Ordinal variable representing the flavonoids content.
Order: Increasing flavonoids content.

'Nonflavanoids':Ordinal Column
Description: Ordinal variable representing the non-flavanoids content.
Order: Increasing non-flavanoids content.

'Proanthocyanins':Ordinal Column
Description: Ordinal variable representing the proanthocyanins content.
Order: Increasing proanthocyanins content.

'Color':Ordinal Column
Description: Ordinal variable representing the color intensity.
Order: Increasing color intensity.

'Hue':Ordinal Column
Description: Ordinal variable representing the hue of the wine.
Order: Increasing hue.

'Dilution':Ordinal Column
Description: Ordinal variable representing the dilution factor.
Order: Increasing dilution factor.

'Proline':Ordinal Column
Description: Ordinal variable representing the proline content.
Order: Increasing proline content.

'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
alco=pd.read_csv("E:/datascience/Association_Rule/wine.csv")
alco.shape
#data set having 178 row and 14 columns
#############################################################


alco.columns
#columns are present in the dataset 'Type',
# 'Alcohol', 'Malic', 'Ash', 'Alcalinity', 'Magnesium',
    #'Phenols','Flavanoids', 'Nonflavanoids', 'Proanthocyanins', 'Color', 
    #'Hue','Dilution', 'Proline'
################################################################



alco.info()
#there is no any null value is present
#type, Magnesium and  Proline columns datatype is int and 
#other data column data type is  float
###############################################################


alco.describe()
#there total 3 type of wine
#the Alcohol is present is 14.8 in high and low is 11.03
#the Dilution is the 4 max and 1.27 min 
#the proline is the std is 314.90 and mean is 746.89
#hue is the 0.95 mean and 0.22 std
#############################################################


sns.boxplot(alco[['Alcohol', 'Malic', 'Ash', 'Alcalinity', 'Magnesium']])
#In above columnn Malic,Ash,Alcalinity,Magnesium having outlier
#remove outlier by winsolizer or masking or triming
#############################################################


#remove outlier by winsolizzer on Malic columns
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr', tail='both',fold=1.4,variables=['Malic'])
df1=winsor.fit_transform(alco[['Malic']]).astype(int)
alco['Malic']=df1#outlier can be removed
#let check removed or not
sns.boxplot(alco[['Alcohol', 'Malic', 'Ash', 'Alcalinity', 'Magnesium']])
##############################################################



#remove outlier by winsolizzer on Ash columns
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr', tail='both',fold=1.4,variables=['Ash'])
df2=winsor.fit_transform(alco[['Ash']]).astype(int)
alco['Ash']=df2#outlier can be removed
#let check removed or not
sns.boxplot(alco[['Alcohol', 'Malic', 'Ash', 'Alcalinity', 'Magnesium']])
 #############################################################   
#apply masking to  Ash
import numpy as np
IQR=alco.Ash.quantile(0.75) - alco.Ash.quantile(0.25)
IQR
lower_limit=alco.Ash.quantile(0.25) - 1.5 * IQR
upper_limit=alco.Ash.quantile(0.75) + 1.5 * IQR
outlier=np.where(alco.Ash > upper_limit,upper_limit,np.where(alco.Ash < lower_limit ,lower_limit,alco.Ash))
alco.Ash=outlier
#let check removed or not
sns.boxplot(alco.Ash)
 ##############################################################   
#remove outlier by winsolizzer on Alcalinity columns
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr', tail='both',fold=1.4,variables=['Alcalinity'])
df2=winsor.fit_transform(alco[['Alcalinity']]).astype(int)
alco['Alcalinity']=df2#outlier can be removed
#let check removed or not
sns.boxplot(alco[['Alcohol', 'Malic', 'Ash', 'Alcalinity', 'Magnesium']])    
 ###########################################################3   
#remove outlier by winsolizzer on Alcalinity columns
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr', tail='both',fold=1.4,variables=['Magnesium'])
df2=winsor.fit_transform(alco[['Magnesium']]).astype(int)
alco['Magnesium']=df2#outlier can be removed
#let check removed or not
sns.boxplot(alco[['Alcohol', 'Malic', 'Ash', 'Alcalinity', 'Magnesium']])    
#above all column have removed outliers
#############################################################
sns.boxplot(alco[['Phenols','Flavanoids','Nonflavanoids','Proanthocyanins','Color','Hue','Dilution','Proline']])
#'Proanthocyanins','Color','Hue' having a outliers
#remove outlier by winsolizzer on Proanthocyanins columns
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr', tail='both',fold=1.4,variables=['Proanthocyanins'])
df2=winsor.fit_transform(alco[['Proanthocyanins']]).astype(int)
alco['Proanthocyanins']=df2#outlier can be removed
#let check removed or not
sns.boxplot(alco[['Proanthocyanins','Color','Hue']])


##############################################################
#apply masking to  Proanthocyanins
import numpy as np
IQR=alco.Proanthocyanins.quantile(0.75) - alco.Proanthocyanins.quantile(0.25)
IQR
lower_limit=alco.Proanthocyanins.quantile(0.25) - 1.5 * IQR
upper_limit=alco.Proanthocyanins.quantile(0.75) + 1.5 * IQR
outlier=np.where(alco.Proanthocyanins > upper_limit,upper_limit,np.where(alco.Proanthocyanins < lower_limit ,lower_limit,alco.Proanthocyanins))
alco.Proanthocyanins=outlier
#let check removed or not
sns.boxplot(alco.Proanthocyanins)
##############################################################3

#remove outlier by winsolizzer on Color columns
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr', tail='both',fold=1.4,variables=['Color'])
df2=winsor.fit_transform(alco[['Color']]).astype(int)
alco['Color']=df2#outlier can be removed
#let check removed or not
sns.boxplot(alco[['Proanthocyanins','Color','Hue']])
##############################################################



#remove outlier by winsolizzer on Hue columns
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr', tail='both',fold=1.4,variables=['Hue'])
df2=winsor.fit_transform(alco[['Hue']]).astype(int)
alco['Hue']=df2#outlier can be removed
#let check removed or not
sns.boxplot(alco[['Proanthocyanins','Color','Hue']])

sns.boxplot(alco)
#all columns removed the outlier by useing maskinmg or winsolizer

#the data having some ordinal value let convert it into dummie value
df1=pd.get_dummies(alco)
#apply normalization on data
def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

#apply norm funtion to dataset
df_norm=norm_fun(df1)

#data is converted
df_norm.drop(['Ash','Proanthocyanins'],axis=1,inplace=True)

#############################################################


#let apply the clustering algoritham 


from sklearn.cluster import KMeans
TWSS=[]
k=list(range(2,8))
k
#we donts know the k value os the cluster that way 
#we aplying all the possible number in the range of 2 to 8 in TWWS
#and plot the fig and understand the k value
for i in k:
    kmean=KMeans(n_clusters=i).fit(df_norm)
    TWSS.append(kmean.inertia_)
TWSS   
plt.plot(k,TWSS,'-ro')
plt.plot(k, TWSS,'ro-')
plt.xlabel("no of cluster")
plt.ylabel("total within ss")

#from the fig we understant the cluster value k=3
model=KMeans(n_clusters=3).fit(df_norm).labels_
mb=pd.Series(model)
#add one columns in original dataset and add new column of cluster
alco['clust']=mb

#---------------- PCA ------------------------------
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
X_sclaer=scaler.fit_transform(df_norm)
X_sclaer


pca=PCA(n_components=6)
pca_values=pca.fit_transform(X_sclaer)

pca_values.shape
#178 and 6 columns

#convert new feture into datafream
pca_data=pd.DataFrame(pca_values)
#give new name to datafream feture
pca_data.columns='comp0','comp1','comp2','comp3','comp4','comp5'
pca_data
#-----------------------------------------













