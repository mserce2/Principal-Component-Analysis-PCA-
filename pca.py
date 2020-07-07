# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 02:25:29 2020

@author: Mete
"""

#sklearn kütüphanesi içindeki hazır datasetlerden birini kullanacağız

from sklearn.datasets import load_iris
import pandas as pd

# %%
iris = load_iris()

data = iris.data
feature_names = iris.feature_names
y = iris.target

df = pd.DataFrame(data,columns = feature_names)
df["sinif"] = y

x = data

#%% PCA
from sklearn.decomposition import PCA

#datamızda birden çok feature olduğu için bunu iki boyuta indiriyoruz(pca yapmamızun amacı bu)
pca = PCA(n_components = 2, whiten= True )  # whitten = normalize
pca.fit(x) #x,y dememezin sebebi label ile işimiz yok sadece boyut azaltma işlemi yapıyoruz

x_pca = pca.transform(x) #bu metod ile 4 boyutlu olan datımızı 2 boyutlu dataya çevirdik

print("variance ratio: ", pca.explained_variance_ratio_) #first ve second companentleri ayırma işlemi yapıyoruz pca.png resmindeki gibi

#datadaki boyut düşürme işleminden sonra varyans düzgün mü bakıyoruz aksi halde datalar bozulabilir
print("sum: ",sum(pca.explained_variance_ratio_))

#%%2d


#Bura da p1 p2 companentleri ayarlıyoruz
df["p1"] = x_pca[:,0]
df["p2"] = x_pca[:,1]

color = ["red","green","blue"]

import matplotlib.pyplot as plt
for each in range(3):
    plt.scatter(df.p1[df.sinif == each],df.p2[df.sinif == each],color = color[each],label = iris.target_names[each])
    
plt.legend()
plt.xlabel("p1")
plt.ylabel("p2")
plt.show()












