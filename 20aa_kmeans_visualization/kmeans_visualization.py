#!/usr/bin/env python
# coding: utf-8

# In[68]:


import pandas as pd
import numpy as np
import sklearn 
import sklearn 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans


# In[69]:


data = pd.read_excel('F:/Amino acid-classifier/classifier/training_set_20aa.xlsx')

data


# In[72]:


x_data = data.iloc[:, 0:5]
pca = PCA(n_components=2, random_state=42)
pca.fit(x_data)
feature_names = x_data.columns


# In[73]:


components = pca.components_
for i, comp in enumerate(components):
  print(f"主成分 {i+1} 最重要的变量:")
  print({feature_names[idx]: weight  
        for idx, weight in enumerate(comp) if abs(weight) > 0.5})


# In[106]:


X = data[['mean', 'std']]

X = StandardScaler().fit_transform(X)


# In[107]:


kmeans = KMeans(n_clusters=20, init='k-means++', n_init=20, random_state=42)
y_pred = kmeans.fit_predict(X)


# In[109]:


x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predict the cluster labels for each point in the grid
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Create a contour plot of the predicted labels
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, alpha=0.8)
plt.xlabel('Delta I')
plt.ylabel('SD')
plt.show()


# In[110]:


kmeans.inertia_


# In[111]:


from sklearn.metrics import silhouette_score
silhouette_score(X, kmeans.labels_)


# In[90]:


#解释inertia，解释重叠部分


# In[ ]:




