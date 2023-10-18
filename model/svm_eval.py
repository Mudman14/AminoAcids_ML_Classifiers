#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd 
import sklearn 
from sklearn.model_selection import cross_validate 
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.svm import SVC 


# In[19]:


traindata = pd.read_excel('F:/Amino acid-classifier/classifier/training_set_20aa.xlsx')

X = traindata[['mean', 'std', 'skew', 'kurt', 'toff']]
y = traindata['label']
X = StandardScaler().fit_transform(X)
y = LabelEncoder().fit_transform(y)


# In[20]:


testdata = pd.read_excel('F:/Amino acid-classifier/classifier/testing_set_20aa.xlsx')
test_X = testdata.iloc[:, 0:5]
test_X = StandardScaler().fit_transform(test_X)
target = LabelEncoder().fit_transform(testdata['label'])


# In[29]:


model = SVC(kernel='linear', degree=2, decision_function_shape='ovo')


# In[30]:


def cross_validation(model, _X, _y, _cv=10):

      _scoring = ['accuracy', 'precision', 'recall', 'f1','roc_auc']
      results = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring=_scoring,
                               return_train_score=True)
        
    
      
      return {
              "Validation Accuracy scores": results['test_accuracy'],
              "Mean_Validation_Accuracy": results['test_accuracy'].mean(),

              
              }


# In[31]:


train_validation = cross_validation(model, X, y, 10)
test_validation = cross_validation(model, test_X, target, 10)



# In[32]:


print('train_validation_accuracy:', train_validation['Mean_Validation_Accuracy'])
print('test_validation_accuracy:', test_validation['Mean_Validation_Accuracy'])


# In[ ]:





# In[ ]:




