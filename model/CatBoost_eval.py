#!/usr/bin/env python
# coding: utf-8

# In[4]:


from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import LabelEncoder
import pandas as pd 
import numpy as np 
from sklearn.model_selection import cross_validate 
from sklearn import metrics 
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score 
import warnings 
warnings.filterwarnings('ignore') 
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix
from catboost import CatBoostClassifier


# In[5]:


traindata = pd.read_excel('F:/Amino acid-classifier/classifier/training_set_20aa.xlsx')
X = traindata[['mean', 'std', 'skew', 'kurt', 'toff']]
y = traindata['label']

X = np.array(X)
X = StandardScaler().fit_transform(X)
y = LabelEncoder().fit_transform(y)


# In[6]:


testdata = pd.read_excel('F:/Amino acid-classifier/classifier/testing_set_20aa.xlsx')
test_X = testdata.iloc[:, 0:5]
test_X = StandardScaler().fit_transform(test_X)
target = LabelEncoder().fit_transform(testdata['label'])


# In[7]:


model = CatBoostClassifier(depth=6,
                          learning_rate=0.1)


# In[9]:


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


# In[10]:


train_validation = cross_validation(model, X, y, 10)
test_validation = cross_validation(model, test_X, target, 10)


# In[11]:


print('train_validation_accuracy:', train_validation['Mean_Validation_Accuracy'])
print('test_validation_accuracy:', test_validation['Mean_Validation_Accuracy'])


# In[ ]:




