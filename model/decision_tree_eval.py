#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np 
import sklearn 
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.model_selection import cross_validate 
from sklearn.tree import DecisionTreeClassifier 


# In[4]:


traindata = pd.read_excel('F:/Amino acid-classifier/classifier/training_set_20aa.xlsx')
X = traindata[['mean', 'std', 'skew', 'kurt', 'toff']]
y = traindata['label']

X = np.array(X)
X = StandardScaler().fit_transform(X)
y = LabelEncoder().fit_transform(y)


# In[5]:


testdata = pd.read_excel('F:/Amino acid-classifier/classifier/testing_set_20aa.xlsx')
test_X = testdata.iloc[:, 0:5]
test_X = StandardScaler().fit_transform(test_X)
target = LabelEncoder().fit_transform(testdata['label'])


# In[6]:


model = DecisionTreeClassifier(max_depth=20)


# In[13]:


def cross_validation(model, _X, _y, _cv=10):
      '''Function to perform 5 Folds Cross-Validation
       Parameters
       ----------
      model: Python Class, default=None
              This is the machine learning algorithm to be used for training.
      _X: array
           This is the matrix of features.
      _y: array
           This is the target variable.
      _cv: int, default=5
          Determines the number of folds for cross-validation.
       Returns
       -------
       The function returns a dictionary containing the metrics 'accuracy', 'precision',
       'recall', 'f1' for both training set and validation set.
      '''
      _scoring = ['accuracy', 'precision', 'recall', 'f1','roc_auc']
      results = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring=_scoring,
                               return_train_score=True)
        
        # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
      
      return {
              "Validation Accuracy scores": results['test_accuracy'],
              "Mean_Validation_Accuracy": results['test_accuracy'].mean(), 
          

              
              }


# In[14]:


train_validation = cross_validation(model, X, y, 10)
test_validation = cross_validation(model, test_X, target, 10)


# In[15]:


print('train_validation_accuracy:', train_validation['Mean_Validation_Accuracy'])
print('test_validation_accuracy:', test_validation['Mean_Validation_Accuracy'])


# In[ ]:




