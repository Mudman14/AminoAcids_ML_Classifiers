#!/usr/bin/env python
# coding: utf-8

# In[179]:


import pandas as pd 
import numpy as np 
import sklearn 
from sklearn.model_selection import cross_validate 
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, precision_score, recall_score, f1_score
from sklearn import metrics
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt


# In[169]:


traindata = pd.read_excel('F:/Amino acid-classifier/classifier/training_set_20aa.xlsx')

X = traindata[['mean', 'std', 'skew', 'kurt', 'toff']]
y = traindata['label']
X = StandardScaler().fit_transform(X)
y = LabelEncoder().fit_transform(y)


# In[170]:


testdata = pd.read_excel('F:/Amino acid-classifier/classifier/testing_set_20aa.xlsx')
test_X = testdata.iloc[:, 0:5]
test_X = StandardScaler().fit_transform(test_X)
target = LabelEncoder().fit_transform(testdata['label'])


# In[175]:


model = RandomForestClassifier(n_estimators=200, random_state=42)


# In[176]:


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



# In[177]:


train_validation = cross_validation(model, X, y, 10)
test_validation = cross_validation(model, test_X, target, 10)


# In[178]:


print('train_validation_accuracy:', train_validation['Mean_Validation_Accuracy'])
print('test_validation_accuracy:', test_validation['Mean_Validation_Accuracy'])


# In[183]:


train_sizes = np.linspace(0.1, 1.0, 10)


train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=10, train_sizes=train_sizes, scoring='accuracy')

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)


# In[184]:


plt.figure()
plt.title('Learning Curve')
plt.xlabel('Training Examples')
plt.ylabel('Score')
plt.ylim(0.0, 1.1)
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color='r')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color='g')
plt.plot(train_sizes, train_scores_mean, 'o-', color='r',
         label='Training Score')
plt.plot(train_sizes, test_scores_mean, 'o-', color='g',
         label='Cross-Validation Score')
plt.legend(loc='best')
plt.show()


# In[ ]:




