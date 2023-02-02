#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt


# In[20]:


frame = pd.read_csv("stage2_woman_men_feature_tidy.csv")
X = np.array(frame["intensity"]).reshape(-1, 1)
Y = np.array(frame["class"]).reshape(-1, 1)


# In[21]:


random_state=85
# Split Data into Train (2/3) and Test (1/3)
DataTrain, DataTest, YTrain, YTest = train_test_split(X, Y, test_size=1/3)


# In[17]:


from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier


# In[22]:


clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    max_depth=1, random_state=0).fit(DataTrain, YTrain)
clf.score(DataTest, YTest)


# In[24]:


y_pred = clf.predict(DataTest)
confusion_matrix(y_pred, YTest)


# In[23]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Create the model 
model = LinearDiscriminantAnalysis()

# Fit on training data
model.fit(DataTrain, YTrain)

#Test 
model.score(DataTest, YTest)

