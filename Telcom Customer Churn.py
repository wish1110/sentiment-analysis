#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


# In[8]:


df = pd.read_csv("telcom customer churn - Telco_Churn.csv")


# In[9]:


df.head()


# In[10]:


df.drop(['Churn Label','Churn Score', 'CLTV','Churn Reason'],axis=1, inplace=True)
df.head()


# In[11]:


df['Count'].unique()


# In[12]:


df['Country'].unique()


# In[13]:


df['State'].unique()


# In[14]:


df.drop(['Count','Country','CustomerID','State','Lat Long'], axis=1, inplace=True)


# In[15]:


df['City'].replace(' ','_',regex=True,inplace=True)


# In[16]:


df['City'].unique()[0:10]


# In[17]:


df.columns = df.columns.str.replace(' ','_')


# In[18]:


df.dtypes


# In[19]:


df['Phone_Service'].unique()


# In[20]:


df['Total_Charges'].unique()


# In[21]:


len(df.loc[df['Total_Charges'] == ' '])


# In[22]:


df.loc[df['Total_Charges']== ' ']


# In[23]:


df.loc[(df['Total_Charges'] == ' '), 'Total_Charges'] =0


# In[24]:


df.loc[df['Tenure_Months'] == 0]


# In[25]:


df['Total_Charges'] = pd.to_numeric(df['Total_Charges'])
df.dtypes


# In[26]:


df.replace(' ','_',regex=True, inplace =True)
df.head()


# In[27]:


X = df.drop('Churn_Value', axis = 1).copy()
X.head()


# In[28]:


y = df['Churn_Value'].copy()


# In[29]:


pd.get_dummies(X, columns=['Payment_Method']).head()


# In[30]:


X_encoded = pd.get_dummies(X, columns=[
    'City',
    'Gender',
    'Senior_Citizen',
    'Partner',
    'Dependents',
    'Phone_Service',
    'Multiple_Lines',
    'Internet_Service',
    'Online_Security',
    'Online_Backup',
    'Device_Protection',
    'Tech_Support',
    'Streaming_TV',
    'Streaming_Movies',
    'Contract',
    'Paperless_Billing',
    'Payment_Method'
])

X_encoded.head()


# In[31]:


y.unique()


# In[32]:


sum(y)/len(y)


# In[33]:


X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=42, stratify=y)


# In[34]:


sum(y_train)/len(y_train)


# In[35]:


sum(y_test)/len(y_test)


# In[36]:


clf_xgb = xgb.XGBClassifier(objective='binary:logistic', seed=42)
clf_xgb.fit(X_train,
           y_train,
)


# In[37]:


predictions = clf_xgb.predict(X_test)


# In[38]:


predictions


# In[39]:


accuracy_score(y_test,predictions)


# In[40]:


balanced_accuracy_score(y_test,predictions)


# In[41]:


ConfusionMatrixDisplay.from_estimator(clf_xgb, X_test, y_test, display_labels=['Did not leave', 'Left'])


# In[42]:


f1_score(y_test, predictions, average='macro')


# In[43]:


roc_auc_score(y_test,predictions)


# From the confusion matrix above, we can get the true postitive 245 , true negative is 1143, false negative(type 2 error) is 151 and false positive(type 1 error) is 222. 
# 
# Sensitivity: True Positive/(True Positive+False Negative) = 245/(245+151) = 61.8%
# Specificity : True Negative/(True Negative+False Positive) = 1143/(1143+222)= 83.7%
# 
# Recall:
# Precision:
# 
# F-score:
# 

# In[44]:


clf_xgb = xgb.XGBClassifier(seed=42,
                           objective='binary:logistic',
                           gamma=0.25,
                           learn_rate=0.1,
                           max_depth=4,
                           reg_lambda=10,
                           scale_pos_weight=3,
                           susample=0.9,
                           colsample_bytree=0.5)
clf_xgb


# In[45]:


clf_xgb.fit(X_train, 
            y_train)


# In[46]:


ConfusionMatrixDisplay.from_estimator(clf_xgb, X_test, y_test, display_labels=['Did not leave', 'Left'])


# From the confusion matrix above, we can get the true postitive 364 , true negative is 974, false negative(type 2 error) is 103 and false positive(type 1 error) is 320. 
# 
# Sensitivity: True Positive/(True Positive+False Negative) = 364/(364+103) = 77.9%
# Specificity : True Negative/(True Negative+False Positive) = 974/(974+320)= 75.3%
# 
# Recall:
# Precision:
# 
# F-score:

# In[ ]:




