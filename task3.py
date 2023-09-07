#!/usr/bin/env python
# coding: utf-8

# In[48]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
iris=pd.read_csv(r"C:\Users\pujit\Desktop\509\iris.csv")
iris


# In[49]:


iris=iris.drop(columns=['Id'])
iris


# In[50]:


iris.info()


# In[51]:


iris['Species'].value_counts()


# In[52]:


iris.isnull().sum()


# In[53]:


x=iris.drop('Species',axis=1)
x


# In[54]:


y=iris['Species']
y


# In[55]:


from sklearn.preprocessing import LabelEncoder
lab=LabelEncoder()
y=lab.fit_transform(y)
iris['Species']=y
iris


# In[56]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)


# In[57]:


x_test.shape


# In[58]:


x_train.shape


# In[59]:


from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(criterion='entropy')
dtc.fit(x_train,y_train)


# In[60]:


y_pred=dtc.predict(x_test)
y_pred


# In[61]:


from sklearn.metrics import accuracy_score


# In[62]:


print("Acurracy:",accuracy_score(y_pred,y_test)*100)


# In[63]:


x=[[5.8,3.1,5.2,1.8]]
y_pred=dtc.predict(x)
print("Iris-virginica",y_pred)


# In[64]:


x=[[5.1,3.5,1.4,0.2]]
y_pred=dtc.predict(x)
print(" Iris-setosa",y_pred)


# In[ ]:





# In[ ]:





# In[ ]:




