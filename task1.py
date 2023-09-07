#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[37]:


data=pd.read_csv(r"C:\Users\pujit\Desktop\509\kc.csv")
data


# In[38]:


features=['sqft_living','bedrooms']


# In[39]:


x=data[features]
y=data['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)


# In[40]:


plt.scatter(x['sqft_living'],y)
plt.xlabel('Square foot')
plt.ylabel('Price')
plt.title('Relationship between square foot and Price')
plt.show()


# In[41]:


x_train=pd.DataFrame(x_train,columns=features)
x_test=pd.DataFrame(x_test,columns=features)


# In[42]:


model=LinearRegression()


# In[43]:


model.fit(x_train,y_train)


# In[46]:


y_pred=model.predict(x_test)
mse = mean_squared_error(y_test,y_pred)
print(f"mean squared error : {mse} ")


# In[47]:


new=[[2000,4]]
predicted_price=model.predict(new)
print(f"Predicted Price for New House :{predicted_price}")


# In[ ]:





# In[ ]:





# In[ ]:




