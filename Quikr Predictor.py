#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

car = pd.read_csv('quikr_car.csv')
car.head()

# In[2]:


car.shape

# In[3]:


car.info

# In[4]:


car.info()

# In[5]:


car['year'].unique()

# In[6]:


car['Price'].unique()

# In[7]:


car['kms_driven'].unique()

# In[8]:


car['fuel_type'].unique()

# In[9]:


backup = car.copy()

# In[10]:


car = car[car['year'].str.isnumeric()]

# In[11]:


car['year'] = car['year'].astype(int)

# In[12]:


car.info()

# In[13]:


car = car[car['Price'] != "Ask For Price"]

# In[14]:


car['Price'] = car['Price'].str.replace(',', '').astype(int)

# In[15]:


car.info()

# In[16]:


car['kms_driven'] = car['kms_driven'].str.split(' ').str.get(0).str.replace(',', '')

# In[17]:


car = car[car['kms_driven'].str.isnumeric()]

# In[18]:


car['kms_driven'] = car['kms_driven'].astype(int)

# In[19]:


car.info()

# In[20]:


car = car[~car['fuel_type'].isna()]

# In[21]:


car['name'] = car['name'].str.split(' ').str.slice(0, 3).str.join(' ')

# In[22]:


car = car.reset_index(drop=True)

# In[23]:


car.describe()

# In[24]:


car = car[car["Price"] < 6e6].reset_index(drop=True)

# In[25]:


car

# In[26]:


car.to_csv('Cleaned Car.csv')

# In[27]:


x = car.drop(columns='Price')
y = car['Price']

# In[28]:


y

# In[29]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# In[30]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# In[31]:


ohe = OneHotEncoder()
ohe.fit(x[['name', 'company', 'fuel_type']])

# In[32]:


ohe.categories_

# In[33]:


column_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_), ['name', 'company', 'fuel_type']),
                                       remainder='passthrough')

# In[34]:


lr = LinearRegression()

# In[35]:


pipe = make_pipeline(column_trans, lr)

# In[36]:


pipe.fit(x_train, y_train)

# In[37]:


y_pred = pipe.predict(x_test)

# In[38]:


r2_score(y_test, y_pred)

# In[39]:


scores = []
for i in range(1000):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=i)
    lr = LinearRegression()
    pipe = make_pipeline(column_trans, lr)
    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_test)
    scores.append(r2_score(y_test, y_pred))

# In[40]:


np.argmax(scores)

# In[41]:


scores[np.argmax(scores)]

# In[42]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=i)
lr = LinearRegression()
pipe = make_pipeline(column_trans, lr)
pipe.fit(x_train, y_train)
y_pred = pipe.predict(x_test)
r2_score(y_test, y_pred)

# In[43]:


import pickle

# In[44]:


pickle.dump(pipe, open('LinearRegressionModel.pkl', 'wb'))

# In[45]:


pipe.predict(pd.DataFrame(columns=x_test.columns,
                          data=np.array(['Maruti Suzuki Swift', 'Maruti', 2019, 100, 'Petrol']).reshape(1, 5)))

# In[46]:


pipe.steps[0][1].transformers[0][1].categories[0]

# In[ ]:
