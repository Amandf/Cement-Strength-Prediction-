#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy import stats


# In[2]:


df= pd.read_excel('Concrete_Data.xls')


# In[3]:


df.head()


# In[4]:


df.columns = ['cement', 'blastFurance', 'flyAsh', 'water', 'superplasticizer', 'courseAggregate', 'fineaggregate','age', 'strenght'] 


# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


df.isnull().sum()


# In[8]:


df.duplicated().sum()


# In[9]:


df.info()


# In[10]:


df.describe()


# In[11]:


corr = df.corr()
corr


# In[12]:


sns.heatmap(corr, annot=True, cbar=True)


# In[13]:


X = df.drop('strenght',axis=1)
y = df['strenght']


# In[14]:


X.shape


# In[15]:


y.shape


# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[18]:


for col in X_train.columns:
    plt.figure(figsize=(10,10))
    plt.subplot(121)
    sns.distplot(X_train[col])
    plt.title(col)
    plt.show()
    
    plt.figure(figsize=(10,10))
    plt.subplot(122)
    stats.probplot(X_train[col],dist='norm',plot=plt)
    plt.title(col)
    plt.show()


# In[19]:


from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer()


# In[20]:


X_train_transformed = pt.fit_transform(X_train)
X_test_transformed = pt.transform(X_test)


# In[ ]:





# In[21]:


for col in X_train.columns:
    plt.figure(figsize=(14,4))
    plt.subplot(121)
    sns.distplot(X_train[col])
    plt.title(col)
 
    
    

    plt.subplot(122)
    stats.probplot(X_train[col], dist='norm',plot=plt)
    plt.title(col)
    plt.show()


# In[22]:


X_train_transformed = pd.DataFrame(X_train_transformed, columns=X_train.columns)

for col in X_train_transformed.columns:
    plt.figure(figsize=(14,4))
    plt.subplot(121)
    sns.distplot(X_train[col])
    plt.title(col)
 
    
    

    plt.subplot(122)
    sns.distplot(X_train_transformed[col])
    plt.title(col)
    plt.show()


# In[23]:


from sklearn.preprocessing import StandardScaler


# In[24]:


scaler = StandardScaler()


# In[25]:


scaler.fit(X_train)


# In[26]:


X_train_transformed = scaler.transform(X_train)
X_test_transoformed = scaler.transform(X_test)


# In[27]:


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score


# In[28]:


get_ipython().system('pip install xgboost')


# In[29]:


models = {
    'lin_reg': LinearRegression(),
    'ridge': Ridge(),
    'lasso': Lasso(),
    'rf_reg': RandomForestRegressor(n_estimators=100, random_state=42),
    'xgb_reg': XGBRegressor(),
    
}


for name ,model in models.items():
    model.fit(X_train_transformed,y_train)
    y_pred = model.predict(X_test_transoformed)
    
    print(f"{name} : mse: {mean_squared_error(y_test,y_pred)}, r2 score : {r2_score(y_test,y_pred)}")


# In[ ]:





# In[30]:


xgb = XGBRegressor()
xgb.fit(X_train_transformed,y_train)
y_pred = xgb.predict(X_test_transoformed)
r2_score(y_test,y_pred)


# In[31]:


def predicion_system(cem,blastf,flyas,water,superplaster,courseagg,fineagg,age):
    features = np.array([[cem,blastf,flyas,water,superplaster,courseagg,fineagg,age]])
    prediction = xgb.predict(features).reshape(1,-1)
    
    return prediction[0]


# In[32]:


X_train


# In[ ]:





# In[33]:


cem = 158.60
blastf = 148.90
flyas = 116.00
water = 175.10
superplaster = 15.00
courseagg = 953.3
fineagg = 719.70
age = 28

prediction = predicion_system(cem,blastf,flyas,water,superplaster,courseagg,fineagg,age)
print("strength is : ",prediction)


# In[ ]:





# In[34]:


import pickle
pickle.dump(xgb,open('model.pkl','wb'))


# In[35]:


conda install tk 


# In[36]:


import numpy as np
import pickle
from xgboost import XGBRegressor
import tkinter as tk
from tkinter import ttk


# In[37]:


xgb = pickle.load(open('model.pkl', 'rb'))


# In[46]:


def prediction_system(cem, blastf, flyas, water, superplaster, courseagg, fineagg, age):
    features = np.array([[cem, blastf, flyas, water, superplaster, courseagg, fineagg, age]])
    print("Original Features:", features)
    features = pt.transform(features)  # Apply power transformation
    print("Transformed Features (Power):", features)
    features = scaler.transform(features)  # Apply scaling
    print("Transformed Features (Scaled):", features)
    prediction = xgb.predict(features)
    print("Prediction:", prediction)
    return prediction[0]


# In[47]:


def handle_prediction():
    try:
        values = [float(entries[label].get()) for label in labels]
        print("Input Values:", values)
        prediction = prediction_system(*values)
        result_label.config(text=f"Predicted Strength: {prediction:.2f}")
    except ValueError:
        result_label.config(text="Error: Please enter valid numbers")


# In[48]:


root = tk.Tk()
root.title("Concrete Strength Prediction")


# In[49]:


labels = ['Cement', 'Blast Furnace', 'Fly Ash', 'Water', 'Superplasticizer', 'Course Aggregate', 'Fine Aggregate', 'Age']
entries = {}

for i, label in enumerate(labels):
    lbl = ttk.Label(root, text=label)
    lbl.grid(row=i, column=0, padx=5, pady=5)
    entry = ttk.Entry(root)
    entry.grid(row=i, column=1, padx=5, pady=5)
    entries[label] = entry


# In[50]:


predict_button = ttk.Button(root, text="Predict", command=handle_prediction)
predict_button.grid(row=len(labels), column=0, columnspan=2, padx=5, pady=5)


# In[51]:


result_label = ttk.Label(root, text="Predicted Strength: N/A")
result_label.grid(row=len(labels)+1, column=0, columnspan=2, padx=5, pady=5)


# In[ ]:


root.mainloop()


# In[ ]:





# In[ ]:




