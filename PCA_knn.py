
# coding: utf-8

# In[69]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[62]:


#Shrink Dimension via PCA
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

data = pd.read_excel('C:/Users/chsjk/OneDrive/Documents/MATLAB/drtoolbox/실험결과.xlsx', 'Sheet1')
name = np.array(data.username)
label = np.array(data.label)
x = data.drop('username', axis = 1)
x = np.round(x.drop('label', axis = 1),3)


pca = PCA(n_components = 3)
x_p = np.round(pca.fit(x).transform(x),3)


# In[63]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[64]:


X_train, X_test, y_train, y_test = train_test_split(x_p, label, test_size = 0.33, random_state = 42)
estimator = KNeighborsClassifier(n_neighbors = 3)
estimator.fit(X_train, y_train)


# In[65]:


#Prediction
label_predict = estimator.predict(X_test)
print('Accuracy: %.9f'%accuracy_score(y_test, label_predict))


# In[66]:


#검증하기

target_names = np.array(['normal', 'abnormal'])
x = x_p
y = estimator.predict(x_p)


# In[68]:


plt.figure(figsize = (10,10))
colors = ['blue','red']
for color,i,target_name in zip(colors,[1,0],target_names):
    plt.scatter(x[y==i,1], x[y==i,0], color = color, label = target_name)
plt.legend() #도표 설명을 표시
plt.xlabel('bart_gain')
plt.ylabel('bart_success')
plt.show()


# ## TOL 자료만 이용해서 KNN으로 나눠보기

# In[84]:


data = pd.read_excel('C:/Users/chsjk/OneDrive/Documents/MATLAB/drtoolbox/TOLz.xlsx')
name = np.array(data.username)
label = np.array(data.label)
x = data.drop('username', axis = 1)
x = np.array(x.drop('label', axis = 1))
x = np.round(x,3)


# In[85]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(x, label, test_size = 0.33, random_state = 42)
estimator = KNeighborsClassifier(n_neighbors = 3)
estimator.fit(X_train, y_train)

label_predict = estimator.predict(X_test)
print('Accuracy: %.9f'%accuracy_score(y_test, label_predict))
y = estimator.predict(x)


# In[86]:


type(x)


# In[92]:


y = estimator.predict(x)

plt.figure(figsize = (10,10))
colors = ['blue','red']
for color,i,target_name in zip(colors,[1,0],target_names):
    plt.scatter(x[y==i,1], x[y==i,0], color = color, label = target_name)
plt.legend() #도표 설명을 표시
plt.xlabel('tol_step_m')
plt.ylabel('tol_rt_m_int')
plt.show()

