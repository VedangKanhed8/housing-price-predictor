#!/usr/bin/env python
# coding: utf-8

# # Mini Project II - House Price Predictor
# 

# In[1]:


import pandas as pd


# In[2]:


housing = pd.read_csv("data.csv")


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing['CHAS'].value_counts()


# In[6]:


housing['AGE'].value_counts()


# In[7]:


housing.describe()


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


import matplotlib.pyplot as plt


# In[10]:


housing.hist(bins = 50 , figsize=(20,15))


# # Training and Testing Dataset Splitting

# In[11]:


import numpy as np
def split_dataset(data,test_ratio) :
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    print(shuffled)
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices] , data.iloc[test_indices]
    


# In[12]:


train_set, test_set = split_dataset(housing , 0.2)


# In[13]:


print(f"Rows in train set : {len(train_set)}\nRows in test set : {len(test_set)}\n")

from sklearn.model_selection import train_test_split
train_set , test_set = train_test_split(housing , test_size = 0.2 , random_state = 42)
print(f"Rows in train set : {len(train_set)}\nRows in test set : {len(test_set)}\n")
# In[14]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2 , random_state = 42)
for train_index, test_index in split.split(housing , housing['CHAS']) :
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[15]:


strat_test_set['CHAS'].value_counts()


# In[16]:


strat_train_set['CHAS'].value_counts()


# In[17]:


housing = strat_train_set.copy()


# # Looking for Correlations

# In[18]:


corr_matrix = housing.corr()


# In[19]:


corr_matrix['MEDV'].sort_values(ascending = False)


# In[20]:


from pandas.plotting import scatter_matrix
attributes = ["MEDV" , "RM" , "ZN" , "PTRATIO" , "LSTAT"]
scatter_matrix(housing[attributes] , figsize = (15,10))


# In[21]:


housing.plot(kind = "scatter", x = "RM" , y = "MEDV" , alpha = 0.9)


# In[22]:


housing["TAXRM"] = housing["TAX"] / housing["RM"]


# In[23]:


housing.head()


# In[24]:


housing.plot(kind = "scatter", x = "TAXRM" , y = "MEDV" , alpha = 0.9)


# In[25]:


housing = strat_train_set.drop("MEDV" , axis = 1)
housing_labels = strat_train_set["MEDV"].copy()


# In[26]:


housing.describe() 


# # PipeLining

# In[27]:


from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer' , SimpleImputer(strategy = "median")) , ('std_scaler',StandardScaler()),
])


# In[28]:


housing_num_tr = my_pipeline.fit_transform(housing)


# In[29]:


housing_num_tr


# # Selecting a desired Model

# In[30]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
#model = LinearRegression()
#model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr , housing_labels)


# In[31]:


some_data = housing.iloc[:5]


# In[32]:


some_labels = housing_labels.iloc[:5]


# In[33]:


prepared_data = my_pipeline.transform(some_data)


# In[34]:


model.predict(prepared_data)


# In[35]:


list(some_labels)


# # Evaluating Model

# In[36]:


from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels,housing_predictions)
rmse = np.sqrt(mse)


# In[37]:


rmse # overfitting


# # Cross Validation Technique to overcome Overfitting

# In[38]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model , housing_num_tr , housing_labels , scoring = "neg_mean_squared_error" , cv = 10)
rmse_scores = np.sqrt(-scores)


# In[39]:


rmse_scores


# In[40]:


def print_scores(scores) :
    print("Scores : ", scores)
    print("Mean : " , scores.mean())
    print("Std. Deviation : ", scores.std())


# In[41]:


print_scores(rmse_scores)


# # Saving the Model

# In[42]:


from joblib import dump, load
dump(model,'house_price_predictor.joblib')


# # Testing the Model

# In[43]:


X_test = strat_test_set.drop("MEDV" , axis = 1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test , final_predictions)
final_rmse = np.sqrt(final_mse)

print(final_predictions,list(Y_test))


# In[44]:


final_rmse


# In[45]:


print(X_test_prepared[0])


# # Using The Model

# In[46]:

from joblib import dump , load
import numpy as np

model = load('house_price_predictor.joblib')
features = np.array([[-0.44228927, -0.4898311,  -1.37640684, -0.27288841, -0.34321545,  0.36738857,
 -0.33092752,  1.20235683, -1.0016859,   0.05733231, -1.21003475,  0.38110555,
 -0.57309194]
])

model.predict(features)


# In[ ]:




