#!/usr/bin/env python
# coding: utf-8

# ![](https://www.santander.co.uk/themes/custom/santander_web18/logo.svg)
# 
# [image-source](https://www.santander.co.uk/themes/custom/santander_web18/logo.svg)

# ## Table of Contents
# 
# - [Problem Definition and Objectives](#intro)
# - [Exploratory Data Analysis](#EDA)
# - [Machine Learning Modeling](#ML)
#     - [Feature Engineering](#FE)
#     - [Decision Tree](#DT)
#     - [Logistic Regression](#LG)
#     - [Random Forest](#RD)
#     - [Navey Bayes](#NB)
#     - [Tuned Model Training](#tuned)
# - [Conclusion](#conclusion)

# ## Problem Definition and Objectives
# <a id="intro"></a>

# At Santander our mission is to help people and businesses prosper. We are always looking for ways to help our customers understand their financial health and identify which products and services might help them achieve their monetary goals.
# 
# Santander is continually challenging its machine learning algorithms, working with the global data science community to make sure we can more accurately identify new ways to solve our most common challenge, binary classification problems such as: is a customer satisfied? Will a customer buy this product? Can a customer pay this loan?
# 
# In this challenge, Kagglers are invited to help Santander identify which customers will make a specific transaction in the future, irrespective of the amount of money transacted. The data provided for this competition has the same structure as the real data we have available to solve this problem.
# 
# Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.
# 
# You are provided with an anonymized dataset containing numeric feature variables, the binary target column, and a string ID_code column.
# 
# The task is to predict the value of target column in the test set.

# ## Exploratory Data Analysis
# <a id="EDA"></a>
# 
# ![](http://blog.k2analytics.co.in/wp-content/uploads/2016/12/Exploratory_Data_Analysis.png)
# 
# [image-source](http://blog.k2analytics.co.in/wp-content/uploads/2016/12/Exploratory_Data_Analysis.png)

# In[2]:


import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import lightgbm as lgb
import os


# In[3]:


#set working directory-
os.chdir("F:\Edvisor Project\Santender_Project")

#check current working directory-
os.getcwd()


# In[4]:


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# In[5]:


train.head()


# In[6]:


train.info()


# In[7]:


train.columns


# In[8]:


train.shape


# In[9]:


train.describe()


# #### Missing Data Analysis

# In[10]:


miss_train = pd.DataFrame(train.isnull().sum())
np.transpose(miss_train)            #for better visibility and no missing values throughout


# In[11]:


miss_test = pd.DataFrame(test.isnull().sum())
np.transpose(miss_test)  


# #### Outlier Analysis

# In[12]:


#checking outliers using Chauvenet's criterion
def chauvenet(array):
    mean = array.mean()           # Mean of incoming array
    stdv = array.std()            # Standard deviation
    N = len(array)                # Lenght of incoming array
    criterion = 1.0/(2*N)         # Chauvenet's criterion
    d = abs(array-mean)/stdv      # Distance of a value to mean in stdv's
    prob = erfc(d)                # Area normal dist.    
    return prob < criterion       # Use boolean array outside this function


# In[13]:


numerical_features=train.columns[2:]


# In[14]:


from scipy.special import erfc
train_outliers = dict()
for col in [col for col in numerical_features]:
    train_outliers[col] = train[chauvenet(train[col].values)].shape[0]
train_outliers = pd.Series(train_outliers)

train_outliers.sort_values().plot(figsize=(14, 40), kind='barh').set_xlabel('Number of outliers');


# In[15]:


print('Total number of outliers in training set: {} ({:.2f}%)'.format(sum(train_outliers.values), (sum(train_outliers.values) / train.shape[0]) * 100))


# In[16]:


#outliers in each variable in test data 
test_outliers = dict()
for col in [col for col in numerical_features]:
    test_outliers[col] = test[chauvenet(test[col].values)].shape[0]
test_outliers = pd.Series(test_outliers)

test_outliers.sort_values().plot(figsize=(14, 40), kind='barh').set_xlabel('Number of outliers');


# In[17]:


print('Total number of outliers in testing set: {} ({:.2f}%)'.format(sum(test_outliers.values), (sum(test_outliers.values) / test.shape[0]) * 100))


# In[18]:


#remove these outliers in train and test data
for col in numerical_features:
    train=train.loc[(~chauvenet(train[col].values))]
for col in numerical_features:
    test=test.loc[(~chauvenet(test[col].values))]


# In[19]:


def plot_feature_distribution(df1, df2, label1, label2, features):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(10,10,figsize=(18,22))

    for feature in features:
        i += 1
        plt.subplot(10,10,i)
        sns.kdeplot(df1[feature], bw=0.5,label=label1)
        sns.kdeplot(df2[feature], bw=0.5,label=label2)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)
        plt.tick_params(axis='y', which='major', labelsize=6)
    plt.show()


# ##### Feature Distributions

# In[20]:


import seaborn as sns
t0 = train.loc[train['target'] == 0]
t1 = train.loc[train['target'] == 1]
features = train.columns.values[2:102]
plot_feature_distribution(t0, t1, '0', '1', features)


# Distribution of the mean values per row in the train and test set.

# In[21]:


# distribution of the mean values per row in the train and test set.
plt.figure(figsize=(16,6))
features = train.columns.values[2:202]
plt.title("Distribution of mean values per row in the train and test set")
sns.distplot(train[features].mean(axis=1),color="green", kde=True,bins=120, label='train')
sns.distplot(test[features].mean(axis=1),color="blue", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# distribution of the mean values per columns in the train and test set.

# In[22]:


#distribution of the mean values per columns in the train and test set.
    
plt.figure(figsize=(16,6))
plt.title("Distribution of mean values per column in the train and test set")
sns.distplot(train[features].mean(axis=0),color="magenta",kde=True,bins=120, label='train')
sns.distplot(test[features].mean(axis=0),color="darkblue", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# Distribution of standard deviation of values per row for train and test datasets.

# In[23]:


# distribution of standard deviation of values per row for train and test datasets.

plt.figure(figsize=(16,6))
plt.title("Distribution of std values per row in the train and test set")
sns.distplot(train[features].std(axis=1),color="black", kde=True,bins=120, label='train')
sns.distplot(test[features].std(axis=1),color="red", kde=True,bins=120, label='test')
plt.legend();plt.show()


# Distribution of the standard deviation of values per columns in the train and test datasets.

# In[24]:


# distribution of the standard deviation of values per columns in the train and test datasets.

plt.figure(figsize=(16,6))
plt.title("Distribution of std values per column in the train and test set")
sns.distplot(train[features].std(axis=0),color="blue",kde=True,bins=120, label='train')
sns.distplot(test[features].std(axis=0),color="green", kde=True,bins=120, label='test')
plt.legend(); plt.show()


# Distribution of skew per row in the train and test set.

# In[25]:


plt.figure(figsize=(16,6))
plt.title("Distribution of skew per row in the train and test set")
sns.distplot(train[features].skew(axis=1),color="red", kde=True,bins=120, label='train')
sns.distplot(test[features].skew(axis=1),color="orange", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# Distribution of skew per column in the train and test set

# In[26]:


plt.figure(figsize=(16,6))
plt.title("Distribution of skew per column in the train and test set")
sns.distplot(train[features].skew(axis=0),color="magenta", kde=True,bins=120, label='train')
sns.distplot(test[features].skew(axis=0),color="darkblue", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# In[27]:


correlations = train[features].corr().abs().unstack().sort_values(kind="quicksort").reset_index()
correlations = correlations[correlations['level_0'] != correlations['level_1']]
correlations.head(10)


# In[28]:


correlations.tail(10)


# In[29]:


import seaborn as sns
#count of both class(number of classes)
train['target'].value_counts()
sns.factorplot('target', data=train, kind='count')


# IN OUR CASE GIVEN DATA IS IMBALANCED……WHERE 90% OF SAMPLES BELONGS TO CLASS 0 AND ONLY 10% BELONGS TO CLASS 1

# In[30]:


train.shape


# ## Model Development

# ![](https://cmci.colorado.edu/classes/INFO-4604/fa17/wordcloud.png)
# [image-source](https://cmci.colorado.edu/classes/INFO-4604/fa17/wordcloud.png)

# ### Feature Engineering

# WE seperate the dataset whose target class is belong to class 0

# In[31]:


#WE seperate the dataset whose target class is belong to class 0
data=train.loc[train['target'] == 0]
#choose starting 30000 rows
data2=data.loc[:30000]
data2


# WE seperate the dataset whose target class is belong to class 1

# In[32]:


#WE seperate the dataset whose target class is belong to class 1
data1=train.loc[train['target'] == 1]
data1


# Add both Dataframe data1 and data2 in one dataframe

# In[33]:


#Add both Dataframe data1 and data2 in one dataframe
newdata=pd.concat([data1, data2], ignore_index=True)
newdata


# Shuffle the Dataframe

# In[34]:


#Shuffle the Dataframe
newdata=newdata.sample(frac=1)
newdata


# In[35]:


sns.factorplot('target', data=newdata, kind='count')


# After applying undersampling technique,we get the balanced data as shown above

# In[36]:


#import libraries
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import cross_val_score


# In[37]:


# Replace target variable categories with Yes or No
newdata['target'] = newdata['target'].replace(0, 'No')
newdata['target'] = newdata['target'].replace(1, 'Yes')

# Converted entries of 'target' variable from o/1 to No/Yes
newdata.head(5)


# In[38]:


# Divide data into train and test for model Development
X = newdata.values[:, 2:202]
Y = newdata.values[:,1]

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.2)


# #### Decision tree

# In[39]:


# Decision tree

C50_model = tree.DecisionTreeClassifier(criterion='entropy').fit(X_train, y_train)

#test scores

C50_predictions = C50_model.predict(X_test)
C50_predictions


# In[40]:


#confusion matrix

CM = pd.crosstab(y_test, C50_predictions)

#let check for belw values
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]

CM


# In[41]:


from sklearn.metrics import classification_report
print(classification_report(y_test,C50_predictions))

from sklearn import metrics

print('Accuracy:', metrics.accuracy_score(y_test, C50_predictions)*100)


# #### Random Forest

# In[42]:


# Random Forest
# Import Libraries
from sklearn.ensemble import RandomForestClassifier

# Develop and train random forest model
RF_model = RandomForestClassifier(n_estimators = 10).fit(X_train, y_train)

# Predict new test cases
RF_Predictions = RF_model.predict(X_test)
RF_Predictions


# In[43]:


# Build confusion matrix
from sklearn.metrics import confusion_matrix 
CM = confusion_matrix(y_test, RF_Predictions)

# To get confusion matrix in tabular form
CM = pd.crosstab(y_test, RF_Predictions)

CM


# In[44]:


from sklearn.metrics import classification_report
print(classification_report(y_test,RF_Predictions))


# In[45]:


from sklearn import metrics

print('Accuracy:', metrics.accuracy_score(y_test, RF_Predictions)*100)


# #### logistic regression

# In[46]:


# logistic regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#instantiate the model

logreg = LogisticRegression()

logreg.fit(X_train, y_train)


# In[47]:


# Predict new test cases
log_Pred = logreg.predict(X_test)
log_Pred


# In[48]:


# Build confusion matrix
from sklearn.metrics import confusion_matrix 
CM = confusion_matrix(y_test, log_Pred)

# To get confusion matrix in tabular form
CM = pd.crosstab(y_test, log_Pred)

CM


# In[49]:


from sklearn.metrics import classification_report
print(classification_report(y_test,log_Pred))


# In[50]:


from sklearn import metrics

print('Accuracy:', metrics.accuracy_score(y_test, log_Pred)*100)


# #### Naive Bayes

# In[51]:


# Naive Bayes
from sklearn.naive_bayes import GaussianNB
#implementation
Nave_model = GaussianNB().fit(X_train, y_train)
NB_Predictions= Nave_model.predict(X_test)
NB_Predictions


# In[52]:


# Build confusion matrix
from sklearn.metrics import confusion_matrix 
CM = confusion_matrix(y_test, NB_Predictions)
CM = pd.crosstab(y_test, NB_Predictions)
CM


# In[53]:


from sklearn.metrics import classification_report
print(classification_report(y_test,NB_Predictions))


# In[54]:


from sklearn import metrics
print('Accuracy:', metrics.accuracy_score(y_test, NB_Predictions)*100)


# # SUMMARY
# "To get the most accurate model out of various models the value of recall, precision, AUC should be high".
# As per the directions of our project we have to predict the results based on recall, precision and accuracy 
# of all machine learning algorithm. Out of all the above developed Machine Learning algorithms we can deduce 
# that "Naive Bayes" is giving all the quantities highest among all other algorithms. Hence "Naive Bayes" is 
# selected to predict target variable from our given test data.
# 

# #### FINDING THE TARGET VALUE OF TEST DATA
# * Now since we have tested all the Machine Learning Algorithms and Statistical Models on our Training Data and retrieved the accuracy from each model 
# * So we choose Naive Bayes method of Supervised Learning to predict the value of our target variable on our test Data

# In[56]:


# Load Data testing data
test = pd.read_csv('test.csv')
test.shape


# In[57]:


test.head()


# In[60]:


# Drop the ID_code column from the Dataset as our model is not trained for it
test = test.drop(['ID_code'], axis=1)
test.head(5)


# In[61]:


test['target'] = Nave_model.predict(test)
test.head()


# In[62]:


test.to_csv('santander_test_predict_py.csv',index=False)

