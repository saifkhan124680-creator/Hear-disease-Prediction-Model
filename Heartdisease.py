#!/usr/bin/env python
# coding: utf-8

# In[54]:


import numpy as numpy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[55]:


df=pd.read_csv("heart.csv")


# In[56]:


df


# In[57]:


df.info()


# In[58]:


df.shape


# In[59]:


df.columns


# In[60]:


df.describe()


# In[61]:


df.duplicated().sum()


# In[62]:


df['HeartDisease'].value_counts()


# In[63]:


df['HeartDisease'].value_counts().plot(kind='bar')


# In[64]:


df.isnull().sum()


# In[65]:


df.head()


# In[66]:


def plotting(var,num):
    plt.subplot(2,2,num)
    sns.histplot(df[var],kde=True)


plotting('Age',1)
plotting('RestingBP',2)
plotting('Cholesterol',3)
plotting('MaxHR',4)

plt.tight_layout()


# In[67]:


df['Cholesterol'].value_counts()


# In[68]:


ch_mean=df.loc[df['Cholesterol'] != 0,'Cholesterol'].mean()


# In[69]:


ch_mean


# In[70]:


df['Cholesterol']=df['Cholesterol'].replace(0,ch_mean)
df['Cholesterol']=df['Cholesterol'].round(2)


# In[71]:


resting_bp_mean = df.loc[df['RestingBP'] != 0, 'RestingBP'].mean()

df['RestingBP'] = df['RestingBP'].replace(0, resting_bp_mean)

df['RestingBP'] = df['RestingBP'].round(2)


# In[72]:


def plotting(var,num):
    plt.subplot(2,2,num)
    sns.histplot(df[var],kde=True)


plotting('Age',1)
plotting('RestingBP',2)
plotting('Cholesterol',3)
plotting('MaxHR',4)

plt.tight_layout()


# In[73]:


sns.countplot(x=df['Sex'])


# In[74]:


sns.countplot(x=df['ChestPainType'],hue=df['HeartDisease'])


# In[75]:


sns.countplot(x=df['FastingBS'],hue=df['HeartDisease'])


# In[76]:


sns.boxplot(x=df['HeartDisease'],y=df['Cholesterol'])


# In[77]:


sns.violinplot(x='HeartDisease', y='Age', data=df)


# In[78]:


sns.heatmap(df.corr(numeric_only=True), annot=True)


# #DATA CELANING

# In[79]:


df_encoded=pd.get_dummies(df,drop_first=True)


# In[80]:


df_encoded


# In[81]:


df_encoded=df_encoded.astype(int)


# In[82]:


df_encoded


# In[83]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier



# In[84]:


df_encoded.head()


# In[85]:


X=df_encoded.drop('HeartDisease',axis=1)
y=df_encoded['HeartDisease']


# In[86]:


X


# In[87]:


y


# In[88]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42)



# In[89]:


scaler=StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[90]:


models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM (RBF Kernel)": SVC(probability=True)
}


# In[91]:


results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results.append({
        'Model': name,
        'Accuracy': round(acc, 4),
        'F1 Score': round(f1, 4)
    })  


# In[92]:


results


# In[93]:


import joblib
joblib.dump(models['KNN'],'KNN_heart.pkl')
joblib.dump(scaler,'scaler.pkl')
joblib.dump(X.columns.tolist(),'columns.pkl')


# In[ ]:
