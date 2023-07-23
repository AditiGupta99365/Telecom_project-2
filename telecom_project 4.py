#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist


# In[2]:


# Load the dataset
df=pd.read_csv('telcom_data.csv')


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


def preprocessing(df):
    categorical_cols = []
    continious_cols = []
    for col in df.columns:
        if df[col].dtypes == "object":
            categorical_cols.append(col)
            df[col].fillna(df[col].mode()[0], inplace = True)
            
        else:
            df[col].fillna(df[col].mean(), inplace = True)
            continious_cols.append(col)
    return categorical_cols,continious_cols, df
categorical_cols,continious_cols, df_preprocessed = preprocessing(df)


# ## Task 4. 1 - Write a Python program to assign:
# #### a. Engagement score to each user. Consider the engagement score as the Euclidean distance between the user data point &         the    less engaged cluster (use the first clustering for this) (Euclidean Distance)
# 
# #### b. Experience score for each user. Consider the experience score as the Euclidean distance between the user data point &             the worst experience cluster. 
# 

# In[7]:


df['Total_Traffic'] = df['Total UL (Bytes)']+df['Total DL (Bytes)']


# ##### Engagement Metrics
# 
# 

# In[8]:


engagement_metrics = df.groupby('MSISDN/Number').agg({'Bearer Id': 'count', 'Dur. (ms)': 'sum', 'Total_Traffic': 'sum'})
engagement_metrics.columns = ['Sessions Frequency', 'Session Duration', 'Session Total Traffic']


# In[9]:


scaler = MinMaxScaler()
normalized_engagement_metrics = scaler.fit_transform(engagement_metrics)


# In[10]:


kmeans = KMeans(n_clusters=3)
kmeans.fit(normalized_engagement_metrics)
engagement_metrics['Cluster'] = kmeans.labels_


# In[11]:


## Getting TCP Retransmission

df['TCP Retransmission'] = df['TCP DL Retrans. Vol (Bytes)'] + df['TCP UL Retrans. Vol (Bytes)']
tcp_retransmission = df.groupby('MSISDN/Number')['TCP Retransmission'].mean()
tcp_retransmission


# In[12]:


df['RTT'] = (df['Avg RTT DL (ms)'] + df['Avg RTT UL (ms)']) 
rtt = df.groupby('MSISDN/Number')['RTT'].mean()

handset_type = df.groupby('MSISDN/Number')['Handset Type'].agg(lambda x: x.mode())

df['Throughput'] = df['Avg Bearer TP DL (kbps)']+df['Avg Bearer TP UL (kbps)']
throughput = df.groupby('MSISDN/Number')['Throughput'].mean()

experience_metrics = pd.concat([tcp_retransmission, rtt, throughput], axis=1)
experience_metrics.columns = ['TCP Retransmission', 'RTT', 'Throughput']


# In[13]:


rtt


# In[14]:


experience_metrics


# In[15]:


## Process the df for K-Means

def preprocessing(df):
    categorical_cols = []
    continious_cols = []
    for col in df.columns:
        if df[col].dtypes == "object":
            categorical_cols.append(col)
            df[col].fillna(df[col].mode()[0], inplace = True)
            
        else:
            df[col].fillna(df[col].mean(), inplace = True)
            continious_cols.append(col)
    return categorical_cols,continious_cols, df

categorical_cols,continious_cols, df_preprocessed = preprocessing(df)


# In[16]:


scaler = MinMaxScaler()
normalized_experience_metrics = scaler.fit_transform(experience_metrics)


# In[17]:


kmeans = KMeans(n_clusters=3)
kmeans.fit(normalized_experience_metrics)
experience_metrics['Cluster'] = kmeans.labels_


# In[18]:


less_engaged_cluster_center = kmeans.cluster_centers_[np.argmin(kmeans.cluster_centers_[:, 0])]
engagement_score = cdist(normalized_engagement_metrics, [less_engaged_cluster_center], metric='euclidean').flatten()

# Assign an experience score for each user. Consider the experience score as the Euclidean distance between the user data point & the worst experience cluster.
worst_experience_cluster_center = kmeans.cluster_centers_[np.argmax(kmeans.cluster_centers_[:, 0])]
experience_score = cdist(normalized_experience_metrics, [worst_experience_cluster_center], metric='euclidean').flatten()


# In[19]:


less_engaged_cluster_center


# In[20]:


engagement_score


# In[21]:


worst_experience_cluster_center


# In[22]:


experience_score


# In[23]:


engagement_metrics['engagement_score'] = engagement_score
engagement_metrics['experience_score'] = experience_score


# In[24]:


engagement_metrics


# ####  Task 4.2   Consider the average of both engagement & experience scores as  the satisfaction score & report the top 10         satisfied customer 
# 

# In[25]:


engagement_metrics['satisfaction_score'] = (engagement_metrics['engagement_score'] + engagement_metrics['experience_score'])/2


# In[26]:


engagement_metrics


# In[27]:


## Top 10 satisfied Customers

engagement_metrics['satisfaction_score'].nlargest(10)


# In[28]:


engagement_metrics


# ##### Task 4.3  Build a regression model of your choice to predict the satisfaction score of a customer. 
# 

# In[29]:


## Regression Model  --> Linear Regression Model

# Define X and y
X,y =  engagement_metrics.drop(["satisfaction_score"], axis =1 ), engagement_metrics["satisfaction_score"]


# In[30]:


# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)


# In[31]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[32]:


# Modelling
from sklearn.linear_model import LinearRegression

model = LinearRegression()

# Starting training
model.fit(X_train, y_train)


# In[33]:


y_pred = model.predict(X_test)


# In[34]:


y_pred


# In[35]:


from sklearn.metrics import r2_score


# In[36]:


r2_score(y_test, y_pred)


# In[37]:


y_pred_train = model.predict(X_train)


# In[38]:


y_pred_train


# In[39]:


r2_score(y_train, y_pred_train)


# In[40]:


import matplotlib.pyplot as plt


# In[41]:


index = []
for i in range(len(X_test)):
    index.append(i)


# In[42]:


plt.scatter( index,y_test, color =  "darkblue")
plt.scatter( index,y_pred, color =  "red")


# ##### Task 4.4   Run a k-means (k=2) on the engagement & the experience score. 
# 

# In[43]:


cluster_data = engagement_metrics.iloc[:,4:6]
cluster_data


# In[44]:


km = KMeans(init="random",n_clusters=2,n_init=10,max_iter=300,random_state=42)
label = km.fit_predict(cluster_data)
centroids = km.cluster_centers_
print(f'# Centroids of the clustering:\n{centroids}')
print(f'# The number of iterations required to converge: {km.inertia_}')
print(f'# The number of iterations required to converge: {km.n_iter_}')


# In[45]:


u_labels = np.unique(label)
 
#plotting the results:
plt.figure(figsize=(10,5))
plt.title(f'User clustering based on engagement score and experience score')
for i in u_labels:
    plt.scatter(cluster_data[label == i].iloc[:,0] , cluster_data[label == i].iloc[:,1], marker='o', label = i)
plt.scatter(centroids[:,0] , centroids[:,1], marker='x', color = 'black')
plt.xlabel("engagement score")
plt.ylabel("experience score")
plt.legend()
plt.savefig('kmeans.png')
plt.show()


# ##### Task 4.5  Aggregate the average satisfaction & experience score per cluster. 

# In[46]:


Aggregate_data = engagement_metrics.copy(deep=True)
Aggregate_data['cluster'] = label
Aggregate_data


# In[47]:


Aggregate_columns = {'engagement_score':'mean','experience_score':'mean','satisfaction_score':'mean'}
#Group and Aggregate
Aggregate_data =Aggregate_data.groupby('cluster').agg(Aggregate_columns)


# In[48]:


Aggregate_data


# ##### Task 4.6  Export your final table containing all user id + engagement, experience & satisfaction scores in your local MySQL database. Report a screenshot of a select query output on the exported table. 

# In[49]:


Final_Table = engagement_metrics.copy(deep=True)
Final_Table.reset_index(inplace=True)
Final_Table = Final_Table.rename(columns={'MSISDN/Number': 'user_id'})


# In[50]:



Final_Table


# In[51]:


Final_Table.info()


# In[52]:


Final_Table.columns


# In[53]:


Final_table = Final_Table.drop(['Sessions Frequency','Session Duration','Session Total Traffic','Cluster' ],axis=1)


# In[54]:


Final_table


# In[ ]:





# In[ ]:





# In[ ]:




