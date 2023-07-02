# %% [markdown]
# ## Kaggle Dataset - Customer Personality Analysis
# 
# Dataset:
# -https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis/code?datasetId=1546318&sortBy=voteCount 
# 
# Reference notebook: 
# -https://thecleverprogrammer.com/2021/02/08/customer-personality-analysis-with-python/

# %%

import numpy as np
import pandas as pd
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler, normalize
from sklearn import metrics
from sklearn.mixture import GaussianMixture

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

import matplotlib.pyplot as plt
import seaborn as sns

from dataprep.eda import plot, plot_correlation, create_report, plot_missing

pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings('ignore')

# %%
#Loading the dataset
data = pd.read_csv("D:/Documents/Personal/Self Projects/Kaggle-Customer Personality Analysis/Customer Personality Analysis/marketing_campaign.csv", sep="\t")
print("Number of datapoints:", len(data))

# %%
df_raw = data.copy()

df_raw['Dt_Customer'] = pd.to_datetime(df_raw['Dt_Customer'],format = '%d-%m-%Y')
df_raw.info()
df_raw.head()

# %%
# plot(data)

# %%
# There are 24 null values in'Income' column, will remove these null values

df_raw = df_raw.dropna()

# %%
# New data column of Seniority
print(df_raw['Dt_Customer'].max())
last_date = df_raw['Dt_Customer'].max()
df_raw['Seniority'] = (last_date - df_raw['Dt_Customer']).dt.days

# New data column of Age, to replace 'birth year' 
df_raw['Age'] = 2014 - df_raw['Year_Birth']

# New data column 'No. of kids' 
df_raw['No_of_kids'] = df_raw['Kidhome'] + df_raw['Teenhome']


# %%
# Rename Columns

df_raw=df_raw.rename(columns={'NumWebPurchases': "Web_purchase",'NumCatalogPurchases':'Catalog_purchase','NumStorePurchases':'Store_purchase', 'NumDealsPurchases':'Deal_purchase','NumWebVisitsMonth':'Web_visits_last_mth'})
df_raw=df_raw.rename(columns={'MntWines': "Amt_Wines",'MntFruits':'Amt_Fruits','MntMeatProducts':'Amt_Meat','MntFishProducts':'Amt_Fish','MntSweetProducts':'Amt_Sweets','MntGoldProds':'Amt_Gold'})

# %%
df_raw.info()
df_raw.head()

# %%
df_raw.describe().T

# %%
df_raw.columns

# %%
# Choose specific columns for Apriori Algorithm 

df_cleaned = df_raw[['Education', 'Marital_Status','Seniority', 'Age', 'No_of_kids', 'Income','Recency', 'Amt_Wines', 'Amt_Fruits',
       'Amt_Meat', 'Amt_Fish', 'Amt_Sweets', 'Amt_Gold', 'Deal_purchase',
       'Web_purchase', 'Catalog_purchase', 'Store_purchase',
       'Web_visits_last_mth']]

df_cleaned.info()
df_cleaned.head()

# %% [markdown]
# ### Trying out Apriori Algorithm 
# 
# - Apriori Algorithm is desgined for handling categorical data. 
# - Will need to discretize numerical variables into bins or convert them into categorical variables based on specific thresholds or domain knowledge. This process is known as binning or discretization.

# %%
#Create Seniority segment
cut_labels_Seniority = ['New customers', 'Discovering customers', 'Experienced customers', 'Old customers']
df_cleaned['Seniority_group'] = pd.qcut(df_cleaned['Seniority'], q=4, labels=cut_labels_Seniority)

#Create Age segment
cut_labels_Age = ['Young', 'Adult', 'Mature', 'Senior']
df_cleaned['Age_group'] = pd.qcut(df_cleaned['Age'], q=4, labels=cut_labels_Age)

#Create Income segment
cut_labels_Age = ['Low income', 'Low to medium income', 'Medium to high income', 'High income']
df_cleaned['Income_group'] = pd.qcut(df_cleaned['Income'], q=4, labels=cut_labels_Income)

#Child 

df_cleaned['Has_Child'] = np.where(df_cleaned['No_of_kids']> 0, 'Has child', 'No child')


# %%
df_cleaned.columns

# %%
cut_labels = ['Low consumer', 'Frequent consumer', 'Biggest consumer']
df_cleaned['Wines_segment'] = pd.qcut(df_cleaned['Amt_Wines'][df_cleaned['Amt_Wines']>0],q=[0, .25, .75, 1], labels=cut_labels).astype("object")
df_cleaned['Fruits_segment'] = pd.qcut(df_cleaned['Amt_Fruits'][df_cleaned['Amt_Fruits']>0],q=[0, .25, .75, 1], labels=cut_labels).astype("object")
df_cleaned['Meat_segment'] = pd.qcut(df_cleaned['Amt_Meat'][df_cleaned['Amt_Meat']>0],q=[0, .25, .75, 1], labels=cut_labels).astype("object")
df_cleaned['Fish_segment'] = pd.qcut(df_cleaned['Amt_Fish'][df_cleaned['Amt_Fish']>0],q=[0, .25, .75, 1], labels=cut_labels).astype("object")
df_cleaned['Sweets_segment'] = pd.qcut(df_cleaned['Amt_Sweets'][df_cleaned['Amt_Sweets']>0],q=[0, .25, .75, 1], labels=cut_labels).astype("object")
df_cleaned['Gold_segment'] = pd.qcut(df_cleaned['Amt_Gold'][df_cleaned['Amt_Gold']>0],q=[0, .25, .75, 1], labels=cut_labels).astype("object")

df_cleaned.replace(np.nan, "Non consumer",inplace=True)

# %%
# Purchase Type
## cut by bins. ("pd.cut" instead of "pd.qcut")

cut_labels = ['Low consumer', 'Low-Frequent consumer', 'Frequent-Big consumer', 'Biggest consumer']
df_cleaned['Deal_purchase_segment'] = pd.cut(df_cleaned['Deal_purchase'],bins=4,labels=cut_labels)
df_cleaned['Web_purchase_segment'] = pd.cut(df_cleaned['Web_purchase'],bins=4,labels=cut_labels)
df_cleaned['Catalog_purchase_segment'] = pd.cut(df_cleaned['Catalog_purchase'],bins=4,labels=cut_labels)
df_cleaned['Store_purchase_segment'] = pd.cut(df_cleaned['Store_purchase'],bins=4,labels=cut_labels)
df_cleaned['Web_visits_last_mth_segment'] = pd.cut(df_cleaned['Web_visits_last_mth'],bins=4,labels=cut_labels)


# %%
df_cleaned.info()
df_cleaned.head()

# %%
df_cleaned['Deal_purchase_segment'].value_counts()

# %%
df_cleaned['Sweets_segment'].value_counts()

# %%
df_cleaned.columns

# %%
# Choose the categorical columns for Apriori Algorithm

df_categ = df_cleaned[['Education', 'Marital_Status','Income_group', 'Has_Child','Seniority_group', 'Age_group', 'Wines_segment', 'Fruits_segment',
       'Meat_segment', 'Fish_segment', 'Sweets_segment', 'Gold_segment',
       'Deal_purchase_segment', 'Web_purchase_segment',
       'Catalog_purchase_segment', 'Store_purchase_segment',
       'Web_visits_last_mth_segment']]

df_categ.info()

# %%
df = pd.get_dummies(df_categ)

# %%
df.info()
df.head()

# %%
df.columns

# %%
# Apply the Apriori algorithm
frequent_itemsets = apriori(df, min_support=0.08, use_colnames=True)
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)

# %%
print(frequent_itemsets.shape)

# %%
print("\nAssociation Rules:")
print(rules)

# %%
# For the "target" , look at the different columns in the 'df' dataframe

target = 'Web_visits_last_mth_segment_Low consumer'
results_personnal_care = rules[rules['consequents'].astype(str).str.contains(target, na=False)].sort_values(by='confidence', ascending=False)
results_personnal_care.head()


