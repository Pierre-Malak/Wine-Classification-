import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('wine.csv')

#Make a copy of DF
df_tr = df

#Transsform the timeOfDay to dummies
df_tr = pd.get_dummies(df_tr, columns=['quality'])

#Standardize
clmns = ['fixed acidity', 'density','pH', 'alcohol','high_quality']
df_tr_std = stats.zscore(df_tr[clmns])

#Cluster the data
kmeans = KMeans(n_clusters=2, random_state=42).fit(df_tr_std)
labels = kmeans.labels_

#Glue back to originaal data
df_tr['cluster'] = labels

#Add the column into our list
clmns.extend(['cluster'])

#Lets analyze the clusters
print (df_tr[clmns].groupby(['cluster']).mean())


#Scatter plot of Wattage and Duration
sns.lmplot('chlorides', 'fixed acidity', data=df_tr, fit_reg=False, hue="color", scatter_kws={"marker": "D", "s": 100})
plt.title('Chlorides vs fixed acidity')
plt.xlabel('Chlorides')
plt.ylabel('Fixed acidity')