
# coding: utf-8

# # Companies Analisys: exploring data of companing with same name
# 
# This notebook provides an exploratory analysis on companies with the same name but different CNPJ's. On this analysis it'll be tried to know more about their existence through an exploratory analysis, and possibly get more insights for new irregularities.

# In[35]:

from serenata_toolbox.datasets import Datasets
from pylab import rcParams
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

get_ipython().magic('matplotlib inline')

# Charts styling
plt.style.use('ggplot')
rcParams['figure.figsize'] = 15, 8
matplotlib.rcParams.update({'font.size': 14})
pd.options.mode.chained_assignment = None
pd.options.display.max_rows = 100000
pd.options.display.max_columns = 10000


# In[2]:

# First, lets download all the needed datasets for this analysis
datasets = Datasets('../data/')

reimbursments_path = Path("../data/2017-07-04-reimbursements.xz")
companies_path = Path("../data/2017-05-21-companies-no-geolocation.xz")

if not reimbursments_path.exists():
    datasets.downloader.download('2017-07-04-reimbursements.xz')

if not companies_path.exists():
    datasets.downloader.download('2017-05-21-companies-no-geolocation.xz')


# In[73]:

# Loading companies dataset
CP_DTYPE =dict(cnpj=np.str, name=np.str, main_activity_code=np.str,
               legal_entity='category', situation='category', status='category')

companies = pd.read_csv('../data/2017-05-21-companies-no-geolocation.xz', dtype=CP_DTYPE,
                        low_memory=False, parse_dates=['last_updated', 'situation_date', 'opening'])

c = companies[['cnpj', 'main_activity_code', 'name', 'opening', 'situation', 'status']]

c['cnpj'] = c['cnpj'].str.replace(r'\D+', '')
c['main_activity_code'] = c['main_activity_code'].str.replace(r'\D+', '')

# Only companies that are OK and ATIVA will be analyzed
c = c[c.status == 'OK']
c = c[c.situation == 'ATIVA']

# Drop columns
c = c.drop('situation', 1).drop('status', 1)

print(c.shape)
c.head(5)


# In[45]:

# Loading reimbursments dataset
R_DTYPE =dict(cnpj_cpf=np.str, supplier=np.str, total_net_value=np.float,
              subquota_group_description='category')

reimbursements = pd.read_csv('../data/2017-07-04-reimbursements.xz',
                             dtype=R_DTYPE, low_memory=False, parse_dates=['issue_date'])

r = reimbursements[reimbursements.year >= 2015]
r = r[['total_net_value', 'subquota_description', 'supplier', 'cnpj_cpf']]

r.rename(columns={'cnpj_cpf':'cnpj'}, inplace=True)
r = r[r.cnpj.str.len() == 14]

r.head(10)


# In[74]:

filtered_c = c[c.cnpj.isin(r.cnpj.unique())]
data = r.merge(filtered_c, on='cnpj', how='left')

data.head(10)


# In[75]:

# count objects with invalid main_activity_code
d = dict()

invalid_main_activity = "0000000"
data_len = len(data)

d['valid'] = len(data[data.main_activity_code != invalid_main_activity]) / data_len * 100
d['invalid'] = len(data[data.main_activity_code == invalid_main_activity]) / data_len * 100

s = pd.Series(d)
s.plot(kind='pie', autopct='%.2f')
plt.title('Number of valid and invalid main_activity_code in dataset')


# In[76]:

# remove items with invalid main_activity_code
data = data[data.main_activity_code != "0000000"]
print('dataset shape: {}.'.format(data.shape))

data.head(5)


# In[77]:

labels = ['subquota_description', 'supplier', 'cnpj', 'name',
          'main_activity_code', 'situation', 'status']

df = data
for l in labels:
    df[l] = data[l].astype('category').cat.codes

df['opening'] = df.opening.astype(np.str).str.replace(r'\D+', '').astype('category').cat.codes

df.head()


# In[62]:

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

X = scale(df)

# # Benchmark clusters
X, _, = train_test_split(df, train_size=0.1, random_state=2)
X.shape


# In[66]:

from time import time
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

print(42 * '_')
print('init\t\ttime\tinertia\tsilhouette')

def bench_k_means(estimator, name, data, labels=0):
    t0 = time()
    estimator.fit(data)
    avg = metrics.silhouette_score(data, estimator.labels_, metric='euclidean', sample_size=500)
    print('%-9s\t%.2fs\t%i\t%.3f' % (name, (time() - t0), estimator.inertia_, avg))

bench_k_means(KMeans(init='k-means++', n_clusters=3, n_init=10),
              name="KMeans (3)", data=X)
bench_k_means(KMeans(init='k-means++', n_clusters=2, n_init=10),
              name="KMeans (2)", data=X)

pca = PCA(n_components=1).fit_transform(X)
bench_k_means(KMeans(init='k-means++', n_clusters=3, n_init=10),
              name="PCA (3)", data=pca)
pca = PCA(n_components=1).fit_transform(X)
bench_k_means(KMeans(init='k-means++', n_clusters=2, n_init=10),
              name="PCA (2)", data=pca)

print(42 * '_')


# In[67]:

from sklearn.cluster import DBSCAN

# Compute DBSCAN
db = DBSCAN(eps=0.3).fit(X)

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)

print('Estimated number of clusters: %d' % n_clusters_)
print("Silhouette Coefficient: %0.3f" %
      metrics.silhouette_score(X, db.labels_, metric='euclidean', sample_size=500))


# In[ ]:



