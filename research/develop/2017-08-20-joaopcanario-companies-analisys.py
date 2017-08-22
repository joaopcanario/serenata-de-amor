
# coding: utf-8

# # Companies Analisys: exploring data of companing with same name
# 
# This notebook provides an exploratory analysis on companies with the same name but different CNPJ's. On this analysis it'll be tried to know more about their existence through an exploratory analysis, and possibly get more insights for new irregularities.

# In[1]:

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


# In[3]:

# Loading companies dataset
CP_DTYPE =dict(cnpj=np.str, name=np.str,
               main_activity_code='category', legal_entity='category',
               partner_1_name=np.str, partner_1_qualification='category',
               partner_2_name=np.str, partner_2_qualification='category',
               situation='category', state='category',
               status='category', type='category')

companies = pd.read_csv('../data/2017-05-21-companies-no-geolocation.xz',
                        dtype=CP_DTYPE, low_memory=False,
                        parse_dates=['last_updated', 'situation_date', 'opening'])

# Cleaning columns with more then 30000 NaN values
# companies = companies.dropna(axis=[0, 1], how='all').dropna(axis=1, thresh=30000)
companies['cnpj'].replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)

c = companies[['cnpj', 'last_updated', 'legal_entity', 'main_activity_code',
               'name', 'opening', 'partner_1_name', 'partner_1_qualification',
               'partner_2_name', 'partner_2_qualification',
               'situation', 'situation_date', 'state', 'status', 'type']]

c.columns.values[0] = 'cnpj_cpf'

c.head(5)


# In[4]:

# Loading reimbursments dataset
R_DTYPE =dict(cnpj_cpf=np.str, year=np.int16, month=np.int16,
              installment='category', term_id='category',
              term='category', document_type='category',
              subquota_group_id='category',
              subquota_group_description='category',
              subquota_number='category', state='category',
              party='category')

reimbursements = pd.read_csv('../data/2017-07-04-reimbursements.xz',
                             dtype=R_DTYPE, low_memory=False, parse_dates=['issue_date'])

r = reimbursements[['year', 'month', 'total_net_value', 'party',
                    'state', 'term', 'issue_date', 'congressperson_name',
                    'subquota_description','supplier', 'cnpj_cpf']]

r.head(10)


# In[5]:

# r.groupby(['supplier', 'congressperson_name', 'year'])['total_net_value'].sum().sort_values(ascending=False).head(20)
filtered_c = c[c['cnpj_cpf'].isin(r.cnpj_cpf.unique())]
data = r.merge(filtered_c, on='cnpj_cpf', how='left')
data = data[data.year >= 2016]

data.head(10)


# In[6]:

# count objects with invalid main_activity_code
d = dict()

invalid_main_activity = "00.00-0-00"
data_len = len(data)

d['valid'] = len(data[data.main_activity_code != invalid_main_activity]) / data_len * 100
d['invalid'] = len(data[data.main_activity_code == invalid_main_activity]) / data_len * 100

s = pd.Series(d)
s.plot(kind='pie', autopct='%.2f')
plt.title('Number of valid and invalid main_activity_code in dataset')


# In[7]:

# remove items with invalid main_activity_code
data = data[data.main_activity_code != "00.00-0-00"]
print('dataset shape: {}.'.format(data.shape))

data.head(5)


# In[8]:

labels = ['party', 'state_x', 'term', 'issue_date', 'congressperson_name', 
          'subquota_description', 'supplier', 'cnpj_cpf', 'legal_entity', 
          'main_activity_code', 'name', 'partner_1_name', 'partner_1_qualification', 
          'partner_2_name', 'partner_2_qualification', 'situation', 'state_y',
          'status', 'type']

df = pd.DataFrame()
for l in labels:
    df[l] = data[l].astype('category').cat.codes
    
df.head()


# In[9]:

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

X = scale(df)

# # Benchmark clusters
# X, _, = train_test_split(df, train_size=0.1, random_state=2)
X.shape


# In[11]:

from time import time
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

print(42 * '_')
print('init\t\ttime\tinertia\tsilhouette')

def bench_k_means(estimator, name, data, labels=0):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=10000)))

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


# In[17]:

from sklearn.cluster import DBSCAN

# Compute DBSCAN
db = DBSCAN(eps=0.3).fit(X)

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)

print('Estimated number of clusters: %d' % n_clusters_)
print("Silhouette Coefficient: %0.3f" %
      metrics.silhouette_score(X, db.labels_, metric='euclidean', sample_size=10000))


# In[ ]:

# from sklearn.cluster import AgglomerativeClustering
# from sklearn import manifold

# # Compute clustering
# print("Compute unstructured hierarchical clustering...")
# # st = time.time()


# X_red = manifold.SpectralEmbedding(n_components=2).fit_transform(X)

# clustering = AgglomerativeClustering(n_clusters=2, linkage='ward')
# ward = clustering.fit(X_red)

# # print("Elapsed time: %.2fs" % (time.time() - st))
# print("Number of points: %i" % ward.labels_.size)

# # # Define the structure A of the data. Here a 10 nearest neighbors
# # from sklearn.neighbors import kneighbors_graph
# # connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)

# # # Compute clustering
# # print("Compute structured hierarchical clustering...")
# # st = time.time()
# # ward = AgglomerativeClustering(n_clusters=6, connectivity=connectivity,
# #                                linkage='ward').fit(X)
# # elapsed_time = time.time() - st
# # label = ward.labels_
# # print("Elapsed time: %.2fs" % elapsed_time)
# # print("Number of points: %i" % label.size)


# In[ ]:



