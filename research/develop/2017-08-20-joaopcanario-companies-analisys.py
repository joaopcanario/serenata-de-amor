
# coding: utf-8

# # Companies Analisys: exploring data of companing with same name
# 
# This notebook provides an exploratory analysis on companies with the same name but different CNPJ's. On this analysis it'll be tried to know more about their existence through an exploratory analysis, and possibly get more insights for new irregularities.

# In[ ]:

from serenata_toolbox.datasets import Datasets
from pylab import rcParams

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

# First, lets download all the needed datasets for this analysis
datasets = Datasets('../data/')
                             
datasets.downloader.download('2017-07-04-reimbursements.xz')
datasets.downloader.download('2017-05-21-companies-no-geolocation.xz')


# In[ ]:

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


# In[ ]:

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


# In[ ]:

# r.groupby(['supplier', 'congressperson_name', 'year'])['total_net_value'].sum().sort_values(ascending=False).head(20)
filtered_c = c[c['cnpj_cpf'].isin(r.cnpj_cpf.unique())]
data = r.merge(filtered_c, on='cnpj_cpf', how='left')
data = data[data.year >= 2016]

data.head(10)


# In[ ]:

# count objects with invalid main_activity_code
d = dict()

invalid_main_activity = "00.00-0-00"
data_len = len(data)

d['valid'] = len(data[data.main_activity_code_x != invalid_main_activity]) / data_len * 100
d['invalid'] = len(data[data.main_activity_code_x == invalid_main_activity]) / data_len * 100

s = pd.Series(d)
s.plot(kind='pie', autopct='%.2f')
plt.title('Number of valid and invalid main_activity_code in dataset')


# In[ ]:

# remove items with invalid main_activity_code
data = data[data.main_activity_code_x != "00.00-0-00"]
data.head(5)


# In[ ]:

# remove items previous than 2015
data = data[data['year'] >= 2015]
data.head(5)


# In[ ]:

data.shape


# In[ ]:



