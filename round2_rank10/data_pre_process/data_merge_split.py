import time
import re
import pandas as pd
from odps import ODPS
from odps.df import DataFrame
import numpy as np
from collections import Iterable
from sklearn import preprocessing
from itertools import combinations

import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

# my all features
train = odps.get_table('meinian_round2_train').to_df().to_pandas()
test = odps.get_table('meinian_round2_submit_b').to_df().to_pandas()
num_data = odps.get_table('juz_num_data_5_31').to_df().to_pandas()
word_data = odps.get_table('juz_word_data_5_30').to_df().to_pandas()

# add wl word features
wl_word = odps.get_table('pre_txt_features_b').to_df().to_pandas()

gene_data = odps.get_table('meinian_round2_snp').to_df().to_pandas()

word_data = pd.merge(word_data, wl_word, on='vid', how='inner')
# fix feature 314
num_data.loc[num_data['314']<=1, '314'] = num_data.loc[num_data['314']<=1, '314'] * 100

lbl = preprocessing.LabelEncoder()

for c in gene_data.columns:
    if c not in ['vid']:
        gene_data[c] = lbl.fit_transform(gene_data[c])

print('final word data shape: ', word_data.shape)
print('final num data shape: ', num_data.shape)
print('final gene data shape: ', gene_data.shape)
        
merge_tmp = pd.merge(num_data, word_data, on='vid', how='inner')
merge_tmp = pd.merge(merge_tmp, gene_data, on='vid', how='left')

print('final data shape: ', merge_tmp.shape)

train_merge = pd.merge(train, merge_tmp, on='vid', how='left')
test_merge = pd.merge(test, merge_tmp, on='vid', how='left')

# fix some value of hdl and dia
train_merge.loc[train_merge['vid'] == '605ebf5c6173cd3aab071060c9618b79', 'hdl'] = 1.28
train_merge.loc[train_merge['vid'] == 'c6aec5461b1c5cca1c4ead3d4c2b83d9', 'dia'] = 90
train_merge.fillna(-999, inplace=True)
test_merge.fillna(-999, inplace=True)
print('final train shape:{}, test shape:{} '.format(train_merge.shape, test_merge.shape))
juz_train = DataFrame(train_merge)
juz_test = DataFrame(test_merge)
juz_train.persist('juz_train_6_6_final')
juz_test.persist('juz_test_6_6_final')