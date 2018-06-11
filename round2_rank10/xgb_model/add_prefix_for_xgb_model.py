from odps import ODPS
import pandas as pd
from odps.df import DataFrame
from sklearn.model_selection import KFold
import time
import numpy as np

import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

label = 'tl'
train = odps.get_table('{}_juz_train_6_6_snp_onehot_22'.format(label)).to_df().to_pandas()
test = odps.get_table('{}_juz_test_6_6_snp_onehot_22'.format(label)).to_df().to_pandas()
print(train.shape, test.shape)

train['log_{}'.format(label)] = np.log(train[label])
test['log_{}'.format(label)] = np.log(test[label])
predict_features = ['sys', 'dia', 'tl', 'hdl', 'ldl']

for i in train.columns:
    if i != 'vid' and not 'snp' in i and not 'log' in i and i not in predict_features:
        train['jz_{}'.format(i)] = train[i]
        test['jz_{}'.format(i)] = test[i]
        predict_features.append(i)

train.drop(predict_features, axis=1, inplace=True)
test.drop(predict_features, axis=1, inplace=True)
print(train.shape, test.shape)

juz_train = DataFrame(train)
juz_test = DataFrame(test)
juz_train.persist('juz_train_6_7_xgb')
juz_test.persist('juz_test_6_7_xgb')