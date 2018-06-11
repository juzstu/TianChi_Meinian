import time
import re
import pandas as pd
from odps import ODPS
from odps.df import DataFrame
import numpy as np
from collections import Iterable
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import preprocessing
from itertools import combinations

import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

# 选择超过设置阈值的snp数据进行one hot
def get_one_hot_list(data_frame, pred_feac, threshold=10):
    print('Now we extract snp features for  {} ...'.format(pred_feac))
    predict_features = ['sys', 'dia', 'tl', 'hdl', 'ldl','vid']
    use_features = [i for i in data_frame.columns if i not in predict_features]
    x_train = data_frame.loc[:, use_features]
    label = data_frame[pred_feac]

    gbdt = GradientBoostingRegressor(random_state=1, n_estimators=100)
    gbdt.fit(x_train, label)
    feature_imp = gbdt.feature_importances_
    df = pd.DataFrame()
    df['feature'] = x_train.columns
    df['imp'] = feature_imp
    df.sort_values(by='imp', ascending=False, inplace=True)
    snp_list = [s for s in df['feature'] if s.startswith('snp')][:threshold]
    return snp_list

	
# sys snp one-hot threshold:14  use_data_set: juz_train_6_6_add_wzm_for145_final2  generate_data_set: sys_juz_train_6_6_snp_onehot_22
# dia snp one-hot threshold:10  use_data_set: juz_train_6_6_add_wzm_for145_final2  generate_data_set: dia_juz_train_6_6_snp_onehot_22
# tl  snp one-hot threshold:14  use_data_set: juz_train_6_6_add_wzm_onlytl_final   generate_data_set: tl_juz_train_6_6_snp_onehot_22
# hdl snp one-hot threshold:10  use_data_set: juz_train_6_6_add_wzm_for145_final   generate_data_set: ldl_juz_train_6_6_snp_onehot_22
# ldl snp one-hot threshold:1   use_data_set: juz_train_6_6_add_wzm_for145_final   generate_data_set: hdl_juz_train_6_6_snp_onehot_22


if __name__ == "__main__":
    use_label = 'hdl'
    train = odps.get_table('juz_train_6_6_add_wzm_for145_final').to_df().to_pandas()
    test = odps.get_table('juz_test_6_6_add_wzm_for145_final').to_df().to_pandas()
    print(train.shape, test.shape)
    gene_list = get_one_hot_list(train, use_label, 14)
    train.replace(-999, np.nan,inplace=True)
    test.replace(-999, np.nan, inplace=True)
    
    drop_snp = [s for s in train.columns if 'snp' in s]
    train.drop(drop_snp, axis=1, inplace=True)
    test.drop(drop_snp, axis=1, inplace=True)

    gene_data = odps.get_table('meinian_round2_snp').to_df().to_pandas()
    snp_data = pd.get_dummies(gene_data.loc[:, gene_list])
    snp_data['vid'] = gene_data['vid'].values
    for s in snp_data.columns:
        if s != 'vid':
            snp_data[s] = snp_data[s].astype(int)

    train_merge = pd.merge(train, snp_data, on='vid', how='left')
    test_merge = pd.merge(test, snp_data, on='vid', how='left')

    train_merge.fillna(-999, inplace=True)
    test_merge.fillna(-999, inplace=True)
    print('final train shape:{}, test shape:{} '.format(train_merge.shape, test_merge.shape))

    juz_train = DataFrame(train_merge)
    juz_test = DataFrame(test_merge)
    juz_train.persist('{}_juz_train_6_6_snp_onehot_22'.format(use_label))
    juz_test.persist('{}_juz_test_6_6_snp_onehot_22'.format(use_label))