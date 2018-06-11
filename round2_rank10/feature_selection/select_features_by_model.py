import numpy as np
import pandas as pd
from odps.df import DataFrame
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV

train = odps.get_table('jz_combine_tl_train_6_2').to_df().to_pandas()
test = odps.get_table('jz_combine_tl_test_6_2').to_df().to_pandas()

predict_features = ['sys', 'dia', 'tl', 'hdl', 'ldl']
use_features = [t for t in train.columns if t != 'vid' and t not in predict_features]
x_train = train.loc[:, use_features]
label = train['tl']

gbdt = GradientBoostingRegressor(random_state=1)
rf = RandomForestRegressor(random_state=1)
l2 = RidgeCV()

sfm_gbdt = SelectFromModel(gbdt, threshold=0.001)
sfm_gbdt.fit_transform(x_train, label)
gbdt_features = set(x_train.columns[sfm_gbdt.get_support()])
print('*************************************')
print(gbdt_features)


sfm_rf = SelectFromModel(rf, threshold=0.001)
sfm_rf.fit_transform(x_train, label)
rf_features = set(x_train.columns[sfm_rf.get_support()])
print('*************************************')
print(rf_features)

print(gbdt_features & rf_features)
sfm_l2 = SelectFromModel(l2, threshold=0.5)
sfm_l2.fit_transform(x_train, label)
l2_features = set(x_train.columns[sfm_l2.get_support()])
print('*************************************')
print(l2_features)

final_features = list(gbdt_features | rf_features | l2_features)
# choose top k features
#final_features = list((gbdt_features & rf_features) | l2_features)
print('gbdt model has {} features'.format(len(gbdt_features)))
print('rf model has {} features'.format(len(rf_features)))
print('l2 model has {} features'.format(len(l2_features)))
print('final has {} features'.format(len(final_features)))
print('*************************************')
print(final_features)
print('*************************************')          

final_features.extend(['vid', 'tl'])
train_final = DataFrame(train.loc[:, final_features])
train_final.persist('combine_tl_train_6_2')
test_final = DataFrame(test.loc[:, final_features])
test_final.persist('combine_tl_test_6_2')