from odps import ODPS
import pandas as pd
from odps.df import DataFrame
from sklearn.model_selection import KFold

train_data = odps.get_table('juz_train_6_7_xgb').to_df().to_pandas()
kf = KFold(n_splits=5, shuffle=True, random_state=1024)
for t, (train_index, test_index) in enumerate(kf.split(train_data), start=1):
    print('第{}次拆分...'.format(t))
    x_train, x_test = train_data.iloc[train_index], train_data.iloc[test_index]
    print(x_train.shape, x_test.shape)
    train_odps = DataFrame(x_train)
    test_odps = DataFrame(x_test)
    train_odps.persist('tl_xgb_train_{}'.format(t))
    test_odps.persist('tl_xgb_test_{}'.format(t))
    
