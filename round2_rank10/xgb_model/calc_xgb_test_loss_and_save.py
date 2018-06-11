from odps import ODPS
import pandas as pd
from odps.df import DataFrame
import numpy as np


def eval_metric(pred, labels):
	return np.mean(np.power(np.log(pred + 1) - np.log(labels + 1), 2))


'''
('fold 1: ', 0.088066106616791956)
('fold 2: ', 0.087444759182314105)
('fold 3: ', 0.097941499769017726)
('fold 4: ', 0.078793753494365307)
('fold 5: ', 0.086734232908105002)
('total loss: ', 0.087796043720344913)
'''

# 这边的val都是预测了test b榜的值，然后保存表
# 如果要计算xgb在验证集上的loss，需要在xgb的模型修改代码，并使用底端的注释内容进行计算
val_1 = odps.get_table('jz_xgb_pred_val_1').to_df().to_pandas().loc[:, ['vid', 'log_tl', 'result']]
val_2 = odps.get_table('jz_xgb_pred_val_2').to_df().to_pandas().loc[:, ['vid', 'log_tl', 'result']]
val_3 = odps.get_table('jz_xgb_pred_val_3').to_df().to_pandas().loc[:, ['vid', 'log_tl', 'result']]
val_4 = odps.get_table('jz_xgb_pred_val_4').to_df().to_pandas().loc[:, ['vid', 'log_tl', 'result']]
val_5 = odps.get_table('jz_xgb_pred_val_5').to_df().to_pandas().loc[:, ['vid', 'log_tl', 'result']]

xgb_result = val_1.loc[:, ['vid']]

xgb_result['tl'] = np.exp((val_1['result'] + val_2['result'] + val_3['result'] + val_4['result'] + val_5['result'])/5)
test_odps = DataFrame(xgb_result)
test_odps.persist('tl_xgb_result')

'''
val = pd.concat([val_1, val_2, val_3, val_4, val_5])
print('fold 1: ', eval_metric(np.exp(val_1['result']), np.exp(val_1['log_tl'])))
print('fold 2: ', eval_metric(np.exp(val_2['result']), np.exp(val_2['log_tl'])))
print('fold 3: ', eval_metric(np.exp(val_3['result']), np.exp(val_3['log_tl'])))
print('fold 4: ', eval_metric(np.exp(val_4['result']), np.exp(val_4['log_tl'])))
print('fold 5: ', eval_metric(np.exp(val_5['result']), np.exp(val_5['log_tl'])))
print('total loss: ', eval_metric(np.exp(val['result']), np.exp(val['log_tl'])))
'''