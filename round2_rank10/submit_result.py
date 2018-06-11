import pandas as pd
import numpy as np

sys = odps.get_table('sys_jz_5_fold_6_6_submit_22').to_df().to_pandas().loc[:,['vid', 'sys']]
dia = odps.get_table('dia_jz_5_fold_6_6_submit_22').to_df().to_pandas().loc[:,['vid', 'dia']]
tl = odps.get_table('tl_jz_5_fold_6_6_submit_22').to_df().to_pandas().loc[:,['vid', 'tl']]
hdl = odps.get_table('hdl_jz_5_fold_6_6_submit_22').to_df().to_pandas().loc[:, ['vid', 'hdl']]
ldl = odps.get_table('ldl_jz_5_fold_6_6_submit_22').to_df().to_pandas().loc[:, ['vid', 'ldl']]

print(tl.sort_values(by=['tl'], ascending=False).head(15))

tl_xgb = odps.get_table('tl_xgb_result').to_df().to_pandas().loc[:,['vid', 'tl']]
tl['tl'] = tl['tl']*0.7 + tl_xgb['tl']*0.35

sys_dia = pd.merge(sys, dia, on=['vid'], how='inner')
sys_dia_tl = pd.merge(sys_dia, tl, on=['vid'], how='inner')
sys_dia_tl_hdl = pd.merge(sys_dia_tl, hdl, on=['vid'], how='inner')
submit = pd.merge(sys_dia_tl_hdl, ldl, on=['vid'], how='inner')

submit.loc[submit['vid'] == '7b437e2632c91be2a0789adabce4b953', 'tl'] = 6
print(submit.describe())
print(submit.head(5))
print(submit.sort_values(by=['tl'], ascending=False).head(15))

juz_submit = DataFrame(submit)
juz_submit.persist('meinian_round2_submit_b')