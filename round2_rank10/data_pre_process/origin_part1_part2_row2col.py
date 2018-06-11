import time
import pandas as pd
from odps import ODPS
from odps.df import DataFrame

import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

# 读取数据
part_1 = odps.get_table('meinian_round2_data_part1').to_df().to_pandas()
part_2 = odps.get_table('meinian_round2_data_part2').to_df().to_pandas()
part_1_2 = pd.concat([part_1,part_2])
part_1_2 = pd.DataFrame(part_1_2).sort_values('vid').reset_index(drop=True)
begin_time = time.time()

# 重复数据的拼接操作
def merge_table(df):
    df['results'] = df['results'].astype(str)
    if df.shape[0] > 1:
        merge_df = "$".join(list(df['results']))
    else:
        merge_df = df['results'].values[0]
    return merge_df
# 数据简单处理
print(part_1_2.shape)
is_happen = part_1_2.groupby(['vid','test_id']).size().reset_index()
# 重塑index用来去重
is_happen['new_index'] = is_happen['vid'] + '_' + is_happen['test_id']
is_happen_new = is_happen[is_happen[0]>1]['new_index']

part_1_2['new_index'] = part_1_2['vid'] + '_' + part_1_2['test_id']

unique_part = part_1_2[part_1_2['new_index'].isin(list(is_happen_new))]
unique_part = unique_part.sort_values(['vid','test_id'])
no_unique_part = part_1_2[~part_1_2['new_index'].isin(list(is_happen_new))]
print('begin')
part_1_2_not_unique = unique_part.groupby(['vid','test_id']).apply(merge_table).reset_index()
part_1_2_not_unique.rename(columns={0:'results'},inplace=True)
tmp = pd.concat([part_1_2_not_unique,no_unique_part[['vid','test_id','results']]])
# 行列转换
print('finish')
tmp = tmp.pivot(index='vid',values='results',columns='test_id')
print(tmp.shape)
combine_data = DataFrame(tmp,unknown_as_string=True)
combine_data.persist('origin_data_combine_part1_part2')
print('total time',time.time() - begin_time)