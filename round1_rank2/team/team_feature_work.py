#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2018/5/8 0008 下午 16:40
# @Author  : Juzphy

import time
import pandas as pd
from math import isnan
start_time=time.time()

def filter_None(data):
    data=data[data['field_results']!='']
    data=data[data['field_results']!='未查']
    return data

# 重复数据的拼接操作
def merge_table(df):
    df['field_results'] = df['field_results'].astype(str)
    if df.shape[0] > 1:
        merge_df = " ".join(list(df['field_results']))
    else:
        merge_df = df['field_results'].values[0]
    return merge_df

# 删除掉一些出现次数低，缺失比例大的字段，保留超过阈值的特征
def remain_feat(df,thresh=0.9):
    exclude_feats = []
    print('----------移除数据缺失多的字段-----------')
    print('移除之前总的字段数量',len(df.columns))
    num_rows = df.shape[0]
    for c in df.columns:
        num_missing = df[c].isnull().sum()
        if num_missing == 0:
            continue
        missing_percent = num_missing / float(num_rows)
        if missing_percent > thresh:
            exclude_feats.append(c)
    print("移除缺失数据的字段数量: %s" % len(exclude_feats))
    # 保留超过阈值的特征
    feats = []
    for c in df.columns:
        if c not in exclude_feats:
            feats.append(c)
    print('剩余的字段数量',len(feats))
    return feats

def map_deal_3601(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        if "严重" in temp:
            return 4
        elif "中度" in temp:
            return 3
        elif "减少" in temp or "降低" in temp or "疏松":
            return 2
        else:
            return 1

def map_deal_0102(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "脂肪" in temp:
            if "重" in temp:
                value = 4
            elif "中" in temp:
                value = 3
            elif "轻" in temp:
                value = 2
            else:
                value = 1
        else:
            value = 0.0
        if "多发" in temp:
            value += 0.5
    return value

def map_deal_0113(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "弥漫性" in temp:
            value = 5
        if "欠清晰" in temp:
            value += 2
        if "粗" in temp:
            value += 0.5
        if "多发" in temp:
            value += 0.5
        if "斑点状" in temp:
            value += 1
        if "回声区" in temp:
            value += 1
    return value

def map_deal_0114(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "毛糙" in temp:
            value = 4
        if "强回声" in temp:
            value += 1
    return value

def map_deal_0115(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "不清晰" in temp:
            value = 4
        if "增强" in temp:
            value += 1
    return value

def map_deal_0115(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "不清晰" in temp:
            value = 4
        if "增强" in temp:
            value += 1
    return value

def map_deal_0116(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "不清晰" in temp:
            value = 4
        if "增强" in temp:
            value += 1
    return value

def map_deal_0117(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "强回声" in temp:
            value = 4
        if "无回声" in temp:
            value += 1
        if "欠均匀" in temp:
            value += 1
    return value

def map_deal_0118(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "强回声" in temp:
            value = 4
        if "无回声" in temp:
            value += 1
        if "欠均匀" in temp:
            value += 1
    return value

def map_deal_0118(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "强回声" in temp:
            value = 4
        if "无回声" in temp:
            value += 1
        if "欠均匀" in temp:
            value += 1
    return value

def map_deal_0503(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "分泌物多" in temp:
            value = 8
        if "分泌物中" in temp:
            value = 5
        if "分泌物少" in temp:
            value = 3
        if "浓性" in temp:
            value += 1
        if "充血" in temp:
            value += 1
        if "黄色" in temp:
            value += 0.5
    return value

def map_deal_0509(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "充血" in temp:
            value = 8
        if "肥大" in temp:
            value = 5
        if "轻糜" in temp:
            value += 1
        if "中糜" in temp:
            value += 1.5
        if "囊" in temp:
            value += 0.5
    return value

def map_deal_0516(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "前位" in temp:
            value = 8
        if "后位" in temp:
            value = 5
        if "平位" in temp:
            value = 3
        if "增大" in temp:
            value += 1
        if "硬" in temp:
            value += 0.5
    return value

def map_deal_0539(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "分泌物" in temp:
            value += 1
        if "肥大" in temp:
            value += 2
        if "充血" in temp:
            value += 3
        if "炎" in temp:
            value += 0.5
    return value

def map_deal_2302(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "亚健康" in temp:
            value = 3
        else:
            value = 1
    return value

def map_deal_1316(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "正常" in temp or "未见" in temp:
            pass
        else:
            value += 2
    return value

def map_deal_0101(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "低回声" in temp or "回声区" in temp:
            value += 1
    return value

def map_deal_0119(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "欠佳" in temp:
            value = 2
    return value

def map_deal_0121(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "低回声" in temp or "回声区" in temp:
            value += 1
    return value

def map_deal_0122(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "回声团" in temp or "回声区" in temp:
            value += 1
    return value

def map_deal_0123(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "回声团" in temp or "回声区" in temp:
            value += 1
    return value

def map_deal_A705(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "衰减" in temp:
            value += 5
    return value

def map_deal_0911(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "肿大" in temp:
            value += 2
    return value

def map_deal_0912(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "无肿大" in temp or "未见" in temp:
            pass
        else:
            value += 2
    return value

def map_deal_0929(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "不全" in temp:
            value = 3
        if "增生" in temp:
            value = 6
    return value

def map_deal_A202(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "陈旧" in temp:
            value = 5
        if "灶" in temp:
            value += 1
    return value

def map_deal_1102(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "增生" in temp:
            value += 1
    return value

def map_deal_0208(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "正常" in temp or "未见" in temp:
            pass
        else:
            value += 1
    return value

def map_deal_0209(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "正常" in temp or "未见" in temp:
            pass
        else:
            value += 1
    return value

def map_deal_0210(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "正常" in temp or "未见" in temp:
            pass
        else:
            value += 1
    return value

def map_deal_0215(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "充血" in temp:
            value = 5

        if "正常" in temp or "未见" in temp:
            pass
        else:
            value += 1
    return value

def map_deal_0217(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "肿" in temp:
            value = 3
    return value

def map_deal_4001(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "轻度" in temp:
            value = 3
        if "中度" in temp:
            value = 5
        if "重度" in temp:
            value = 8
    return value

def map_deal_1001(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "过缓" in temp or "不齐" in temp or "偏" in temp:
            value += 3
    return value

def map_deal_0409(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "血压" in temp:
            value += 9
        if "糖尿" in temp:
            value += 3
        if "脂肪" in temp:
            value += 5
    return value

def map_deal_0421(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "不齐" in temp:
            value += 3
    return value

def map_deal_0424(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "次" in temp:
            if "70" in temp:
                value = 70
            else:
                value = 80
    return value

def map_deal_0434(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "血压" in temp:
            value += 9
        if "糖尿" in temp:
            value += 3
        if "脂肪" in temp:
            value += 5
        if "心" in temp:
            value += 1
    return value

def map_deal_1402(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "硬" in temp:
            value += 5
        if "低" in temp:
            value += 1
        if "慢" in temp:
            value += 1
    return value

def map_deal_0120(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "强回声" in temp:
            value += 5
        if "低" in temp:
            value += 1
    return value

def map_deal_0984(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "增" in temp:
            value += 5
    return value

def map_deal_100010(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "+" in temp:
            value += 5
    return value

def map_deal_3190(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "+" in temp:
            value += 5
    return value

def map_deal_3191(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "+" in temp:
            value += 5
    return value

def map_deal_3192(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "+" in temp:
            value += 5
    return value


def map_deal_3195(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "+" in temp:
            value += 5
    return value


def map_deal_3196(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "+" in temp:
            value += 5
    return value


def map_deal_3197(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "+" in temp:
            value += 5
    return value


def map_deal_3430(temp):
    value = 0
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        if "+" in temp:
            value += 5
    return value


def map_deal_3399(temp):
    try:
        if isnan(float(temp)):
            return -1
        else:
            return float(temp)
    except Exception:
        temp = str(temp)
        value = 0
        if "淡" in temp:
            value += 5
    return value


item_list = ['3601', '0102', '0113', '0114', '0115', '0116',
             '0117', '0118', '0503', '0509', '0516', '0539',
             '2302', '1316', '0101', '0119', '0121', '0122',
             '0123', 'A705', '0911', '0912', '0929', 'A202',
             '1102', '0208', '0209', '0210', '0215', '0217',
             '4001', '1001', '0409', '0421', '0424', '0434',
             '1402', '0120', '0984', '100010', '3190', '3191',
             '3192', '3195', '3196', '3197', '3430', '3399']

map_list = [map_deal_3601, map_deal_0102, map_deal_0113, map_deal_0114, map_deal_0115, map_deal_0116,
            map_deal_0117, map_deal_0118, map_deal_0503, map_deal_0509, map_deal_0516, map_deal_0539,
            map_deal_2302, map_deal_1316, map_deal_0101, map_deal_0119, map_deal_0121, map_deal_0122,
            map_deal_0123, map_deal_A705, map_deal_0911, map_deal_0912, map_deal_0929, map_deal_A202,
            map_deal_1102, map_deal_0208, map_deal_0209, map_deal_0210, map_deal_0215, map_deal_0217,
            map_deal_4001, map_deal_1001, map_deal_0409, map_deal_0421, map_deal_0424, map_deal_0434,
            map_deal_1402, map_deal_0120, map_deal_0984, map_deal_100010, map_deal_3190, map_deal_3191,
            map_deal_3192, map_deal_3195, map_deal_3196, map_deal_3197, map_deal_3430, map_deal_3399
           ]


def get_file():
    train = pd.read_csv('../data/meinian_round1_train_20180408.csv', sep=',', encoding='gbk')
    test = pd.read_csv('../data//meinian_round1_test_b_20180505.csv', sep=',', encoding='gbk')
    data_part1 = pd.read_csv('../data/meinian_round1_data_part1_20180408.txt', sep='$', encoding='utf-8')
    data_part2 = pd.read_csv('../data/meinian_round1_data_part2_20180408.txt', sep='$', encoding='utf-8')

    # data_part1和data_part2进行合并，并剔除掉与train、test不相关vid所在的行
    part1_2 = pd.concat([data_part1, data_part2], axis=0)  # {0/'index', 1/'columns'}, default 0
    part1_2 = pd.DataFrame(part1_2).sort_values('vid').reset_index(drop=True)
    vid_set = pd.concat([train['vid'], test['vid']], axis=0)
    vid_set = pd.DataFrame(vid_set).sort_values('vid').reset_index(drop=True)
    part1_2 = part1_2[part1_2['vid'].isin(vid_set['vid'])]
    # 根据常识判断无用的'检查项'table_id，过滤掉无用的table_id
    part1_2 = filter_None(part1_2)
    # 数据简单处理
    print(part1_2.shape)
    vid_tabid_group = part1_2.groupby(['vid', 'table_id']).size().reset_index()
    print('------------------------------去重和组合-----------------------------')
    vid_tabid_group['new_index'] = vid_tabid_group['vid'] + '_' + vid_tabid_group['table_id']
    vid_tabid_group_dup = vid_tabid_group[vid_tabid_group[0] > 1]['new_index']

    # print(vid_tabid_group_dup.head()) #000330ad1f424114719b7525f400660b_0102
    part1_2['new_index'] = part1_2['vid'] + '_' + part1_2['table_id']

    dup_part = part1_2[part1_2['new_index'].isin(list(vid_tabid_group_dup))]
    dup_part = dup_part.sort_values(['vid', 'table_id'])
    unique_part = part1_2[~part1_2['new_index'].isin(list(vid_tabid_group_dup))]

    part1_2_dup = dup_part.groupby(['vid', 'table_id']).apply(merge_table).reset_index()
    part1_2_dup.rename(columns={0: 'field_results'}, inplace=True)
    part1_2_res = pd.concat([part1_2_dup, unique_part[['vid', 'table_id', 'field_results']]])

    # 行列转换
    print('--------------------------重新组织index和columns---------------------------')
    merge_part1_2 = part1_2_res.pivot(index='vid', values='field_results', columns='table_id')
    merge_part1_2.to_csv('../data/merge_part1_2.csv', encoding='utf-8')
    del merge_part1_2
    time.sleep(10)
    print('------------------------重新读取数据merge_part1_2--------------------------')
    merge_part1_2 = pd.read_csv('../data/merge_part1_2.csv', sep=',', encoding='utf-8')
    print('--------------新的part1_2组合完毕----------')
    print(merge_part1_2.shape)
    feats = remain_feat(merge_part1_2, thresh=0.96)
    merge_part1_2 = merge_part1_2[feats]

    for i in range(len(item_list)):
        merge_part1_2[item_list[i]] = merge_part1_2[item_list[i]].apply(map_list[i])

    tran_kind_dict = {}
    for x in merge_part1_2.columns:
        if merge_part1_2[x].dtype == 'object':
            a = len(merge_part1_2[x].unique())
            tran_kind_dict[x] = a

    drop_list = []
    onehot_list = []
    for x in tran_kind_dict.keys():

        if tran_kind_dict[x] <= 200:
            onehot_list.append(x)
        else:
            if x != 'vid':
                drop_list.append(x)

    from sklearn import preprocessing
    lbl = preprocessing.LabelEncoder()
    for x in onehot_list:
        merge_part1_2[x] = lbl.fit_transform(merge_part1_2[x].map(lambda x: str(x)))

    merge_part1_2.drop(drop_list, axis=1, inplace=True)
    merge_part1_2 = merge_part1_2.convert_objects(convert_numeric=True)
    train_of_part = merge_part1_2[merge_part1_2['vid'].isin(train['vid'])]
    test_of_part = merge_part1_2[merge_part1_2['vid'].isin(test['vid'])]
    train = pd.merge(train, train_of_part, on='vid')
    test = pd.merge(test, test_of_part, on='vid')
    return train, test


def do_map(merge_part1_2):
    for i in range(len(item_list)):
        merge_part1_2[item_list[i]] = merge_part1_2[item_list[i]].apply(map_list[i])

    merge_part1_2.info()
    tran_kind_dict = {}
    for x in merge_part1_2.columns:
        if merge_part1_2[x].dtype == 'object':
            a = len(merge_part1_2[x].unique())
            tran_kind_dict[x] = a

    drop_list = []
    onehot_list = []
    for x in tran_kind_dict.keys():

        if tran_kind_dict[x] <= 200:
            onehot_list.append(x)
        else:
            if x != 'vid':
                drop_list.append(x)

    from sklearn import preprocessing
    lbl = preprocessing.LabelEncoder()
    for x in onehot_list:
        merge_part1_2[x] = lbl.fit_transform(merge_part1_2[x].map(lambda x: str(x)))

    merge_part1_2.drop(drop_list, axis=1, inplace=True)
    merge_part1_2 = merge_part1_2.convert_objects(convert_numeric=True)
    return merge_part1_2
