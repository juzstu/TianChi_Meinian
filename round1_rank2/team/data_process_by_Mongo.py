#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2018/4/10 0010 上午 9:10
# @Author  : Juzphy

import pandas as pd
import time
from pymongo import MongoClient
from collections import defaultdict


'''
    save data to MongoDB and export the mongo data to csv file.
'''


def feature_data():
    df = pd.read_csv('../data/meinian_round1_data_part1_20180408.txt', delimiter='$')
    df2 = pd.read_csv('./data/meinian_round1_data_part2_20180408.txt', delimiter='$')
    df = df.append(df2, ignore_index=True)
    df = df.fillna('')
    return df


def data_load(path):
    data = pd.read_csv(path, encoding='gbk')
    data = data.fillna('')
    return data


def match(feature, data_frame, hostname, db_name, set_name, port=27017):
    count = 0
    time_start = time.time()
    df_group = (d for d in feature.groupby(by='vid'))
    vid_value = set(data_frame['vid'].values)
    mongo_conn = MongoClient(hostname, port)
    db_set = mongo_conn[db_name][set_name]
    for j in df_group:
        if j[0] in vid_value:
            count += 1
            temp_dict = data_frame[data_frame['vid'] == j[0]].to_dict()
            temp_dict = {'_id' if k == 'vid' else k: n for k, v in temp_dict.items() for n in v.values()}
            if len(j[1]['table_id']) > len(j[1]['table_id'].unique()):
                j[1].index = range(j[1].shape[0])
                table_dict = defaultdict(int)
                for t in j[1]['table_id']:
                    table_dict[t] += 1
                beyond_one = [k for k, v in table_dict.items() if v > 1]
                other = [k for k, v in table_dict.items() if v == 1]
                other_index = j[1][j[1]['table_id'].isin(other)].index
                temp = dict(zip(j[1]['table_id'].iloc[other_index], j[1]['field_results'].iloc[other_index]))
                beyond_dict = {k: '$'.join(j[1]['field_results'].iloc[j[1][j[1]['table_id'] == k].index].fillna('')) for
                               k in beyond_one}
                temp.update(beyond_dict)
            else:
                temp = dict(zip(j[1]['table_id'], j[1]['field_results']))
            temp_dict.update(temp)
            db_set.save(temp_dict)
            print('vid of {} has writen.'.format(j[0]))
    print("total writen {0} records and spend {1} s.".format(count, round(time.time() - time_start), 2))


def feature_count(hostname, db_name, set_name, port=27017, count_threshold=0.4, name='train_set'):
    mongo_conn = MongoClient(hostname, port)
    mongo_set = mongo_conn[db_name][set_name]
    cursor = mongo_set.find()
    feature_dict = defaultdict(list)
    size = 0
    for c in cursor:
        size += 1
        for k in c.keys():
            if k != '_id':
                feature_dict[k].append(c['_id'])

    feature_lt_threshold = [code for code, b_list in feature_dict.items() if len(b_list)/size < count_threshold]
    feature_gt_threshold = set(feature_dict.keys()) - set(feature_lt_threshold)
    barcode_gt_threshold = list({f for fd in feature_gt_threshold for f in feature_dict[fd]})
    temp = {t: 0 for t in feature_lt_threshold}
    data = pd.DataFrame(list(mongo_set.find({"_id": {"$in": barcode_gt_threshold}}, temp)))
    data.to_csv('../data/{}.csv'.format(name), index=None, encoding='gbk')
    print('{}.csv has been successfully saved.'.format(name))


def mongo2csv(hostname, db_name, set_name, port=27017, name='train_set'):
    mongo_conn = MongoClient(hostname, port)
    mongo_set = mongo_conn[db_name][set_name]
    data = pd.DataFrame(list(mongo_set.find()))
    data.to_csv('./data/{}.csv'.format(name), index=None, encoding='gbk')
    print('{}.csv has been successfully saved.'.format(name))


if __name__ == '__main__':
    all_features = feature_data()
    # train = data_load('./data/origin_train.csv')
    # a_test = data_load('./data/origin_test_a.csv')
    b_test = data_load('./data/origin_test_b.csv')
    # print(train.shape, a_test.shape)
    match(all_features, b_test, 'localhost', 'meinian', 'test_b')
    # match(feature_data, a_test, 'localhost', 'meinian', 'test_data')
    # feature_count('10.10.0.7', 'meinian', 'train_data', name='new_meinian_train')
    # feature_count('10.10.0.7', 'meinian', 'test_data', name='new_meinian_test')
    mongo2csv('localhost', 'meinian', 'test_b', name='meinian_test_b')
