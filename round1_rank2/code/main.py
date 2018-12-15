#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2018/5/7 0007 下午 13:02
# @Author  : Juzphy
import time
import pandas as pd
import lightgbm as lgb
import numpy as np
import re
from collections import Iterable
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
import warnings
from team.team_feature_work import get_file

warnings.filterwarnings('ignore')


class DataPreProcess(object):
    def __init__(self, threshold):
        self.thresh = threshold

    def pre_process(self):
        # 过滤掉无用的table_id
        def filter_none(data):
            data = data[data['field_results'] != '']
            data = data[data['field_results'] != '未查']
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
        def get_remain_feats(df):
            exclude_feats = set()
            print('----------目前移除缺失的阈值为{}-----------'.format(self.thresh))
            print('----------移除数据缺失多的字段-----------')
            print('移除之前总的字段数量', len(df.columns))
            num_rows = df.shape[0]
            for c in df.columns:
                num_missing = df[c].isnull().sum()
                if num_missing == 0:
                    continue
                missing_percent = num_missing / float(num_rows)
                if missing_percent > self.thresh:
                    exclude_feats.add(c)
            print("移除后数据的字段数量: %s" % len(exclude_feats))
            # 保留超过阈值的特征
            remain_feats = set(df.columns) - exclude_feats
            print('剩余的字段数量', len(remain_feats))
            return list(remain_feats)

        origin_train = pd.read_csv('../data/meinian_round1_train_20180408.csv', sep=',', encoding='gbk')
        origin_test = pd.read_csv('../data/meinian_round1_test_b_20180505.csv', sep=',', encoding='gbk')
        data_part1 = pd.read_csv('../data/meinian_round1_data_part1_20180408.txt', sep='$', encoding='utf-8')
        data_part2 = pd.read_csv('../data/meinian_round1_data_part2_20180408.txt', sep='$', encoding='utf-8')
        # data_part1和data_part2进行合并，并剔除掉与train、test不相关vid所在的行
        # {0/'index', 1/'columns'}, default 0
        part1_2 = pd.concat([data_part1, data_part2], axis=0)
        part1_2 = pd.DataFrame(part1_2).sort_values('vid').reset_index(drop=True)
        vid_set = pd.concat([origin_train['vid'], origin_test['vid']], axis=0)
        vid_set = pd.DataFrame(vid_set).sort_values('vid').reset_index(drop=True)
        part1_2 = part1_2[part1_2['vid'].isin(vid_set['vid'])]
        part1_2 = filter_none(part1_2)
        print(part1_2.shape)
        vid_tabid_group = part1_2.groupby(['vid', 'table_id']).size().reset_index()
        print('------------------------------去重和组合-----------------------------')
        vid_tabid_group['new_index'] = vid_tabid_group['vid'] + '_' + vid_tabid_group['table_id']
        vid_tabid_group_dup = vid_tabid_group[vid_tabid_group[0] > 1]['new_index']
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
        merge_part1_2.to_csv('../data/merge_part1_2_{}.csv'.format(self.thresh), encoding='utf-8')
        del merge_part1_2
        time.sleep(10)
        print('------------------------重新读取数据merge_part1_2--------------------------')
        merge_part1_2 = pd.read_csv('../data/merge_part1_2_{}.csv'.format(self.thresh), sep=',', encoding='utf-8')
        print('--------------新的part1_2组合完毕----------')
        print(merge_part1_2.shape)
        feats = get_remain_feats(merge_part1_2)
        return merge_part1_2[feats]


class FeatureWork(object):
    def __init__(self, thresh_num):
        self.thresh_num = thresh_num

    def get_features(self):
        # 脂肪肝程度
        def transform_101_102_113(df):
            if df:
                if '脂肪肝趋势' in df:
                    return 1
                elif '轻度' in df:
                    if '中' not in df:
                        return 2
                    else:
                        return 3
                elif '中度' in df:
                    if '重' not in df:
                        return 3
                    else:
                        return 4
                elif '重度' in df:
                    return 4
                else:
                    return 0
            else:
                return np.nan

        def transform_2302(df):
            try:
                if '健康' in df:
                    if '亚健康' in df:
                        return 1
                    else:
                        return 0
                elif '疾病' in df:
                    return 2
            except Exception:
                return df

        def high_sugar(df):
            if df:
                if '血糖偏高' in df or '降糖' in df or '血糖' in df:
                    return 1
                else:
                    return 0
            else:
                return np.nan

        def high_fat(df):
            if df:
                if '血脂偏高' in df or '低脂' in df or '血脂' in df:
                    return 1
                else:
                    return 0
            else:
                return np.nan

        def high_pressure(df):
            if df:
                if '血压偏高' in df or '降压' in df or '血压' in df:
                    return 1
                else:
                    return 0
            else:
                return np.nan

        def higher_pressure(df):
            if df:
                if '血压偏高' not in df:
                    if '高血压' in df:
                        return 1
                    else:
                        return 0
            else:
                return np.nan

        def higher_fat(df):
            if df:
                if '血脂偏高' not in df:
                    if '高血脂' in df:
                        return 1
                    else:
                        return 0
            else:
                return np.nan

        def higher_sugar(df):
            if df:
                if '血糖偏高' not in df:
                    if '高血糖' in df or '糖尿病' in df:
                        return 1
                    else:
                        return 0
            else:
                return np.nan

        def fatty_liver(df):
            if df:
                if '脂肪肝' in df:
                    return 1
                else:
                    return 0
            else:
                return np.nan

        def coronary_heart_disease(df):
            if df:
                if '冠心病' in df or '冠状' in df:
                    return 1
                else:
                    return 0
            else:
                return np.nan

        def kidney(df):
            if df:
                if '肾' in df:
                    return 1
                else:
                    return 0
            else:
                return np.nan

        def smoke(df):
            if df:
                if '烟' in df:
                    return 1
                else:
                    return 0
            else:
                return np.nan

        def strQ2B(df):
            """全角转半角"""
            if isinstance(df, Iterable):
                rstring = ""
                for uchar in df:
                    inside_code = ord(uchar)
                    # 全角空格直接转换
                    if inside_code == 12288:
                        inside_code = 32
                    elif 65281 <= inside_code <= 65374:
                        inside_code -= 65248
                    rstring += chr(inside_code)
                return rstring
            else:
                return df

        def extract_num(df):
            try:
                df = float(df)
                if df <= 0:
                    return np.nan
                return df
            except Exception:
                if '.' in df:
                    temp = re.findall('(\d+\.\d+)', df)
                else:
                    temp = re.findall('(\d+)', df)
                if temp:
                    return float(temp[0])
                else:
                    return np.nan

        def blood_pipe_style(df):
            try:
                if '良好' in df or '正常' in df:
                    return 0
                elif '趋势' in df:
                    return 1
                elif '轻度' in df:
                    return 2
                elif '中度' in df:
                    return 3
                elif '重度' in df:
                    return 4
                elif '硬化' in df:
                    return 5
                else:
                    return np.nan
            except Exception:
                return df

        def ying_yang(df):
            try:
                if '+' in df and '-' in df:
                    return 1
                elif '+' in df and '-' not in df:
                    return 2
                elif ('-' in df or '阴' in df or '正常' in df or 'Normal' in df) and '+' not in df:
                    return 0
                else:
                    return 0
            except Exception:
                return df

        def HP_yy(df):
            try:
                if '阳' in df:
                    return 1
                else:
                    return 0
            except Exception:
                return df

        # 尿
        def urine(df):
            try:
                if '>=' in df:
                    return 1
                else:
                    return 0
            except Exception:
                return df

        def heart_rate(df):
            try:
                if df != '强弱不等':
                    if '弱' in df or '远' in df or '低' in df:
                        return 1
                    elif '强' in df or '力' in df:
                        return 3
                    else:
                        return 0
                else:
                    return 2
            except Exception:
                return df

        def transform_421(df):
            try:
                if '齐' in df and '不' not in df:
                    return 0
                else:
                    return 1
            except Exception:
                return df

        def transform_430(df):
            try:
                if df == '软':
                    return 1
                elif df == '中':
                    return 2
                elif df == '硬':
                    return 3
                else:
                    return 0
            except Exception:
                return df

        def transform_403(df):
            try:
                if '大' in df and '无' not in df:
                    return 1
                else:
                    return 0
            except Exception:
                return df

        def transform_3399(df):
            try:
                if df == '黄色' or df == 'yellow':
                    return 2
                elif df == '淡黄色' or df == '浅黄色':
                    return 1
                elif df == '无色':
                    return 0
                elif '红' in df:
                    return 3
                elif df == '混浊':
                    return 4
                else:
                    return 5
            except Exception:
                return df

        def lung_voice(df):
            try:
                if '干啰' in df:
                    return 1
                elif '湿啰' in df:
                    return 2
                elif '哮鸣' in df:
                    return 3
                elif '湿鸣' in df:
                    return 4
                else:
                    return 0
            except Exception:
                return df

        def one_hot(data_frame):
            one_hot_list = ['101', '102', '113', '409', '413', '434', '439', 'A201', 'A202', '4001', '705', 'A301', '709', '985']
            data_frame.loc[:, one_hot_list] = data_frame.loc[:, one_hot_list].fillna('')
            data_frame['4001'] = data_frame['4001'].astype(str)
            data_frame['705'] = data_frame['705'].astype(str)
            data_frame['709'] = data_frame['709'].astype(str)
            data_frame['A301'] = data_frame['A301'].astype(str)
            data_frame['985'] = data_frame['985'].astype(str)
            data_frame['439'] = data_frame['439'].astype(str)
            frame_409_434 = data_frame['409'] + data_frame['434'] + data_frame['413'] + data_frame['4001'] + \
                            data_frame['A201'] + data_frame['A301'] + data_frame['A202'] + data_frame['705'] + \
                            data_frame['709'] + data_frame['985'] + data_frame['439']
            data_frame['血压偏高'] = frame_409_434.apply(high_pressure)
            data_frame['血脂偏高'] = frame_409_434.apply(high_fat)
            data_frame['血糖偏高'] = frame_409_434.apply(high_sugar)
            data_frame['高血糖'] = frame_409_434.apply(higher_sugar)
            data_frame['高血脂'] = frame_409_434.apply(higher_fat)
            data_frame['高血压'] = frame_409_434.apply(higher_pressure)
            data_frame['脂肪肝'] = frame_409_434.apply(fatty_liver)
            data_frame['冠心病'] = frame_409_434.apply(coronary_heart_disease)
            data_frame['肾问题'] = frame_409_434.apply(kidney)
            data_frame['吸烟'] = frame_409_434.apply(smoke)
            fat_liver_num = data_frame['101'] + data_frame['102'] + data_frame['113']
            data_frame['脂肪肝程度'] = fat_liver_num.apply(transform_101_102_113)

        def cm2mm(df):
            try:
                if 'cm' in df:
                    temp_cm = re.findall('\d+(?:\.\d+)?.*?x?\d+(?:\.\d+)?', df)
                    if temp_cm:
                        return float(temp_cm[0][0]) * float(temp_cm[0][1]) * 100
                elif 'mm' in df:
                    temp_mm = re.findall('\d+(?:\.\d+)?.*?x?\d+(?:\.\d+)?', df)
                    if temp_mm:
                        return float(temp_mm[0][0]) * float(temp_mm[0][1])
                else:
                    return np.nan
            except Exception:
                return np.nan

        def get_num_from_102_front(df):
            try:
                temp_x = re.findall('(\d+)/(\d+)', df)
                if temp_x:
                    return float(temp_x[0][0])
            except Exception:
                return np.nan

        def get_num_from_102_back(df):
            try:
                temp_x = re.findall('(\d+)/(\d+)', df)
                if temp_x:
                    return float(temp_x[0][1])
            except Exception:
                return np.nan

        def word2num(data_frame):
            drop_list = ['3193', '420', '431', '976', '429', '422', '423', '426', '3400', '3485', '3486', '30007']
            drop_list2 = ['101', '102', '113', '409', '413', '434', 'A201', 'A202', '4001', '705', 'A301', '709']
            drop_list3 = ['1001', '114', '116', '117', '118', '121', '985', '439']
            drop_list.extend(drop_list2)
            drop_list.extend(drop_list3)
            yy_list = ['3190', '3191', '3192', '3194', '3195', '3196', '3197', '3430', '100010']
            for y in yy_list:
                data_frame[y] = data_frame[y].apply(ying_yang)
            data_frame['尿比重'] = data_frame['3193'].apply(urine)
            data_frame['心音'] = data_frame['420'].apply(heart_rate)
            data_frame['430'] = data_frame['430'].apply(transform_430)
            data_frame['3399'] = data_frame['3399'].apply(transform_3399)
            data_frame['3301'] = data_frame['3301'].apply(HP_yy)
            data_frame['403'] = data_frame['3301'].apply(transform_403)
            data_frame['421'] = data_frame['421'].apply(transform_421)
            data_frame['405'] = data_frame['405'].apply(lung_voice)
            data_frame['gender'] = data_frame['121'].apply(lambda n: 1 if isinstance(n, Iterable) else 0)
            data_frame['血管弹性'] = data_frame['4001'].apply(blood_pipe_style)
            data_frame['2302'] = data_frame['2302'].apply(transform_2302)
            one_hot(data_frame)
            for x, y in zip(['113', '114', '116', '117', '118'], ['肝脏回声', '胆囊回声', '脾脏回声', '左肾回声', '右肾回声']):
                data_frame[x] = data_frame[x].apply(strQ2B)
                data_frame[x] = data_frame[x].apply(lambda n: n.lower().replace('×', 'x').replace('*', 'x') if
                isinstance(n, Iterable) else n)
                data_frame[y] = data_frame[x].apply(cm2mm)
            data_frame['血压_front'] = data_frame['102'].apply(get_num_from_102_front)
            data_frame['血压_back'] = data_frame['102'].apply(get_num_from_102_back)
            data_frame['心跳次数'] = data_frame['1001'].apply(extract_num)
            data_frame.drop(drop_list, axis=1, inplace=True)
            return data_frame

        def file_split(data_frame, path):
            with open(path, encoding='utf8') as f:
                feature_list = [i for i in f.read().split(', ')]
            features = data_frame[feature_list]
            return features

        def save_all_num(data_frame):
            for c in data_frame.columns:
                if c != 'vid':
                    data_frame[c] = data_frame[c].apply(extract_num)
                    q_num = data_frame[c].quantile(0.9) * 1.5
                    data_frame[c] = data_frame[c].apply(lambda x: x if x < q_num else np.nan)
            return data_frame

        # 添加队友的数据特征
        def add_new_feature(df_mine, df_team, save_path):
            columns = list(set(df_team.columns) - set(df_mine.columns) -
                           set(['舒张压', '收缩压', '血清高密度脂蛋白', '血清低密度脂蛋白', '血清甘油三酯']))
            columns.append('vid')
            new_data = df_team[columns]
            final_data = pd.merge(df_mine, new_data, on='vid')
            final_data.to_csv(save_path, encoding='utf8', index=False)

        dpp = DataPreProcess(threshold=self.thresh_num)
        all_data = dpp.pre_process()
        all_data.columns = [a[1:] if a.startswith('0') else a for a in all_data.columns]
        train_set = pd.read_csv('../data/meinian_round1_train_20180408.csv', sep=',', encoding='gbk')
        for t in train_set.columns:
            if t != 'vid':
                train_set[t] = train_set[t].apply(extract_num)
        test_set = pd.read_csv('../data/meinian_round1_test_b_20180505.csv', sep=',', encoding='gbk')
        num_data_temp = file_split(all_data, '../features/num_label.txt')
        word_data_temp = file_split(all_data, '../features/word_label.txt')
        num_data_temp.to_csv('../data/num_data.csv', encoding='utf8', index=False)
        word_data_temp.to_csv('../data/word_data.csv', encoding='utf8', index=False)
        num_data = pd.read_csv('../data/num_data.csv', encoding='utf8')
        word_data = pd.read_csv('../data/word_data.csv', encoding='utf8')
        num_data = save_all_num(num_data)
        word_data = word2num(word_data)
        transform_data = pd.merge(num_data, word_data, on='vid')
        train_of_part = transform_data[transform_data['vid'].isin(train_set['vid'])]
        test_of_part = transform_data[transform_data['vid'].isin(test_set['vid'])]
        train_set = pd.merge(train_set, train_of_part, on='vid')
        train_set.loc[train_set['vid'] == '7685d48685028a006c84070f68854ce1', '舒张压'] = 64
        train_set.loc[train_set['vid'] == 'fa04c8db6d201b9f705a00c3086481b0', '舒张压'] = 74
        train_set.loc[train_set['vid'] == 'de82a4130c4907cff4bfb96736674bbc', '血清低密度脂蛋白'] = 1.22
        train_set.loc[train_set['vid'] == 'd9919661f0a45fbcacc4aa2c1119c3d2', '血清低密度脂蛋白'] = 0.12
        train_set.loc[train_set['vid'] == '798d859a63044a8a5addf1f8c528629e', '血清低密度脂蛋白'] = 0.06
        test_set = pd.merge(test_set, test_of_part, on='vid')
        team_train, team_test = get_file()
        add_new_feature(train_set, team_train, '../data/train_set_merge.csv')
        add_new_feature(test_set, team_test, '../data/test_set_merge.csv')
        print('*************************训练集和测试集数据已成功写入。*************************')


class LGBRegression(object):
    def __init__(self):
        self.params = {
            'learning_rate': 0.01,
            'boosting_type': 'gbdt',
            'objective': 'mse',
            'num_leaves': 62,
            'reg_sqrt': True,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 2,
            'num_threads': -1,
            'min_data_in_leaf': 5,
            'verbose': -1
        }

    def eval_metric(self, pred, labels):
        return np.mean(np.power(np.log(pred + 1) - np.log(labels + 1), 2))

    def eval_error(self, pred, train_data):
        labels = train_data.get_label()
        score = np.mean(np.power(np.log(pred + 1) - np.log(labels + 1), 2))
        return 'meinian', score, False

    def lgb_regression_model(self, df, label, use_feature, true_test, submission_data):
        print("基于lightgbm： 开始训练 label 为{}...".format(label))
        value4preds = df[label]
        value4preds = value4preds[value4preds.isnull().values == False]
        df = df.iloc[value4preds.index]
        train_data = df.loc[:, use_feature]
        print(train_data.shape)
        scores = np.zeros(len(value4preds))
        submission_scores = np.zeros((len(submission_data), 5))
        num_round = 8000
        kf = KFold(n_splits=5, shuffle=True, random_state=1024)
        for t, (train_index, test_index) in enumerate(kf.split(train_data, value4preds), start=1):
            print('第{}次训练...'.format(t))
            x_train, x_test = train_data.iloc[train_index], train_data.iloc[test_index]
            y_train, y_test = value4preds.iloc[train_index], value4preds.iloc[test_index]
            lgb_train = lgb.Dataset(x_train, y_train)
            lgb_test = lgb.Dataset(x_test, y_test)
            gbm = lgb.train(self.params,
                            lgb_train,
                            num_boost_round=num_round,
                            valid_sets=lgb_test,
                            verbose_eval=100,
                            feval=self.eval_error,
                            early_stopping_rounds=100)
            scores[test_index] = gbm.predict(x_test)
            submission_scores[:, t - 1] = gbm.predict(true_test)
        submission_data[label] = np.mean(submission_scores, axis=1).round(3)
        return self.eval_metric(scores, value4preds)


class LGBClassification(object):
    def __init__(self):
        self.params = {
            'learning_rate': 0.01,
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            'num_leaves': 62,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 2,
            'verbose': -1,
            'min_data_in_leaf': 5,
        }

    # pos: 分类的分界线
    # df: 训练集
    # label: 主要针对血清甘油三脂(分界线4)和血清低密度脂蛋白(分界线5)
    # use_feature: 训练使用的特征
    # save_path: 分类结果保存路径
    def lgb_classification_model(self, pos, df, label, use_feature, test_class, save_path):
        print("开始训练分界线为{}...".format(pos))
        df['pos_{}'.format(pos)] = df[label].apply(lambda x: 1 if x > pos else 0)
        test_preds = df['pos_{}'.format(pos)]
        test4lgb = test_class.loc[:, use_feature]
        train_preds = df[use_feature]
        kf = KFold(n_splits=5, random_state=1024, shuffle=True)
        pred_labels = np.zeros(df.shape[0])
        submission_label = np.zeros((test4lgb.shape[0], 5))
        for t, (train_index, test_index) in enumerate(kf.split(train_preds, test_preds), start=1):
            print('第{}次训练...'.format(t))
            X_train, X_test = train_preds.iloc[train_index], train_preds.iloc[test_index]
            y_train, y_test = test_preds.iloc[train_index], test_preds.iloc[test_index]
            pos_weight = y_train.sum() / y_train.size
            print(pos_weight)
            self.params.update({'scale_pos_weight': pos_weight})
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_test = lgb.Dataset(X_test, y_test)
            gbm = lgb.train(self.params,
                            lgb_train,
                            num_boost_round=8000,
                            valid_sets=lgb_test,
                            verbose_eval=100,
                            early_stopping_rounds=100)
            pred_labels[X_test.index] = np.where(gbm.predict(X_test) > 0.5, 1, 0)
            self.params.pop('scale_pos_weight')
            submission_label[:, t - 1] = np.where(gbm.predict(test4lgb) > 0.5, 1, 0)
        test_class['pos_{}'.format(pos)] = np.where(np.sum(submission_label, axis=1) >= 1, 1, 0)
        print(classification_report(pred_labels, test_preds))
        test_class.to_csv(save_path, index=False, encoding='utf8')


if __name__ == "__main__":
    fw = FeatureWork(thresh_num=0.9)
    fw.get_features()
    train = pd.read_csv('../data/train_set_merge.csv', encoding='utf8', low_memory=False)
    test = pd.read_csv('../data/test_set_merge.csv', encoding='utf8', low_memory=False)
    print(train.shape, test.shape)
    predict_features = ['舒张压', '收缩压', '血清高密度脂蛋白', '血清低密度脂蛋白', '血清甘油三酯']
    train[predict_features] = train[predict_features]
    test[predict_features] = test[predict_features]
    use_features = [t for t in test.columns if t != 'vid' and t not in predict_features]
    test_data = test.loc[:, use_features]
    submission = test.loc[:, ['vid', '收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白']]
    base_line_score = np.zeros(5)
    start = time.time()
    lgb_reg = LGBRegression()
    for i, j in enumerate(predict_features):
        base_line_score[i] = lgb_reg.lgb_regression_model(train, j, use_features, test_data, submission)
    print(dict(zip(predict_features, base_line_score)))
    print('CV训练用时{}秒'.format(time.time() - start))
    print('线下得分为：', np.mean(base_line_score))
    date1 = time.strftime('%Y%m%d_%H%M%S')
    submission.to_csv('../submit/submit_{}.csv'.format(date1), index=None, header=None, encoding='utf8')
    time.sleep(10)
    lgr_class = LGBClassification()
    lgr_class.lgb_classification_model(4, train, '血清甘油三酯', use_features, test, '../data/fat_class_pos4.csv')
    time.sleep(10)
    reg_test = pd.read_csv('../data/fat_class_pos4.csv', encoding='utf8', low_memory=False)
    pos_eq_1 = reg_test[reg_test['pos_4'] == 1]
    test_eq_1 = pos_eq_1.loc[:, use_features]
    submission_gt_4 = pos_eq_1.loc[:, ['vid', '血清甘油三酯']]
    train_gt_4 = train[train['血清甘油三酯'] >= 4]
    train_gt_4.index = list(range(train_gt_4.shape[0]))
    lgb_reg.lgb_regression_model(train_gt_4, '血清甘油三酯', use_features, test_eq_1, submission_gt_4)
    submission_gt_4.to_csv('../data/submit_gt_4.csv', index=None, header=None, encoding='utf8')
    gt_4_index = submission[submission['vid'].isin(submission_gt_4['vid'])].index
    submission_temp = submission.loc[gt_4_index, ['vid', '血清甘油三酯']]
    merge_fat = pd.merge(submission_temp, submission_gt_4, on='vid')
    temp_columns = [tc for tc in merge_fat.columns if tc != 'vid']
    replace_num = np.max(merge_fat.loc[:, temp_columns], axis=1)
    submission.loc[gt_4_index, '血清甘油三酯'] = replace_num.values
    date2 = time.strftime('%Y%m%d_%H%M%S')
    submission.to_csv('../submit/submit_{}.csv'.format(date2), index=None, encoding='utf8')
