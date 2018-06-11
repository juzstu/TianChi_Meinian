from odps import ODPS
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge
from odps.df import DataFrame
from sklearn.model_selection import KFold
import time
import numpy as np


def eval_metric(pred, labels):
	return np.mean(np.power(np.log(pred + 1) - np.log(labels + 1), 2))

def gbdt_model(df, label, use_feature, true_test, submission_data, gbdt_model):
    print("基于GBDT： 开始训练 label 为{}...".format(label))
    value4preds = np.log(df[label])
    train_data = df.loc[:, use_feature]
    print(train_data.shape)
    scores = np.zeros(len(value4preds))
    submission_scores = np.zeros((len(submission_data), 5))
    kf = KFold(n_splits=5, shuffle=True, random_state=1024)
    five_fold_index = list(kf.split(train_data, value4preds))
    
    train_index_1, test_index_1 = five_fold_index[0]
    print('第1次训练...')
    x_train_1, x_test_1 = train_data.iloc[train_index_1], train_data.iloc[test_index_1]
    y_train_1, y_test_1 = value4preds.iloc[train_index_1], value4preds.iloc[test_index_1]
    gbdt_model.fit(x_train_1, y_train_1)
    scores[test_index_1] = np.exp(gbdt_model.predict(x_test_1))
    submission_scores[:, 0] = gbdt_model.predict(true_test)
    print('第1次训练结束')
    print('*******************************************************************')
    train_index_2, test_index_2 = five_fold_index[1]
    print('第2次训练...')
    x_train_2, x_test_2 = train_data.iloc[train_index_2], train_data.iloc[test_index_2]
    y_train_2, y_test_2 = value4preds.iloc[train_index_2], value4preds.iloc[test_index_2]
    gbdt_model.fit(x_train_2, y_train_2)
    scores[test_index_2] = np.exp(gbdt_model.predict(x_test_2))
    submission_scores[:, 1] = gbdt_model.predict(true_test)
    print('第2次训练结束')
    print('*******************************************************************')
    train_index_3, test_index_3 = five_fold_index[2]
    print('第3次训练...')
    x_train_3, x_test_3 = train_data.iloc[train_index_3], train_data.iloc[test_index_3]
    y_train_3, y_test_3 = value4preds.iloc[train_index_3], value4preds.iloc[test_index_3]
    gbdt_model.fit(x_train_3, y_train_3)
    scores[test_index_3] = np.exp(gbdt_model.predict(x_test_3))
    submission_scores[:, 2] = gbdt_model.predict(true_test)
    print('第3次训练结束')
    print('*******************************************************************')
    train_index_4, test_index_4 = five_fold_index[3]
    print('第4次训练...')
    x_train_4, x_test_4 = train_data.iloc[train_index_4], train_data.iloc[test_index_4]
    y_train_4, y_test_4 = value4preds.iloc[train_index_4], value4preds.iloc[test_index_4]
    gbdt_model.fit(x_train_4, y_train_4)
    scores[test_index_4] = np.exp(gbdt_model.predict(x_test_4))
    submission_scores[:, 3] = gbdt_model.predict(true_test)
    print('第4次训练结束')
    print('*******************************************************************')
    train_index_5, test_index_5 = five_fold_index[4]
    print('第5次训练...')
    x_train_5, x_test_5 = train_data.iloc[train_index_5], train_data.iloc[test_index_5]
    y_train_5, y_test_5 = value4preds.iloc[train_index_5], value4preds.iloc[test_index_5]
    gbdt_model.fit(x_train_5, y_train_5)
    scores[test_index_5] = np.exp(gbdt_model.predict(x_test_5))
    submission_scores[:, 4] = gbdt_model.predict(true_test)
    print('第5次训练结束')
    print('*******************************************************************')
    submission_data[label] = np.exp(np.mean(submission_scores, axis=1)).round(3)

    
# A榜使用了tl的高低值分类，B榜没有
if __name__ == "__main__":
    train = odps.get_table('juz_train_6_6').to_df().to_pandas()
    test = odps.get_table('juz_test_6_6').to_df().to_pandas()
    submission = odps.get_table('tl_jz_5_fold_6_6_submit_22').to_df().to_pandas()
    vid_gt_4 = odps.get_table('tl_gt_4_vid_6_6').to_df().to_pandas()['vid']
    predict_features = ['sys', 'dia', 'tl', 'hdl', 'ldl']
    use_features = [t for t in train.columns if t != 'vid' and t not in predict_features and t != 'pos_4' and not 'log' in t]
    pos_eq_1 = test[test['vid'].isin(vid_gt_4)]
    test_eq_1 = pos_eq_1.loc[:, use_features]
    submission_gt_4 = pos_eq_1.loc[:, ['vid', 'tl']]
    train_gt_4 = train[train['tl'] >= 4]
    train_gt_4.index = list(range(train_gt_4.shape[0]))
    model =  GradientBoostingRegressor(learning_rate=0.01, n_estimators=800, max_depth=5, subsample=0.8,
                                               random_state=1, verbose=1, min_samples_leaf=20)
    gbdt_model(train_gt_4, 'tl', use_features, test_eq_1, submission_gt_4, model)
    gt_4_index = submission[submission['vid'].isin(submission_gt_4['vid'])].index
    submission_temp = submission.loc[gt_4_index, ['vid', 'tl']]
    merge_fat = pd.merge(submission_temp, submission_gt_4, on='vid')
    temp_columns = [tc for tc in merge_fat.columns if tc != 'vid']
    replace_num = np.max(merge_fat.loc[:, temp_columns], axis=1)
    submission.loc[gt_4_index, 'tl'] = replace_num.values
    print(submission.sort_values(by=['tl'], ascending=False))
    sub_final = DataFrame(submission)
    sub_final.persist('tl_jz_5_fold_6_6_22_submit_modified_high_value')
