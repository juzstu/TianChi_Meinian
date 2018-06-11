from odps import ODPS
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from odps.df import DataFrame
from sklearn.model_selection import KFold
import time
import numpy as np


def gbdt_model(df, label, use_feature, true_test, submission_data, gbdt_model):
    print(submission_data.head())
    print("基于GBDT： 开始训练 label 为{}...".format(label))
    value4preds = df['pos_4']
    train_data = df.loc[:, use_feature]
    print(train_data.shape, true_test.shape)
    pred_labels = np.zeros(df.shape[0])
    submission_label = np.zeros((true_test.shape[0], 5))
    kf = KFold(n_splits=5, shuffle=True, random_state=1024)
    five_fold_index = list(kf.split(train_data, value4preds))
    
    train_index_1, test_index_1 = five_fold_index[0]
    print('第1次训练...')
    x_train_1, x_test_1 = train_data.iloc[train_index_1], train_data.iloc[test_index_1]
    y_train_1, y_test_1 = value4preds.iloc[train_index_1], value4preds.iloc[test_index_1]
    gbdt_model.fit(x_train_1, y_train_1)
    pred_labels[x_test_1.index] = np.where(gbdt_model.predict(x_test_1) > 0.5, 1, 0)
    submission_label[:, 0] = np.where(gbdt_model.predict(true_test) > 0.5, 1, 0)
    print('第1次训练结束')
    print('*******************************************************************')
    train_index_2, test_index_2 = five_fold_index[1]
    print('第2次训练...')
    x_train_2, x_test_2 = train_data.iloc[train_index_2], train_data.iloc[test_index_2]
    y_train_2, y_test_2 = value4preds.iloc[train_index_2], value4preds.iloc[test_index_2]
    gbdt_model.fit(x_train_2, y_train_2)
    pred_labels[x_test_2.index] = np.where(gbdt_model.predict(x_test_2) > 0.5, 1, 0)
    submission_label[:, 1] = np.where(gbdt_model.predict(true_test) > 0.5, 1, 0)
    print('第2次训练结束')
    print('*******************************************************************')
    train_index_3, test_index_3 = five_fold_index[2]
    print('第3次训练...')
    x_train_3, x_test_3 = train_data.iloc[train_index_3], train_data.iloc[test_index_3]
    y_train_3, y_test_3 = value4preds.iloc[train_index_3], value4preds.iloc[test_index_3]
    gbdt_model.fit(x_train_3, y_train_3)
    pred_labels[x_test_3.index] = np.where(gbdt_model.predict(x_test_3) > 0.5, 1, 0)
    submission_label[:, 2] = np.where(gbdt_model.predict(true_test) > 0.5, 1, 0)
    print('第3次训练结束')
    print('*******************************************************************')
    train_index_4, test_index_4 = five_fold_index[3]
    print('第4次训练...')
    x_train_4, x_test_4 = train_data.iloc[train_index_4], train_data.iloc[test_index_4]
    y_train_4, y_test_4 = value4preds.iloc[train_index_4], value4preds.iloc[test_index_4]
    gbdt_model.fit(x_train_4, y_train_4)
    pred_labels[x_test_4.index] = np.where(gbdt_model.predict(x_test_4) > 0.5, 1, 0)
    submission_label[:, 3] = np.where(gbdt_model.predict(true_test) > 0.5, 1, 0)
    print('第4次训练结束')
    print('*******************************************************************')
    train_index_5, test_index_5 = five_fold_index[4]
    print('第5次训练...')
    x_train_5, x_test_5 = train_data.iloc[train_index_5], train_data.iloc[test_index_5]
    y_train_5, y_test_5 = value4preds.iloc[train_index_5], value4preds.iloc[test_index_5]
    gbdt_model.fit(x_train_5, y_train_5)
    pred_labels[x_test_5.index] = np.where(gbdt_model.predict(x_test_5) > 0.5, 1, 0)
    submission_label[:, 4] = np.where(gbdt_model.predict(true_test) > 0.5, 1, 0)
    print('第5次训练结束')
    print('*******************************************************************')
    submission_data['pos_4'] = np.where(np.sum(submission_label, axis=1) >= 1, 1, 0)
    print(classification_report(pred_labels, value4preds))
    print(submission_data[submission_data['pos_4']==1])
    sub_class = DataFrame(submission_data[submission_data['pos_4']==1], unknown_as_string=True)
    sub_class.persist('tl_gt_4_vid_6_6')

# A榜使用了tl的高低值分类，B榜没有
if __name__ == "__main__":
    train = odps.get_table('juz_train_6_6_final').to_df().to_pandas()
    train['pos_4'] = train['tl'].apply(lambda x: 1 if x > 4 else 0)
    test = odps.get_table('juz_test_6_6_final').to_df().to_pandas()
    class_result = test.loc[:, ['vid', 'tl']]
    predict_features = ['sys', 'dia', 'tl', 'hdl', 'ldl']
    use_features = [t for t in train.columns if t != 'vid' and t not in predict_features]
    test_data = test.loc[:, use_features]
    start = time.time()
    model =  GradientBoostingClassifier(learning_rate=0.01, n_estimators=1500, max_depth=5, subsample=0.7,
                                               random_state=1, verbose=0, min_samples_leaf=50)
    for i, j in enumerate(predict_features):
        if j in ['tl']:
            gbdt_model(train, j, use_features, test_data, class_result, model)