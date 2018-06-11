###  团队：Unreal	

###  Rank：10


### 代码说明


#### data_pre_process

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.origin_part1_part2_row2col：进行原始数据转换，包括行转列，去重等；

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2. get_num_features，生成数值特征的表；

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3. get_word_features，生成文字特征的表；

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4. data_merge_split，合并数值、文字以及snp数据。


#### feature_selection

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1. 分别针对sys, dia, tl, hdl, ldl 运行snp_drop_one_hot, 得出五个对应特征的数据集，这一步骤主要是删去gbdt预训练中不重要的snp特征，然后进行one_hot编码；

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2. 分别针对sys, dia, tl, hdl, ldl 运行get_best_rounds, 得出a步骤五个数据对应的五折最优迭代次数。

#### every_prediction_model

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;运行所有文件，得出sys，dia，tl，hdl，ldl在测试集上的预测结果。

#### xgb_model

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1. add_prefix_for_xgb_model，得出带有前缀的特征数据集；

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2. split_5_fold_data_xgb，分割五折训练的数据；

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3. baseline_xgboost_jz，训练xgb模型；

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4. calc_xgb_test_loss_and_save, 将c步骤中的五个tl的预测结果融合并取均值。
   

####  submit_result
提交最终结果，最终结果是sys，dia，hdl，ldl为gbdt单模型，tl为gbdt和xgb的加权融合，比例为0.7和0.35。
A榜单模型GBDT最优得分为0.0318，B榜单模型GBDT最优0.0321，tl加权融合后最优成绩0.0319。

### 队友Github
zhuifeng414: https://github.com/Zhuifeng414

   wzm	   : https://github.com/w-zm
