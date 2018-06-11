# README
###  团队：Unreal	

###  Rank：2

### 文件夹说明：
- data:数据文件夹
- features: 手动整理好的数值型和文字型特征，分别为num_label.txt和word_label.txt,数据清洗过程中需要使用这两个文件
- code: 主运行代码，我的数据融合了zhuifeng414的数据，可直接运行main.py
- team: 队友特征工程代码以及我的Mongodb操作代码
- submit: 提交结果文件夹



### PS：
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;我之前数据预处理部分并没有使用豆腐大佬分享的开源代码，是通过把数据存入Mongodb文件里，然后再转存为csv文件的。原始数据其中有个别同一个vid的table_id对应了多个结果，我这边对他们进行了拼接操作，具体逻辑可见压缩包中的team文件夹的data_process_by_Mongo.py。考虑到主办方工作任务比较繁重且代码整理的时间比较少，我这里还是和队友一起使用了开源的数据预处理代码，但两者结果可能存在一定的差异，最终也许会对提交的成绩有所影响，所以这里特此说明下。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**团队目前使用的是去掉缺失值为96%的数据**，A,B榜的最优成绩的舒张压，收缩压，血清高密度脂蛋白是基于去掉缺失值96%，血清低密度脂蛋白和血清甘油三酯是基于98%的，代码位置在team_feature_work.py的886行。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;这份代码是融合了我和zhuifeng414的特征，成绩A榜可以到0.02817，B榜可以到0.02792，B榜的最优成绩0.02764是用xgb进行融合的，那部分xgb代码是由wzm提供的。

### 队友Github
zhuifeng414: https://github.com/Zhuifeng414

   wzm	   : https://github.com/w-zm
