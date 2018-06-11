import time
import re
import pandas as pd
from odps import ODPS
from odps.df import DataFrame
import numpy as np
from collections import Iterable

import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

def extract_num_norm(df):
    if isinstance(df, Iterable):
        temp = re.findall(r'\-*\d+(?:\.\d+)?', df)
        if temp:
            return np.mean([float(i.replace('--', '')) for i in temp])
        else:
            return np.nan
    else:
        return np.nan

def transform_0424(df):
        if isinstance(df, Iterable):
            temp = re.findall(r'\-*\d+(?:\.\d+)?', df)
            if temp:
                return np.mean([float(i.replace('--', '')) for i in temp])
            else:
                if '常' in df:
                    return 75
                elif '过速' in df:
                    return 100
                elif '过缓' in df:
                    return 50
                else:
                    return np.nan
        else:
            return np.nan
                
def transform_0425(df):
        if isinstance(df, Iterable):
            temp = re.findall(r'\-*\d+(?:\.\d+)?', df)
            if temp:
                return np.mean([float(i.replace('--', '')) for i in temp])
            else:
                # median
                if '常' in df:
                    return 17
                # min
                elif '粗糙' in df:
                    return 14
                else:
                    return np.nan
        else:
            return np.nan
                
def transform_1308(df):
    if isinstance(df, Iterable):
        if '裸眼' in df:
            temp1 = re.findall(r'\-*\d+(?:\.\d+)?', df)
            if temp1:
            	luo_yan = np.mean([float(i.replace('--', '')) for i in temp1])
                if luo_yan >= 1:
                    return 4
                else:
                    return 3
        elif '矫正' in df:
            temp2 = re.findall(r'\-*\d+(?:\.\d+)?', df)
            if temp2:
                jiao_zheng = np.mean([float(i.replace('--', '')) for i in temp2])
                if jiao_zheng >= 1:
                    return 2
                else:
                    return 1
        else:
            return np.nan
    else:
        return np.nan
    
def transform_1321_1322(df):
        if isinstance(df, Iterable):
            temp = re.findall(r'\-*\d+(?:\.\d+)?', df)
            if temp:
                return np.mean([float(i.replace('--', '')) for i in temp])
            else:
                if '失明' in df or '义眼' in df:
                    return 0
                elif '指数' in df:
                    return 0.003
                elif '手动' in df:
                    return 0.002
                elif '光感' in df:
                    return 0.001
                else:
                    return np.nan
        else:
            return np.nan
    
def calc_voice_area(df, desc):
    if isinstance(df, Iterable) and desc in df:
        temp = re.findall(r'\-*\d+(?:\.\d+)?', df)
        if temp:
            if 'cm' in df:
                if len(temp) == 2:
                	return float(temp[0]) * float(temp[1]) * 100
                if len(temp) == 4:
                    return (float(temp[0]) * float(temp[1]) + float(temp[2]) * float(temp[3]))*100.0/2
            if 'mm' in df:
                if len(temp) == 2:
                	return float(temp[0]) * float(temp[1])
                if len(temp) == 4:
                    return (float(temp[0]) * float(temp[1]) + float(temp[2]) * float(temp[3]))*1.0/2
        else:
            return 0
    else:
        return np.nan
    
# 眼压        
def transform_1319_1320(df):
        if isinstance(df, Iterable):
            temp = re.findall(r'\-*\d+(?:\.\d+)?', df)
            if temp:
                return float(temp[0])
            else:
                if '正常' in df:
                    return 15
                elif '偏高' in df:
                    return 22
                else:
                    return np.nan
        else:
            return np.nan

def get_pure_num_features(data_frame, threshold):
    pure_num_list = ['vid']
    for c in data_frame.columns:
        if c != 'vid':
            data_frame[c] = pd.to_numeric(data_frame[c], errors='ignore')
        if data_frame[c].dtypes != 'object' and (data_frame[c].isnull().sum() * 1.0 / data_frame.shape[0] <= threshold):
            #if np.abs(ex_num[c].skew()) <= pian_tai:
        	pure_num_list.append(c)
    return data_frame.loc[:, pure_num_list]

def split_data(data_series, desc):
    check_array = ['' for _ in range(data_series.shape[0])]
    for pos, j in enumerate(data_series):
        if isinstance(j, Iterable):
            tmp = set(j.split('$'))
            for t in tmp:
                if isinstance(t, Iterable) and desc in t:
                    check_array[pos] = t
    return check_array

def qian_lie_xian(df, pos):
    if isinstance(df, Iterable):
        temp = re.findall(r'\-*\d+(?:\.\d+)?', df)
        if temp:
            if 'cm' in df:
                if len(temp) >= 3:
                	return float(temp[pos]) * 10
            if 'mm' in df:
                if len(temp) >= 3:
                	return float(temp[pos])
    else:
        return np.nan

def dpm_check(df):
    if isinstance(df, Iterable):
        temp = re.findall(r'\-*\d+(?:\.\d+)?', df)
        if temp and 'dpm' in df:
        	return float(temp[0])
    else:
        return np.nan

def ex_num_from_str(data_frame):
    word_096_norm = ['004997', '0107', '100008', '100013', '100014','10002', '10003', '1106', '1107', '1110','1112', '1115',
                     '1117', '1345', '139', '141', '143', '1474', '155', '1814', '1815', '183', '1845', '1850','1873', '191', 
                     '192', '193', '20002', '2165', '2174', '2371', '2376', '2390', '2403', '2404', '2405', '2406','2420',
                     '269011', '300001', '300021', '300035', '300051','300069', '300070', '300073', '300074', '300076', '300078',
                     '300093', '300113', '300119', '300125', '300129', '314','3193', '321', '3804', '3807', '669003', '809021', 
                     '979001', '979002', '979003', 'a701', 'a703']
    shili = ['1308','1319','1320','1321','1322']
    heart = ['0424','0425', 'vid']
    for w in word_096_norm:
        data_frame[w] = data_frame[w].apply(extract_num_norm)
    data_frame['0424'] =  data_frame['0424'].apply(transform_0424)
    data_frame['0425'] =  data_frame['0425'].apply(transform_0425)
    data_frame['1308'] =  data_frame['1308'].apply(transform_1308)
    data_frame['1319'] =  data_frame['1319'].apply(transform_1319_1320)
    data_frame['1320'] =  data_frame['1320'].apply(transform_1319_1320)
    data_frame['1321'] =  data_frame['1321'].apply(transform_1321_1322)
    data_frame['1322'] =  data_frame['1322'].apply(transform_1321_1322)
    data_frame['left_shen_no_voice'] =  data_frame['left_shen'].apply(calc_voice_area, args=('无回声',))
    #data_frame['left_shen_strong_voice'] =  data_frame['left_shen'].apply(calc_voice_area, args=('强回声',))
    data_frame['right_shen_no_voice'] =  data_frame['right_shen'].apply(calc_voice_area, args=('无回声',))
    data_frame['right_shen_strong_voice'] =  data_frame['right_shen'].apply(calc_voice_area, args=('强回声',))
    data_frame['jzx_no_voice_area'] =  data_frame['jia_zx'].apply(calc_voice_area, args=('无回声区',))
    data_frame['jzx_no_voice_jiejie'] =  data_frame['jia_zx'].apply(calc_voice_area, args=('无回声结节',))
    data_frame['jzx_low_voice_area'] =  data_frame['jia_zx'].apply(calc_voice_area, args=('低回声区',))
    data_frame['jzx_low_voice_jiejie'] =  data_frame['jia_zx'].apply(calc_voice_area, args=('低回声结节',))
    data_frame['liver_no_voice'] =  data_frame['0113'].apply(calc_voice_area, args=('无回声',))
    data_frame['liver_strong_voice'] =  data_frame['0113'].apply(calc_voice_area, args=('强回声',))
    data_frame['dan_strong_voice'] =  data_frame['0114'].apply(calc_voice_area, args=('强回声',))
    data_frame['qian_lie_xian_1'] =  data_frame['0120'].apply(qian_lie_xian, args=(0,))
    data_frame['qian_lie_xian_2'] =  data_frame['0120'].apply(qian_lie_xian, args=(1,))
    data_frame['qian_lie_xian_3'] =  data_frame['0120'].apply(qian_lie_xian, args=(2,))
    data_frame['dpm_from_3301'] =  data_frame['3301'].apply(dpm_check)
    huishen = ['left_shen_no_voice','right_shen_no_voice','right_shen_strong_voice','jzx_no_voice_area','qian_lie_xian_2','qian_lie_xian_3','dpm_from_3301',
               'jzx_no_voice_jiejie','jzx_low_voice_area','jzx_low_voice_jiejie','liver_no_voice','liver_strong_voice','dan_strong_voice','qian_lie_xian_1']
    total = word_096_norm  + shili + heart + huishen
    num_ex_str = data_frame.loc[:, total]
    return num_ex_str

if __name__ == "__main__":
    part_1_2 = odps.get_table('origin_data_combine_part1_part2').to_df().to_pandas()
    part_1_2['jia_zx'] = split_data(part_1_2['0101'], '甲状腺')
    part_1_2['left_shen'] = split_data(part_1_2['0117'], '左肾')
    part_1_2['right_shen'] = split_data(part_1_2['0118'], '右肾')
    part_1_2_copy = part_1_2.copy(deep=True)
    ex_num_data = ex_num_from_str(part_1_2)
    print('the shape of the num_data get from word: ', ex_num_data.shape)
    pure_num_data = get_pure_num_features(part_1_2_copy, 0.96)
    pure_columns = [p for p in pure_num_data.columns if p != 'vid']
    ex_num_columns = [i for i in ex_num_data.columns if i not in ['vid', '314','1308','1319','1320','1321','1322','0424','0425']]
    print('the shape of origin num data: ', pure_num_data.shape)
    numeric_data = pd.merge(pure_num_data, ex_num_data, on='vid', how='inner')
    exm_drop = []
    for w in pure_columns + ex_num_columns:
		if np.abs(numeric_data[w].skew()) > 12:
			exm_drop.append(w)
    print(exm_drop)
    numeric_data.drop(exm_drop, axis=1, inplace=True)
    print('total data shape: ', numeric_data.shape)
    juz_num_data = DataFrame(numeric_data)
    juz_num_data.persist('juz_num_data_5_31')