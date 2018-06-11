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
    
def dannan_xirou(df):
    if df:
        if '胆囊息肉' in df:
            return 1
        else:
            return 0
    else:
        return np.nan


def dannan_jieshi(df):
    if df:
        if '胆囊结石' in df:
            return 1
        else:
            return 0
    else:
        return np.nan


def shen_jieshi(df):
    if df:
        if '肾结石' in df:
            return 1
        else:
            return 0
    else:
        return np.nan


def shen_nangzhong(df):
    if df:
        if '肾囊肿' in df:
            return 1
        else:
            return 0
    else:
        return np.nan


def gan_nangzhong(df):
    if df:
        if '肝囊肿' in df:
            return 1
        else:
            return 0
    else:
        return np.nan
    
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

def gan_ying_hua(df):
    if df:
        if '肝脏' in df:
            return 1
        else:
            return 0
    else:
        return np.nan
    
def strQ2B(ustring):  
    """全角转半角""" 
    ustring = str(ustring)
    rstring = ""  
    for uchar in ustring:  
        inside_code=ord(uchar)  
        if inside_code == 12288:                                            
            inside_code = 32   
        elif (inside_code >= 65281 and inside_code <= 65374):
            inside_code -= 65248  
  
        rstring += chr(inside_code)
    return rstring

def extract_num_norm(df):
    if isinstance(df, Iterable):
        temp = re.findall(r'\-*\d+(?:\.\d+)?', df)
        if temp:
            return np.mean([float(i.replace('--', '')) for i in temp])
        else:
            return np.nan
    else:
        return np.nan


def is_sex(x):
    x = str(x)
    if ('阴道' in x)|('子宫' in x)|('妇' in x)|('乳' in x)|('孕' in x)|('卵巢' in x)|('女' in x)|('宫颈' in x)|('妊娠' in x)|('剖腹产' in x):
        return 1
    elif ('前列腺' in x)|('包皮' in x)|('包茎' in x)|('男' in x)|('阴茎' in x)|('睾丸' in x):
        return 2
    else:
        return 0

def word2num(data_frame):
    one_hot_list = ['0101', '0102', '0113', '0409', '0413', '0434', '0439', 'a201', 'a202', '4001', '0705', 'a301', '0709',
                    '0985', 'a705']
    data_frame.loc[:, one_hot_list] = data_frame.loc[:, one_hot_list].fillna('')
    frame_409_434 = data_frame['0409'] + data_frame['0434'] + data_frame['0413'] + data_frame['4001'] + \
                    data_frame['a201'] + data_frame['a301'] + data_frame['a202'] + data_frame['0705'] + \
                    data_frame['0709'] + data_frame['0985'] + data_frame['0439']
    data_frame['xue_ya_pian_gao'] = frame_409_434.apply(high_pressure)
    data_frame['gan_by_ts'] = data_frame['0113'].apply(map_deal_0113)
    data_frame['xue_zhi_pian_gao'] = frame_409_434.apply(high_fat)
    data_frame['xue_tang_pian_gao'] = frame_409_434.apply(high_sugar)
    data_frame['high_sugar'] = frame_409_434.apply(higher_sugar)
    data_frame['guan_xin_bin'] = frame_409_434.apply(coronary_heart_disease)
    data_frame['shen'] = frame_409_434.apply(kidney)
    data_frame['smoke'] = frame_409_434.apply(smoke)
    fat_liver_num = data_frame['0101'] + data_frame['0102'] + data_frame['0113'] + data_frame['a202']
    data_frame['dannan_jieshi'] = fat_liver_num.apply(dannan_jieshi)
    data_frame['dannan_xirou'] = fat_liver_num.apply(dannan_xirou)
    data_frame['shen_jieshi'] = fat_liver_num.apply(shen_jieshi)
    data_frame['shen_nanz'] = fat_liver_num.apply(shen_nangzhong)
    data_frame['gan_nanz'] = fat_liver_num.apply(gan_nangzhong)
    data_frame['gan_ying_hua'] = data_frame['a705'].apply(gan_ying_hua)
    yy_list = ['3190', '3191', '3192', '3194', '3195', '3197', '3430', '100010']
    for y in yy_list:
        data_frame[y] = data_frame[y].apply(ying_yang)
    data_frame['niao'] = data_frame['3193'].apply(urine)
    data_frame['heart_rate'] = data_frame['0420'].apply(heart_rate)
    data_frame['3399_w'] = data_frame['3399'].apply(transform_3399)
    data_frame['3301_w'] = data_frame['3301'].apply(HP_yy)
    data_frame['0403_w'] = data_frame['0403'].apply(transform_403)
    data_frame['0421_w'] = data_frame['0421'].apply(transform_421)
    data_frame['0405_w'] = data_frame['0405'].apply(lung_voice)
    data_frame['blood_pipe_style'] = data_frame['4001'].apply(blood_pipe_style)
    data_frame['health'] = data_frame['2302'].apply(transform_2302)
    data_frame['pres_front'] = data_frame['0102'].apply(get_num_from_102_front)
    data_frame['pres_back'] = data_frame['0102'].apply(get_num_from_102_back)
    data_frame['heart_times'] = data_frame['1001'].apply(extract_num_norm)

    data_frame['all_result'] = '_'
    for p in data_frame.columns:
        if p != 'vid':
        	data_frame['all_result'] = data_frame['all_result'] +  '_' + data_frame[p].astype('str')
 
    data_frame['gender'] = data_frame['all_result'].apply(is_sex)
    del data_frame['all_result']

    new_add = ['xue_ya_pian_gao', 'xue_zhi_pian_gao', 'xue_tang_pian_gao', 'high_sugar', 'guan_xin_bin', 'shen', 'smoke','niao', 'heart_rate', '3399_w', 
               '3301_w', '0403_w', '0421_w', '0405_w', 'gender','blood_pipe_style', 'health','pres_front', 'pres_back','heart_times', 'vid', 'dannan_jieshi',
               'dannan_xirou', 'shen_jieshi', 'shen_nanz', 'gan_nanz','gan_ying_hua']
    yy_list.extend(new_add)
    return data_frame.loc[:, yy_list]


if __name__ == "__main__":
    part_1_2 = odps.get_table('origin_data_combine_part1_part2').to_df().to_pandas()
    word_data = word2num(part_1_2)
    print('the shape of word_data: ',word_data.shape)
    juz_word_data = DataFrame(word_data)
    juz_word_data.persist('juz_word_data_5_30')