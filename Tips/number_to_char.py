''' 
可将万亿以下的正
负（可带小数点）阿
拉伯数字转换为中文
'''
import sys
import re

def prev_deal(data_0):
    
    if '-' in data_0:  
        data_0_abs = data_0[1:]
    else:
        data_0_abs = data_0
    digit_range = re.match('(\d+)', data_0_abs)
    if len(digit_range.group(1)) > 12:
    	return '仅支持转换正负万亿范围内的数'
    if '.' in data_0_abs:  
        add = data_0_abs.find('.')
        #切割成整数和小数部分
        data_0_d =  data_0_abs[:add]
        data_0_f = data_0_abs[add:]
        #整数部分需插入单位
        result = digit_cutting(arabic2chinese(data_0_d)) + arabic2chinese(data_0_f)
    else:
        result = digit_cutting(arabic2chinese(data_0_abs))
    if '-' in data_0:
        return '负' + result
    else:
        return result
        
def arabic2chinese(data_1):

    dir = {   '1':'一', '2':'二',
              '3':'三', '4':'四',
              '5':'五', '6':'六',
              '7':'七', '8':'八',
              '9':'九', '0':'零',
              '.':'点'
          }
    return ''.join(dir[ch] for ch in data_1)

def digit_cutting(data_2):
    '''对不同长度的数进行切割'''
    length = len(data_2)
    if length == 1:        
        pass
    elif 1 < length <=4:
        data_2 = unit_insert(data_2)
        if data_2[0] == '零':
            data_2 = data_2[1:]
    elif 4 < length <= 8:
        last_four = data_2[-4:]
        prev_four = data_2[:-4] 
        if len(prev_four) != 1:
            m = unit_insert(prev_four)
        else:
            m = prev_four
        data_2 = m + '万' + unit_insert(last_four)
    elif 8 < length <=12:
        last_four = data_2[-4:]
        mid_four  = data_2[-8:-4]
        prev_four = data_2[:-8]
        if len(prev_four) == 1:
            m = prev_four
        else:
            m = unit_insert(prev_four)
        if unit_insert(mid_four) == '' and unit_insert(last_four) != '':
            data_2 = m + '零' + unit_insert(last_four)
        elif unit_insert(mid_four) == '' and unit_insert(last_four) == '':
            data_2 = m + '亿'
        else:
            data_2 = m + '亿' + unit_insert(mid_four) + '万' + unit_insert(last_four)
    if data_2[:2] == '一十':
        data_2 = data_2[1:]
    return data_2

def unit_insert(data_3):
    '''在四位以下的数字切片中插入单位'''
    string = '十百千'
    length = len(data_3)
    data_3_list = list(data_3)
    for i in range(length - 1):        
        data_3_list.insert(-(2*i+1), string[i])
    data_3 = ''.join(data_3_list)
    #数字切片会有  一千零百零十五  的情况，以下进行多余零的切除
    for i in ['零千', '零百', '零十', '零零零','零零']:
        k = data_3.find(i)
        if k != -1:
            data_3 = re.sub(i, '零', data_3)
    if data_3[-1] == '零':
        return data_3[:-1]
    else:
        return data_3

while True:
    line = sys.stdin.readline()
    line = line.strip()
    if line == '':
        break
    print(prev_deal(line))
原文链接：https://blog.csdn.net/dongzhen5535/article/details/104356717
