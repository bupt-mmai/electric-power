# 基于语法规则过滤

# 过滤以在stop_starts中开头的词
# 过滤以在stop_ends中结尾的词

import csv

stop_ends_dict = set()
with open('../resources/stop_ends_dict.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        stop_ends_dict.add(line.strip())
max_end_length = 4

stop_starts_dict = set()
with open('../resources/stop_starts_dict.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        stop_starts_dict.add(line.strip())

max_start_length = 3


def compute_max_word_length(dict):
    max_length=0
    for word in dict:
        if len(word)>max_length:
            max_length=len(word)
    return max_length

def writeCsv(row):
    csvFile = open("../out/result8.csv", "a")            # 创建csv文件
    writer = csv.writer(csvFile)                  # 创建写的对象
    # 先写入columns_name
    # writer.writerow(["","new_word","count","ie_l","ie_r","tag","av_l","av_r","c_p_max","c_p_min"])     #写入列的名称
    writer.writerow(row)
    csvFile.close()
# writeCsv([])

def readCsv():
    with open('/Users/zc/ztt/GitHub/dianli/src/test/new_word2.csv','r') as f:
        rander = csv.reader(f)
        next(f)
        next(f)
        #对数据循环获取
        for i in rander:

            flag = 1
            word = i[2]
            word_length = len(word)

            # 过滤start
            index = 0
            window_size = word_length if word_length < max_start_length else max_start_length
            for size in range(window_size + index, index, -1):
                piece = word[index:size]
                if piece not in stop_starts_dict:
                    continue
                else:
                    flag = 0
                    break

            # 过滤end
            index2 = word_length
            window_size2 = word_length if word_length < max_end_length else max_end_length
            for size in range(index2 - window_size2, index2):
                piece = word[size:index2]
                if piece not in stop_ends_dict:
                    continue
                else:
                    flag = 0
                    break

            # 过滤叠词

            if flag == 1:
                writeCsv(i)
readCsv()

