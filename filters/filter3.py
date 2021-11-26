# 过滤单字
# 过滤论文词

import csv


paper_dict = set()
with open('../resources/negtive_paper_word.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        paper_dict.add(line.strip())

sample_dict = set()
with open('../resources/negtive_sample.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        sample_dict.add(line.strip())

nz_dict = set()
with open('../resources/negtive_nz.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        nz_dict.add(line.strip())

manner_dict = set()
with open('../resources/negtive_manner.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        manner_dict.add(line.strip())

todo_dict = set()
with open('../resources/negtive_manner.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        todo_dict.add(line.strip())


def writeCsv(row):
    csvFile = open("../out/result9.csv", "a")            # 创建csv文件
    writer = csv.writer(csvFile)                  # 创建写的对象
    # 先写入columns_name
    # writer.writerow(["","new_word","count","ie_l","ie_r","tag","av_l","av_r","c_p_max","c_p_min"])     #写入列的名称
    writer.writerow(row)
    csvFile.close()
# writeCsv([])

def readCsv():
    with open( '/Users/zc/ztt/GitHub/dianli/src/asserts/new_word(1).csv','r') as f:
        rander = csv.reader(f)
        next(f)
        next(f)
        #对数据循环获取
        for i in rander:
            if len(i[2]) > 2: # 词长 > 1
                if i[2] not in paper_dict: # 过滤论文词汇
                    if i[2] not in manner_dict: # 不正确词
                        if i[2] not in nz_dict: # 公司词
                            if i[2] not in todo_dict: # 中间过程词
                                if i[2] not in sample_dict: 
                                    writeCsv(i)

readCsv()