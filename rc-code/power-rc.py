import hanlp
tokenizer = hanlp.load('LARGE_ALBERT_BASE')

def rc(sentence):
    tokenizer_result = tokenizer(sentence)

    tagger = hanlp.load(hanlp.pretrained.pos.CTB9_POS_ALBERT_BASE)
    tagger_result = tagger(tokenizer_result)

    token_and_tagger_dic = [] 
    for i in range(len(tokenizer_result)):
        token_and_tagger_dic.append((tokenizer_result[i],tagger_result[i]))

    syntactic_parser = hanlp.load(hanlp.pretrained.dep.CTB7_BIAFFINE_DEP_ZH)
    sparser_result = syntactic_parser(token_and_tagger_dic)

    for i in range(len(sparser_result)):
        word = sparser_result[i]
        word = str(word)
        print(word.split('	'))
        word_dic = word.split('	')
        if(word_dic[7]=='root'):
            print('核心是：'+word_dic[1])
            # 找找有没有谓语补语
            dong_bu = []
            for j in range(len(sparser_result)):
                word2 = str(sparser_result[j]).split('	')
                if (word_dic[0]==word2[6] and word2[7]!='punct' and word2[7]!=' '):
                    dong_bu.append(word2[1])
                    if(word2[7]=='attr' and word2[7]!=' '):
                        print('属性关系是：',word2[1])
                    if(word2[7]=='cop'):
                        print('核心词修正是：',word2[1]+word_dic[1])
            print(dong_bu)
            
        # if(word_dic[7]=='nsubj'):
        #     # print('名词主语是：'+word_dic[1])
        #     zhuyu_bu = []
        #     # 主语的补语
        #     for j in range(len(sparser_result)):
        #         word2 = str(sparser_result[j]).split('	')
        #         if (word_dic[0]==word2[6]):
        #             zhuyu_bu.append(word2[1])
        #     zhuyu_bu.append(word_dic[1])
        # if(word_dic[7]=='dobj'):
        #     # print('直接宾语是：'+word_dic[1])

