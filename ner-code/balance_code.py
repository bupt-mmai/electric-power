#节点类
class Node(object):
    def __init__(self,name=None,value=None):
        self._name = name
        self._value = value
        self._left = None
        self._right = None

#哈夫曼树类
class HuffmanTree(object):

    #根据Huffman树的思想：以叶子节点为基础，反向建立Huffman树
    def __init__(self,char_weights,save_path=""):
        self.a=[Node(part[0],part[1]) for part in char_weights]  #根据输入的字符及其频数生成叶子节点
        while len(self.a)!=1:
            self.a.sort(key=lambda node:node._value,reverse=True)
            c=Node(value=(self.a[-1]._value+self.a[-2]._value))
            c._left=self.a.pop(-1)
            c._right=self.a.pop(-1)
            self.a.append(c)
        self.root=self.a[0]
        self.b=list(range(10))         #self.b用于保存每个叶子节点的Haffuman编码,range的值只需要不小于树的深度就行
        self.c=[] #self.c用于保存每个叶子节点最终的Haffuman编码
        self.save_path = save_path

    #用递归的思想生成编码
    def pre(self,tree,length):
        node=tree
        if (not node):
            return
        elif node._name:
            print(node._name + '的编码为:')
            # for i in range(length):
            #     print(self.b[i])
            self.c = self.b[0:length]
            print(self.c)
            code = ''
            for i in range(length):
                code += str(self.b[i])
            if self.save_path!='':
                write_line = node._name+','+code
                with open(self.save_path,'a') as fa:
                    fa.write(write_line)
                    fa.write('\n')
            # print('\n')
            return
        self.b[length]=0
        self.pre(node._left,length+1)
        self.b[length]=1
        self.pre(node._right,length+1)
     #生成哈夫曼编码
    def get_code(self):
        self.pre(self.root,0)


class CreateData(object):

    def __init__(self):
        self.labels_frequency = {}

    def count_label_frequency(self, file_path):
        labels_frequency = {}
        with open(file_path, 'r') as fr:
            for line in fr.readlines():
                label_arr = line.split(' ')
                if len(label_arr) == 2:
                    if label_arr[1].strip() in labels_frequency:
                        labels_frequency[label_arr[1].strip()] += 1
                    else:
                        labels_frequency[label_arr[1].strip()] = 1
        self.labels_frequency = labels_frequency
        return labels_frequency

    def trans_labels_frequency(self):
        char_weights = []
        for k, v in self.labels_frequency.items():
            char_weights.append((k,v))
        return char_weights


if __name__=='__main__':
    # 输入的是字符及其频数
    createData = CreateData()
    createData.count_label_frequency('/Users/zc/ztt/ner_code/data/cluner/cluener_txt/all.txt')
    char_weights = createData.trans_labels_frequency()
    print(char_weights)
    tree = HuffmanTree(char_weights, '/Users/zc/ztt/ner_code/data/cluner/cluener_txt/all_Balance_labels.csv')
    tree.get_code()


