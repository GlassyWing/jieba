import unittest
import jieba


class TokenizerTest(unittest.TestCase):

    # def test_initialize(self):
    #     self.tokenizer.initialize()

    def test_cut(self):
        seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
        print("Full Mode: " + "/ ".join(seg_list))  # 全模式
        seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
        print("Default Mode: " + "/ ".join(seg_list))  # 精确模式
        seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
        print(", ".join(seg_list))
        seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
        print(", ".join(seg_list))

    def test_load_userdict(self):
        set_list = jieba.cut('李小福是创新办主任也是云计算方面的专家')
        print("Before:", "/ ".join(set_list))
        jieba.load_userdict('userdict.txt')
        set_list = jieba.cut('李小福是创新办主任也是云计算方面的专家')
        print("After:", "/ ".join(set_list))

    def test_adjust_dict(self):
        print('/'.join(jieba.cut('如果放到post中将出错。', HMM=False)))
        print('Adjusted freq: ', jieba.suggest_freq(('中', '将'), True))
        print('/'.join(jieba.cut('如果放到post中将出错。', HMM=False)))
        print('/'.join(jieba.cut('「台中」正确应该不会被切开', HMM=False)))
        print('Adjusted freq:', jieba.suggest_freq('台中', True))
        print('/'.join(jieba.cut('「台中」正确应该不会被切开', HMM=False)))

if __name__ == '__main__':
    unittest.main()
