import unittest
import jieba


class NotifierTest(unittest.TestCase):

    def setUp(self):
        jieba.register_notifier(lambda changed_list: print("changed list: {}".format(changed_list[0])))

    def test_notifier(self):
        jieba.add_word('台中')

    def test_notifier_2nd(self):
        seg_list = jieba.cut('今天天气不错')
        print("Before: {}".format('/'.join(seg_list)))
        jieba.suggest_freq(('今天', '天气'), True)
        seg_list = jieba.cut('今天天气不错')
        print("After: {}".format('/'.join(seg_list)))

    def test_notifier_3rd(self):
        jieba.load_userdict('userdict.txt')


if __name__ == '__main__':
    unittest.main()
