import unittest
import jieba


class ForceSplitWordsTest(unittest.TestCase):

    def setUp(self):
        jieba.get_dictionary.get_cache_strategy().remove_cache()

    def test_init(self):
        pass

    def test_suggest_freq(self):
        print('/'.join(jieba.cut('如果放到post中将出错。', HMM=False)))
        jieba.suggest_freq(('中', '将'), True)
        print('/'.join(jieba.cut('如果放到post中将出错。', HMM=False)))

    def test_suggest_freq_01(self):
        print('/'.join(jieba.cut('「台中」正确应该不会被切开', HMM=False)))
        jieba.suggest_freq('台中', True)
        print(jieba.get_force_split_words())
        print('/'.join(jieba.cut('「台中」正确应该不会被切开', HMM=False)))
