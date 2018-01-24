import unittest
from jieba.dataprocess.cache_strategies import *
import jieba


class CacheTest(unittest.TestCase):
    def test_file_cache(self):
        cache = FileCacheStrategy()
        print(cache.get_cache_file())
        print(cache.get_tmp_dir())
        cache.dump(({'a': 20}, 20))
        print(cache.loads())

    # def test_src(self):
    #     jieba.initialize()


if __name__ == '__main__':
    unittest.main()
