import unittest
from jieba.dataprocess import *


class TestResource(unittest.TestCase):

    def test_resource_eq(self):
        a = FileDictResource('foobar.txt')
        b = FileDictResource('foobar.txt')
        c = FileDictResource('userdict.txt')
        self.assertTrue(a is not b)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    # def test_default_file_resource(self):
    #     for word, freq, word_class in DEFAULT_DICT.get_record():
    #         print(word, freq, word_class)

    def test_file_resource(self):
        resource = FileDictResource('foobar.txt')
        self.assertEqual(resource.get_lrecord(), [('好人', '12', 'n')])

    def test_file_record(self):
        resource = FileDictResource('foobar.txt')

    def test_to_string(self):
        self.assertEqual(
            str(FileDictResource('foobar.txt'))
            , r'E:\python\jieba\test\foobar.txt'
        )



if __name__ == '__main__':
    unittest.main()
