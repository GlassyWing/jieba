import unittest

from jieba.tokenizer import default_logger


class LoggerTest(unittest.TestCase):
    def test_print(self):
        default_logger.debug('h')


if __name__ == '__main__':
    unittest.main()
