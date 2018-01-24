# encoding=utf-8
import marshal
import tempfile
from abc import abstractmethod, ABCMeta
from hashlib import md5

# Defining the replace function depends on os
import os

if os.name == 'nt':
    from shutil import move as _replace_file
else:
    _replace_file = os.rename


class CacheStrategy(metaclass=ABCMeta):
    """
    The class represent the strategy to cache inner data
    """

    @abstractmethod
    def dump(self, data):
        """
        Dump data to cache
        :param data: the data to be cached
        :return:
        """
        pass

    @abstractmethod
    def loads(self):
        """
        Load data from cache
        :return: the data cached
        """
        pass

    @abstractmethod
    def is_expires(self):
        """
        Check is the cache expires
        :return: True if expired, or to False
        """
        pass

    @abstractmethod
    def is_cache_exist(self):
        """
        Check is the cache exist
        :return:
        """
        pass

    @staticmethod
    @abstractmethod
    def get(*args):
        """
        This is a function method to get strategy instance
        :param args:
        :return:
        """
        pass


class FileCacheStrategy(CacheStrategy):
    """
    This class used to process file cache
    """

    def __init__(self, source, cache_file=None, tmp_dir=None):
        """
        Initialize the instance
        :param source: dict source, which is the instance of DictSource
        :param cache_file: the path of cache file
        :param tmp_dir: the path of tmp dir
        """
        self._source = source
        if cache_file is None:
            self._cache_file = "jieba.u%s.cache" \
                               % md5(str(source).encode('utf-8', 'replace')).hexdigest()
        else:
            self._cache_file = cache_file
        self._cache_file = \
            os.path.join(tmp_dir or tempfile.gettempdir(), self._cache_file)

        self._tmp_dir = os.path.dirname(self._cache_file)

    def get_tmp_dir(self):
        return self._tmp_dir

    def get_cache_file(self):
        return self._cache_file

    def dump(self, data):
        fd, fpath = tempfile.mkstemp(dir=self._tmp_dir)
        with os.fdopen(fd, 'wb') as temp_cache_file:
            marshal.dump(data, temp_cache_file)
        _replace_file(fpath, self._cache_file)

    def loads(self):
        with open(self._cache_file, 'rb') as cf:
            return marshal.load(cf)

    def is_cache_exist(self):
        return os.path.isfile(self._cache_file)

    @staticmethod
    def get(*args):
        return FileCacheStrategy(*args)

    def is_expires(self):
        return self._source.get_last_modify_time() > os.path.getmtime(self._cache_file)

    def __str__(self):
        return self._cache_file
