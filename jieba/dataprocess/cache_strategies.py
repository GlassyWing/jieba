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
    def set_data_source(self, source):
        pass

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
    def is_cache_exist(self):
        """
        Check is the cache exist
        :return:
        """
        pass

    @abstractmethod
    def remove_cache(self):
        pass


class FileCacheStrategy(CacheStrategy):
    """
    This class used to process file cache
    """

    def __init__(self, cache_file=None, tmp_dir=None):
        """
        Initialize the instance
        :param cache_file: the path of cache file
        :param tmp_dir: the path of tmp dir
        """
        self._cache_file = cache_file
        self._tmp_dir = tmp_dir

    def set_data_source(self, source):
        self._source = source

    def get_tmp_dir(self):
        cache_path = os.path.join(self._tmp_dir or tempfile.gettempdir(), self.get_cache_file())
        return os.path.dirname(cache_path)

    def get_cache_path(self):
        return os.path.join(self.get_tmp_dir(), self.get_cache_file())

    def get_cache_file(self):
        if self._cache_file:
            return self._cache_file
        if self._source:
            return "jieba.u%s.cache" \
                   % md5(str(self._source).encode('utf-8', 'replace')).hexdigest()
        return "jieba.cache"

    def dump(self, data):
        fd, fpath = tempfile.mkstemp(dir=self.get_tmp_dir())
        with os.fdopen(fd, 'wb') as temp_cache_file:
            marshal.dump(data, temp_cache_file)
        _replace_file(fpath, self.get_cache_path())

    def loads(self):
        with open(self.get_cache_path(), 'rb') as cf:
            return marshal.load(cf)

    def is_cache_exist(self):
        return os.path.isfile(self.get_cache_path())

    def remove_cache(self):
        if self.is_cache_exist():
            return os.remove(self.get_cache_path())
        return True

    def __str__(self):
        return self.get_cache_path()

    def __eq__(self, other):
        if self is other: return True
        if isinstance(other, FileCacheStrategy):
            return self.get_cache_path() == other.get_cache_path()
        return False
