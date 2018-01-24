# encoding=utf-8

from abc import ABCMeta, abstractmethod

from jieba._common import *
from jieba._compat import *


class DictResource(metaclass=ABCMeta):
    """
    This abstract class can be represent a source that can get dict record
    which one contains 3 elements at least, they are: word, freq, tag
    """

    @abstractmethod
    def get_record(self):
        """
        The method should return a generator
         which generate a tuple(word, freq)
        :return: a generator
        """
        pass

    @abstractmethod
    def get_lrecord(self):
        """
        Get record sequences which
        in the form of [(word, freq),...]
        :return:
        """
        pass

    @abstractmethod
    def get_data_source(self):
        """
        Return the data source
        :return:
        """
        pass

    @abstractmethod
    def get_last_modify_time(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __hash__(self):
        pass


class FileDictResource(DictResource):
    """
    This class represent data source from a file
    """

    def __init__(self, path):
        path = get_abs_path(path)
        self._path = path

    def get_lrecord(self):
        return list(self.get_record())

    def get_data_source(self):
        return self._path

    def get_record(self):
        with open(self._path, 'rb') as f:
            for no, ln in enumerate(f, 1):
                line = ln.strip()
                if not isinstance(line, text_type):
                    try:
                        line = line.decode('utf-8').lstrip('\ufeff')
                    except UnicodeDecodeError:
                        raise ValueError('The dictionary %s must be utf-8' % f.name)
                if not line:
                    continue
                try:
                    word, freq, tag = re_userdict.match(line).groups()
                    yield (word, freq, tag)
                except ValueError:
                    raise ValueError(
                        'invalid dictionary entry in %s at Line %s: %s' % (f.name, no, line))

    def get_last_modify_time(self):
        return os.path.getmtime(self._path)

    def __str__(self):
        return self._path

    def __eq__(self, other):
        return self._path == other._path

    def __hash__(self) -> int:
        return hash(self._path)
