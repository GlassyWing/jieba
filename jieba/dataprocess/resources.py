# encoding=utf-8

from abc import ABCMeta, abstractmethod

from jieba._common import *
from jieba._compat import *


class Resource(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_record(self):
        """
        The method should return a generator
         which generate a tuple(word, freq)
        :return: a generator
        """
        pass

    def get_lrecord(self):
        """
        Get record sequences which
        in the form of [(word, freq),...]
        :return:
        """
        return list(self.get_record())

    def __str__(self):
        return self.__class__.__name__

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __hash__(self):
        pass


class DictResource(Resource):
    """
    This abstract class can be represent a source that can get dict record
    which one contains 3 elements at least, they are: word, freq, tag
    """

    @abstractmethod
    def get_record(self):
        pass


class PureDictResource(DictResource):
    """
    This class represent dictionary source from a python sequence
    """

    def __init__(self, words_seq):
        self._words_seq = words_seq

    def get_record(self):
        for record in iter(self._words_seq):
            yield convert_to_word_record(*record)

    def get_lrecord(self):
        return self._words_seq

    def __eq__(self, other):
        if self is other: return True
        if isinstance(other, PureDictResource):
            return self._words_seq == other._words_seq
        return False

    def __str__(self):
        return '{}:{}'.format(super(PureDictResource, self).__str__(), self._words_seq)

    def __hash__(self):
        return hash(self.__str__())


class FileDictResource(DictResource):
    """
    This class represent dictionary source from a file
    """

    def __init__(self, path):
        path = get_abs_path(path)
        self._path = path

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

    def __str__(self):
        return '{}:{}'.format(super(FileDictResource, self).__str__(), self._path)

    def __eq__(self, other):
        if self is other: return True
        if isinstance(other, FileDictResource):
            return self._path == other._path
        return False

    def __hash__(self) -> int:
        return hash(self.__str__())
