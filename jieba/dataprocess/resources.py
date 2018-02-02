# encoding=utf-8

from abc import ABCMeta, abstractmethod

from jieba._common import *
from jieba._compat import *
from .cache_strategies import CacheStrategy, FileCacheStrategyForDict
import typing

DEFAULT_CACHE_STRATEGY = FileCacheStrategyForDict()


class Resource(object):
    __metaclass__ = ABCMeta

    def __init__(self, cache_strategy: CacheStrategy):
        self._cache_strategy = cache_strategy
        self._set_cache_strategy()

    def set_cache_strategy(self, cache_strategy: CacheStrategy):
        self._cache_strategy = cache_strategy
        self._set_cache_strategy()

    def _set_cache_strategy(self):
        if self._cache_strategy:
            self._cache_strategy.set_data_source(self)

    def get_cache_strategy(self):
        return self._cache_strategy

    @abstractmethod
    def get_record(self) -> typing.Iterable:
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

    def dump_to_cache(self, data=None):
        if data:
            self._cache_strategy.dump(data)
        else:
            self._cache_strategy.dump(self.get_lrecord())

    def load_from_cache(self):
        if self._cache_strategy.is_cache_exist():
            return self._cache_strategy.loads()
        return None

    def __str__(self):
        return self.__class__.__name__

    @abstractmethod
    def __eq__(self, other):
        return self.get_cache_strategy() == other.get_cache_strategy()

    @abstractmethod
    def __hash__(self):
        pass


class DictResource(Resource):
    """
    This abstract class can be represent a source that can get dict record
    which one contains 3 elements at least, they are: word, freq, tag
    """

    def __init__(self, cache_strategy):
        super(DictResource, self).__init__(cache_strategy)

    @abstractmethod
    def get_record(self) -> typing.Iterable:
        pass

    def gen_pfdict(self):
        """
        Read from a dictionary source, and count the numbers of words
        :param dict_source: dictionary source
        :return: A tuple in the form of ({'word1':freq1,...}, total_freq)
        """
        lfreq = {}
        ltotal = 0
        for word, freq, tag in self.get_record():
            freq = int(freq)
            lfreq[word] = freq
            ltotal += freq
            for ch in xrange(len(word)):
                wfrag = word[:ch + 1]
                if wfrag not in lfreq:
                    lfreq[wfrag] = 0
        return lfreq, ltotal

    def dump_to_cache(self, data=None):
        if data:
            self.get_cache_strategy().dump(data)
        else:
            data = self.gen_pfdict()
            self.get_cache_strategy().dump(data)


class PureDictResource(DictResource):
    """
    This class represent dictionary source from a python sequence which item
    in this is in form of (word,freq=None,tag=None)
    """

    def __init__(self, words_seq, cache_strategy=DEFAULT_CACHE_STRATEGY):
        super(PureDictResource, self).__init__(cache_strategy)
        self._words_seq = words_seq

    def get_record(self):
        for record in iter(self._words_seq):
            yield convert_to_word_record(*record)

    def get_lrecord(self):
        return self._words_seq

    def __eq__(self, other):
        if self is other: return True
        if isinstance(other, PureDictResource):
            return self._words_seq == other._words_seq and super(PureDictResource, self).__eq__(other)
        return False

    def __str__(self):
        return '{}:{}'.format(super(PureDictResource, self).__str__(), self._words_seq)

    def __hash__(self):
        return hash(self.__str__())


class FileDictResource(DictResource):
    """
    This class represent dictionary source from a file
    """

    def __init__(self, path, cache_strategy=DEFAULT_CACHE_STRATEGY):
        super(FileDictResource, self).__init__(cache_strategy)
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
            return self._path == other._path and super(FileDictResource, self).__eq__(other)
        return False

    def __hash__(self) -> int:
        return hash(self.__str__())
