import time
import threading
from math import log
from . import finalseg
from .dataprocess import *
from ._compat import *
from ._common import *
from rx.subjects import Subject
import typing

DICT_WRITING = {}

default_logger = init_log('Tokenizer')


class Tokenizer(object):
    """
    The Tokenizer used to cut words
    """

    def __init__(self, dict_source=DEFAULT_DICT, dict_cache_strategy=DEFAULT_DICT_CACHE_STRATEGY):
        """
        Initialize the Tokenizer
        :param dict_source: the path of dictionary, or the instance of DictResource.
        :param dict_cache_strategy: the cache strategy which used to process the inner data
        """
        self.lock = threading.RLock()

        if isinstance(dict_source, string_types):
            self.dictionary = FileDictResource(dict_source)
        else:
            self.dictionary = dict_source
        self.dict_cache_strategy = dict_cache_strategy
        self.FREQ = {}
        self.total = 0
        self.dict_sub = Subject()
        self.user_word_tag_tab = {}
        self.initialized = False
        # Indicates whether to add words in bulk
        self.batch = False

    def __repr__(self):
        return '<Tokenizer dictionary=%r>' % self.dictionary

    def gen_pfdict(self, dict_source):
        """
        Read from a dictionary source, and count the numbers of words
        :param dict_source: dictionary source
        :return: A tuple in the form of ({'word1':freq1,...}, total_freq)
        """
        lfreq = {}
        ltotal = 0
        for word, freq, tag in dict_source.get_record():
            freq = int(freq)
            lfreq[word] = freq
            ltotal += freq
            for ch in xrange(len(word)):
                wfrag = word[:ch + 1]
                if wfrag not in lfreq:
                    lfreq[wfrag] = 0
        return lfreq, ltotal

    def register_notifier(self, dict_change_notifier):
        """
        Register an notifier that will send emission when dict has changed
        :param dict_change_notifier: the notifier which is instance of Observer
        :return:
        """
        self.dict_sub.subscribe(dict_change_notifier)

    def initialize(self, dict_source=None, dict_cache_strategy=None):
        """
        Initialize the Tokenizer
        :param dict_source: the path of dictionary, or the instance of DictResource.
        :param dict_cache_strategy: the cache strategy which used to process the inner data
        :return:
        """

        # If it's already initialized.
        if self.dictionary == dict_source and self.dict_cache_strategy == dict_cache_strategy and self.initialized:
            return
        else:
            self.initialized = False

        if dict_source:
            if isinstance(dict_source, string_types):
                dict_source = FileDictResource(dict_source)
            self.dictionary = dict_source
        else:
            dict_source = self.dictionary

        if dict_cache_strategy:
            self.dict_cache_strategy = dict_cache_strategy
        else:
            dict_cache_strategy = self.dict_cache_strategy

            dict_cache_strategy.set_data_source(dict_source)

        with self.lock:
            try:
                with DICT_WRITING[dict_source]:
                    pass
            except KeyError:
                pass

            # If other thread had initialized, just return
            if self.initialized:
                return

            default_logger.debug("Building prefix dict from %s ..." % (dict_source or 'the default dictionary'))
            t1 = time.time()

            load_from_cache_fail = True

            # If the cache file existed
            if dict_cache_strategy.is_cache_exist():
                default_logger.debug(
                    "Loading model from cache: %s" % dict_cache_strategy)
                try:
                    self.FREQ, self.total = dict_cache_strategy.loads()
                    load_from_cache_fail = False
                except Exception:
                    load_from_cache_fail = True

            if load_from_cache_fail:
                self.cache_dict_resource(dict_source)

            self.initialized = True
            default_logger.debug(
                "Loading model cost %.3f seconds." % (time.time() - t1))
            default_logger.debug("Prefix dict has been built succesfully.")

    def cache_dict_resource(self, dict_source=None, change_dict_source=False):
        """
        Forced cache dictionary source.
        :param dict_source: dictionary source.
        :param change_dict_source: Indicates should the dictionary source used be changed.
        :return:
        """
        if dict_source and change_dict_source:
            self.set_dictionary(dict_source)
        else:
            dict_source = self.get_dict_source()

        wlock = DICT_WRITING.get(dict_source, threading.RLock())
        DICT_WRITING[dict_source] = wlock
        with wlock:
            self.FREQ, self.total = self.gen_pfdict(dict_source)
            default_logger.debug(
                "Dumping model to cache: %s" % self.dict_cache_strategy)
            try:
                self.dict_cache_strategy.dump((self.FREQ, self.total))
            except Exception:
                default_logger.exception("Dump cache file failed.")
            default_logger.debug(
                "Dumping model to cache: %s done!" % self.dict_cache_strategy)

        try:
            del DICT_WRITING[dict_source]
        except KeyError:
            pass

    def check_initialized(self):
        if not self.initialized:
            self.initialize()

    def calc(self, sentence, DAG, route):
        N = len(sentence)
        route[N] = (0, 0)
        logtotal = log(self.total)
        for idx in xrange(N - 1, -1, -1):
            route[idx] = max((log(self.FREQ.get(sentence[idx:x + 1]) or 1) -
                              logtotal + route[x + 1][0], x) for x in DAG[idx])

    def get_DAG(self, sentence):
        self.check_initialized()
        DAG = {}
        N = len(sentence)
        for k in xrange(N):
            tmplist = []
            i = k
            frag = sentence[k]
            while i < N and frag in self.FREQ:
                if self.FREQ[frag]:
                    tmplist.append(i)
                i += 1
                frag = sentence[k:i + 1]
            if not tmplist:
                tmplist.append(k)
            DAG[k] = tmplist
        return DAG

    def __cut_all(self, sentence):
        dag = self.get_DAG(sentence)
        old_j = -1
        for k, L in iteritems(dag):
            if len(L) == 1 and k > old_j:
                yield sentence[k:L[0] + 1]
                old_j = L[0]
            else:
                for j in L:
                    if j > k:
                        yield sentence[k:j + 1]
                        old_j = j

    def __cut_DAG_NO_HMM(self, sentence):
        DAG = self.get_DAG(sentence)
        route = {}
        self.calc(sentence, DAG, route)
        x = 0
        N = len(sentence)
        buf = ''
        while x < N:
            y = route[x][1] + 1
            l_word = sentence[x:y]
            if re_eng.match(l_word) and len(l_word) == 1:
                buf += l_word
                x = y
            else:
                if buf:
                    yield buf
                    buf = ''
                yield l_word
                x = y
        if buf:
            yield buf
            buf = ''

    def __cut_DAG(self, sentence):
        DAG = self.get_DAG(sentence)
        route = {}
        self.calc(sentence, DAG, route)
        x = 0
        buf = ''
        N = len(sentence)
        while x < N:
            y = route[x][1] + 1
            l_word = sentence[x:y]
            if y - x == 1:
                buf += l_word
            else:
                if buf:
                    if len(buf) == 1:
                        yield buf
                        buf = ''
                    else:
                        if not self.FREQ.get(buf):
                            recognized = finalseg.cut(buf)
                            for t in recognized:
                                yield t
                        else:
                            for elem in buf:
                                yield elem
                        buf = ''
                yield l_word
            x = y

        if buf:
            if len(buf) == 1:
                yield buf
            elif not self.FREQ.get(buf):
                recognized = finalseg.cut(buf)
                for t in recognized:
                    yield t
            else:
                for elem in buf:
                    yield elem

    def cut(self, sentence, cut_all=False, HMM=True):
        """
        The main function that segments an entire sentence that contains
        Chinese characters into seperated words.

        :param sentence: The str(unicode) to be segmented.
        :param cut_all: Model type. True for full pattern, False for accurate pattern.
        :param HMM: Whether to use the Hidden Markov Model.
        :return:
        """
        sentence = strdecode(sentence)

        if cut_all:
            re_han = re_han_cut_all
            re_skip = re_skip_cut_all
        else:
            re_han = re_han_default
            re_skip = re_skip_default
        if cut_all:
            cut_block = self.__cut_all
        elif HMM:
            cut_block = self.__cut_DAG
        else:
            cut_block = self.__cut_DAG_NO_HMM
        blocks = re_han.split(sentence)
        for blk in blocks:
            if not blk:
                continue
            if re_han.match(blk):
                for word in cut_block(blk):
                    yield word
            else:
                tmp = re_skip.split(blk)
                for x in tmp:
                    if re_skip.match(x):
                        yield x
                    elif not cut_all:
                        for xx in x:
                            yield xx
                    else:
                        yield x

    def cut_for_search(self, sentence, HMM=True):
        """
        Finer segmentation for search engines.
        """
        words = self.cut(sentence, HMM=HMM)
        for w in words:
            if len(w) > 2:
                for i in xrange(len(w) - 1):
                    gram2 = w[i:i + 2]
                    if self.FREQ.get(gram2):
                        yield gram2
            if len(w) > 3:
                for i in xrange(len(w) - 2):
                    gram3 = w[i:i + 3]
                    if self.FREQ.get(gram3):
                        yield gram3
            yield w

    def lcut(self, *args, **kwargs):
        return list(self.cut(*args, **kwargs))

    def lcut_for_search(self, *args, **kwargs):
        return list(self.cut_for_search(*args, **kwargs))

    _lcut = lcut
    _lcut_for_search = lcut_for_search

    def _lcut_no_hmm(self, sentence):
        return self.lcut(sentence, False, False)

    def _lcut_all(self, sentence):
        return self.lcut(sentence, True)

    def _lcut_for_search_no_hmm(self, sentence):
        return self.lcut_for_search(sentence, False)

    def get_dict_source(self):
        """
        Get the dictionary file object
        :return: dictionary file object
        """
        return self.dictionary

    def load_userdict(self, dictionary):
        """
        Load personalized dict to improve detect rate.

        Structure of dict records:
        word1 freq1 word_type1
        word2 freq2 word_type2
        ...

        Word type may be ignored
        :param dictionary: A plain text file contains words and their ocurrences.
                  Can be a DictResource object, or the path of the dictionary file,
                  whose encoding must be utf-8.
        :return:
        """

        self.check_initialized()
        if isinstance(dictionary, string_types):
            user_dict = FileDictResource(dictionary)
        elif isinstance(dictionary, DictResource):
            user_dict = dictionary
        elif isinstance(dictionary, typing.List):
            user_dict = PureDictResource(dictionary)
        else:
            raise ValueError("The expected 'dictionary' should be file path or sequence or instance of DictResource")
        self.batch = True
        change_list = []
        for word, freq, tag in user_dict.get_record():
            if freq is not None and isinstance(freq, string_types):
                freq = freq.strip()
            if tag is not None:
                tag = tag.strip()
            change_list += self.add_word(word, freq, tag)
        self._notify(change_list)
        self.batch = False

    def add_word(self, word, freq=None, tag=None):
        """
        Add a word to dictionary.

        freq and tag can be omitted, freq defaults to be a calculated value
        that ensures the word can be cut out.
        """
        self.check_initialized()
        word = strdecode(word)
        freq = int(freq) if freq is not None else self.suggest_freq(word, False)
        self.FREQ[word] = freq
        self.total += freq
        if tag:
            self.user_word_tag_tab[word] = tag

        # count the words that have changed.
        changed_list = []
        changed_list.append((word, freq, tag))
        for ch in xrange(len(word) - 1):
            wfrag = word[:ch + 1]
            wfrag = wfrag.strip()
            if wfrag not in self.FREQ:
                self.FREQ[wfrag] = 0
                changed_list.append((wfrag, 0, None))
        if freq == 0:
            finalseg.add_force_split(word)
        if not self.batch and len(changed_list) != 0:
            self._notify(changed_list)
        return changed_list

    def del_word(self, word):
        """
        Convenient function for deleting a word.
        """
        self.add_word(word, 0)

    def suggest_freq(self, segment, tune=False):
        """
        Suggest word frequency to force the characters in a word to be
        joined or splitted.

        Note that HMM may affect the final result. If the result doesn't change,
        set HMM=False.

        :param segment: The segments that the word is expected to be cut into,
                        If the word should be treated as a whole, use a str.
        :param tune: If True, tune the word frequency.
        :return:
        """
        self.check_initialized()
        ftotal = float(self.total)
        freq = 1
        if isinstance(segment, string_types):
            word = segment
            for seg in self.cut(word, HMM=False):
                freq *= self.FREQ.get(seg, 1) / ftotal
            freq = max(int(freq * self.total) + 1, self.FREQ.get(word, 1))
        else:
            segment = tuple(map(strdecode, segment))
            word = ''.join(segment)
            for seg in segment:
                freq *= self.FREQ.get(seg, 1) / ftotal
            freq = min(int(freq * self.total), self.FREQ.get(word, 0))
        if tune:
            self.add_word(word, freq)
        return freq

    def tokenize(self, unicode_sentence, mode="default", HMM=True):
        """
        Tokenize a sentence and yields tuples of (word, start, end)

        :param unicode_sentence: the str(unicode) to be segmented.
        :param mode: "default" or "search", "search" is for finer segmentation.
        :param HMM: whether to use the Hidden Markov Model.
        :return:
        """
        if not isinstance(unicode_sentence, text_type):
            raise ValueError("jieba: the input parameter should be unicode.")
        start = 0
        if mode == 'default':
            for w in self.cut(unicode_sentence, HMM=HMM):
                width = len(w)
                yield (w, start, start + width)
                start += width
        else:
            for w in self.cut(unicode_sentence, HMM=HMM):
                width = len(w)
                if len(w) > 2:
                    for i in xrange(len(w) - 1):
                        gram2 = w[i:i + 2]
                        if self.FREQ.get(gram2):
                            yield (gram2, start + i, start + i + 2)
                if len(w) > 3:
                    for i in xrange(len(w) - 2):
                        gram3 = w[i:i + 3]
                        if self.FREQ.get(gram3):
                            yield (gram3, start + i, start + i + 3)
                yield (w, start, start + width)
                start += width

    def set_dictionary(self, dict_source):
        """
        Set the path for dictionary
        :param dict_source: the path to dictionary
        :return:
        """
        with self.lock:
            if isinstance(dict_source, text_type):
                dict_source = FileDictResource(dict_source)
            self.dictionary = dict_source
            self.initialized = False

    def _notify(self, changed_dict_list):
        self.dict_sub.on_next((changed_dict_list, self.dictionary))


def _lcut_all(s):
    return dt._lcut_all(s)


def _lcut(s):
    return dt._lcut(s)


def _lcut_no_hmm(s):
    return dt._lcut_no_hmm(s)


def _lcut_all(s):
    return dt._lcut_all(s)


def _lcut_for_search(s):
    return dt._lcut_for_search(s)


def _lcut_for_search_no_hmm(s):
    return dt._lcut_for_search_no_hmm(s)


def _pcut(sentence, cut_all=False, HMM=True):
    parts = strdecode(sentence).splitlines(True)
    if cut_all:
        result = pool.map(_lcut_all, parts)
    elif HMM:
        result = pool.map(_lcut, parts)
    else:
        result = pool.map(_lcut_no_hmm, parts)
    for r in result:
        for w in r:
            yield w


def _pcut_for_search(sentence, HMM=True):
    parts = strdecode(sentence).splitlines(True)
    if HMM:
        result = pool.map(_lcut_for_search, parts)
    else:
        result = pool.map(_lcut_for_search_no_hmm, parts)
    for r in result:
        for w in r:
            yield w


def enable_parallel(processnum=None):
    """
    Change the module's `cut` and `cut_for_search` functions to the
    parallel version.

    Note that this only works using dt, custom Tokenizer
    instances are not supported.
    """
    global pool, dt, cut, cut_for_search
    from multiprocessing import cpu_count
    if os.name == 'nt':
        raise NotImplementedError(
            "jieba: parallel mode only supports posix system")
    else:
        from multiprocessing import Pool
    dt.check_initialized()
    if processnum is None:
        processnum = cpu_count()
    pool = Pool(processnum)
    cut = _pcut
    cut_for_search = _pcut_for_search


def disable_parallel():
    global pool, dt, cut, cut_for_search
    if pool:
        pool.close()
        pool = None
    cut = dt.cut
    cut_for_search = dt.cut_for_search
