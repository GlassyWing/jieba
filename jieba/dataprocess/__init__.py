from __future__ import absolute_import, unicode_literals
from .resources import *
from .cache_strategies import *
import os

DEFAULT_DICT = FileDictResource(os.path.normpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../dict.txt')))
DEFAULT_CACHE_STRATEGY = FileCacheStrategy
