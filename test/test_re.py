import re

re_userdict = re.compile(r'^([^\s]+)\s*([0-9]+)?\s*([a-z]+)?$', re.U)

print(re_userdict.match('好人 h').groups())
