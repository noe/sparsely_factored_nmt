import sys
import regex as re
from unidecode import unidecode
import unicodedata


PUNCTUATION_REGEX_EXCEPT_LEADING_MARKS = re.compile(r'[^\P{P}¿¡]+')


# taken from https://stackoverflow.com/a/93029/674487
ALL_CHARS = (chr(i) for i in range(sys.maxunicode))
CONTROL_CHARS = ''.join(c for c in ALL_CHARS if unicodedata.category(c) == 'Cc')
CONTROL_CHAR_REGEX = re.compile('[%s]' % re.escape(CONTROL_CHARS))


def _replace(match):
    old_puntuation = match.group()
    return unidecode(old_puntuation)


def normalize_punctuation(s: str):
    s = s.replace('„', '"')  # unidecode does not handle this properly
    s = PUNCTUATION_REGEX_EXCEPT_LEADING_MARKS.sub(_replace, s)
    return s


def remove_control_chars(s: str):
    return CONTROL_CHAR_REGEX.sub('', s)
