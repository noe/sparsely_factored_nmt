from .attrs import NOT_LEMMATIZABLE_CATEGORIES
from ling.constants import Attr


def is_lemmatizable(terminal: dict):
    category = terminal[Attr.CATEGORY]
    banned_cat = category in NOT_LEMMATIZABLE_CATEGORIES
    has_alo = Attr.ALOMORPH in terminal
    is_under_phrasal = 'PHR-S' in terminal.get(Attr.UNDER, [])
    return not banned_cat and has_alo and not is_under_phrasal


def should_ignore(terminal: dict, ignore_stringp: bool):
    return ignore_stringp and terminal[Attr.CATEGORY] == "STRINGP"


def remove_under(terminal: dict):
    if terminal[Attr.CATEGORY] not in ["AST", "VST"] and Attr.UNDER in terminal:
        # Attribute "UNDER" is used in the lammatized vocabs to identify:
        # - adverbs that end in "-ly / -mente", which are marked as
        #   adjectives (AST). For other categories, this attribute is
        #   counter productive.
        # - adverbs created from participles (CAT VST), that is,a VST
        #   (UNDER ADVB) indicates the same (e.g. "organizedly")
        terminal.pop(Attr.UNDER)
    return terminal


def postprocess_terminal(terminal):
    # return remove_under(terminal)
    return terminal
