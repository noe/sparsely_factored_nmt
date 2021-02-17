from ling.constants import Attr


BASIC_ATTRS = [
    Attr.CANONICAL_FORM,
    Attr.ALOMORPH,
    Attr.CATEGORY,
    Attr.WORD,
    Attr.CUSTOM_WORD_INITIAL,
    Attr.CUSTOM_WORD_FINAL,
    Attr.WORD_N,
    ]


TRANSVERSAL_FLEXION_ATTRS = tuple([
    Attr.NUMBER,
    Attr.GENDER,
    Attr.PERSON,
    Attr.TENSE,
    Attr.PREDICATE_FORM,
    Attr.VERBAL_MODE,
    Attr.PERFECTION,  # relevant for some languages that featurize
    # part of the tense information, like Euskera
    Attr.CASE,  # for languages with cases, but also for English' saxon genitive
    Attr.DEGREE,  # for things like superlatives
    Attr.INFLECTION,  # for German strong and weak forms
    Attr.UNDER,  # "-ly"/"-mente" adverbs appear as adjectives in terminals,
                 #  with UNDER we distinguish them from normal adjectives
    Attr.NEGATION,  # in some languages the negation is featurized
                           # in the verb, e.g. Spanish, Catalan, Portuguese
    Attr.ABSOLUTE_SUPERLATIVE,
    ])


TARGET_ATTRS = BASIC_ATTRS + list(TRANSVERSAL_FLEXION_ATTRS)

NOT_LEMMATIZABLE_CATEGORIES = ['NUM', 'UNK', 'TOKEN', 'N-FLEX', 'V-FLEX']
