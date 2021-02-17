

class Attr:
    CATEGORY = "CAT"
    CANONICAL_FORM = "CAN"
    ALOMORPH = "ALO"
    WORD = "WORD"
    WORD_N = "WORD#"
    WORD_INITIAL = "WI"
    WORD_FINAL = "WF"
    NUMBER = "NU"  # values (set): SG, PL
    GENDER = "GD"
    PERSON = "PS"  # values (set): 1, 2 ,3
    TENSE = "TN"   # values (set): PA, PR, INDF, FUT
    PREDICATE_FORM = "PF"    # values (set): FIN, PAPL
    VERBAL_MODE = "MD"  # values: IND
    PERFECTION = "PERF"
    SEX = "SX"     # values: M, F
    TYPE_OF_NOUN = "TYN"
    KIND = "KN"
    CASE = "CA"
    DEGREE = "DG"  # for comparatives and superlatives
    INFLECTION = "INFLECTION"  # German's weak/strong forms
    UNDER = "UNDER"
    CUSTOM_WORD_INITIAL = "IN-WI"
    CUSTOM_WORD_FINAL = "IN-WF"
    EN_FOLLOWS_VOWEL = "FOLL-ON"  # in English, to distinguis "a" (CO) from "an" (VO)
    NEGATION = "NEG"  # featurized negation of the verb
    EU_NOR = "NOR"  # euskera
    EU_NORI = "NORI"
    EU_NORK = "NORK"
    DE_CONTRACTED_DETERMINANT = "CONTRED-DET"  # mark DE contracted determinants e.g. zur = zu der
    ABSOLUTE_SUPERLATIVE = "SUPABS"
