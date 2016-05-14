from nose import tools as nosey
from candidate_classifier.token_dictionary import TokenDictionary


__author__ = 'Eric Lind'


def test_dictionary_works():
    d = TokenDictionary()
    s = "Anything you can do I can do better.".split()
    nosey.assert_equal(d[s], [0, 1, 2, 3, 4, 2, 3, 5])
