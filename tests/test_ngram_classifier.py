from nose import tools as nosey
import unittest
from candidate_classifier.nltk_model import NgramModel
from candidate_classifier.nltk_model.ngram_classifier import NgramClassifier
import ujson


__author__ = 'Eric Lind'

# TODO:
# - Create data just for this test.
# - Test fit and predict


with open('candidate_classifier/data/processed/processed.json', 'rb') as _f:
    processed = ujson.load(_f)

trump_sents = processed['TRUMP']['sents'][:1000]
trump_labels = ['T']*len(trump_sents)
hillary_sents = processed['CLINTON']['sents'][:1000]
hillary_labels = ['H']*len(hillary_sents)


data = trump_sents+hillary_sents
labels = trump_labels + hillary_labels


def test_ngramclassifier_builds_models():
    c = NgramClassifier()
    c.fit(data, labels)

    nosey.assert_is_instance(c, NgramClassifier)
    nosey.assert_is_instance(c.m1, NgramModel)
    nosey.assert_is_instance(c.m2, NgramModel)


def test_ngramclassifier_trains_models():
    c = NgramClassifier()
    c.fit(data, labels)

    nosey.assert_greater(len(c.m1._ngrams), 0)
    nosey.assert_greater(len(c.m2._ngrams), 0)


