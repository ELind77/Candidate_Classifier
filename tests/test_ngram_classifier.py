from nose import tools as nosey
import unittest
from candidate_classifier.nltk_model import NgramModel
from candidate_classifier.nltk_model.ngram_classifier import \
    NgramClassifier, NgramClassifierMulti
from candidate_classifier.token_dictionary import TokenDictionary
import ujson
import os
import numpy as np

from candidate_classifier.utils import flatten


__author__ = 'Eric Lind'

# TODO:
# - Test multi-class
# - Generators as input
# - pad-ngrams


with open(os.path.join('tests', 'data', 'ngram_preprocessed_test_data.json')) as _f:
    ALL_DATA = ujson.load(_f)

BINARY_DATA = ALL_DATA['TRUMP']['sents'][:1000] + ALL_DATA['CLINTON']['sents'][:1000]
BINARY_LABELS = ([1]*1000) + ([0]*1000)
TEST_DATA = ALL_DATA['TRUMP']['sents'][1000:1200]


def test_ngramclassifier_builds_models():
    c = NgramClassifier()
    c.fit(BINARY_DATA, BINARY_LABELS)

    nosey.assert_is_instance(c, NgramClassifier)
    nosey.assert_is_instance(c.m1, NgramModel)
    nosey.assert_is_instance(c.m2, NgramModel)


def test_ngramclassifier_trains_models():
    c = NgramClassifier()
    c.fit(BINARY_DATA, BINARY_LABELS)

    nosey.assert_greater(len(c.m1._ngrams), 0)
    nosey.assert_greater(len(c.m2._ngrams), 0)


def test_ngramclf_predict_dtype():
    c = NgramClassifier()
    c.fit(BINARY_DATA, BINARY_LABELS)

    predictions = c.predict(TEST_DATA)
    nosey.assert_equal(len(predictions), len(TEST_DATA))

def test_ngramclf_get_params():
    c = NgramClassifier()
    expected1 = {
        'n': 4,
        'alpha': 0.01,
        'pad_ngrams': False,
        'use_dictionary': False
    }
    expected2 = {
        'n': c.n,
        'alpha': c.alpha,
        'pad_ngrams': c.pad_ngrams,
        'use_dictionary': c.use_dictionary
    }

    nosey.assert_dict_equal(c.get_params(), expected1)
    nosey.assert_dict_equal(c.get_params(), expected2)


def test_ngramclf_predict_proba_dtype():
    c = NgramClassifier()
    c.fit(BINARY_DATA, BINARY_LABELS)

    probs = c.predict_proba(TEST_DATA)
    nosey.assert_is_instance(probs, np.ndarray)
    nosey.assert_equal(probs.shape, (len(TEST_DATA), 2))


# Test Multiclass


# Test  using dictionary
def test_ngramclf_with_dictionary():
    c = NgramClassifier(use_dictionary=True)
    c.fit(BINARY_DATA, BINARY_LABELS)

    # Check the dictionary
    nosey.assert_is_instance(c.dictionary, Dictionary)
    # check contents
    for k, v in c.dictionary.d.iteritems():
        nosey.assert_is_instance(k, basestring)
        nosey.assert_is_instance(v, int)

    # Check length
    nosey.assert_equal(len(c.dictionary.d), len(set(filter( None, flatten(BINARY_DATA)))))
