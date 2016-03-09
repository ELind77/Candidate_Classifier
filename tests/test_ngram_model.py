from nose import tools as nosey
import unittest

__author__ = 'Eric Lind'


import nltk
from candidate_classifier.nltk_model import NgramModel
from nltk.probability import LaplaceProbDist
from nltk.probability import LidstoneProbDist


DOC1 = ['foo', 'foo', 'foo', 'foo', 'bar', 'baz']


def test_with_laplace():
    num_word_types = len(set(DOC1))
    est = LaplaceProbDist
    lm = NgramModel(3, DOC1, estimator=est, bins=num_word_types)

    nosey.assert_almost_equal(lm.prob('foo', ('foo', 'foo')), 0.5)

    # Try with unseen context
    nosey.assert_almost_equal(lm.prob('baz', ('foo', 'foo')), 1/6.0)


# def test_with_lidstone():
#     num_word_types = len(set(DOC1))
#     est = lambda freqdist, bins: LidstoneProbDist(freqdist, 0.002, bins)
#     lm = NgramModel(3, DOC1, estimator=est, bins=num_word_types)
#
#     nosey.assert_almost_equal(lm.prob('foo', ('foo', 'foo')), 0.5)
#
#     # Try with unseen context
#     nosey.assert_almost_equal(lm.prob('baz', ('foo', 'foo')), 1/6.0)



# word_seq = ['foo', 'foo', 'foo', 'foo', 'bar', 'baz']
# word_types = set(word_seq)
#
# lm = NgramModel(3, word_seq, estimator=LaplaceProbDist, bins=len(word_types))

# def test(m):
#     print m.prob('foo', ('foo', 'foo'))
#     print m.prob('baz', ('foo', 'foo'))
#     print m.prob('foo', ('bar', 'baz'))
#
# test(lm)
#
# # From https://github.com/nltk/nltk/issues/367
# word_seq = ['foo' for i in range(0, 10000)]
# word_seq.append('bar')
# word_seq.append('baz')
#
# est = lambda freqdist, bins: LidstoneProbDist(freqdist, 0.2, bins)
# model = NgramModel(3, word_seq, True, True, est, bins=3)
#
# # Consider the ngram ['bar', 'baz', 'foo']
# # We've never seen this before, so the trigram model will fall back
# context = ('bar', 'baz',)
# word = 'foo'
# print "P(foo | bar, baz) = " + str(model.prob(word, context))


# Try pickling
# =========================
# This doesn't work...

# import cPickle as pickle
#
# with open('test_lm_pickle.p', 'wb') as _f:
#     pickle.dump(model, _f)
#
# print "Pickled"
#
# with open('test_lm_pickle.p', 'rb') as _f:
#     model = pickle.load(_f)
#
# print "unpickled"
#
# print model.prob(word, context)


# Try different probdists
# =========================
# This doesn't work...

# from nltk.probability import SimpleGoodTuringProbDist
#
# lm = NgramModel(3, word_seq, estimator=SimpleGoodTuringProbDist, bins=len(word_types))
#
# test(lm)







