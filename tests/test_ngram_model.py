from nose import tools as nosey
from nose.tools import with_setup
from candidate_classifier.nltk_model import NgramModel
from nltk.probability import LaplaceProbDist, LidstoneProbDist, \
    ConditionalFreqDist, ConditionalProbDist
import itertools

__author__ = 'Eric Lind'


# TODO:
# - Calculate probabilities for Lidstone with alpha=0.001 by hand
# - Test for save/load the model
# - Test generate, prob, prob_seq


DOC1 = ['foo', 'foo', 'foo', 'foo', 'bar', 'baz']

def gen_docs():
    for d in DOC1:
        yield d

def test_with_laplace():
    est = LaplaceProbDist
    lm = NgramModel(3, docs=DOC1, estimator=est)

    nosey.assert_almost_equal(lm.prob('foo', ('foo', 'foo')), 0.5)
    # Try with unseen context
    nosey.assert_almost_equal(lm.prob('baz', ('foo', 'foo')), 1/6.0)


# I'm going to need to do this out by hand at some point
# def test_with_lidstone():
#     num_word_types = len(set(DOC1))
#     est = lambda freqdist, bins: LidstoneProbDist(freqdist, 0.002, bins)
#     lm = NgramModel(3, DOC1, estimator=est, bins=num_word_types)
#
#     nosey.assert_almost_equal(lm.prob('foo', ('foo', 'foo')), 0.5)
#
#     # Try with unseen context
#     nosey.assert_almost_equal(lm.prob('baz', ('foo', 'foo')), 1/6.0)


def test_trains_model():
    lm = NgramModel(3, DOC1)
    nosey.assert_is_instance(lm._model, ConditionalProbDist)


def test_creates_lower_order_models():
    lm = NgramModel(3, DOC1)
    nosey.assert_is_instance(lm._backoff, NgramModel)


def test_trains_backoff_models():
    lm = NgramModel(3, DOC1)
    nosey.assert_is_instance(lm._backoff._model, ConditionalProbDist)


def test_sets_params():
    lm = NgramModel(3, DOC1)
    # backoff alphas
    nosey.assert_is_instance(lm._backoff_alphas, dict)
    nosey.assert_equal(len(lm._cfd.conditions()), len(lm._backoff_alphas))


def test_sets_backoff_model_params():
    lm = NgramModel(3, DOC1)
    nosey.assert_is_instance(lm._backoff._backoff_alphas, dict)
    nosey.assert_equal(len(lm._backoff._cfd.conditions()),
                       len(lm._backoff._backoff_alphas))


def test_unigram_model_doesnt_create_backoff():
    lm = NgramModel(1, DOC1)
    nosey.assert_is_none(lm._backoff)


def test_model_can_use_generator_of_strings():
    lm = NgramModel(3, gen_docs())
    nosey.assert_is_instance(lm, NgramModel)
    nosey.assert_is_instance(lm._backoff, NgramModel)
    nosey.assert_is_instance(lm._model, ConditionalProbDist)
    nosey.assert_is_instance(lm._backoff._model, ConditionalProbDist)


# NB: This is a one-use iterator, is it technically a generator as well?
def test_model_can_use_generator_of_list_of_strings():
    docs = itertools.chain(DOC1, DOC1)
    lm = NgramModel(3, docs)
    nosey.assert_is_instance(lm, NgramModel)
    nosey.assert_is_instance(lm._backoff, NgramModel)
    nosey.assert_is_instance(lm._model, ConditionalProbDist)
    nosey.assert_is_instance(lm._backoff._model, ConditionalProbDist)


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








