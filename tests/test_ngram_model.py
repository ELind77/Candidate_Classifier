from nose import tools as nosey
from nose.tools import with_setup
import itertools
import math
from nltk.probability import LaplaceProbDist, LidstoneProbDist, \
    ConditionalFreqDist, ConditionalProbDist

from candidate_classifier.nltk_model import NgramModel


__author__ = 'Eric Lind'


# TODO:
# - Calculate probabilities for Lidstone with alpha=0.001 by hand
# - Test generate, prob, prob_seq
# - Correctness tests for entropy and perplexity
# - Test for save/load the model


DOC1 = ['foo', 'foo', 'foo', 'foo', 'bar', 'baz']

def gen_docs():
    for d in DOC1:
        yield d

def test_nmodel_gives_correct_prob_with_laplace_smoothing():
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


def test_nmodel_prob_returns_float():
    lm = NgramModel(3, DOC1)
    nosey.assert_is_instance(lm.prob('bar'), float)


def test_nmodel_logprob_returns_float():
    lm = NgramModel(3, DOC1)
    nosey.assert_is_instance(lm.logprob('bar'), float)


def test_nmodel_logprob_returns_neg_log_of_prob():
    lm = NgramModel(3, DOC1)
    prob = lm.prob('bar')
    logprob = lm.logprob('bar')
    nosey.assert_almost_equal(-math.log(prob, 2), logprob)


def test_generate_returns_list_of_string():
    lm = NgramModel(3, DOC1)
    gen = lm.generate(5)
    nosey.assert_is_instance(gen, list)
    nosey.assert_is_instance(gen[0], basestring)
    nosey.assert_equal(len(gen), 5)

def test_choose_random_word_returns_string():
    lm = NgramModel(3, DOC1)
    rw = lm.choose_random_word(('foo'))

    nosey.assert_is_instance(rw, basestring)



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
