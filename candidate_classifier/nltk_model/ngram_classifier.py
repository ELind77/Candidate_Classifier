#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import warnings
from nltk.probability import LidstoneProbDist
import itertools
from collections import Sequence
import operator
import numpy as np
import math

from candidate_classifier.nltk_model import NgramModel



__author__ = 'Eric Lind'

# TODO: Different estimators
# - Good-Turing is screwed so that's out
# - WrittenBell should be fine though

def make_estimator(alpha):
    def est(fdist, bins):
        return LidstoneProbDist(fdist, alpha, bins)
    return est


class NgramClassifier(object):
    """
    sklearn-compatible binary classifier using an NgramModel and bayes rule
    for deciding which class to assign.

    Inspired by:
    Grégoire Mesnil, Tomas Mikolov, Marc'Aurelio and Yoshua Bengio:
    Ensemble of Generative and Discriminative Techniques for Sentiment Analysis \
    of Movie Reviews; Submitted to the workshop track of ICLR 2015.
    http://arxiv.org/abs/1412.5335
    """
    def __init__(self, n=4, alpha=0.01):
        """
        :param n: The degree of the NgramModel
        :type n: int
        :param alpha: The additive smoothing parameter for the distribution of unseen
            events.  Defaults to 0.01.  If 1 is specified, you're getting Laplace
            smoothing, anything else is Lidstone.  It is a good idea to tune this
            parameter.
        :type alpha: float
        """
        # Check params
        if n > 6:
            warnings.warn("You have specified an n-gram degree greater than 6."
                          "be aware that this is likely to use a lot of memory, "
                          "and may result in overfitting your data.")
        if alpha < 0:
            raise ValueError("Negative value given for alpha parameter. "
                             "[A]lpha must be greater than 0.")
        self.n = n
        self.alpha = alpha
        # self.estimator = lambda freqdist, bins: LidstoneProbDist(freqdist, alpha, bins)
        self.est = make_estimator(alpha)
        self.classes = [0, 1]

        self.x1 = None
        self.x2 = None
        self.m1 = None
        self.m2 = None
        self.n_y1 = 0
        self.n_y2 = 0
        self.y1_prob = 0
        self.y2_prob = 0
        self.y_ratio = 0

    # FIXME: y needs to be a list, but what if it's HUGE?
    def fit(self, X, y, classes=None, **kwargs):
        """
        X should be an array-like where each element is an iterable of tokens.
        The classifier will train two NgramModels, one for each class.
        Not specifying classes will result in an error. And specifying more
        than two classes will result in an error (this should be fixed soon).

        y is a list of classes.  Y has to be some kind of sequence that has
        __getitem__.  If it's a one-use iterator, it will be transformed into a
        list and held in memory during training.

        Right now, this only works as a binary classifier

        and classes must evaluate to True and False when tested
        as booleans.  That said, it could be expanded to use a different
        model for every class.
        """
        # Check y
        if not isinstance(y, Sequence):
            y = list(y)

        # Check classes
        # if classes is None:
        #     raise ValueError("No classes specified for NGram Classifier.  This class "
        #                      "requires exactly two classes to be specified.")
        if classes is None:
            classes = list(set(y))
        if len(classes) != 2:
            raise ValueError("Number of classes not equal to two. "
                             "NGramClassifier is a binary classifier and requires exactly "
                             "two classes to be specified. %s" % classes)
        self.classes = sorted(classes)

        # Get docs for each class
        self.x1 = itertools.imap(operator.itemgetter(1),
                                 itertools.ifilter(lambda e: y[e[0]] == classes[0], enumerate(X)))
        self.x2 = itertools.imap(operator.itemgetter(1),
                                 itertools.ifilter(lambda e: y[e[0]] == classes[1], enumerate(X)))

        # Get class distribution
        # FIXME: Use operator module to optimize
        self.n_y1 = len(list(itertools.ifilter(lambda e: e == classes[0], y)))
        self.n_y2 = len(list(itertools.ifilter(lambda e: e == classes[1], y)))
        # Should be y1/y2
        self.y1_prob = np.float64(self.n_y1) / np.float64(self.n_y1 + self.n_y2)
        self.y2_prob = np.float64(self.n_y2) / np.float64(self.n_y1 + self.n_y2)
        self.y_ratio = self.y1_prob / self.y2_prob

        # Build models
        self.m1 = NgramModel(self.n, self.x1, estimator=self.est)
        self.m2 = NgramModel(self.n, self.x2, estimator=self.est)

    def predict(self, X):
        """X is a 2D array-like where each element is a list of tokens.

        Should return an array of length n-samples
        """
        return [self._get_prediction(sent) for sent in X]

    def _get_prediction(self, sequence):
        r = self._calc_prob_ratio(sequence)

        if r > 1:
            return self.classes[0]
        else:
            return self.classes[1]

    def _calc_prob_ratio(self, sequence):
        """Calculates the ratio of the tow class probabilities"""
        p1 = self.m1.prob_seq(sequence)
        p2 = self.m2.prob_seq(sequence)

        # Handle 0 for both
        if p1 == p2 == 0:
            # FIXME: Refactor this
            # Return a random value based on training data frequency
            choice = np.random.choice(self.classes, 1, p=[self.y1_prob, self.y2_prob])[0]
            if choice == self.classes[0]:
                return 2.0
            else:
                return 0.1
        # Handle division by zero
        if p2 == 0:
            # If only the second class has zero probability, return a ratio
            # value greater than 1 so the first class is picked
            return 2.0

        # FIXME:
        # Move out of log space
        # This seems like a really crummy way to do things, but I'm not really sure
        # what a better way would be.
        # p1 = d_ctx.power(2, -decimal.Decimal(p1))
        # p2 = d_ctx.power(2, -decimal.Decimal(p2))
        p_arr = np.power(np.array([2.0, 2.0], dtype=np.float128),
                         np.array([-p1, -p2], dtype=np.float128))

        # Calculate the ratio of the probability that the sequence has
        # class one given the sequence, to the probability of class2 given
        # the sequence
        # return (p1 / p2) * decimal.Decimal(self.y_ratio)
        return ((p_arr / p_arr[1]) * self.y_ratio)[0]


    # FIXME: Better documentation explanation
    # The better way to accurately estimate these probabilities would be to find a
    # way to get the probability of the sequence over the space of all possible
    # sentences as in LDAInfVoc.  But that's also a bit dubious, and very involved
    # so I'll stick with this for now and see how it goes.

    # Actually, I think that the P(x) term can be ignored, since it's the same
    # regardless of training data.  It's just a scaling factor and drops out
    # when making comparisons.
    def predict_proba(self, X):
        """
        Returns probability estimates for each value in X.
        """
        out = []
        for s in X:
            s_probs = self._get_probs(s)
            y_probs = np.asarray([self.y1_prob, self.y2_prob], dtype=np.float64)
            c_probs = s_probs * y_probs
            # Make the class probabilities sum to 1
            # normalized_probs = c_probs / np.sum(c_probs)
            # normalized_probs = c_probs / np.max(c_probs)
            out.append(c_probs)
        return np.asarray(out, dtype=np.float64)
        # return np.apply_along_axis(self._get_class_probs, 1, X)

    # def _get_class_probs(self, seq):
    #     probs = self._get_probs(seq)
    #     y_probs = np.asarray([self.y1_prob, self.y2_prob])
    #     return probs * y_probs

    def _get_probs(self, seq):
        p1 = self.m1.prob_seq(seq)
        p2 = self.m2.prob_seq(seq)

        # FIXME:
        # Move out of log space
        # This seems like a really crummy way to do things, but I'm not really sure
        # what a better way would be.
        # Maybe numpy can do it?
        # p1 = d_ctx.power(2, -decimal.Decimal(p1))
        # p2 = d_ctx.power(2, -decimal.Decimal(p2))
        p_arr = np.power(np.array([2.0, 2.0], dtype=np.float128),
                         np.array([-p1, -p2], dtype=np.float128))
        return p_arr

    def get_params(self, deep=False):
        return {
            'n': self.n,
            'alpha': self.alpha
        }
