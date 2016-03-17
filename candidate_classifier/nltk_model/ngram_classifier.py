#!/usr/bin/env python2
# -*- coding: utf-8 -*-


from math import log
import warnings
from nltk.probability import LidstoneProbDist
import itertools
from collections import Sequence
import operator
import numpy as np

from candidate_classifier.nltk_model import NgramModel


__author__ = 'Eric Lind'

# TODO: Different estimators
# - Good-Turing is screwed so that's out
# - WrittenBell should be fine though


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
        :param alpha: The additive smoothing parameter for the distribution of unseen
            events.  Defaults to 0.01.  If 1 is specified, you're getting Laplace
            smoothing, anything else is Lidstone.  It is a good idea to tune this
            parameter.
        # :param threshold: A probability threshold that needs to be met for a given
        #     sequence to be said to have the desired class. Should be a float [0,1]
        # :type threshold: float|int
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
        self.estimator = lambda freqdist, bins: LidstoneProbDist(freqdist, alpha, bins)
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
                             "two classes to be specified.")
        self.classes = classes

        # Get docs for each class
        self.x1 = itertools.imap(operator.itemgetter(1),
                                 itertools.ifilter(lambda e: y[e[0]] == classes[0], enumerate(X)))
        self.x2 = itertools.imap(operator.itemgetter(1),
                                 itertools.ifilter(lambda e: y[e[0]] == classes[1], enumerate(X)))

        # Get class distribution
        # FIXME: Use operator module
        self.n_y1 = len(list(itertools.ifilter(lambda e: e == classes[0], y)))
        self.n_y2 = len(list(itertools.ifilter(lambda e: e == classes[1], y)))
        # Should be y1/y2
        self.y1_prob = self.n_y1/float(self.n_y1+self.n_y2)
        self.y2_prob = self.n_y2/float(self.n_y1+self.n_y2)
        self.y_ratio = self.y1_prob / self.y2_prob

        # Build models
        self.m1 = NgramModel(self.n, self.x1, estimator=self.estimator)
        self.m2 = NgramModel(self.n, self.x2, estimator=self.estimator)

    def predict(self, X):
        """X is a 2D array-like where each element is a list of tokens."""
        return [self._get_prediction(sent) for sent in X]

    def _get_prediction(self, sequence):
        p1 = self.m1.prob_seq(sequence, lg=False)
        p2 = self.m2.prob_seq(sequence, lg=False)

        # Handle 0 for both
        if p1 == p2 == 0:
            # Return a random value based on training data frequency
            return np.random.choice(self.classes, 1, p=[self.y1_prob, self.y2_prob])[0]
        # Handle division by zero
        if p2 == 0:
            return self.classes[0]

        # Calculate the ratio of the probability that the sequence has
        # class one given the sequence, to the probability of class2 given
        # the sequence
        r = (p1 / p2) * self.y_ratio

        if r > 1:
            return self.classes[0]
        else:
            return self.classes[1]

    def get_params(self, deep=False):
        return {
            'n': self.n,
            'alpha': self.alpha
        }
