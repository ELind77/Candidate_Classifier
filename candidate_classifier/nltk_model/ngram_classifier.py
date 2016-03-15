#!/usr/bin/env python2
# -*- coding: utf-8 -*-


from math import log
from candidate_classifier.nltk_model import NgramModel
import warnings
from nltk.probability import LidstoneProbDist
import itertools


__author__ = 'Eric Lind'

# TODO: Different estimators
# - Good-Turing is screwed so that's out
# - WrittenBell should be fine though


class NgramModelClassifier(object):
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
            smoothing, anything else is Lidstone.
        # :param threshold: A probability threshold that needs to be met for a given
        #     sequence to be said to have the desired class. Should be a float [0,1]
        # :type threshold: float|int
        """
        # Check params
        if n > 6:
            warnings.warn("You have specified an n-gram degree greater than 6."
                          "be aware that this is likely to use a lot of memory, "
                          "and my result in overfitting your data.")
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

        # check threshold
        # if threshold < 0 or threshold > 1:
        #     raise ValueError("Out of range value given for threshold")
        # # Map threshold to logspace
        # self.threshold = threshold
        self.log_threshold = -log(threshold, 2)

    # FIXME: y needs to be a list...
    def fit(self, X, y, classes=None, **kwargs):
        """
        X should be an array-like where each element is an iterable of tokens.
        The classifier will train an two NgramModels, one for each class.
        Not specifying classes will result in an error. And specifying more
        than two classes will result in an error (this should be fixed soon).

        y is a list of classes.  Right now, this only works as a binary
        classifier

        and classes must evaluate to True and False when tested
        as booleans.  That said, it could be expanded to use a different
        model for every class.
        """
        # Check classes
        if classes is None:
            raise ValueError("No classes specified for NGram Classifier.  This class "
                             "requires exactly two classes to be specified.")
        if len(classes) != 2:
            raise ValueError("Number of classes not equal to two. "
                             "NGramClassifier is a binary classifier and requires exactly "
                             "two classes to be specified.")
        self.classes = classes

        # Get docs for each class
        self.x1 = itertools.ifilter(lambda e: y[e[0]] == classes[0], enumerate(X))
        self.x2 = itertools.ifilter(lambda e: y[e[1]] == classes[1], enumerate(X))

        # Get class distribution
        self.n_y1 = len(itertools.tee(self.x1))
        self.n_y2 = len(itertools.tee(self.x2))

        # Build models
        self.m1 = NgramModel(self.n, self.x1, estimator=self.estimator)
        self.m2 = NgramModel(self.n, self.x2, estimator=self.estimator)

    def predict(self, X):
        """X is an array-like where each element is a list of tokens."""
        # return [self.model.prob_seq(seq) >= self.log_threshold for seq in X]

        p1 = self.m1.prob_seq(X)
        p2 = self.m2.prob_seq(X)

        r = (p1 / p2) * (self.n_y1 / self.n_y2)

        if r > 1:
            return self.classes[0]
        else:
            return self.classes[1]
        

    def get_params(self, deep=False):
        return {
            'n': self.n,
            'alpha': self.alpha
        }
