#!/usr/bin/env python2


from math import log
from candidate_classifier.nltk_model import NgramModel


__author__ = 'Eric Lind'


class NgramModelClassifier(object):
    """sklearn-compatible classifier using an NgramModel and a probability threshold"""
    def __init__(self, n=4, threshold=0.5):
        """
        :param n: The degree of the NgramModel
        :param threshold: A probability threshold that needs to be met for a given
            sequence to be said to have the desired class. Should be a float [0,1]
        :type threshold: float|int
        """
        # check threshold
        if threshold < 0 or threshold > 1:
            raise ValueError("Out of range value given for threshold")
        # Map threshold to logspace
        self.threshold = threshold
        self.log_threshold = -log(threshold, 2)

        self.n = 4
        self.model = None


    def fit(self, X, y, **kwargs):
        """X should be an array-like where each element is a list of tokens.
        The classifier will train an NgramModel using the token lists that
        have the label 1 as documents.

        y is a list of classes.  Right now, this only works as a binary
        classifier and classes must evaluate to True and False when tested
        as booleans.  That said, it could be expanded to use a different
        model for every class I suppose.
        """
        if 'n' in kwargs:
            n = kwargs['n']
            self.n = n

        # TODO: Allow different estimators?
        self.model = NgramModel(self.n, X)

    def predict(self, X):
        """X is an array-like where each element is a list of tokens."""
        return [self.model.prob_seq(seq) >= self.log_threshold for seq in X]

    def get_params(self, deep=False):
        return {
            'n': self.n,
            'threshold': self.threshold
        }
