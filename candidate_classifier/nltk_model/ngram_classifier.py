#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import warnings
from nltk.probability import LidstoneProbDist
import itertools
from collections import Sequence, Iterable, Iterator
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from candidate_classifier.nltk_model import NgramModel
from candidate_classifier import utils



__author__ = 'Eric Lind'

# TODO: Different estimators
# - Good-Turing is screwed so that's out
# - WrittenBell should be fine though

def make_estimator(alpha):
    def est(fdist, bins):
        return LidstoneProbDist(fdist, alpha, bins)
    return est


# TODO: Add derivation to docs
class NgramClassifier(BaseEstimator, ClassifierMixin):
    """
    scikit-learn compatible binary classifier using an NgramModel and bayes rule
    for deciding which class to assign.

    Inspired by:
    Grégoire Mesnil, Tomas Mikolov, Marc'Aurelio and Yoshua Bengio:
    Ensemble of Generative and Discriminative Techniques for Sentiment Analysis \
    of Movie Reviews; Submitted to the workshop track of ICLR 2015.
    http://arxiv.org/abs/1412.5335
    """
    def __init__(self, n=4, alpha=0.01, pad_ngrams=False):
        """
        :param n: The degree of the NgramModel
        :type n: int
        :param alpha: The additive smoothing parameter for the distribution of unseen
            events.  Defaults to 0.01.  If 1 is specified, you're getting Laplace
            smoothing, anything else is Lidstone.  It is a good idea to tune this
            parameter.
        :type alpha: float
        :param pad_ngrams: Whether to add additional padding to sentences when making ngrams
            in order to give more context to the documents.
        :type pad_ngrams: bool
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
        self.pad_ngrams = pad_ngrams
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
    def fit(self, X, y, classes=None):
        """
        Fit an Ngram Classifier on training documents in X and labels in y.

        X should be an array-like where each element is an iterable of tokens.
        The classifier will train two NgramModels, one for each class.
        Not specifying classes will result in an error. And specifying more
        than two classes will result in an error (this should be fixed soon).

        y is a list of classes.  Y has to be some kind of sequence that has
        __getitem__.  If it's a one-use iterator, it will be transformed into a
        list and held in memory during training.

        Right now, this only works as a binary classifier.

        :param X: An iterable of lists, where each list is a sequence to learn.
        :type X: Iterable
        :param y: An iterable of labels for each sequence in X.  There must be exactly
            two unique labels
        :type y: Iterable
        :returns: self
        """
        # Check y
        y = np.asarray(y)

        # Check classes
        # if classes is None:
        #     raise ValueError("No classes specified for NGram Classifier.  This class "
        #                      "requires exactly two classes to be specified.")
        # FIXME: y can't be a generator...
        uniqs = np.unique(y, return_counts=True)
        self.classes = uniqs[0]
        self.n_y1 = uniqs[1][0]
        self.n_y2 = uniqs[1][1]
        if self.classes.shape[0] != 2:
            raise ValueError("Number of classes not equal to two. "
                             "NGramClassifier is a binary classifier and requires exactly "
                             "two classes to be specified. %s" % classes)


        # # Get class distribution
        # self.n_y1 = len(list(itertools.ifilter(lambda e: e == classes[0], y)))
        # self.n_y2 = len(list(itertools.ifilter(lambda e: e == classes[1], y)))
        # Should be y1/y2
        self.y1_prob = np.float64(self.n_y1) / np.float64(self.n_y1 + self.n_y2)
        self.y2_prob = np.float64(self.n_y2) / np.float64(self.n_y1 + self.n_y2)
        self.y_ratio = self.y1_prob / self.y2_prob

        # Build models
        self.m1 = NgramModel(self.n,
                             pad_left=self.pad_ngrams, pad_right=self.pad_ngrams,
                             estimator=self.est)
        self.m2 = NgramModel(self.n,
                             pad_left=self.pad_ngrams, pad_right=self.pad_ngrams,
                             estimator=self.est)

        # An ngram model should be able to train on very large datasets, too large for RAM.
        # In order to accommodate this, both models are trained in batches so that a shuffled
        # one-use generator can be used for training data without having to separate out all
        # examples of each class before training.
        # Most ML models behave differently based on the order of the training data, ngram
        # models don't care.
        b1 = []
        b2 = []
        y_itr = iter(y)
        batch_size = 1000

        for chunk in utils.chunked(X, batch_size):
            for d in chunk:
                try:
                    lbl = y_itr.next()
                except StopIteration:
                    raise ValueError("Labels and training data for NgramClassifier were of unequal "
                                     "length")
                # TODO: This basically acts like OneVsRest...
                if lbl == self.classes[0]:
                    b1.append(d)
                else:
                    b2.append(d)
            if max(len(b1), len(b2)) >= batch_size:
                self._train_models(b1, b2)
                # Clear old values
                b1 = []
                b2 = []
        self._train_models(b1, b2)

        # Set up models
        self.m1._build_model(self.est, {})
        self.m2._build_model(self.est, {})

        return self

    def _train_models(self, b1, b2):
        for d in b1:
            self.m1.train(d)
        for d in b2:
            self.m2.train(d)

    def predict(self, X):
        """X is a 2D array-like where each element is a list of tokens.

        Should return a list of length n-samples.

        :param X: An iterable of lists, where each list is a sequence to learn.
        :type X: Iterable
        :return: A list of the same length as X
        :rtype: list
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

        # Move out of log space
        # If there's a better way to do this, I'm not sure what it is
        p_arr = np.power(np.array([2.0, 2.0], dtype=np.float128),
                         np.array([-p1, -p2], dtype=np.float128))

        # Calculate the ratio of the probability that the sequence has
        # class one given the sequence, to the probability of class2 given
        # the sequence
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
            # NB: I tried multiple normalizations for c_probs:
            # - raw, no normalization
            # - Forcing sum to one by dividing by sum
            # - Setting chosen prob to 1 by dividing by max
            # Forcing sum to 1 performed terribly in evaluation.
            # Setting the chosen prob := 1 scored marginally (~0.01 weighted f1)
            # better, than raw scores, but as the purpose is to break ties, it
            # seemed a bit silly to just give 1 as the predicted probability
            # so I just stuck with the raw scores.
            out.append(c_probs)
        return np.asarray(out, dtype=np.float64)


    def _get_probs(self, seq):
        p1 = self.m1.prob_seq(seq)
        p2 = self.m2.prob_seq(seq)

        # Move out of log space
        # If there's a better way to do this, I'm not sure what it is
        p_arr = np.power(np.array([2.0, 2.0], dtype=np.float128),
                         np.array([-p1, -p2], dtype=np.float128))
        return p_arr
