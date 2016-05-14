#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
A scikit-learn compatible classifier using an n-gram language model.

The base NgramClassifier is only a binary classifier.  For multi-class
problems use NgramClassifierMulti.
"""

from __future__ import division

import warnings
from collections import Sequence, Iterable, Iterator
import dill
import types
import copy_reg
import multiprocessing as mp

import numpy as np
from nltk.probability import LidstoneProbDist
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.multiclass import OneVsOneClassifier
from sklearn.utils.validation import check_consistent_length
from sklearn.multiclass import _fit_ovo_binary
from joblib import Parallel, delayed

from candidate_classifier.nltk_model import NgramModel
from candidate_classifier import utils
from candidate_classifier.token_dictionary import TokenDictionary



__author__ = 'Eric Lind'

# TODO: Different estimators
# - Good-Turing is screwed so that's out
# - WrittenBell should be fine though
# - make NgramClassifier a base class and use NgramMulti as the main class
# - Test that it's pickleable




# Helpers to make functions pickleable
def _pickle_fxn(func):
    return _unpickle_fxn, (dill.dumps(func),)

def _unpickle_fxn(data):
    return dill.loads(data)
copy_reg.pickle(types.FunctionType, _pickle_fxn, _unpickle_fxn)


class Worker(mp.Process):
    """A queue consumer for multi-process training of ngram models.
    Consumes training data from a queue and when done, dumps the
    trained model into the output queue.

    Use of the queue allows training from a corpus that is larger
    than RAM when the corpus/labels have been shuffled.
    For example, if the labels are: [0, 0, 0, 1, 0, 1] then in order
    to train sequentially, you would need to read in the entire
    training set and the labels in order to start training.  But
    if each model is trained in it's own process, the training data
    only has to be looped over once.

    The downside to this is that the (very large) models have to be
    pickled and sent back to the main thread and that has to be done
    sequentially.  Also, in python 2.7 it looks like pickle is used
    instead of cPickle, which makes everything worse.

    Possible enhancements:
    - Register cPickle as the pickler for Ngram Model
    - Have NgramModel write all of the large data stores to disk before
        sending model back to the main thread and then asynchronously
        load all of them in the background (relying on the OS for
        parallelizing the file reads) and return from fit method when
        all models are loaded.
    """
    def __init__(self, in_q, out_q, model):
        self.in_q = in_q
        self.out_q = out_q
        self.model = model
        super(Worker, self).__init__()

    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

    def run(self):
        while True:
            try:
                doc = self.in_q.get()
                # Poison pill
                if doc is None:
                    self.in_q.task_done()
                    # Export the now-trained model
                    self.out_q.put(self.model)
                    # Shut down the worker
                    # print "%s Exiting" % self.name
                    break
                # Train the model
                # print doc
                self.model.train(doc)
                # print self.model
                # Tell the queue it's done
                self.in_q.task_done()
            except Exception, e:
                print "Worker encountered error %s" % e
        return


# TODO: LaTeX


def _calc_prob_ratio(sequence, m1, m2, y1_prob, y2_prob, y_log_ratio):
    """
    Calculates the ratio of the two class probabilities.

    Given a sequence, a language model returns the probability of
    that sequence, given the data that it was trained on, P(x|y).
    According to Bayes' Theorem this gives us:

    P(x|y) = (P(y|x) * P(x)) / P(y)

    Now if we have two classes, trained on Y+ and Y-, respectively,
    and we divide P(x|Y+)/P(x|Y-) and multiply by P(Y+)/P(Y-) we get:
        P(Y+|x) / P(Y-|x)
    Which is the ratio of the probability of the sequence having one
    of the two classes.  This gives the classification rule: If this
    ratio is greater than 1 assign Y+, otherwise Y-.

    :param sequence:
    :type sequence: Iterable
    :param m1:
    :type m1: NgramModel
    :param m2:
    :type m2: NgramModel
    :type y1_prob: float
    :type y2_prob: float
    :type y_log_ratio: float
    """
    # Get the negative log probabilities
    # NB: ngram model returns negative log probability so need to
    # make negative to get the actual log probability
    p1 = np.float128(-m1.prob_seq(sequence))
    p2 = np.float128(-m2.prob_seq(sequence))

    # Handle 0 for both
    # This is actually the tie-breaking rule...
    # TODO: Return 0 for ties and handle tie-breaking in get_prediction
    if p1 == p2 == 0:
        # FIXME: Refactor this
        # Return a random value based on training data frequency
        choice = np.random.choice([0, 1], 1, p=[y1_prob, y2_prob])[0]
        if choice == 0:
            return 2.0
        else:
            return -0.1
    # Handle division by zero
    # FIXME: Do I still need this in log-space?
    # No, but it's nice to have
    if p2 == 0:
        # If only the second class has zero probability, return a ratio
        # value greater than 0 so the first class is picked
        return 2.0

    # Calculate the ratio of the probability that the sequence has
    # class one given the sequence, to the probability of class2 given
    # the sequence.
    # Calculate the ratio using log rules:
    # r = (p1/p2) * (py1/py2) becomes this in log space:
    # log(r) = log((p1*py1)/(p2*py2))
    # log(r) = (log(p1) - log(p2)) + (log(y1) - log(y2))
    return (p1 - p2) + y_log_ratio

    # Calculate the ratio of the probability that the sequence has
    # class one given the sequence, to the probability of class2 given
    # the sequence
    # p1 = np.exp2(-self.m1.prob_seq(sq))
    # p2 = np.exp2(-self.m2.prob_seq(sq))
    # return (p1/p2) * self.y_ratio



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
    def __init__(self, n=4, alpha=0.01, pad_ngrams=False, use_dictionary=False, parallel=False):
        """
        :param n: The degree of the NgramModel
        :type n: int
        :param alpha: The additive smoothing parameter for the distribution of unseen
            events.  Defaults to 0.01.  If 1 is specified, you're getting Laplace
            smoothing, anything else is Lidstone.  It is a good idea to tune this
            parameter.
        :type alpha: float
        :param pad_ngrams: Whether to add additional padding to sentences when making
            ngrams in order to give more context to the documents.
        :type pad_ngrams: bool
        :param use_dictionary: If True, convert all inputs into lists of integers before
            training/predicting.  This can be particularly useful when training on a
            large number of documents.
        :type use_dictionary: bool
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
        self.use_dictionary = use_dictionary
        self.dictionary = TokenDictionary() if use_dictionary else None
        self.parallel = parallel
        # self.estimator = lambda freqdist, bins: LidstoneProbDist(freqdist, alpha, bins)
        # self.est = make_estimator(alpha)
        self.classes = [0, 1]

        self.x1 = None
        self.x2 = None
        self.m1 = None
        self.m2 = None
        self.n_y1 = 0
        self.n_y2 = 0
        self.y1_prob = 0
        self.y2_prob = 0
        self.y_probs = np.zeros(2)
        self.y_log_probs = np.zeros(2)
        self.y_ratio = 0
        self.y_log_ratio = 0

    @staticmethod
    def _make_setimator(alpha):
        def est(fdist, bins):
            return LidstoneProbDist(fdist, alpha, bins)
        return est

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
        :param classes: A list of classes to fit against
        :type classes: list
        :returns: self
        """
        # Check y
        y = np.asarray(y)

        # Check classes
        # if classes is None:
        #     raise ValueError("No classes specified for NGram Classifier.  This class "
        #                      "requires exactly two classes to be specified.")
        # FIXME: y can't be a generator...
        # NB: unique sorts the output (but that's a good thing)
        uniqs = np.unique(y, return_counts=True)
        self.classes = uniqs[0]
        self.n_y1 = uniqs[1][0]
        try:
            self.n_y2 = uniqs[1][1]
        except IndexError:
            pass
        if self.classes.shape[0] > 2:
            raise ValueError("Number of classes not equal to two. "
                             "NGramClassifier is a binary classifier and requires exactly "
                             "two classes to be specified. %s" % classes)


        # # Get class distribution
        self.y1_prob = np.float64(self.n_y1) / np.float64(self.n_y1 + self.n_y2)
        self.y2_prob = np.float64(self.n_y2) / np.float64(self.n_y1 + self.n_y2)
        self.y_probs = np.asarray([self.y1_prob, self.y2_prob], dtype=np.float64)
        self.y_log_probs = np.log2(self.y_probs)
        self.y_ratio = self.y1_prob / self.y2_prob
        self.y_log_ratio = np.log2(self.y_ratio)


        # Build models
        self.m1 = NgramModel(self.n,
                             pad_left=self.pad_ngrams, pad_right=self.pad_ngrams,
                             estimator=self._make_setimator(self.alpha))
        self.m2 = NgramModel(self.n,
                             pad_left=self.pad_ngrams, pad_right=self.pad_ngrams,
                             estimator=self._make_setimator(self.alpha))

        # An ngram model should be able to train on very large datasets, too large for RAM.
        # In order to accommodate this, both models are trained in batches so that a shuffled
        # one-use generator can be used for training data without having to separate out all
        # examples of each class before training.
        # Most ML models behave differently based on the order of the training data, ngram
        # models don't care.
        # FIXME: Add param and switch for multi-threaded
        # FIXME: Don't let queues fill with docs because that will defeat the purpose of using a generator

        if self.parallel:
            self._train_parallel(X, y)
        else:
            self._train_sync(X, y)

        # print "Finished training"

        # Set up models
        # self.m1 = self.workers[0].model
        # self.m2 = self.workers[1].model
        # FIXME: don't let that error through
        try:
            self.m1._build_model(self._make_setimator(self.alpha), {})
            self.m2._build_model(self._make_setimator(self.alpha), {})
        except RuntimeError:
            pass

        # print "Built models"

        return self

    def _train_parallel(self, X, y):
        models = [self.m1, self.m2]

        # Create workers
        queues = [mp.JoinableQueue(), mp.JoinableQueue()]
        result_queues = [mp.JoinableQueue(), mp.JoinableQueue()]
        workers = [Worker(queues[i], result_queues[i], models[i]) for i in (0, 1)]
        workers[0].start()
        workers[1].start()

        for i, doc in enumerate(X):
            if y[i] == self.classes[0]:
                queues[0].put(doc)
            else:
                queues[1].put(doc)
        # Poison pills
        for q in queues:
            q.put(None)

        # Wait for training to finish
        for q in queues:
            q.join()

        # Get the models
        self.m1 = result_queues[0].get()
        self.m2 = result_queues[1].get()

    def _train_sync(self, X, y):
        """
        Synchronous training for models
        """
        b1 = []
        b2 = []
        y_itr = iter(y)
        batch_size = 1000

        for chunk in utils.chunked(X, batch_size):
            for d in chunk:
                lbl = y_itr.next()
                if self.use_dictionary:
                    d = self.dictionary[d]

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
        r = _calc_prob_ratio(sequence, self.m1, self.m2, self.y1_prob, self.y2_prob, self.y_log_ratio)

        # FIXME: Tie breaker?
        # I handle this in calculating the ratio, but it would be good to be more
        # explicit about it.  And it only handles the case where both are 0

        # NB: The ratio is calculated in log-space
        # so the threshold for decision-making is 0, not 1
        if r > 0:
            return self.classes[0]
        else:
            return self.classes[1]

    # def _calc_prob_ratio(self, sequence):
    #     """Calculates the ratio of the two class probabilities"""
    #     # Get the negative log probabilities
    #     # NB: ngram model returns negative log probability so need to
    #     # make negative to get the actual log probability
    #     p1 = np.float128(-self.m1.prob_seq(sequence))
    #     p2 = np.float128(-self.m2.prob_seq(sequence))
    #
    #     # Handle 0 for both
    #     # This is actually the tie-breaking rule...
    #     # TODO: Return 0 for ties and handle tie-breaking in get_prediction
    #     if p1 == p2 == 0:
    #         # FIXME: Refactor this
    #         # Return a random value based on training data frequency
    #         choice = np.random.choice(self.classes, 1, p=[self.y1_prob, self.y2_prob])[0]
    #         if choice == self.classes[0]:
    #             return 2.0
    #         else:
    #             return -0.1
    #     # Handle division by zero
    #     # FIXME: Do I still need this in log-space?
    #     # No, but it's nice to have
    #     if p2 == 0:
    #         # If only the second class has zero probability, return a ratio
    #         # value greater than 0 so the first class is picked
    #         return 2.0
    #
    #     # Calculate the ratio of the probability that the sequence has
    #     # class one given the sequence, to the probability of class2 given
    #     # the sequence.
    #     # Calculate the ratio using log rules:
    #     # r = (p1/p2) * (py1/py2) becomes this in log space:
    #     # log(r) = log((p1*py1)/(p2*py2))
    #     # log(r) = (log(p1) - log(p2)) + (log(y1) - log(y2))
    #     return (p1 - p2) + self.y_log_ratio
    #
    #     # Calculate the ratio of the probability that the sequence has
    #     # class one given the sequence, to the probability of class2 given
    #     # the sequence
    #     # p1 = np.exp2(-self.m1.prob_seq(sq))
    #     # p2 = np.exp2(-self.m2.prob_seq(sq))
    #     # return (p1/p2) * self.y_ratio

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
        Return value has shape (n_samples, 2)
        """
        # NB: I tried multiple normalizations for c_probs:
        # - raw, no normalization
        # - Forcing sum to one by dividing by sum
        # - Setting chosen prob to 1 by dividing by max
        # Forcing sum to 1 performed terribly in evaluation.
        # Setting the chosen prob := 1 scored marginally (~0.01 weighted f1)
        # better, than raw scores, but as the purpose is to break ties, it
        # seemed a bit silly to just give 1 as the predicted probability
        # so I just stuck with the raw scores.
        # return np.asarray([self._get_probs(s) * self.y_probs for s in X])
        return np.exp2(self.predict_log_proba(X))
        # probs = self.predict_log_proba(X)
        # Normalize by (dividing by) sum of probabilities
        # return np.exp2(probs - np.atleast_2d(logsumexp2(probs, axis=1)).T)

    # TODO: Don't think I need this any more
    # def _get_probs(self, seq):
    #     p1 = self.m1.prob_seq(seq)
    #     p2 = self.m2.prob_seq(seq)
    #     # Move out of log space
    #     p_arr = np.exp2(np.array([-p1, -p2], dtype=np.float128))
    #     return p_arr

    def _get_log_probs(self, seq):
        """Get the negative log probabilities for a sequence"""
        # NB: NgramModel returns negative log probs
        return np.negative(np.asarray([self.m1.prob_seq(seq), self.m2.prob_seq(seq)],
                                      dtype=np.float128))

    def predict_log_proba(self, X):
        """Returns the log probabilities of the samples.
        Should return an array of shape: (n_samples, 2)
        """
        # Use log properties to multiply the two probabilities in log-space
        # FIXME: The mathematical formulation calls for weighting by class
        # probabilities, but that doesn't seem to have any meaningful impact on the
        # results.
        return np.asarray([self._get_log_probs(s) + self.y_log_probs for s in X],
                          dtype=np.float128)
        # return np.asarray([self._get_log_probs(s) for s in X],
        #                   dtype=np.float128)


def _fit_one(estimator, X, y, cls):
    """Fit a single NgramModel on a given class"""
    X = X[y == cls]
    y = y[y == cls]
    estimator = clone(estimator)
    estimator.fit(X, y)
    return estimator


class NgramClassifierMulti(OneVsOneClassifier):
    """
    Multi-class classifier using an n-gram language model as
    an estimator and an One vs One approach for multi-class classification"""

    def __init__(self, n=4, alpha=0.01, pad_ngrams=False, use_dictionary=False, n_jobs=1):
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
        self.use_dictionary = use_dictionary
        self.n_jobs = n_jobs
        self.classes_ = []
        self.class_probs = []
        self.estimators_ = []

        super(NgramClassifierMulti, self).__init__(NgramClassifier(n=n,
                                                                   alpha=alpha,
                                                                   pad_ngrams=pad_ngrams,
                                                                   use_dictionary=use_dictionary),
                                                   n_jobs=n_jobs)

    def fit(self, X, y):
        """Fit underlying estimators.
        Modified from the method in `OneVsOne` in order to account for the
        nature of the N-gram model.  Because a ngram model is only really
        trained on a single class, there's no real need to train a bunch
        of classifiers on each pairing of classes.  Instead one model is
        created for each class and their outputs are combined in order to
        make predictions.

        Parameters
        ----------
        X : array-like, shape = [n_samples, unk]
            Every sample should be a list of strings (or integers
            representing tokens).

        y : array-like, shape = [n_samples]
            Multi-class targets.

        Returns
        -------
        self
        """
        # FIXME: should work for generators...
        y = np.asarray(y)
        check_consistent_length(X, y)

        uniqs = np.unique(y, return_counts=True)
        self.classes_ = uniqs[0]
        class_counts = uniqs[1]
        self.overall_class_probs = class_counts / np.sum(class_counts)

        pair_class_probs = []
        for i in xrange(self.n_classes_):
            for j in xrange(i+1, self.n_classes_):
                sm = class_counts[i] + class_counts[j]
                y1_prob = class_counts[i]/sm
                y2_prob = class_counts[j]/sm
                pair_class_probs.append([y1_prob, y2_prob])
        self.pair_class_probs = pair_class_probs


        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one)(self.estimator, X, y, c)
            for c in self.classes_
        )
        return self

    # TODO: This could probably be supported
    def partial_fit(self, X, y, classes=None):
        raise NotImplementedError("Partial fit is not implemented for NgramClassifierMulti")


    def predict(self, X):
        """
        Estimate the best class label for each sample in X.
        Parameters
        ----------
        X : array-like, shape = [n_samples, unk]
            Every sample should be a list of strings (or integers
            representing tokens).

        Returns
        -------
        y : numpy array of shape [n_samples]
            Predicted multi-class targets.
        """
        out = []
        for s in X:
            # I shouldn't actually need to move out of log-space here...
            probs = np.asarray([-e.m1.prob_seq(s) for e in self.estimators_], dtype=np.float128)
            # probs = probs + np.log2(self.class_probs)
            out.append(self.classes_[probs.argmax()])
        return np.asarray(out)

        #out = []
#        # Now with voting
#
#        # For every pairing of models,
#        for y_idx, i in enumerate(xrange(self.n_classes_)):
#            for j in xrange(i+1, self.n_classes_):
#                m1 = self.estimators_[i].m1
#                m2 = self.estimators_[j].m1
#                y1_prob = self.pair_class_probs[y_idx][0]
#                y2_prob = self.pair_class_probs[y_idx][1]
#                # calc class log prob ratio
#                y_log_ratio = np.log2(y1_prob/y2_prob)
#
#                predictions = []
#                # For every sequence:
#                for s in X:
#                    # calc r
#                    r = _calc_prob_ratio(s, m1, m2, y1_prob, y2_prob, y_log_ratio)
#
#                    # Add prediction
#                    if r > 0:
#                        predictions.append(self.classes_[0])
#                    else:
#                        predictions.append(self.classes_[1])
#
#                out.append(predictions)
#        # Out should become (n_samples, n_classifiers)
#        out = np.vstack(out).T
#        return self.classes_[out.argmax(axis=1)]

    def predict_proba(self, X):
        """
        Returns probability estimates for each value in X.

        For now, just going with the average, though I'm not really convinced
        that's the best approach yet.

        X is an array with n-samples rows

        :returns: An array of shape (n_samples, n_classes)
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.
        """
        # probs = self._get_probs(X)
        # return (probs.sum(0) / (len(self.classes_) - 1)).T
        return np.exp2(self.predict_log_proba(X))

    def predict_log_proba(self, X):
        probs = self._get_probs(X)
        # Calculate the average while in logspace by dividing (subtracting)
        # the number of non-zero values in each column
        return (logsumexp2(probs, axis=0) - self.n_classes_).T


    def _get_probs(self, X):
        """
        :param X:
        :return: array (n_estimators, n_classes, n_estimators)
        """
        # Need to create an array (n_classes, n_samples)
        # Populate an array (n_estimators, n_classes, n_samples)
        # and taking the mean along axis 0
        # I do this by creating a list of tuples where each tuple corresponds to
        # a result from model.predict_proba and the values in the tuple are the
        # columns those predictions should go in the probs array.
        # You have to think about it in 3D fo it to really make any sense, but
        # the idea is to take this list of 2D arrays and put each column from those
        # 2D arrays into the right "stack" (depth column) of the probs array.
        n_estimators = len(self.estimators_)
        n_classes = len(self.classes_)
        idxs = [(i, j) for i in range(n_classes) for j in range(i + 1, n_classes)]
        confidences = [est.predict_log_proba(X) for est in self.estimators_]

        # FIXME: use scipy sparse?
        probs = np.zeros((n_estimators, n_classes, X.shape[0]), dtype=np.float128)

        # Populate the probs array
        for i, c in enumerate(confidences):
            tup = idxs[i]
            for j, col in enumerate(tup):
                probs[i, col, :] = c[:, j]

        return probs



# Borrowed from sklearn so I could change the exponent base
# and fix issues with sparse matrices
# TODO: switch everything to base e and use scipy logsumexp
def logsumexp2(arr, axis=0):
    """Computes the sum of arr assuming arr is in the log domain.

    Returns log(sum(exp(arr))) while minimizing the possibility of
    over/underflow.


    Examples
    --------

    >>> import numpy as np
    >>> from sklearn.utils.extmath import logsumexp
    >>> a = np.arange(10)
    >>> np.log(np.sum(np.exp(a)))
    9.4586297444267107
    >>> logsumexp(a)
    9.4586297444267107
    """
    arr = np.rollaxis(arr, axis)
    # Use the max to normalize, as with the log this is what accumulates
    # the less errors
    vmax = arr.max(axis=0)
    # I have no idea what the vmax really does, but I do know that when 
    # the input matrix is sparse, this will mess up because log(0) = 1 and 
    # all of those ones will be included in the sum.
    e_arr = np.exp2(arr - vmax)
    # Set ones back to 0 for the sum
    e_arr[arr == 0.0] = 0
    # Finish the calculation
    out = np.log2(np.sum(e_arr, axis=0))
    # out = np.log2(np.sum(np.exp2(arr - vmax), axis=0))
    out += vmax
    return out


if __name__ == '__main__':
    # DOC1 = u"I'm starting to know what God felt like when he sat out " \
    #        "there in the darkness, creating the world."
    # DOC2 = u"And what did he feel like, Lloyd, my dear?"
    # DOC3 = u"Very pleased he'd taken his Valium."
    # data = np.asarray([datum.split() for datum in (DOC1, DOC2, DOC3)])
    # labels = [1, 0, 2]

    import cPickle as pickle

    with open('candidate_classifier/data/processed/pre-processed.pkl', 'rb') as _f:
        preprocessed = pickle.load(_f)

    docs = preprocessed['int_docs']
    labels = preprocessed['labels']

    del preprocessed
    del _f
    candidates = sorted(list(set(labels)))


    ngm = NgramClassifierMulti()
    ngm.fit(np.asarray(docs[:1000]), labels[:1000])
    ngm.predict(np.asarray(docs[1000:1010]))
