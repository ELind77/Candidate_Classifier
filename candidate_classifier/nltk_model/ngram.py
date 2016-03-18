# Natural Language Toolkit: Language Models
#
# Copyright (C) 2001-2013 NLTK Project
# Authors: Steven Bird <stevenbird1@gmail.com>
#          Daniel Blanchard <dblanchard@ets.org>
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT

# This is copied from nltk commit: c54edec
# on the "model" branch.


# SimpleGoodTuring probdist just doesn't work at all with this model:
# https://github.com/nltk/nltk/issues/602

# So far, only lidstone and laplace work.  WrittenBell might work but I haven't
# tested yet and it's going to be a pain to test by hand.


from __future__ import unicode_literals

from math import log
import warnings
from collections import Sequence, Iterator
import itertools
from types import *

from nltk.probability import ConditionalProbDist, ConditionalFreqDist, LidstoneProbDist
from nltk.util import ngrams
from nltk import compat
from candidate_classifier.nltk_model.api import ModelI


def _estimator(fdist, **estimator_kwargs):
    """
    Default estimator function using a LidstoneProbDist.
    """
    # can't be an instance method of NgramModel as they
    # can't be pickled either.
    return LidstoneProbDist(fdist, 0.001, **estimator_kwargs)


@compat.python_2_unicode_compatible
class NgramModel(ModelI):
    """
    A processing interface for assigning a probability to the next word.
    """

    # add cutoff
    def __init__(self,
                 n,
                 docs=(),
                 pad_left=False,
                 pad_right=False,
                 estimator=_estimator,
                 **estimator_kwargs):
        """
        Create an ngram language model to capture patterns in n consecutive
        words of training text.  An estimator smooths the probabilities derived
        from the text and may allow generation of ngrams not seen during
        training. See model.doctest for more detailed testing

            >>> from nltk.corpus import brown
            >>> lm = NgramModel(3, brown.words(categories='news'))
            >>> lm
            <NgramModel with 91603 3-grams>
            >>> lm._backoff
            <NgramModel with 62888 2-grams>
            >>> lm.entropy(brown.words(categories='humor'))
            ... # doctest: +ELLIPSIS
            12.0399...

        NB: If a ``bins`` parameter is given in teh ``estimator_kwargs``
        it will be ignored.  The number of bins to use is the number of
        outcomes (tokens) encountered at each level of the backoff recursion
        and as such, the number must change each time.

        :param n: the order of the language model (ngram size)
        :type n: int
        :param docs: the training text
        :type docs: list(str) or list(list(str))
        :param pad_left: whether to pad the left of each sentence with an (n-1)-gram of empty strings
        :type pad_left: bool
        :param pad_right: whether to pad the right of each sentence with an (n-1)-gram of empty strings
        :type pad_right: bool
        :param estimator: a function for generating a probability distribution
        :type estimator: a function that takes a ConditionalFreqDist and
            returns a ConditionalProbDist
        :param estimator_kwargs: Extra keyword arguments for the estimator
        :type estimator_kwargs: (any)
        """
        super(NgramModel, self).__init__()

        # protection from cryptic behavior for calling programs
        # that use the pre-2.0.2 interface
        assert(isinstance(pad_left, bool))
        assert(isinstance(pad_right, bool))

        # Check for bins argument
        if 'bins' in estimator_kwargs:
            warnings.warn('A value was provided for the `bins` parameter of '
                          '`estimator_kwargs`.  This value will be overridden.'
                          'If you think you have a better idea, write your own '
                          'darn model.')
            # Clear out the bins so we don't throw recursive warnings
            estimator_kwargs.pop('bins', None)

        self._lpad = ('',) * (n - 1) if pad_left else ()
        self._rpad = ('',) * (n - 1) if pad_right else ()
        self._pad_left = pad_left
        self._pad_right = pad_right

        # make sure n is greater than zero, otherwise print it
        assert (n > 0), n
        self._unigram_model = (n == 1)
        self._n = n

        # Declare all other fields
        self._backoff = None
        if not self._unigram_model:
            # FIXME: estimator_kwargs
            self._backoff = NgramModel(n - 1,
                                       [],
                                       pad_left, pad_right,
                                       estimator,
                                       **estimator_kwargs)
        self._backoff_alphas = None
        self._model = None

        # Process training
        self._ngrams = set()
        self.outcomes = set()
        self._cfd = ConditionalFreqDist()


        # ===================
        # Check Docs
        # ===================
        # FIXME: More robust check?
        # I think it's important that the model be able to train on a one-use generator
        # so that it can train on corpora that don't fit in RAM. This requires some robust
        # type-checking though.  What's below could use some improvement, but seems to work
        # for now.

        # Docs need to be able to be a list, tuple, or generator,
        # or an iterable that yields such (CorpusView?)
        # If given a list of strings instead of a list of lists, create enclosing list

        # NB: The Iterator type won't catch lists, or strings, but it will catch things returned
        # by functions in itertools
        if isinstance(docs, GeneratorType) or isinstance(docs, Iterator):
            nxt = docs.next()
            # Either it's a string or a list of string
            if isinstance(nxt, basestring):
                docs = [itertools.chain([nxt], docs)]
            elif isinstance(nxt, Sequence):
                # It should be a list of string...
                # FIXME: Handle generator here as well
                if isinstance(nxt[0], basestring):
                    # So docs is a generator that yields sequences of str
                    docs = itertools.chain([nxt], docs)
            else:
                raise TypeError("Training documents given to NgramModel are a generator "
                                "that yields something other than a string or a list of "
                                "string.  %s" % docs)
        # could also just be a sting
        elif isinstance(docs, basestring):
            raise TypeError("Training documents given to NgramModel must be either a list "
                            "of string or a list of lists of string.  Or a generator that "
                            "acts in the same way as one of the above.  A string was found "
                            "instead: %s" % docs)
        elif isinstance(docs, Sequence):
            # It's some kind of iterable with a __getitem__, not a generator
            # If it's empty, assume training will happen later
            if len(docs) == 0:
                pass
            elif isinstance(docs[0], basestring):
                # Make it into a list of lists
                docs = [docs]
            elif isinstance(docs[0], Sequence):
                # Check inner to make sure it's a string
                if not isinstance(docs[0][0], basestring):
                    raise TypeError("Training documents given to NgramModel were neither a "
                                    "list of string nor a list of list of string: %s" % docs)
                # If it is a string everything is fine, nothing to worry about
        else:
            raise TypeError("Unsupported type supplied to NgramModel for training documents: %s" %
                            docs)


        # Train the model
        for sent in docs:
            self._train(sent)

        # Build model and set the backoff parameters
        if len(self.outcomes) > 0:
            self._build_model(estimator, estimator_kwargs)



    # ===================
    # TRAINING
    # ===================

    # At every stage, in the backoff/recursion the number of bins for
    # the estimator should be equal to the total number of outcomes
    # (tokens) encountered while training.  This means that it needs
    # to be recalculated at each level of the recursion.
    # NB: For the unigram case, this would be the actual vocabulary size
    def _train(self, sent):
        # FIXME: This may use extra memory, but because python 2.7 doesn't
        # support deepcopy for generators, I'm not sure what else to do...
        if isinstance(sent, GeneratorType) or isinstance(sent, Iterator):
            s1, s2 = itertools.tee(sent, 2)
            self._train_one(s1)
            if self._backoff is not None:
                self._backoff._train(s2)
        else:
            self._train_one(sent)
            if self._backoff is not None:
                self._backoff._train(sent)


    # FIXME: Discard cfd after training?
    # Should check if the probdist keeps a reference to it
    def _train_one(self, sent):
        """Train the model on a sequence"""
        for ngram in ngrams(sent, self._n,
                            self._pad_left,
                            self._pad_right,
                            left_pad_symbol=self._lpad,
                            right_pad_symbol=self._rpad):
            self._ngrams.add(ngram)
            context = tuple(ngram[:-1])
            token = ngram[-1]
            self._cfd[context][token] += 1
            self.outcomes.add(token)


    # ===================
    # CREATE MODEL
    # ===================
    # Even if the number of bins is explicitly passed, we should use the number
    # of word types encountered during training as the bins value.
    # If right padding is on, this includes the padding symbol.
    #
    # NB: There is a good reason for this!  If the number of bins isn't set the
    # ConditionalProbDist will choose from a different total number of possible
    # outcomes for each condition and the NgramModel won't give probability
    # estimates that sum to 1.
    def _build_model(self, estimator, estimator_kwargs):
        n_outcomes = len(self.outcomes)
        if n_outcomes <= 0:
            raise Exception("NgramModel can't build a model without training input!")

        estimator_kwargs['bins'] = n_outcomes

        # Create the probability model
        self._model = ConditionalProbDist(self._cfd, estimator, **estimator_kwargs)

        # Clear out the bins so we don't throw recursive warnings
        estimator_kwargs.pop('bins', None)

        # Build backoff model and get backoff parameters
        if not self._unigram_model:
            self._backoff._build_model(estimator, estimator_kwargs)
            self._set_backoff_params()


    # ===================
    # SET BACKOFF PARAMS
    # ===================
    def _set_backoff_params(self):
        # Construct parameters for
        if not self._unigram_model:
            # self._backoff = NgramModel(n-1,
            #                            docs,
            #                            pad_left, pad_right,
            #                            estimator,
            #                            **estimator_kwargs)

            self._backoff_alphas = dict()
            # For each condition (or context)
            for ctxt in self._cfd.conditions():
                prdist = self._model[ctxt]  # prob dist for this context

                backoff_ctxt = ctxt[1:]
                backoff_total_pr = 0.0
                total_observed_pr = 0.0
                for word in self._cfd[ctxt]:
                    # This is the subset of words that we OBSERVED
                    # following this context
                    total_observed_pr += prdist.prob(word)
                    # We normalize it by the total (n-1)-gram probability of
                    # words that were observed in this n-gram context
                    backoff_total_pr += self._backoff.prob(word, backoff_ctxt)

                assert (0 < total_observed_pr <= 1), total_observed_pr
                # beta is the remaining probability weight after we factor out
                # the probability of observed words
                beta = 1.0 - total_observed_pr

                # backoff total has to be less than one, otherwise we get
                # ZeroDivision error when we try subtracting it from 1 below
                assert (0 < backoff_total_pr < 1), backoff_total_pr
                alpha_ctxt = beta / (1.0 - backoff_total_pr)

                self._backoff_alphas[ctxt] = alpha_ctxt


    # ==================
    # API Methods
    # ==================

    # This is a new method (not in original nltk model)
    def prob_seq(self, seq):
        """
        Evaluate the probability of a sequence (list of tokens).

        :param seq: A list of tokens representing a document/sentence/etc.
        :type seq: list(str)
        :param lg: If ``log`` is True, returns value in logspace to avoid underflows.
        :type lg: bool
        :return: float
        :rtype: float
        """
        prob = 0.0
        for ngram in ngrams(seq, self._n, self._pad_left, self._pad_right,
                            left_pad_symbol=self._lpad,
                            right_pad_symbol=self._rpad):
            context = tuple(ngram[:-1])
            token = ngram[-1]
            prob += self.logprob(token, context)
        return float(prob)

    def prob(self, word, context):
        """
        Evaluate the probability of this word in this context using Katz Backoff.

        :param word: the word to get the probability of
        :type word: str
        :param context: the context the word is in
        :type context: list(str)
        """
        context = tuple(context)
        if (context + (word,) in self._ngrams) or (self._unigram_model):
            return self[context].prob(word)
        else:
            return self._alpha(context) * self._backoff.prob(word, context[1:])

    # Updated _alpha function, discarded the _beta function
    def _alpha(self, context):
        """Get the backoff alpha value for the given context
        """
        error_message = "Alphas and backoff are not defined for unigram models"
        assert not self._unigram_model, error_message

        if context in self._backoff_alphas:
            return self._backoff_alphas[context]
        else:
            return 1

    def logprob(self, word, context):
        """
        Evaluate the (negative) log probability of this word in this context.

        :param word: the word to get the probability of
        :type word: str
        :param context: the context the word is in
        :type context: list(str)
        """
        return -log(self.prob(word, context), 2)

    def choose_random_word(self, context):
        """
        Randomly select a word that is likely to appear in this context.

        :param context: the context the word is in
        :type context: list(str)
        """
        return self.generate(1, context)[-1]

    # NB, this will always start with same word if the model
    # was trained on a single text
    def generate(self, num_words, context=()):
        """
        Generate random text based on the language model.

        :param num_words: number of words to generate
        :type num_words: int
        :param context: initial words in generated string
        :type context: list(str)
        """

        text = list(context)
        for i in range(num_words):
            text.append(self._generate_one(text))
        return text

    def _generate_one(self, context):
        context = (self._lpad + tuple(context))[-self._n + 1:]

        if context in self:
            return self[context].generate()
        elif self._n > 1:
            return self._backoff._generate_one(context[1:])
        else:
            return '.'

    def entropy(self, text):
        """
        Calculate the approximate cross-entropy of the n-gram model for a
        given evaluation text.
        This is the average log probability of each word in the text.

        :param text: words to use for evaluation
        :type text: list(str)
        """
        H = 0.0     # entropy is conventionally denoted by "H"
        text = list(self._lpad) + text + list(self._rpad)
        for i in xrange(self._n - 1, len(text)):
            context = tuple(text[(i - self._n + 1):i])
            token = text[i]
            H += self.logprob(token, context)
        return H / float(len(text) - (self._n - 1))

    def perplexity(self, text):
        """
        Calculates the perplexity of the given text.
        This is simply 2 ** cross-entropy for the text.

        :param text: words to calculate perplexity of
        :type text: list(str)
        """
        return pow(2.0, self.entropy(text))

    def __contains__(self, item):
        if not isinstance(item, tuple):
            item = (item,)
        return item in self._model

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            item = (item,)
        return self._model[item]

    def __repr__(self):
        return '<NgramModel with %d %d-grams>' % (len(self._ngrams), self._n)




if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
