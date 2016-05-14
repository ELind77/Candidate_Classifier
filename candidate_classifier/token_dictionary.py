#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from collections import Iterable


__author__ = 'Eric Lind'


class TokenDictionary(object):
    """A dictionary class to map strings/tokens to integer ids.

    >>> from candidate_classifier.nltk_model.ngram_classifier import TokenDictionary
    >>> d = TokenDictionary()
    >>> s = "Anything you can do I can do better.".split()
    >>> d[s]
    [0, 1, 2, 3, 4, 2, 3, 5]
    """

    def __init__(self):
        self.d = dict()

    def __getitem__(self, item):
        if not isinstance(item, Iterable):
            raise ValueError("Non-iterable given/asked of Dictaionary.  "
                             "Are you sure you tokenized the string?")

        out = []
        for i in item:
            if i in self.d:
                idx = self.d[i]
            else:
                idx = len(self.d)
                self.d[i] = idx
            out.append(idx)
        return out
