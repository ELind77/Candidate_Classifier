#!/usr/bin/env python2

__author__ = 'Eric Lind'


from sklearn.cross_validation import cross_val_score
import numpy as np
import ujson
import os
from sklearn.cross_validation import KFold

from candidate_classifier.nltk_model.ngram import NgramModelClassifier

with open('candidate_classifier/data/processed/processed.json', 'rb') as _f:
    processed = ujson.load(_f)

trump_sents = processed['TRUMP']['sents']
trump_labels = [1]*len(trump_sents)
hillary_sents = processed['CLINTON']['sents']
hillary_labels = [0]*len(hillary_sents)

data = trump_sents+hillary_sents
labels = trump_labels + hillary_labels

classifier = NgramModelClassifier(threshold=0.75)
cross_val_score(classifier,
                data,
                y=labels,
                cv=KFold(len(data), n_folds=10, shuffle=True, random_state=1),
                scoring='f1')
