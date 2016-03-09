#! /usr/bin/env python2

import os
import ujson
from nltk.probability import LidstoneProbDist
from nltk_model import NgramModel


__author__ = 'Eric Lind'


PROCESSED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'data', 'processed', 'processed.json')
EST = lambda freqdist, bins: LidstoneProbDist(freqdist, 0.2, bins)


def get_models():
    with open(PROCESSED_PATH, 'rb') as _f:
        processed = ujson.load(_f)

    models = {name: NgramModel(4, data['sents'], estimator=EST, bins=data['num_word_types'])
              for name, data in processed.iteritems() if data['num_word_types'] > 0}

    return models
