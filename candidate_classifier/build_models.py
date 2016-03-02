#! /usr/bin/env python2

from nltk.probability import LidstoneProbDist
import os
import ujson

from nltk_model import NgramModel
# from candidate_classifier.debate_corpus_reader import DebateCorpusReader


__author__ = 'Eric Lind'


PROCESSED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'data', 'processed', 'processed.json')
# CORPUS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
# FILE_PATTERN = '.*\.txt'
# CANDIDATES = ['BUSH', 'CARSON', 'CHRISTIE', 'CRUZ', 'FIORINA', 'KASICH' , 'PAUL', 'RUBIO', 'TRUMP',
#               'CLINTON', 'SANDERS']
# CORPUS = DebateCorpusReader(CORPUS_ROOT, FILE_PATTERN)


EST = lambda freqdist, bins: LidstoneProbDist(freqdist, 0.2, bins)

def get_models():
    # grouped_sents = CORPUS.grouped_sents(speakers=CANDIDATES)

    with open(PROCESSED_PATH, 'rb') as _f:
        processed = ujson.load(_f)

    models = {name: NgramModel(4, data['sents'], estimator=EST, bins=data['num_word_types'])
              for name, data in processed.iteritems() if data['num_word_types'] > 0}

    return models
