#! /usr/bin/env python2

from nltk.probability import LidstoneProbDist
import os

from nltk_model import NgramModel
from candidate_classifier.debate_corpus_reader import DebateCorpusReader


__author__ = 'Eric Lind'


CORPUS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
FILE_PATTERN = '.*\.txt'
CANDIDATES = ['BUSH', 'CARSON', 'CHRISTIE', 'CRUZ', 'FIORINA', 'KASICH' , 'PAUL', 'RUBIO', 'TRUMP',
              'CLINTON', 'SANDERS']
CORPUS = DebateCorpusReader(CORPUS_ROOT, FILE_PATTERN)

EST = lambda freqdist, bins: LidstoneProbDist(freqdist, 0.2, bins)


def get_models():
    grouped_sents = CORPUS.grouped_sents(speakers=CANDIDATES)

    # FIXME: Get num word types
    models = {name: NgramModel(4, sents, estimator=EST, bins=5000)
              for name, sents in grouped_sents.iteritems()}

    return models





