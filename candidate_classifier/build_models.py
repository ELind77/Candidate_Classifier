#! /usr/bin/env python2

from nltk_model import NgramModel
from nltk.probability import LaplaceProbDist, LidstoneProbDist
from nltk.corpus import PlaintextCorpusReader
import os

CORPUS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

__author__ = 'Eric Lind'


