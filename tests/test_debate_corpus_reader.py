from nose import tools as nosey
from nose.tools import with_setup
import unittest
import os

from candidate_classifier.debate_corpus_reader import DebateCorpusReader
from nltk.corpus.reader.util import StreamBackedCorpusView

__author__ = 'Eric Lind'


ROOT = os.path.join('tests', 'data')
PATTERN = '.*\.txt'
TEST_FILE = os.path.join('tests', 'data', 'test_debate.txt')
DCR = DebateCorpusReader(ROOT, PATTERN)


def setup_func():
    print "Setting up"
    global DCR
    DCR = DebateCorpusReader(ROOT, PATTERN)

def teardown_func():
    print "tearing down"
    global DCR
    DCR = None


#
# Tests
#

@with_setup(setup_func, teardown_func)
def test_dcr_instantiates():
    nosey.assert_is_instance(DCR, DebateCorpusReader)


# Test contracts

@with_setup(setup_func, teardown_func)
def test_dcr_speakers_returns_list_of_string():
    speakers = DCR.speakers()
    nosey.assert_is_instance(speakers, list)
    for s in speakers:
        nosey.assert_is_instance(s, basestring)


@with_setup(setup_func, teardown_func)
def test_dcr_words_returns_corpusview_list_of_str():
    words = DCR.words()
    nosey.assert_is_instance(words, StreamBackedCorpusView)
    for w in words:
        nosey.assert_is_instance(w, basestring)

@with_setup(setup_func, teardown_func)
def test_dcr_sents_returns_cv_lolos():
    sents = DCR.sents()
    nosey.assert_is_instance(sents, StreamBackedCorpusView)
    for sent in sents:
        nosey.assert_is_instance(sent, list)
        for w in sent:
            nosey.assert_is_instance(w, basestring)

@with_setup(setup_func, teardown_func)
def test_dcr_utterances_returns_tokenized_tuples():
    uts = DCR.utterances()
    nosey.assert_is_instance(uts, StreamBackedCorpusView)
    for ut in uts:
        nosey.assert_is_instance(ut, tuple)
        nosey.assert_is_instance(ut[0], basestring)
        nosey.assert_is_instance(ut[1], list)
        for sent in ut[1]:
            nosey.assert_is_instance(sent, list)
            for w in sent:
                nosey.assert_is_instance(w, basestring)

@with_setup(setup_func, teardown_func)
def test_dcr_grouped_words_returns_dict_of_tokens():
    gws = DCR.grouped_words()
    nosey.assert_is_instance(gws, dict)
    for speaker, words in gws.iteritems():
        nosey.assert_is_instance(speaker, basestring)
        nosey.assert_is_instance(words, StreamBackedCorpusView)
        for w in words:
            nosey.assert_is_instance(w, basestring)

@with_setup(setup_func, teardown_func)
def test_dcr_grouped_sents_returns_dict_of_tokenized_sents():
    gsents = DCR.grouped_sents()
    nosey.assert_is_instance(gsents, dict)
    for speaker, sents in gsents.iteritems():
        nosey.assert_is_instance(speaker, basestring)
        nosey.assert_is_instance(sents, StreamBackedCorpusView)
        for sent in sents:
            nosey.assert_is_instance(sent, list)
            for w in sent:
                nosey.assert_is_instance(w, basestring)


# Test filter by speakers
@with_setup(setup_func, teardown_func)
def test_dcr_utterances_can_filter_by_speakers():
    uts = DCR.utterances(speakers='PAUL')
    uts = [u for u in uts]
    nosey.assert_equal(len(uts), 1)
    nosey.assert_equal(uts[0][0], 'PAUL')

# Test filter by speakers
@with_setup(setup_func, teardown_func)
def test_dcr_grouped_sents_can_filter_by_speaker():
    gsents = DCR.grouped_sents(speakers='PAUL')
    gsents = gsents.items()
    nosey.assert_equal(len(gsents), 1)
    nosey.assert_equal(gsents[0][0], 'PAUL')


# Test custom tokenizer
def test_dcr_can_use_custom_tokenizer():
    dcr = DebateCorpusReader(ROOT, PATTERN, word_tokenizer=''.split())
    nosey.assert_is_instance(dcr, DebateCorpusReader)

