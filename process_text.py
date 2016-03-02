#! /usr/bin/env python2
"""
Script for processing all debate text into stored JSON
"""

from spacy.en import English
import ujson
import os

from candidate_classifier.string_processing import *
from candidate_classifier.debate_corpus_reader import DebateCorpusReader
from candidate_classifier.utils import flatten


__author__ = 'Eric Lind'


CORPUS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'candidate_classifier', 'data', 'raw')
PROCESSED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'candidate_classifier', 'data', 'processed', 'processed.json')

FILE_PATTERN = '.*\.txt'
CANDIDATES = ['BUSH', 'CARSON', 'CHRISTIE', 'CRUZ', 'FIORINA', 'KASICH' , 'PAUL', 'RUBIO', 'TRUMP',
              'CLINTON', 'SANDERS']
NLP = English(entity=False, tagger=False, load_vectors=False)

BRACKET_PATTERN = re.compile(r"\[[a-zA-Z ]*\]", re.U)
SPACED_ELLIPSIS_PATTERN = re.compile(r"((?:\.\s){3})")
MULTI_ELLIPSIS_PATTERN = re.compile(r"(?:(?:\.){3} ?)+")
ENDS_WITH_ELLIPSIS = lambda s: s[-3:] == '...'
STARTS_WITH_ELLIPSIS = lambda s: s[:3] == '...'
STARTS_WITH_DASH = lambda s: s[0] == '-'
ENDS_WITH_DASH = lambda s: s[-1] == '-'


class TransformerWrapper(object):
    def __init__(self, transformer):
        self.transformer = transformer

    def tokenize(self, s):
        return self.transformer(s)


def sent_tokenizer(s):
    doc = NLP(s)
    return [u''.join(t.text_with_ws for t in sent) for sent in doc.sents]


def word_tokenizer(s):
    toks = NLP(s)
    return ['<S>'] + [t.lower_ for t in toks] + ['</S>']


DOC_TRANSFORMER = TransformerABC(
    prefilter_substitutions=[BRACKET_PATTERN,
                             (SPACED_ELLIPSIS_PATTERN, '...'),
                             (MULTI_ELLIPSIS_PATTERN, '...'),
                             'whitespace',
                             'strip',
                             'deaccent'],
    tokenizer=sent_tokenizer)


SENT_TRANSFORMER = TransformerABC(
    prefilter_substitutions=['strip'],
    filters=[STARTS_WITH_ELLIPSIS, ENDS_WITH_ELLIPSIS, STARTS_WITH_DASH, ENDS_WITH_DASH],
    tokenizer=word_tokenizer)



def main():
    # Load Corpus
    corpus = DebateCorpusReader(CORPUS_ROOT,
                                FILE_PATTERN,
                                sent_tokenizer=TransformerWrapper(DOC_TRANSFORMER),
                                word_tokenizer=TransformerWrapper(SENT_TRANSFORMER))

    grouped_sents = corpus.grouped_sents(speakers=CANDIDATES)

    # FIXME: Can the JSON serializer do this in a streaming fashion?
    # Expand everything into RAM
    processed = {name: {'num_word_types': len(set(flatten(sents))),
                        'sents': sents}
                 for name, sents in grouped_sents.iteritems()}

    # Dump to file
    with open(PROCESSED_PATH, 'wb') as _f:
        ujson.dump(processed, _f)

    print "DONE!"


if __name__ == '__main__':
    main()
