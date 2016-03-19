#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Set of functions to perform different kinds of tokenization.

Eventually this should get rolled into a kind of factory like
string_processing but for now it's just a collection of functions.
"""


from spacy.en import English
from candidate_classifier.sn_grams import sn_grams


__author__ = 'Eric Lind'


# TODO:
# - Spacy tokenization, no lemmatization
# - replace candidate and person names with PERSON
# - substitute numbers
# - Use word shapes
# - Experiment with replacing OOV words with brown cluster
# - WordNet WSD tokenizer
#   - That's kinda mor like a transformer though.


nlp = English(load_vectors=False)


# =================
# Wrappers
# =================

class TokenizerWrapper(object):
    """Class wrapper for tokenizer functions so that they can
    be used with nltk CorpusReaders that require a class with
    a ``tokenize`` method."""
    def __init__(self, transformer):
        self.transformer = transformer

    def tokenize(self, s):
        return self.transformer(s)


class DummyTokenizer(object):
    """Dummy tokenizer for nltk CorpusReaders that require a class
    with a ``tokenize`` method."""
    @staticmethod
    def tokenize(s):
        return s



# =================
# Helpers
# =================

def merge_np(doc):
    for np in doc.noun_chunks:
        while len(np) > 1 and np[0].dep_ not in ('avmod', 'amod', 'compound'):
            np = np[1:]
        np.merge(np.root.tag_, np.text, np.root.ent_type_)
    return doc

def merge_ent(doc):
    for ent in doc.ents:
        # In the sense2vec code they do something a bit
        # different with the entity label and I'm not sure why
        ent.merge(ent.root.tag_, ent.text, ent.label_)
    return doc

def get_brown(doc):
    out = []
    for t in doc:
        if not t.is_space:
            # Should use tok.ent_iob_
            if t.cluster != 0 and t.tag_ != 'NNP':
                out.append("**%s**" % t.cluster)
            else:
                out.append(t.lemma_)
    return out

def get_sn_grams(doc, prop, spanning=False):
    result = []
    for sent in doc.sents:
        result.extend(sn_grams(sent, 1, 3, prop=prop, spanning=spanning))
    return result

def filter_features(toks, attrs):
    return [t for t in toks if not any([getattr(t, a) for a in attrs])]



# =================
# Tokenizers
# =================

def simple_tokenizer(s):
    return s.split()

def lemmas_tokenizer(s):
    return [t.lemma_ for t in nlp(s) if not t.is_space]

def lemmas_no_punct(s):
    return [t.lemma_ for t in nlp(s) if not any([t.is_punct, t.is_space])]

def lemmas_no_punt_no_num(s):
    return [t.lemma_ for t in nlp(s)
            if not any([t.is_punct, t.is_space, t.is_digit, t.is_like_num])]

def lemmas_sub_nums(s):
    raise NotImplementedError

def lemmas_no_punct_sub_nums(s):
    raise NotImplementedError

def lemmas_merge_np(s):
    toks = merge_np(nlp(s))
    return [t.lemma_ for t in toks if not t.is_space]

def lemmas_merge_ents(s):
    toks = merge_ent(nlp(s))
    return [t.lemma_ for t in toks if not t.is_space]

def lemmas_merge_np_merge_ents(s):
    toks = merge_ent(nlp(s))
    toks = merge_np(toks)
    return [t.lemma_ for t in toks if not t.is_space]

def lemmas_cased_tokenizer(s):
    """Tokenizer that lematizes but preserves the case of the first letter of tokens"""
    toks = nlp(s)
    return [t.lemma_.title() if t.is_title else t.lemma_ for t in toks]

def lemmas_cased_merge_ents(s):
    toks = merge_ent(nlp(s))
    return [t.lemma_.title() if t.is_title else t.lemma_ for t in toks]


# So these were basically useless.
# I'm pretty sure I'm using them wrong...
def brown_cluster_tokenizer(s):
    toks = nlp(s)
    return get_brown(toks)

def brown_cluster_merge_ents(s):
    toks = merge_ent(nlp(s))
    return get_brown(toks)


# Sn-gram tokenizers
def sn_tokenizer(s):
    return get_sn_grams(nlp(s), 'lemma_')

def sn_pos_tokenizer(s):
    return get_sn_grams(nlp(s), 'tag_')

def sn_merge_ents(s):
    doc = merge_ent(nlp(s))
    return get_sn_grams(doc, 'lemma_')

def sn_pos_merge_ent(s):
    doc = merge_ent(nlp(s))
    return get_sn_grams(doc, 'tag_')

def sn_merge_np(s):
    doc = merge_np(nlp(s))
    return get_sn_grams(doc, 'lemma_')

def sn_sr_tokenizer(s):
    return get_sn_grams(nlp(s), 'dep_')

def sn_sr_merge_ent(s):
    doc = merge_ent(nlp(s))
    return get_sn_grams(doc, 'dep_')

def sn_spanning_tokenizer(s):
    return get_sn_grams(nlp(s), 'lemma_', spanning=True)

def sn_spanning_merge_ents(s):
    doc = merge_ent(nlp(s))
    return get_sn_grams(doc, 'lemma_', spanning=True)
