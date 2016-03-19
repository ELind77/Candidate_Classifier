#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Functions for generating syntactic n-grams as described in:

Syntactic Dependency-based N-grams as Classification Features by
Grigori Sidorov, Francisco Velasquez, Efstathios Stamatatos,
Alexander Gelbukh, and Liliana Chanona-Hernández
www.cic.ipn.mx/~sidorov

The paper describes creating only those syntactic n-grams that
follow the arrows of the dependency parse.  However, I believe
that there is value in syntactic n-grams that span a syntactic
head, i.e. move up the tree and then back down.  This would
allow for n-grams that behave like syntactic subj-vrb-obj
triples.  When using a token property other than the text/lemma
(as described in the paper) this is unlikely to be particularly
useful as subj-vrb-obj occurs in most sentences.  But when using
the lemma or text of the tokens I think that these features will
be useful.

The functions provided here can behave in either way and can be
controlled using the ``spanning`` keyword argument.
"""


from collections import deque
import copy
import operator
from spacy.tokens.span import Span

__author__ = 'Eric Lind'


# TODO: Make spanning ngrams not be painfully slow



def sn_grams(sent, m, n, prop='lemma_', spanning=False):
    """
    Get all syntactic n-grams from min length m to max length n.
    If ``m`` < ``n`` sn-grams of lengths [m, n] (inclusive)
    will be generated.

    :param sent: A sentence Span object from spacy
    :type sent: Span
    :param m: The minimum n-gram size to generate.
    :type m: int
    :param n: The max n-gram size to generate
    :type n: int
    :param prop: The property of the tokens to build the n-grams
        from.  E.g. lemma_, text, tag.  (See http://spacy.io/docs
        for more info.)
    :type prop: str
    :param spanning: Whether to produce sn-grams that span syntactic
        heads.  Basically, should n-gram construction go both up and
        down the parse tree.
    :type spanning: bool
    """
    if spanning:
        root = sent.root
        return sn_spanning_helper([], deque([], n), root, m, prop=prop)
    else:
        root = sent.root
        return sn_helper([], deque([], n), root, m, prop=prop)

def sn_helper(acc, buff, curr, m, prop='lemma_'):
    # Add curr
    # But don't add ROOT because those relations occur in almost all sentences
    # NB: No need to pop b/c deque has max length specified
    if prop == 'dep_' and curr.dep_ == 'ROOT':
        pass
    else:
        buff.appendleft(getattr(curr, prop))

        # Create the ngrams that end with the current node
        # Having the n-grams in reverse lexical order is a bit odd
        # for humans to read, but because it's deterministic, it
        # shouldn't make any difference to the computer for using them as
        # features
    #     gram = []
    #     if len(buff) >= m:
    #         for i, tok in enumerate(buff):
    #             # Build up the ngram
    #             gram.append(tok)
    #             # Only append ngrams >= the min length
    #             if len(gram) >= m:
    #                 acc.append('_'.join(reversed(gram)))

    # Optimized (at least a bit)
    gram = ''
    count = 0
    if len(buff) >= m:
        for i, tok in enumerate(buff):
            # Build up the ngram
            gram += '_' + tok
            count += 1
            # Only append ngrams >= the min length
            if count >= m:
                acc.append(gram)

    # Add all childrens' ngrams
    for c in curr.children:
        # Add to acc with destructive modification
        # but don't let children modify buffer because it needs
        # to be in the same state for the next child of curr
        sn_helper(acc, copy.copy(buff), c, m, prop=prop)

    # Return the accumulator
    return acc


# TODO:re-write as iterative
# TODO: I'm pretty sure this is a dynamic programming problem
# TODO: skip POBJ?
def sn_spanning_helper(acc, buff, curr, m, prop='lemma_'):
    #     print 'curr:', (getattr(curr, prop))
    # Add curr
    # But don't add ROOT because those relations occur in almost all sentences
    # NB: No need to pop b/c deque has max length specified
    if prop == 'dep_' and curr.dep_ == 'ROOT':
        pass
    else:
        buff.append(curr)
        # TODO: try heapq instead of sorting
        buff = deque(sorted(buff, key=operator.attrgetter('idx')), buff.maxlen)

    # Create the ngrams that end with the current node
    # Having the n-grams in reverse lexical order is a bit odd
    # for humans to read, but because it's deterministic, it
    # shouldn't make any difference to the computer for using them as
    # features
    gram = []
    if len(buff) >= m:
        for i, tok in enumerate(buff):
            # Build up the ngram
            #             gram.append(tok)
            gram.append(getattr(tok, prop))
            # Only append ngrams >= the min length
            if len(gram) >= m:
                acc.append('_'.join(gram))

    if curr.n_lefts > 0:
        # Add all left ngrams
        for c in curr.lefts:
            # Don't revisit self
            if c is curr:
                continue
            # Add to acc with destructive modification
            # but don't let children modify buffer because it needs
            # to be in the same state for the next child of curr
            sn_spanning_helper(acc, copy.copy(buff), c, m, prop=prop)
    else:
        # Add all right ngrams
        for c in curr.rights:
            # Add to acc with destructive modification
            # but don't let children modify buffer because it needs
            # to be in the same state for the next child of curr
            sn_spanning_helper(acc, copy.copy(buff), c, m, prop=prop)

    # Now, descend the parent's rights
    # So long as this isn't the root
    if curr.head is curr:
        return acc
    else:
        for c in curr.head.rights:
            # Don't revisit self
            if c is curr:
                continue
            sn_spanning_helper(acc, copy.copy(buff), c, m, prop=prop)

    # Return the accumulator
    return acc
