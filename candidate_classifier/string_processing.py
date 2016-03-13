#!/usr/bin/env python2

"""
Idea behind the structure:

A StringTransformer is a generalized way to perform an ordered
series of transformations on a string.  There are a number of
pre-built transformations already included and arbitrary
callables can be added to the pipeline.

There are three primary kinds of transformations:
    - substitutions
    - filters
    - tokenization

A substitution is any operation that takes in a string and returns
either the same string or a modification of the original string.

A filter is an operation that takes in a string and either returns
that string or returns a falsey value (indicating that that string
was filtered).

Tokenization takes in a single string and returns a list of string.


The pipeline works in the following order:
    1. Pre-filter substitutions
    2. filters
    3. post-filter substitutions
    4. tokenization
    5. post-tokenization operations
        - This one is still in development and isn't stable

There are two phases of subsitution for a couple of reasons. The pre-
filter subsititutions are important because a substitution may result
in a string that is caught by a filter.  The post-filter subsitution is
for efficiency, why perform a potentially expensive transformation on
a string that is going to be filtered anyways?

"""

import re
import string
import unicodedata
from bs4 import BeautifulSoup as bs
from gensim.utils import any2unicode, decode_htmlentities
from gensim.parsing import preprocessing as gprocessing
from candidate_classifier import utils


__author__ = 'Eric Lind'


# TODO:
# - Stopwords
# - Punctuation
# is non-ascii
# is numeric
# is url
# is email address



PUNCT = frozenset(string.punctuation)



# class StringProcessor(object):
#     """This is supposed to be a composition of StringTransformers but
#     because of tokenization yielding a list, this needs to be reworked
#     to do a kind of context-dependant `yield from` possibly using the
#     flatten generator function."""
#     def __init__(self,
#                  string_transformer,
#                  sentence_transformer=None,
#                  token_transformer=None):
#         self.str_transformer = string_transformer
#         self.sent_transformer = sentence_transformer
#         self.tok_transformer = token_transformer
#
#     def __call__(self, s):
#         # Process string
#         processed = self.str_transformer(s)
#
#         # TODO: Fix this and figure our contract
#         # Sentences
#         if self.sent_transformer is not None:
#             sents = self.sent_transformer(s)
#
#             for sent in sents:
#                 # Tokens
#                 if self.tok_transformer is not None:
#                     sent = self.tok_transformer(sent)
#
#                 yield sent
#
#     # def __call__(self, s):
#     #     Yes, I know this is difficult to read.  But this is
#     #     my library and I'll do as I please thank you very much.
#     #     The contract of this reduce operation is essentially this:
#     #     f([a, b, c], 's') ==> c(b(a('s')))
#     #     The reason for the reversal is so that when a user specifies
#     #     the function they want to call first, it can be first in the list.
#         # return reduce(lambda x, y: y(x), self.transformers, s)



# ========================
# FILTERS
# ========================
# Filters return True if a condition is met.
# If a filter returns True for a given string,
# that string is filtered out.

def length_check(length=3):
    def inner(s):
        return len(s) <= length
    return inner


def pattern_check(pattern):
    def inner(s):
        return pattern.search(s)
    return inner


def is_non_ascii(s):
    """
    Returns True if the string contains non-ascii characters
    Fastest way to check for non-ascii:
    http://stackoverflow.com/questions/196345/how-to-check-if-a-string-in-python-is-in-ascii
    """
    try:
        s.decode('ascii')
    except UnicodeDecodeError:
        return True
    else:
        return False


# ========================
# SUBSTITUTIONS
# ========================
# Substitutions take in a string and return some
# transformed version of that string.

def strip_html(s):
    return bs(s, 'lxml').get_text()


# TODO: Compare this to gensim method.
# NFC vs NKKD?
def strip_accents_ascii(s):
    """
    This is shamelessly ripped off from scikit learn:
    https://github.com/scikit-learn/scikit-learn/blob/51a765a/sklearn/feature_extraction/text.py#L504

    Transform accentuated unicode symbols into ascii or nothing
    Warning: this solution is only suited for languages that have a direct
    transliteration to ASCII symbols.
    See also
    --------
    strip_accents_unicode
        Remove accentuated char for any unicode symbol.
    """
    nkfd_form = unicodedata.normalize('NFKD', s)
    return nkfd_form.encode('ASCII', 'ignore').decode('ASCII')


def pattern_sub(pattern):
    def inner(s):
        return pattern.sub('', s)
    return inner



# ========================
# StringTransformer
# ========================
# TODO: Refactor name to StringTransformer

# TODO:
# Make a repr that shows the current filters/substitutions being used
# Add the available built-in filters/substitutions to the docstring
class TransformerABC(object):
    """Interface for applying transformations to a string/token"""

    re_type = type(re.compile(r''))
    ws_pattern = re.compile(r"white(?:-|_|\s)?space", re.I)
    non_ascii_pattern = re.compile(r"non(?:-|_|\s)?ascii", re.I)
    h_ents_pattern = re.compile(r"html(?:-|_|\s)?entities", re.I)
    # TODO: Make min_len an optional argument
    min_len = 3

    def __init__(self,
                 prefilter_substitutions=(),
                 postfilter_substitutions=(),
                 filters=(),
                 normalize_encoding=True,
                 flatten=True,  # TODO: Document
                 # Tokenization always occurs last.
                 # If you want to process the tokens, use another Transformer
                 # tokenizer must be callable
                 tokenizer=None):
        self.filters = self._process_filters(filters)
        self.pre_substitutions = self._process_substitutions(prefilter_substitutions)
        self.post_substitutions = self._process_substitutions(postfilter_substitutions)
        self.normalize_encoding = normalize_encoding
        self.tokenizer = tokenizer

        # Set processor
        self.processor = self._flat_process if flatten else self._nested_process


    # FIXME More documnetation for empty string behavior
    def __call__(self, s):
        """
        If called with a string, returns either a string (no tokenization) or
        a list of strings (tokenized).

        If called with a list of strings, returns a generator that yields
        either strings (no tokenization) or lists of strings (tokenization).
        """
        if isinstance(s, basestring):
            return self._process(s)
        # TODO: More robust checking for iterable
        else:
            return self.__iter__(s)
        # return (i for i in self.processor(s) if i)

    def __iter__(self, docs):
        """
        Docs is a list of strings (documents), this yields lists of strings (tokenized doc).
        If it were to return everything at once it would return str[][].
        :param docs: str[]
        """
        for doc in docs:
            s = self._process(doc)
            if s:
                yield s


    def _process_filters(self, filts):
        """
        Filters are functions that return booleans.  If a function
        resturns True (or some truthy value) for a given string,
        that string is filtered out.
        """
        t = []

        for f in filts:
            # Length
            if f in {'length', 'len', 'short'}:
                # Default length is 3
                t.append(length_check(self.min_len))
            elif hasattr(f, '__getitem__'):
                try:
                    if f[0] in {'length', 'len', 'short'}:
                        # t.append(lambda s: len(s) <= f[1])
                        t.append(length_check(f[1]))
                # FIXME: More thorough checks
                except (IndexError, TypeError):
                    pass

            # Patterns
            elif isinstance(f, self.re_type):
                t.append(pattern_check(f))
                # t.append(lambda s: re.search(f, s))

            # Non-Ascii
            elif re.search(self.non_ascii_pattern, str(f)):
                t.append(is_non_ascii)

            # Callable
            elif hasattr(f, '__call__'):
                t.append(f)

            # TODO:
            # html tags
            # punctuation

        return t


    # TODO: Use string.translate for punctuation ?
    # Use combinations of string.translate for the various combinations of substitutions
    # Accept translation tables as arguments
    def _process_substitutions(self, subs):
        t = []

        # Go through all subs and prepare/apply them.
        # The order is IMPORTANT.  It is expected that
        # the user knows what order they want!
        for sub in subs:
            # HTML
            if sub == 'html':
                t.append(strip_html)

            # HTML entities
            # if sub in {'htmlentities', 'html entities', 'html_entities', 'html-entities'}:
            elif re.search(self.h_ents_pattern, str(sub)):
                t.append(decode_htmlentities)

            # Daccent
            # TODO: Document that this will also remove all non-ascii characters
            elif sub == 'deaccent':
                t.append(strip_accents_ascii)

            # Punctuation
            # TODO: Compare performance to string.translate
            elif sub in {'punct', 'punctuation', 'puncts'}:
                t.append(gprocessing.strip_punctuation)

            # Case
            # TODO: Compare performance to string.translate
            elif sub == 'lower':
                t.append(string.lower)
            elif sub == 'upper':
                t.append(string.upper)

            # Whitespace
            elif re.search(self.ws_pattern, str(sub)):
                t.append(gprocessing.strip_multiple_whitespaces)
            elif sub == 'strip':
                t.append(string.strip)

            # callable
            # TODO: look at gensim.utils.identity
            elif hasattr(sub, '__call__'):
                t.append(sub)

            # patterns
            elif isinstance(sub, self.re_type):
                t.append(pattern_sub(sub))
                # t.append(lambda s: sub.sub('', s))
            else:
                try:
                    # TODO: Better checking
                    # Dict support?
                    if isinstance(sub[0], self.re_type):
                        # def funcC(sub):
                        #     # def func(s): sub[0].sub(sub[1], s)
                        #     def func(s): return re.sub(sub[0], sub[1], s)
                        #     return func
                        # t.append(funcC(sub))
                        # t.append(lambda s: re.sub(sub[0], sub[1], s))
                        t.append(pattern_sub(sub))
                except IndexError:
                    pass
        return t


    def _flat_process(self, s):
        """Takes in either an iterator or a string and yields strings.
        Calling this on a nested list will result in a flattened list.


        str => str
        str[] => str[]
        """
        for elt in utils.flatten(s):
            # print elt
            # Tokenization may add another level of nesting
            for t in utils.flatten(self._process(elt)):
                # print t
                yield t

    def _nested_process(self, s):
        return utils.nested_map(s, self._process)

    def _process(self, s):
        """Takes in a string and returns either a string (no tokenization) or
        a list of strings (tokenized).
        """
        # TODO: Clarify contract return value for filtered strings
        # TODO: Add docs for: if a filter evaluates as False the string is discarded

        # Skip empty strings:
        if len(s) == 0:
            # Keep return type consistent
            if self.tokenizer:
                return []
            else:
                return ''

        # Normalize the encoding before anything else
        if self.normalize_encoding:
            s = any2unicode(s)

        # Apply sub, filter, sub
        for sub in self.pre_substitutions:
            s = sub(s)

        # If any of the filters return True, filter the string
        if any(f(s) for f in self.filters):
            # Keep return type consistent
            if self.tokenizer:
                return []
            else:
                return ''

        for sub in self.post_substitutions:
            s = sub(s)

        # Tokenize last.  If you want to process the tokens use another
        # Transformer
        if self.tokenizer and s:
            s = filter(None, self.tokenizer(s))

        return s
