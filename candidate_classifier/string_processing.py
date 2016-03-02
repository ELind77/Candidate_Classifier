#! /usr/bin/env python2

"""
Idea behind the structure:

 String Level Transformations
   - Encodings
   - Substitutions
 Tokenize
 Sentence Level Transformations
   - Filters
   - Transformations
 Token Level Transformations
   - Filters
   - Substitutions
   - Additions
       - Append POS, lemmatize, etc

"""

import re
import string
# import itertools
from bs4 import BeautifulSoup as bs
from gensim.utils import any2unicode, any2utf8, decode_htmlentities, deaccent
from gensim.parsing import preprocessing as gprocessing
from candidate_classifier import utils


__author__ = 'Eric Lind'


# TODO:
# pre/postfilter substitutions
#  - What if the substitution you perform changes the string so that it would have been filtered?


class StringProcessor(object):
    def __init__(self,
                 string_transformer,
                 sentence_transformer=None,
                 token_transformer=None):
        """Autobots, roll out!"""
        self.str_transformer = string_transformer
        self.sent_transformer = sentence_transformer
        self.tok_transformer = token_transformer

    def __call__(self, s):
        # Process string
        processed = self.str_transformer(s)

        # TODO: Fix this and figure our contract
        # Sentences
        if self.sent_transformer is not None:
            sents = self.sent_transformer(s)

            for sent in sents:
                # Tokens
                if self.tok_transformer is not None:
                    sent = self.tok_transformer(sent)

                yield sent



    # def __call__(self, s):
    #     Yes, I know this is difficult to read.  But this is
    #     my library and I'll do as I please thank you very much.
    #     The contract of this reduce operation is essentially this:
    #     f([a, b, c], 's') ==> c(b(a('s')))
    #     The reason for the reversal is so that when a user specifies
    #     the function they want to call first, it can be first in the list.
        # return reduce(lambda x, y: y(x), self.transformers, s)


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
                 tokenizer=None,
                 **kwargs):
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
            if f in {'length', 'len'}:
                # Default length is 3
                # Create a closure
                def funcC(fil):
                    def func(s): return len(s) <= self.min_len
                    return func
                t.append(funcC(f))
                # t.append(lambda s: len(s) <= self.min_len)
            elif hasattr(f, '__getitem__'):
                try:
                    if f[0] in {'length', 'len', 'short'}:
                        t.append(lambda s: len(s) <= s[1])
                # FIXME: More thorough checks
                except (IndexError, TypeError):
                    pass

            # Patterns
            elif isinstance(f, self.re_type):
                # Create a closure for the current filter
                def funcC(fil):
                    def func(s): return re.search(fil, s)
                    return func
                t.append(funcC(f))
                # t.append(lambda s: re.search(f, s))

            # Non-Ascii
            elif re.search(self.non_ascii_pattern, str(f)):
                t.append(self._is_non_ascii)

            # Callable
            elif hasattr(f, '__call__'):
                t.append(f)

            # TODO:
            # html tags
            # punctuation

        return t


    # TODO: Use string.translate for punctuation
    # Use combinations of string.translate for the various combinations of substitutions
    # Accept translation tables as agruments
    def _process_substitutions(self, subs):
        t = []

        # Go through all subs and prepare/apply them.
        # The order is IMPORTANT.  It is expected that
        # the user knows what order they want!
        for sub in subs:
            # HTML
            if sub == 'html':
                t.append(lambda s: bs(s, 'lxml').get_text())

            # HTML entities
            # if sub in {'htmlentities', 'html entities', 'html_entities', 'html-entities'}:
            elif re.search(self.h_ents_pattern, str(sub)):
                t.append(decode_htmlentities)

            # Daccent
            elif sub == 'deaccent':
                t.append(deaccent)

            # Punctuation
            elif sub in {'punct', 'punctuation', 'puncts'}:
                t.append(gprocessing.strip_punctuation)

            # Case
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
            elif hasattr(sub, '__call__'):
                t.append(sub)

            # patterns
            elif isinstance(sub, self.re_type):
                # TODO: Make more elegant
                def funcC(sub):
                    def func(s): return sub.sub('', s)
                    return func
                t.append(funcC(sub))
                # t.append(lambda s: sub.sub('', s))
            else:
                try:
                    if isinstance(sub[0], self.re_type):
                        def funcC(sub):
                            # FIXME: Use compiled pattern
                            # def func(s): sub[0].sub(sub[1], s)
                            def func(s): return re.sub(sub[0], sub[1], s)
                            return func
                        t.append(funcC(sub))
                        # t.append(lambda s: re.sub(sub[0], sub[1], s))
                except IndexError:
                    pass

        return t

    @staticmethod
    def _is_non_ascii(s):
        """Returns True if the string contains non-ascii characters
        Fastest way to check for non-ascii:
        http://stackoverflow.com/questions/196345/how-to-check-if-a-string-in-python-is-in-ascii
        """
        try:
            s.decode('ascii')
        except UnicodeDecodeError:
            return True
        else:
            return False

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
            return s

        # Normalize the encoding before anything else
        if self.normalize_encoding:
            s = any2unicode(s)

        # Apply sub, filter, sub
        for sub in self.pre_substitutions:
            s = sub(s)

        # If any of the filters return True, filter the string
        if any(f(s) for f in self.filters):
            return ''

        for sub in self.post_substitutions:
            s = sub(s)

        # Tokenize last.  If you want to process the tokens use another
        # Transformer
        if self.tokenizer:
            s = self.tokenizer(s)

        return s






class StringTransformer(TransformerABC):
    def __init__(self,
                 prefilter_substitutions=(),
                 postfilter_substitutions=(),
                 filters=()):
        super(StringTransformer, self).__init__(prefilter_substitutions,
                                                postfilter_substitutions,
                                                filters)

    def __call__(self, s):
        # Normalize encoding
        # TODO: Find a more robust function for this (e.g. from another library)
        s = any2unicode(s)

        return self._process(s)



class SentenceTransformer(TransformerABC):
    def __init__(self,
                 prefilter_substitutions=(),
                 postfilter_substitutions=(),
                 filters=(),
                 sent_tokenizer=None):
        super(SentenceTransformer, self).__init__(prefilter_substitutions,
                                                  postfilter_substitutions,
                                                  filters)
        # If no tokenizer is specified, it is assumed that
        # this will be called on a pre-tokenized list of sentences
        self.tokenizer = sent_tokenizer

    def __call__(self, s):
        """Takes in a string and yields sentences"""
        if self.tokenizer is not None:
            sents = self.tokenizer(s)
        else:
            sents = s

        for sent in sents:
            yield self._process(sent)



# TODO:
# - Stopwords
# - Punctuation
# is non-ascii
# is numeric
# is url
# is email address
class TokenTransformer(TransformerABC):
    def __init__(self,
                 prefilter_substitutions=(),
                 postfilter_substitutions=(),
                 filters=(),
                 tokenizer=None):
        super(TokenTransformer, self).__init__(prefilter_substitutions,
                                               postfilter_substitutions,
                                               filters)
        # self.filters.extend(self._process_token_filters(filters))
        # self.pre_substitutions.extend(self._process_token_substitutions(prefilter_substitutions))
        # self.post_substitutions.extend(self._process_token_substitutions(postfilter_substitutions))

        # If no tokenizer is specified, it is assumed that
        # this will be called on a pre-tokenized list of words
        self.tokenizer = tokenizer

    def _process_token_filters(self, filters):
        # Stopwords
        # if f in {'stop', 'stops', 'stopwords'}:
        #     pass
        pass


    def _process_token_substitutions(self, substitutions):
        pass

    # def __call__(self, sent):
    #     """Takes in a

