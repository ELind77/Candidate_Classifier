#! /usr/bin/env python2


from nltk.corpus.reader.util import *
from nltk.corpus import PlaintextCorpusReader
import nltk
from nltk.tokenize import WordPunctTokenizer


__author__ = 'Eric Lind'


# TODO: give user access to regexes that will strip out the messy parts of the text e.g. [applause]

# Should have methods for:
#   - List of speakers
#   - Raw
#   - Words
#       - File as a list of words in the oder they appear in the file
#       - Filtered by speaker(s)
#       - no labels
#   - Sents
#       - File as a list of sentences in the order they appear in the file
#       - Filtered by speaker(s)
#       - no labels, no other cleaning
#   - Paras
#       - Filtered by speaker
#       - File as a list of utterances, labeled by the speaker
#       - Labels as a tuple
#   - Grouped_Words
#       - This uses some preprocessing to (hopefully) get better sentence tokenization.
#       - Filtered by speaker
#       - File as a dict {str speaker: str[] words} where each speaker name
#         is a key and all of their words are the list
#   - Grouped Sents
#       - Filtered by speaker
#       - File as a dict {str speaker: str[][] sents}



class DebateCorpusReader(PlaintextCorpusReader):
    """
    Extension of the NLTK `PlaintextCorpusReader` designed to read
    presedential debate transcripts from The American Presidency Project
    http://www.presidency.ucsb.edu/debates.php

    There are two ways to access the tokens of a DebateCorpusReader instance,
    through the standard `words()` and`sents()` functions and through the
     `grouped_words()` and `grouped_sents()` functions.  The standard functions
    simply return the words and sentences of the documents as they appear,
    separated by the 'SPEAKER:' indicator in the file.  But the grouped
    alternatives separate out the text by individual speakers (and
    `grouped_sents()` joins text accross 'SPEAKER:' boundaries to improve
    sentence tokenization) so that you can operate on tokens/sentences for each
    speaker.

    All of the methods above accept `speakers` as a keyword argument so that only
    the tokens of a given speaker or list of speakers will be returned.

    The default NLTK tokenizers are used by default but anything that fits the
    contract can be used.
    """
    _speaker_pattern = re.compile(r"(^\[?[A-Z :\-?_]+\]?:+)", re.U | re.M)

    def __init__(self, root, fileids,
                 word_tokenizer=WordPunctTokenizer(),
                 sent_tokenizer=nltk.data.LazyLoader(
                     'tokenizers/punkt/english.pickle'),
                 encoding='utf8'):
        super(DebateCorpusReader, self).__init__(root,
                                                 fileids,
                                                 word_tokenizer=word_tokenizer,
                                                 sent_tokenizer=sent_tokenizer,
                                                 para_block_reader=self._make_regex_block_reader(),
                                                 encoding=encoding)
        # Property cache
        # The cache is a dictionary where keys are file names joined by _ and values
        # are whatever that property/function would return
        self._speakers = {}
        # TODO: Cache speaker paras?

    def speakers(self, fileids=None):
        """Same as `raw_speakers` but cleans out common typos."""
        return list(set(self._clean_speaker(s) for s in self.raw_speakers(fileids=fileids)))

    @staticmethod
    def _clean_speaker(s):
        return s.strip(' :[]')

    def raw_speakers(self, fileids=None):
        """
        Returns the names of all of the speakers in the given files.
        :return: A list of all of the unique speaker names.
        :rtype: list(str)
        """
        if fileids is None:
            fileids = self._fileids
        elif isinstance(fileids, string_types):
            fileids = [fileids]

        # FIXME: Cleanup
        if hasattr(fileids, '__getitem__'):
            # FIXME: confirm type is str[]
            f_key = '_'.join(fileids)
        else:
            f_key = fileids

        # Used cached value if it exists
        if f_key in self._speakers:
            return self._speakers[f_key]
        else:
            speakers = list(set(m.group(0) for m in
                                self._speaker_pattern.finditer(self.raw(fileids=fileids))))
            self._speakers[f_key] = speakers

        return speakers


    #
    # Access Methods
    # Methods for accessing the various lexical chunks of documents

    def words(self, fileids=None, speakers=None):
        """
        Returns all of the words in the given file(s), filtered by the
        given speaker(s).
        :param fileids: A list of file names
        :param speakers: A list of (cleaned) speaker names
        :return: the given file as a list of words, with the 'SPEAKER:'
            indicator removed.
        """
        # Get speakers
        speakers = self._parse_speakers(fileids, speakers)
        # Make reader
        reader = self._make_word_block_reader(speakers)
        return self._reader_view(fileids, reader)

    def sents(self, fileids=None, speakers=None):
        """
        Return the given file(s) as a list of sentences as they appear in the
        file(s), filtered by the given speaker(s).  If you want the various incidences
        of crosstalk and other interruptions to be ignored so that sentences
        continue across utterance boundaries this can greatly improve sentence
        tokenization), try `grouped_sents`.

        NB: The 'SPEAKER:' indicator is removed.

        :param fileids: A list of file names
        :param speakers: A list of (cleaned) speaker names
        :return: the given file(s) as a list of tokenized sentences, with the
            'SPEAKER:' indicator removed.
        :rtype: list[list[str]]
        """
        # Get speakers
        speakers = self._parse_speakers(fileids, speakers)
        # Make reader
        reader = self._make_sent_block_reader(speakers)
        return self._reader_view(fileids, reader)

    def paras(self, fileids=None, speakers=None):
        raise NotImplementedError("DebateCorpusReader doesn't do paragraphs in the same way as "
                                  "other readers. Please use utterances() instead and note the "
                                  "return type.")

    def utterances(self, fileids=None, speakers=None):
        """
        Return the given file(s) as a list of utterances as they appear in the
        file(s), filtered by the given speaker(s). The utterances are a list
        of tuples where the first element is the (cleaned) name of the speaker
        indicated in the file and the second element is a list of sentences
        that are each a list of tokens.

        If you want the various incidences of crosstalk and other interruptions
        to be ignored so that sentences continue across utterance boundaries
        (this can greatly improve sentence tokenization), try `grouped_sents`.

        Example return value:
        [("WALKER", [["That's", "crazy."], ["What", "you're", "saying", "makes", "no", "sense"]])]

        :param fileids: A list of file names
        :param speakers: A list of (cleaned) speaker names
        :return: the given file(s) as a list of utterances as they appear in the
            file(s), filtered by the given speaker(s), labeled by the speaker names
        """
        # Get speakers
        speakers = self._parse_speakers(fileids, speakers)
        # Make reader
        reader = self._make_utterance_block_reader(speakers)
        return self._reader_view(fileids, reader)

    def grouped_words(self, fileids=None, speakers=None):
        """
        Returns the given file(s) as a dict {str speaker: str[] words} where each
        speaker name is a key and the value is their words as a list.

        :param fileids: A list of file names
        :param speakers: A list of (cleaned) speaker names
        :return: A dictionary mapping a speaker to a list of thier words
        :rtype: dict[str, list[str]]
        """
        speakers = self._parse_speakers(fileids, speakers)
        return {s: self.words(fileids=fileids, speakers=speakers) for s in speakers}


    def grouped_sents(self, fileids=None, speakers=None):
        """
        Because of the nature of the text, interruptions, cross-talk, etc.
        many sentences are split across speaker boundaries.  This can cause unclean
        sentence tokenization.
        This method uses some preprocessing to combine utterances across boundaries
        so as to (hopefully) get better sentence tokenization.  Unfortunately,
        in order to do this, a speakers' entire text (for a given file) has to be
        read completely into memory in order to do this because there is no
        guarantee that sentences will end at utterance bondaries (which is the
        whole point of this method).  The files are typically small though so
        this shouldn't be a big concern.

        It may also be a good idea to use some string substitution to remove the
        various parentheticals, e.g. [applause] before processing, but that
        is not done here.

        Returns the given file(s) as a dict {str speaker: str[][] sentences} where each
        speaker name is a key and the values are their sentences as lists of words.

        :param fileids: A list of file names
        :param speakers: A list of (cleaned) speaker names
        :return: A dictionary mapping a speaker to a list of thier words
        :rtype: dict[str, list[list[str]]
        """
        # Get speakers
        speakers = self._parse_speakers(fileids, speakers)

        # Return dict of sents
        grouped = {}
        for s in speakers:
            reader = self._make_joined_sent_block_reader(s)
            grouped[s] = self._reader_view(fileids, reader)

        return grouped



    # Helpers

    def _parse_speakers(self, fileids, speakers):
        """
        :rtype: list[str]
        """
        if speakers is None:
            return self.speakers(fileids=fileids)
        elif isinstance(speakers, basestring):
            return [self._clean_speaker(speakers).upper()]
        elif isinstance(speakers, (list, tuple)):
            return [self._clean_speaker(s).upper() for s in speakers]
        else:
            raise TypeError('The `speakers` parameter is neither a string nor a list/tuple')

    def _reader_view(self, fileids, reader):
        return concat([self.CorpusView(path, reader, encoding=enc)
                       for (path, enc, fileid)
                       in self.abspaths(fileids, True, True)])


    # def joined_speaker_paras(self, fileids=None, speakers=None):
    #     """Because of the nature of the text, interruptions, cross-talk, etc.
    #     many sentences are split across speaker boundaries.  This can cause unclean
    #     sentence tokenization.  This method is not particularly memory efficient
    #     because it reads in a whole file at a time, but it avoids the tokenization
    #     issue.
    #     """
    #     if fileids is None: fileids = self._fileids
    #     elif isinstance(fileids, string_types): fileids = [fileids]
    #
    #     speakers = self._parse_speakers(fileids, speakers)
    #
    #     paras = {}
    #
    #     for s in speakers:
    #         reader = self._make_joined_reader(s)
    #         paras[s] = concat([self.CorpusView(path, reader, encoding=enc)
    #                            for (path, enc, fileid)
    #                            in self.abspaths(fileids, True, True)])
    #
    #     return paras
    #
    # def all_speaker_paras(self, fileids=None):
    #     return {name: self.speaker_paras(name) for name in self.speakers(fileids=fileids)}
    #
    # def speaker_paras(self, fileids=None, speakers=None):
    #     """Returns a dictionary of
    #     {str speaker_name: CorpusView paragraphs}
    #     """
    #     # FIXME: Uppercase speakers
    #     # speakers = [s.upper() for s in speakers]
    #     # TODO: Tokenize etc.
    #     reader = self._make_speaker_block_reader(speakers)
    #
    #     return concat([self.CorpusView(path, reader, encoding=enc)
    #                    for (path, enc, fileid)
    #                    in self.abspaths(fileids, True, True)])


    #
    # Reader Factories
    # Factory methods for the various block readers

    def _make_word_block_reader(self, speakers):
        """
        Factory method that creates a word block reader that
        is filtered by the given speakers.
        :param speakers: basestring|basestring[]
        :return: func
        """
        reader = self._make_speaker_block_reader(speakers)

        def wbr(stream):
            words = []
            # Read a chunk of paras and remove the speaker label
            try:
                for i in xrange(10):
                    for para in reader(stream):
                        if not para:
                            raise StopIteration
                        words.extend(self._word_tokenizer.tokenize(para[1]))
            except StopIteration:
                return words
            finally:
                return words

        return wbr


    def _make_sent_block_reader(self, speakers):
        """Factory method that creates a sentence block
        reader that is filtered by the given speaker(s)"""
        reader = self._make_speaker_block_reader(speakers)

        def sbr(stream):
            sents = []
            for para in reader(stream):
                sents.extend([self._word_tokenizer.tokenize(sent)
                              for sent in self._sent_tokenizer.tokenize(para[1]) if sent])
            return sents

        return sbr


    def _make_utterance_block_reader(self, speakers):
        """Factory Method that creates an utterance block reader
        that is filtered by the speaker names and includes labels
        for the speaker"""
        reader = self._make_speaker_block_reader(speakers)

        def ubr(stream):
            paras = []
            for para in reader(stream):
                paras.append((para[0],
                              [self._word_tokenizer.tokenize(sent)
                               for sent in self._sent_tokenizer.tokenize(para[1]) if sent]))
            return paras

        return ubr


    def _make_regex_block_reader(self, start_re=None, end_re=None):
        """Factory method for making a paragraph block reader using
        a closure for `read_regexp_block` with pre-defined regexp."""
        if start_re is None: start_re = self._speaker_pattern

        def pbr(stream):
            return read_regexp_block(stream, start_re, end_re=end_re)

        return pbr


    # TODO: Create a cache for this
    def _make_speaker_block_reader(self, speakers):
        """
        Factory method that returns a regexp block reader that only returns
        utterances by the given speaker.

        The reader that is returned doesn't do any tokenization, it only
        pulls out the label and the text.

        :param speakers: basestring|basestring[]
        :return: func
        """
        start_re = self._make_speaker_pattern(speakers)
        end_re = self._speaker_pattern
        reader = self._make_regex_block_reader(start_re=start_re, end_re=end_re)

        def new_reader(stream):
            paras = []
            for para in reader(stream):
                # para = self._speaker_pattern.sub('', para, count=1)
                # paras.append([self._word_tokenizer.tokenize(sent)
                #               for sent in self._sent_tokenizer.tokenize(para) if sent])
                # speaker = self._clean_speaker(self._speaker_pattern.search(para).group(0))
                # paras.append((speaker, [self._word_tokenizer.tokenize(sent)
                #                         for sent in self._sent_tokenizer.tokenize(para) if sent]))
                # TODO: Better error checking
                split = self._speaker_pattern.split(para, maxsplit=1)
                paras.append((self._clean_speaker(split[1]), split[2]))
            return paras

        return new_reader


    def _make_joined_sent_block_reader(self, speaker):
        """Factory method to create a reader that reads everything a given speaker
        says in a stream and then tokenize it."""
        reader = self._make_speaker_block_reader(speaker)

        # FIXME: On most recent python this is no longer true.  += is faster
        # What I've done here is a bit convoluted, but the purpose is to take
        # advantage of the fact that str.join is much faster at string
        # concatenation than +=
        # E.g. https://waymoot.org/home/python_string/
        # I still need to actually benchmark this on this particular task, but
        # it should work
        # I suppose I could also use itertools.takewhile...
        def helper(stream):
            paras = reader(stream)
            while paras:
                for p in paras:
                    yield p[1]
                paras = reader(stream)

        def jsbr(stream):
            text = u' '.join(p for p in helper(stream))
            return [self._word_tokenizer.tokenize(sent)
                    for sent in self._sent_tokenizer.tokenize(text) if sent]

        return jsbr


    # Helpers

    @staticmethod
    def _make_speaker_pattern(names):
        """
        Returns a compiled regular expression to match any speaker
        by name, even accounting for typos.

        If a list of names are given, this will match any of them.
        """
        if isinstance(names, basestring):
            pattern = r"(^\[?{}[ :\-?_]*\]?:+)".format(re.escape(names))
        # FIXME: Better checking for list/tuple
        else:
            name_p = r'(?:' + r'|'.join(re.escape(n) for n in names) + r')'
            pattern = r"(^\[?{}[ :\-?_]*\]?:+)".format(name_p)

        return re.compile(pattern, re.U | re.M)



class DummyTokenizer(object):
    def tokenize(self, s):
        return s
