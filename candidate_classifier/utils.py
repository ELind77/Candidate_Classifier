#! /usr/bin/env python2

import os
import random
from nltk.corpus.reader.util import ConcatenatedCorpusView

__author__ = 'Eric Lind'


def ensure_directories_exist(path):
    """
    Checks whether directories containing the specified file exist
    and attempts to create them is they don't.

    Hyperspec: http://clhs.lisp.se/Body/f_ensu_1.htm

    :param path: str - A path, can optionally end in a file
    :returns str - The path that was created
    """
    # http://stackoverflow.com/questions/273192/in-python-check-if-a-directory-exists-and-create-it-if-necessary
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return path


class DocumentsFromDir(object):
    """
    Yields file names from a directory tree if they have the given extension.
    Usage:
    `for doc in DocumentsFromDir('.', ext='.json'):
        print doc`
    """
    def __init__(self, dirr, ext='.txt'):
        self.dirr = dirr
        # Add a dot if needed
        self.ext = ext if ext.startswith('.') else '.' + ext

    def __iter__(self):
        for root, sub_dir_list, file_list in os.walk(self.dirr):
            for f_name in file_list:
                if os.path.splitext(f_name)[1] == self.ext:
                    yield os.path.join(root, f_name)


def r_sample(source, k, seed=None):
    """
    Implementation of reservoir sampling algorithm.
    Selects k items from a collection with equal probability k/len(source),
    also expressed as k/i.
    Proof by wikipedia: https://en.wikipedia.org/wiki/Reservoir_sampling

    Note: You can also use the builtin random.sample()
          but that won't work on a generator without a __len__,
          so if you want to select from a generator use this :)

    :param source: Any iterator
    :param k: int: The number of samples to take from iter
    :param seed: optional random seed, defaults to None so it uses system time
    :return: A list of length k
    """
    # int[]
    reservoir = []
    # seed random so we get repeatable results
    random.seed(seed)

    for i, elt in enumerate(source):
        # Populate the reservoir
        if i < k:
            reservoir.append(elt)
        else:
            r = random.randint(0, i)
            if r < k:
                reservoir[r] = elt
    # Return the reservoir
    return reservoir


def flatten(nested):
    """
    Completely generalized function to flatten an arbitrarily
    nested iterator/generator.  It's also a generator, so it
    can work on generators without fully expanding them.

    However, because it's depth-first, it will reach the
    recursion limit if given a truely infinitely deep tree.

    http://www.java2s.com/Tutorial/Python/0060__Statement/MakingItSafer.htm
    """
    try:
        # Don't unwind strings
        try:
            nested + ''
        except TypeError:
            pass
        else:
            # TODO: Make this more robust
            if not isinstance(nested, ConcatenatedCorpusView):
                raise TypeError

        for sublist in nested:
            for element in flatten(sublist):
                yield element
    except TypeError:
        yield nested


def nested_map(root, func):
    """Takes in an arbitrarily nested iterator and traverses it
     deapth first to create generators at every level, where
     every element has a function applied to it.
     """
    try:
        # Don't unwind strings
        try:
            root + ''
        except TypeError:
            pass
        else:
            raise TypeError

        return (nested_map(elt, func) for elt in root)

    except TypeError:
        return func(root)
