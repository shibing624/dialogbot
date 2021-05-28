# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import glob
import io
import os


def get_file_path(dotted_path, extension='json'):
    """
    Reads a dotted file path and returns the file path.
    """
    # If the operating system's file path seperator character is in the string
    if os.sep in dotted_path or '/' in dotted_path:
        # Assume the path is a valid file path
        return dotted_path

    parts = dotted_path.split('.')
    corpus_path = os.path.join(*parts)

    if os.path.exists(corpus_path + '.{}'.format(extension)):
        corpus_path += '.{}'.format(extension)

    return corpus_path


def read_corpus(file_name):
    """
    Read and return the data from a corpus json file.
    """
    with io.open(file_name, encoding='utf-8') as data_file:
        return data_file.readlines()


def list_corpus_files(dotted_path, extension='tsv'):
    """
    Return a list of file paths to each data file in the specified corpus.
    """
    corpus_path = get_file_path(dotted_path, extension=extension)
    paths = []

    if os.path.isdir(corpus_path):
        paths = glob.glob(corpus_path + '/**/*.' + extension, recursive=True)
    else:
        paths.append(corpus_path)

    paths.sort()
    return paths


def load_corpus(*data_file_paths, sep='\t'):
    """
    Return the data contained within a specified corpus.
    """
    for file_path in data_file_paths:
        corpus_data = read_corpus(file_path)
        for data in corpus_data:
            query, answer = data.split(sep)
            yield query, answer
