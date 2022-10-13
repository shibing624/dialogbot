# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

from setuptools import setup, find_packages

if sys.version_info < (3,):
    sys.exit('Sorry, Python3 is required for dialogbot.')

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

setup(
    name='dialogbot',
    version='0.1.2',
    description='Dialog Robot, ChatBot',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='XuMing',
    author_email='xuming624@qq.com',
    url='https://github.com/shibing624/dialogbot',
    license="Apache 2.0",
    classifiers=[
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Natural Language :: Chinese (Simplified)',
        'Natural Language :: Chinese (Traditional)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Text Processing',
        'Topic :: Text Processing :: Indexing',
        'Topic :: Text Processing :: Linguistic',
    ],
    keywords='NLP,chatbot,dialogbot,dialogue,dialog',
    install_requires=[
        'transformers',
        'loguru',
        'jieba',
        'gensim',
        'lxml',
        'tqdm',
        'bs4',
        'numpy',
        'nltk'
    ],
    packages=find_packages(exclude=['tests']),
    package_dir={'dialogbot': 'dialogbot'},
    package_data={
        'dialogbot': ['*.*', '../LICENSE', '../*.md', '../*.txt',
                      'data/*', 'data/medical_dict/*',
                      'data/person_graph/*', 'data/taobao/*'],
    },
    test_suite='tests',
)
