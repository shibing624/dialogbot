# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
from __future__ import print_function

import sys

from setuptools import setup, find_packages

from dialogbot import __version__

if sys.version_info < (3,):
    sys.exit('Sorry, Python3 is required for dialogbot.')

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

with open('LICENSE', 'r', encoding='utf-8') as f:
    license = f.read()

with open('requirements.txt', 'r', encoding='utf-8') as f:
    reqs = f.read()

setup(
    name='dialogbot',
    version=__version__,
    description='Dialog robot, ChatBot',
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
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Text Processing',
        'Topic :: Text Processing :: Indexing',
        'Topic :: Text Processing :: Linguistic',
    ],
    keywords='NLP,chatbot,dialogbot,dialogue,dialog',
    install_requires=reqs.strip().split('\n'),
    packages=find_packages(exclude=['tests']),
    package_dir={'dialogbot': 'dialogbot'},
    package_data={
        'dialogbot': ['*.*', 'LICENSE', 'README.*', 'data/*', 'data/medical_dict/*',
                      'data/chat/*', 'data/taobao/*', 'kg/*', 'preprocess/*', 'reader/*', 'searchdialog/*',
                      'seq2seqdialog/*', 'taskdialog/*', 'utils/*', 'web/*', 'web/static/*', 'web/templates/*'],
    },
    test_suite='tests',
)
