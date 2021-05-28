# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import multiprocessing
import os
import sys

pwd_path = os.path.abspath(os.path.dirname(__file__))


workers = 1
chdir = pwd_path

worker_connections = 1000
timeout = 120
max_requests = 2000
graceful_timeout = 60

loglevel = 'debug'

reload = True
debug = True
daemon = True
bind="0.0.0.0:8820"

errorlog = '%s/error.log' % pwd_path
accesslog = '%s/access.log' % pwd_path

