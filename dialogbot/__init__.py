# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

from .kg.bot import KGBot
from .util.logger import get_logger

__version__ = "0.0.2"

logger = get_logger("smart-chatbot" + __version__)
