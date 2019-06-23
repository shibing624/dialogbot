# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
from . import config
from .bot import Bot
from .kg.bot import KGBot
from .searchdialog.bot import SearchBot
from .seq2seqdialog.bot import Seq2SeqBot
from .utils.logger import get_logger

__version__ = "0.0.1"

logger = get_logger("dialogbot" + __version__)
