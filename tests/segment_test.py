# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
from time import time

from dialogbot import config
from dialogbot.util.tokenizer import segment_file

start_time = time()
# 切词
segment_file(config.train_file_path, config.train_seg_file_path, pos=True)
print("spend time:", time() - start_time)
