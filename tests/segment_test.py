# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
from time import time
import sys

sys.path.append('..')
from dialogbot import config
from dialogbot.utils.tokenizer import segment_file

start_time = time()
# 切词
segment_file(config.train_path, 'seg.txt', is_pos=False)
print("spend time:", time() - start_time)
