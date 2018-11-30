# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

# knowledge graph
host = "127.0.0.1"
kg_port = 7474
user = "neo4j"
password = "123456"
answer_num_limit = 20


# word2vec param
train_file_path = '../data/reduce_weight.txt'
train_model_path = '../data/reduce_weight.pkl'
emb_model_path = '../data/reduce_weight.bin'
min_count = 3
num_workers = 4

# mongodb
mongo_host = 'localhost'
mongo_port = 27017
