# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

pwd_path = os.path.abspath(os.path.dirname(__file__))

# -----用户目录，存储模型文件-----
USER_DATA_DIR = os.path.expanduser('~/.cache/torch/shibing624')
os.makedirs(USER_DATA_DIR, exist_ok=True)

# tokenize config file
punctuations_path = os.path.join(pwd_path, "data/punctuations.txt")
stopwords_path = os.path.join(pwd_path, "data/stopwords.txt")
user_define_words_path = os.path.join(pwd_path, "data/user_define_words.txt")
remove_words_path = os.path.join(pwd_path, "data/remove_words.txt")

# search dialog
search_model = 'bm25'
question_answer_path = os.path.join(pwd_path, 'data/taobao/question_answer.txt')
context_response_path = os.path.join(pwd_path, 'data/taobao/context_response.txt')
search_vocab_path = os.path.join(pwd_path, 'data/taobao/vocab.txt')

# seq2seq dialog
dialog_mode = 'single'
vocab_path = os.path.join(pwd_path, "data/taobao/vocab.txt")
model_path = os.path.join(pwd_path, 'output/models')
seq2seq_model_path = os.path.join(model_path, dialog_mode)
predict_result_path = os.path.join(pwd_path, 'output/predict_result.txt')


class Params:
    rnn_size = 256
    num_layers = 1
    embedding_size = 300
    vocab_size = 10000
    learning_rate = 0.001
    batch_size = 80
    epochs = 15
    save_steps = 300
    model_name = "chatbot.ckpt"
    beam_size = 10
    max_gradient_norm = 5.0
    use_attention = True
    bidirectional_rnn = False


# knowledge graph
host = "127.0.0.1"
kg_port = 7474
user = "neo4j"
password = "123456"
answer_num_limit = 20
# mongodb
mongo_host = 'localhost'
mongo_port = 27017
