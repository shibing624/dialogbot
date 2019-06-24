# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

pwd_path = os.path.abspath(os.path.dirname(__file__))

# knowledge graph
host = "127.0.0.1"
kg_port = 7474
user = "neo4j"
password = "123456"
answer_num_limit = 20

# mongodb
mongo_host = 'localhost'
mongo_port = 27017

# preprocess
train_path = os.path.join(pwd_path, 'data/taobao/dev.txt')
dev_path = os.path.join(pwd_path, 'data/taobao/dev.txt')
test_path = os.path.join(pwd_path, 'data/taobao/test.txt')

question_answer_path = os.path.join(pwd_path, 'output/question_answer.txt')
context_response_path = os.path.join(pwd_path, 'output/context_response.txt')
demo_question_answer_path = os.path.join(pwd_path, 'data/taobao/demo_question_answer.txt')
demo_context_response_path = os.path.join(pwd_path, 'data/taobao/demo_context_response.txt')

# Tokenize config file
punctuations_path = os.path.join(pwd_path, "data/punctuations.txt")
stopwords_path = os.path.join(pwd_path, "data/stopwords.txt")
user_define_words_path = os.path.join(pwd_path, "data/user_define_words.txt")
remove_words_path = os.path.join(pwd_path, "data/remove_words.txt")

order_info_path = os.path.join(pwd_path, "data/order/order.txt")
# Tfidf config file
corpus_dict_path = os.path.join(pwd_path, "data/order/corpus_dict.txt")
corpus_tfidf_path = os.path.join(pwd_path, "data/order/corpus_tfidf.txt")

vocab_path = os.path.join(pwd_path, "output/vocab.txt")
log_file = os.path.join(pwd_path, 'output/log.txt')

search_model = 'bm25'
# seq2seq dialog
model_path = os.path.join(pwd_path, 'output/models')
dialog_mode = 'single'
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


class Web:
    """
    前端
    """
    host = "0.0.0.0"
    port = "8820"
    url = "http://" + host + ":" + port
