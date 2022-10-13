# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

from codecs import open

import gensim
import numpy as np
from loguru import logger
from dialogbot.utils.tokenizer import Tokenizer

PAD_TOKEN, GO_TOKEN, EOS_TOKEN, UNK_TOKEN = '<pad>', '<go>', '<eos>', '<unk>'
PAD_ID, GO_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
max_seq_len = 60


class Batch:
    def __init__(self):
        self.encoder_inputs = []
        self.encoder_inputs_length = []
        self.decoder_targets = []
        self.decoder_targets_length = []


def load_dataset(vocab_path, train_path=None, vocab_size=0):
    """
    load data set
    :param vocab_path:
    :param train_path:
    :param vocab_size:
    :return: word2id, id2word, training_samples
    """
    with open(vocab_path, "r", encoding="utf-8") as rfd:
        word2id = {PAD_TOKEN: PAD_ID,
                   GO_TOKEN: GO_ID,
                   EOS_TOKEN: EOS_ID,
                   UNK_TOKEN: UNK_ID}
        id2word = {v: k for k, v in word2id.items()}
        cnt = 4
        vocab_data = rfd.read().splitlines()
        vocab_data_user = vocab_data[:vocab_size] if vocab_size > 0 else vocab_data
        for line in vocab_data_user:
            items = line.split("\t")
            if len(items) != 2:
                logger.error('error file path: {}, line at: {}'.format(vocab_path, line))
                continue
            word = items[0]
            word2id[word] = cnt
            id2word[cnt] = word
            cnt += 1

    if train_path:
        with open(train_path, "r", encoding="utf-8") as rfd:
            data = rfd.read().splitlines()
            data = [line.split("\t") for line in data]
            training_samples = [[text2id(item[0], word2id),
                                 text2id(item[1], word2id)] for item in data]
            training_samples.sort(key=lambda x: len(x[0]))
            training_samples = [item for item in training_samples if
                                (len(item[0]) >= 1 and len(item[1]) >= 1)]
        logger.debug("Load train data from %s done." % train_path)
        return word2id, id2word, training_samples
    else:
        return word2id, id2word


def text2id(text, word2id):
    ids = [word2id[w] for w in text.split() if w in word2id]
    return ids


def create_batch(samples):
    batch = Batch()
    samples = [[item[0][-max_seq_len:], item[1][:max_seq_len]] for item in samples]
    batch.encoder_inputs_length = [len(sample[0]) + 2 for sample in samples]
    batch.decoder_targets_length = [len(sample[1]) + 2 for sample in samples]

    max_source_length = max(batch.encoder_inputs_length)
    max_target_length = max(batch.decoder_targets_length)

    for sample in samples:
        # source = list(reversed(sample[0]))
        source = [GO_ID] + sample[0] + [EOS_ID]
        pad = [PAD_ID] * (max_source_length - len(source))
        batch.encoder_inputs.append(pad + source)

        target = [GO_ID] + sample[1] + [EOS_ID]
        pad = [PAD_ID] * (max_target_length - len(target))
        batch.decoder_targets.append(target + pad)

    return batch


def get_batches(data, batch_size):
    # random.shuffle(data)
    batches = []
    data_len = len(data)

    def gen_next_samples():
        for i in range(0, data_len, batch_size):
            yield data[i:min(i + batch_size, data_len)]

    for samples in gen_next_samples():
        batch = create_batch(samples)
        batches.append(batch)
    return batches


def sentence2enco(sentence, word2id):
    if isinstance(sentence, str):
        tokens = Tokenizer.tokenize(sentence, True)
    else:
        tokens = sentence
    word_ids = [word2id[w] for w in tokens if w in word2id]
    batch = create_batch([[word_ids, []]])
    return batch


def corpus2enco(corpus, word2id):
    corpus = [([word2id[w] for w in s.split()], []) for s in corpus]
    batch = create_batch([corpus])
    return batch


def dump_word_embeddings(word2id, emb_size, word2vec_path, embeddings_path):
    vocab_size = len(word2id)
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(
        word2vec_path, binary=False)
    embeddings = np.random.randn(vocab_size, emb_size)
    for word, idx in word2id.items():
        if word in word2vec.wv.key_to_index:
            embeddings[idx, :] = word2vec.wv.key_to_index[word]
        else:
            embeddings[idx, :] = np.random.randn(emb_size)
    np.save(embeddings_path, embeddings)


def load_corpus_file(corpus_file, word2id, size=None):
    data_all = []
    with open(corpus_file, "r", encoding="utf-8") as rfd:
        data_all = rfd.readlines()
    if size and size > 0:
        data_all = data_all[:size]
    data = [s.strip().split("\t") for s in data_all]
    contexts = [[w for w in s.split() if w in word2id] for s, _ in data]
    responses = [s.replace(" ", "") for _, s in data]
    return contexts, responses
