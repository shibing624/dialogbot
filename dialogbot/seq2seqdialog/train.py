# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import math
import os
import sys
import time

import tensorflow as tf

sys.path.append('../..')
from dialogbot import config
from dialogbot.reader.data_helper import load_dataset, get_batches
from dialogbot.seq2seqdialog.seq2seq import Seq2SeqModel

Params = config.Params


def train(model_path,
          vocab_path,
          question_answer_path=None,
          context_response_path=None,
          dialog_mode='single'):
    for name, value in vars(Params).items():
        t = "%s\t%s" % (name, value)
        print(t)

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_dir = os.path.join(model_path, dialog_mode)
    checkpoint_path = os.path.join(model_dir, Params.model_name)

    if dialog_mode == "single":
        train_path = question_answer_path
    else:
        train_path = context_response_path
    word2id, id2word, samples = load_dataset(vocab_path, train_path, vocab_size=Params.vocab_size)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
    # Define train/eval model.
    with tf.name_scope("Train"):
        with tf.variable_scope("Model", reuse=None):
            Params.beam_search = False
            seq2seq_model = Seq2SeqModel(sess, "train", Params, word2id)
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Reloading model parameters..')
            seq2seq_model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('Created new model parameters..')
            sess.run(tf.global_variables_initializer())

    with tf.name_scope("Eval"):
        with tf.variable_scope("Model", reuse=True):
            Params.beam_search = True
            eval_model = Seq2SeqModel(sess, "decode", Params, word2id)

    current_step = 0
    for epoch in range(Params.epochs):
        time_s = time.time()
        print("\nEpoch %d/%d" % (epoch + 1, Params.epochs))
        batches = get_batches(samples, Params.batch_size)
        for next_batch in batches:
            loss, summary = seq2seq_model.train(next_batch)
            current_step += 1
            if current_step % Params.save_steps == 0:
                perplexity = math.exp(float(loss)) if loss < 300 else float('inf')
                print("step=%d, loss=%.2f, perplexity=%.2f, save=%s" %
                      (current_step, loss, perplexity, checkpoint_path))
                seq2seq_model.saver.save(seq2seq_model.sess, checkpoint_path)

        print("epoch=%d, save." % epoch)
        seq2seq_model.saver.save(seq2seq_model.sess, checkpoint_path)

        time_e = time.time()
        print("Epoch %d training done, time=%.2f minutes" % (epoch + 1, (time_e - time_s) / 60))


if __name__ == "__main__":
    train(config.model_path, config.vocab_path,
          question_answer_path=config.question_answer_path,
          context_response_path=config.context_response_path,
          dialog_mode=config.dialog_mode)
