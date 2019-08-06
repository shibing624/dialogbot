# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description:
"""
from codecs import open

import numpy as np
import tensorflow as tf

from dialogbot import config
from dialogbot.reader.data_helper import load_dataset, sentence2enco
from dialogbot.seq2seqdialog.seq2seq import Seq2SeqModel
from dialogbot.utils.bleu import bleu
from dialogbot.utils.logger import logger

Params = config.Params
Params.beam_search = True


class Model:
    def __init__(self, vocab_path, model_dir_path):
        self.word2id, self.id2word = load_dataset(vocab_path)
        self.model = self.load_model(model_dir_path=model_dir_path, word2id=self.word2id)

    @staticmethod
    def load_model(model_dir_path, word2id):
        tf.reset_default_graph()
        sess = tf.Session()
        with tf.variable_scope("Model"):
            model = Seq2SeqModel(sess, "decode", Params, word2id)
        logger.info("Load seq2seq model from %s" % model_dir_path)
        ckpt = tf.train.get_checkpoint_state(model_dir_path)
        try:
            model.saver.restore(model.sess, ckpt.model_checkpoint_path)
        except AttributeError as e:
            logger.error('load seq2seq model error, check model path:%s, train seq2seq to generate it.' %
                         model_dir_path)
            model = None
        return model

    def _predict_ids_to_seq(self, predict_ids, beam_size=5):
        predicts = []
        for single_predict in predict_ids:
            for i in range(beam_size):
                predict_list = np.ndarray.tolist(single_predict[:, :, i])
                predict_seq = [self.id2word[idx] for idx in predict_list[0]
                               if idx in self.id2word if idx >= 4]
                predicts.append("".join(predict_seq))
        return predicts

    def predicts(self, context, beam_size=5):
        result = []
        batch = sentence2enco(context, self.word2id)
        if self.model:
            predicted_ids = self.model.infer(batch)
            result = self._predict_ids_to_seq(predicted_ids, beam_size)
        return result

    def predict(self, context):
        answer = ''
        answers = self.predicts(context)
        if answers:
            answer = answers[0]
        return answer

    def predict_sent_emb(self, context):
        batch = sentence2enco(context, self.word2id)
        sent_emb = self.model.infer_sent_emb(batch)[0]
        return sent_emb

    @staticmethod
    def _process_gen_msg(text):
        text = text.replace("URL", "http")
        return text

    def single_dialog(self, context):
        answer = self.predict(context)
        answer = self._process_gen_msg(answer)
        return answer

    def multi_dialog(self, questions):
        results = []
        q_len = len(questions)
        for idx in range(q_len):
            context = questions[-1]
            answer = self.single_dialog(context)
            results.append(answer)
        return results

    def evaluate(self, dialog_mode, evaluate_result_path, gold_path):
        if dialog_mode == "single":
            with open(gold_path, "r", "utf-8") as rfd, \
                    open(evaluate_result_path, "w", "utf-8") as wfd:
                for line in rfd:
                    line = line.strip("\r\n")
                    answer = self.single_dialog(line)
                    wfd.write("%s\n" % answer)
            return bleu(evaluate_result_path, gold_path)
        else:
            with open(gold_path, "r", "utf-8") as rfd, \
                    open(evaluate_result_path, "w", "utf-8") as wfd:
                questions = []
                for line in rfd:
                    line = line.strip("\r\n")
                    if line != "":
                        questions.append(line)
                    else:
                        answers = self.multi_dialog(questions)
                        for answer in answers:
                            wfd.write("%s\n" % answer)
                        questions = []
                answers = self.multi_dialog(questions)
                for answer in answers:
                    wfd.write("%s\n" % answer)
            return bleu(evaluate_result_path, gold_path)
