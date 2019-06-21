# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description:
"""

from codecs import open

from dialogbot.seq2seqdialog.infer import predict
from dialogbot.util.bleu import bleu


def _process_gen_msg(text):
    text = text.replace("URL", "http")
    return text


def single_dialog(model, context):
    answer = predict(model, context, beam_size=1)
    answer = _process_gen_msg(answer)
    return answer


def multi_dialog(model, questions):
    results = []
    q_len = len(questions)
    for idx in range(q_len):
        context = questions[-1]
        answer = predict(model, context, beam_size=1)
        answer = _process_gen_msg(answer)
        results.append(answer)
    return results


def evaluate(model, dialog_mode, result_path, question_answer_path=None, context_response_path=None):
    if dialog_mode == "single":
        with open(question_answer_path, "r", "utf-8") as rfd, \
                open(result_path, "w", "utf-8") as wfd:
            for line in rfd:
                line = line.strip("\r\n")
                answer = single_dialog(model, line)
                wfd.write("%s\n" % answer)
        return bleu(result_path, question_answer_path)
    else:
        with open(context_response_path, "r", "utf-8") as rfd, \
                open(result_path, "w", "utf-8") as wfd:
            questions = []
            for line in rfd:
                line = line.strip("\r\n")
                if line != "":
                    questions.append(line)
                else:
                    answers = multi_dialog(model, questions)
                    for answer in answers:
                        wfd.write("%s\n" % answer)
                    questions = []
            answers = multi_dialog(model, questions)
            for answer in answers:
                wfd.write("%s\n" % answer)
        return bleu(result_path, context_response_path)


if __name__ == '__main__':
    from dialogbot.seq2seqdialog.infer import get_infer_model
    from dialogbot import config
    seq2seq_inst = get_infer_model(dialog_mode=config.dialog_mode)
    user_msgs = '你好 苹果 怎么 卖 ？'
    response = predict(seq2seq_inst, user_msgs, 1)
    print('response:', response)
    bleu_score = evaluate(seq2seq_inst, config.dialog_mode, config.predict_result_path,
                          config.question_answer_path, config.context_response_path)
