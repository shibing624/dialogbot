# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import sys
from codecs import open

from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu


def bleu(answer_file, standard_answer_file):
    rf_answer = open(answer_file, 'r', "utf-8")
    rf_standard_answer = open(standard_answer_file, 'r', "utf-8")
    answer_lines = rf_answer.readlines()
    standard_answer_lines = rf_standard_answer.readlines()
    # compute score
    scores = []
    for i in range(len(answer_lines)):
        candidate = list(answer_lines[i].strip())
        each_score = 0
        for j in range(10):
            reference = []
            standard_answer_line = standard_answer_lines[i * 11 + j].strip().split('\t')
            reference.append(list(standard_answer_line[0].strip()))
            standard_score = standard_answer_line[1]
            bleu_score = sentence_bleu(reference, candidate, weights=(0.35, 0.45, 0.1, 0.1),
                                       smoothing_function=SmoothingFunction().method1)
            each_score = bleu_score * float(standard_score) + each_score
        scores.append(each_score / 10)
    rf_answer.close()
    rf_standard_answer.close()
    score_final = sum(scores) / float(len(answer_lines))
    precision_score = round(score_final, 6)
    return precision_score


if __name__ == "__main__":
    candidate_file = sys.argv[1]
    reference_file = sys.argv[2]
    s = bleu(candidate_file, reference_file)
    print(s)
