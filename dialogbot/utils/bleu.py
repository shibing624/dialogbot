# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

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
            references = []
            standard_answer_line = standard_answer_lines[i * 11 + j].strip().split('\t')
            references.append(list(standard_answer_line[0].strip()))
            standard_score = standard_answer_line[1]
            bleu_score = sentence_bleu(references, candidate, weights=(0.35, 0.45, 0.1, 0.1),
                                       smoothing_function=SmoothingFunction().method1)
            each_score = bleu_score * float(standard_score) + each_score
        scores.append(each_score / 10)
    rf_answer.close()
    rf_standard_answer.close()
    score_final = sum(scores) / float(len(answer_lines))
    precision_score = round(score_final, 6)
    return precision_score


def bleu_score(candidate, reference):
    score = sentence_bleu(
        [list(reference)], list(candidate),
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=SmoothingFunction().method1)
    return score


def bleu_similarity(query, docs):
    scores = [(idx, bleu_score(doc, query))
              for idx, doc in enumerate(docs)]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


if __name__ == "__main__":
    c = "我爱你中国，不爱美国"
    t = "中国我爱你，美国我不爱"
    r = '我爱美国，不爱你中国'
    o = '我不爱中国，我也不爱美国'
    a = bleu_score(c, t)
    print(a)

    a = bleu_score(c, r)
    print(a)

    b = bleu_score(c, o)
    print(b)

    l = [r, o, c, t]
    d = bleu_similarity(c, l)
    print(d)

    print(bleu_similarity(c, [c]))
    print(bleu_similarity(c, [c, t]))

