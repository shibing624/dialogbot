# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

from dialogbot.kg.answer_searcher import AnswerSearcher
from dialogbot.kg.question_classifier import QuestionClassifier
from dialogbot.kg.question_parser import QuestionPaser


class KGBot:
    """基于医疗知识图谱的问答机器人"""

    def __init__(self):
        self.classifier = QuestionClassifier()
        self.parser = QuestionPaser()
        self.searcher = AnswerSearcher()

    def answer(self, sent):
        answer = '您好，我是医药智能助理，希望可以帮到您。祝您身体棒棒！'
        res_classify = self.classifier.classify(sent)
        if not res_classify:
            return answer
        res_sql = self.parser.parser(res_classify)
        final_answers = self.searcher.search(res_sql)
        if final_answers:
            return '\n'.join(final_answers)
        return answer
