#!/usr/bin/env python3
# coding: utf-8
# File: chatbot_graph.py
# Author: XuMing（xuming624@qq.com),lhy<lhy_in_blcu@126.com>

from kg.question_classifier import QuestionClassifier
from kg.question_parser import QuestionPaser
from kg.answer_searcher import AnswerSearcher


class ChatBotGraph:
    """问答类"""

    def __init__(self):
        self.classifier = QuestionClassifier()
        self.parser = QuestionPaser()
        self.searcher = AnswerSearcher()

    def chat_main(self, sent):
        answer = '您好，我是医药智能助理，希望可以帮到您。祝您身体棒棒！'
        res_classify = self.classifier.classify(sent)
        if not res_classify:
            return answer
        res_sql = self.parser.parser_main(res_classify)
        final_answers = self.searcher.search_main(res_sql)
        if not final_answers:
            return answer
        else:
            return '\n'.join(final_answers)


if __name__ == '__main__':
    handler = ChatBotGraph()
    print("eg：百日咳什么症状？")
    while 1:
        question = input('query:')
        answer = handler.chat_main(question)
        print('answer:', answer)
