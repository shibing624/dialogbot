# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
from collections import deque

from .bm25model import BM25Model
from .tfidfmodel import TfidfModel
from .vectormodel import VectorModel
from ..reader.data_helper import load_dataset
from ..util.logger import get_logger

logger = get_logger(__name__)


class SearchBot:
    def __init__(self,
                 question_answer_path=None,
                 context_response_path=None,
                 vocab_path=None,
                 search_model="bm25",
                 last_txt_len=100):
        self.last_txt = deque([], last_txt_len)

        self.word2id, _ = load_dataset(vocab_path)
        if search_model == "tfidf":
            self.qa_search_inst = TfidfModel(question_answer_path, word2id=self.word2id)
            self.cr_search_inst = TfidfModel(context_response_path, word2id=self.word2id)
        elif search_model == "bm25":
            self.qa_search_inst = BM25Model(question_answer_path, word2id=self.word2id)
            self.cr_search_inst = BM25Model(context_response_path, word2id=self.word2id)
        elif search_model == "vector":
            self.qa_search_inst = VectorModel(question_answer_path, word2id=self.word2id)
            self.cr_search_inst = VectorModel(context_response_path, word2id=self.word2id)

    def search(self, msg_tokens, mode="qa", filter_pattern=None):
        query = [w for w in msg_tokens if w in self.word2id]
        search_inst = self.qa_search_inst if mode == "qa" else self.cr_search_inst
        sim_items = search_inst.similarity(query, size=10)
        docs, answers = search_inst.get_docs(sim_items)

        # User filter pattern.
        if filter_pattern:
            new_docs, new_answers = [], []
            for doc, ans in zip(docs, answers):
                if not filter_pattern.search(ans):
                    new_docs.append(doc)
                    new_answers.append(ans)
            docs, answers = new_docs, new_answers

        logger.debug("init_query=%s, filter_query=%s" % ("".join(msg_tokens), "".join(query)))
        response, score = answers[0], sim_items[0][1]
        logger.debug("%s_search_sim_doc=%s, score=%.4f" % (mode, "".join(docs[0]), score))
        if score <= 1.0:
            response, score = "亲爱哒，还有什么小妹可以帮您呢~", 2.0
        return response, score
