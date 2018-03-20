# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
from domain.answer import QA
import domain.config as config

qa = QA(train_file_path=config.train_file_path, emb_model_path=config.emb_model_path,
        train_model_path=config.train_model_path)
q = '如何去坚持减肥？'
answer = qa.answer(q)
print('question:', q, '\tanswer', answer)
