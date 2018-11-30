# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
from domain.answer import QA
import domain.config as config

qa = QA(train_file_path=config.train_file_path, emb_model_path=config.emb_model_path,
        train_model_path=config.train_model_path)
querys = ['如何去坚持减肥？', '怎么减肥', '减肥的好处是什么？']
for q in querys:
    answer = qa.answer(q)
    print('question:', q, '\tanswer', answer)
