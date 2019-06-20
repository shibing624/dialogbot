# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
from chatbot.searchdialog.bot import QABot
from chatbot import config
import os

# cur_dir = os.path.abspath(os.path.dirname(__file__))
# qa = QABot(train_file_path=os.path.join(cur_dir, "..", config.train_file_path),
#         emb_model_path=os.path.join(cur_dir, "..", config.emb_model_path),
#         train_model_path=os.path.join(cur_dir, "..", config.train_model_path))
# querys = ['如何去坚持减肥？', '怎么减肥', '减肥的好处是什么？']
# for q in querys:
#     answer = qa.answer(q)
#     print('question:', q, '\tanswer', answer)
