# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import sys

from flask import Flask, render_template, request

sys.path.append('..')
sys.path.append('../..')
from dialogbot.bot import Bot
from dialogbot.utils.logger import start_heartbeat, logger

app = Flask(__name__)

bot = Bot()

# logger each 1 hour
start_heartbeat(60 * 10 * 6, logger)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    user_text = request.args.get('msg')
    return str(bot.answer(user_text))


if __name__ == "__main__":
    from dialogbot.config import Web

    app.run(Web.host, Web.port)
