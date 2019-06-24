# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('../..')
from flask import Flask, render_template, request

from dialogbot.bot import Bot
from dialogbot.utils.logger import start_heartbeat

app = Flask(__name__)

bot = Bot()

start_heartbeat(logger=bot.logger)


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
