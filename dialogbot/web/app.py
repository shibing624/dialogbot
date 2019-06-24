# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

from flask import Flask, render_template, request

from dialogbot import Bot
from dialogbot.utils.logger import start_heartbeat

app = Flask(__name__)

bot = Bot()

start_heartbeat(logger=bot.logger)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return str(bot.answer(userText))


if __name__ == "__main__":
    from dialogbot.config import Web

    app.run(Web.host, Web.port)
