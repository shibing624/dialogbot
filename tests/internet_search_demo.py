# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import sys
sys.path.append("..")
from dialogbot.searchdialog.internet.search_engine import Engine
from dialogbot.utils.logger import logger

if __name__ == '__main__':
    engine = Engine()
    logger.debug(engine.search("北京今天天气如何？"))
    logger.debug(engine.search("上海呢？"))
    logger.debug(engine.search("武汉呢？"))
    logger.debug(engine.search("武汉明天呢？"))
    ans = engine.search("貂蝉是谁")
    logger.debug(ans)
    ans = engine.search("西施是谁")
    logger.debug(ans)
    ans = engine.search("你知道我是谁")
    logger.debug(ans)
    context = engine.contents
    print(context)
