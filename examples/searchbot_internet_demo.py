# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
from dialogbot.searchdialog.internet.search_engine import Engine

if __name__ == '__main__':
    engine = Engine()
    print(engine.search("北京今天天气如何？"))
    print(engine.search("上海呢？"))
    print(engine.search("武汉呢？"))
    print(engine.search("武汉明天呢？"))
    ans = engine.search("貂蝉是谁")
    print(ans)
    ans = engine.search("西施是谁")
    print(ans)
    ans = engine.search("你知道我是谁")
    print(ans)
    context = engine.contents
    print(context)
