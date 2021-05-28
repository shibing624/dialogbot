# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import sys

sys.path.append('..')
from dialogbot.searchdialog.internet.search_engine import Engine

if __name__ == '__main__':
    engine = Engine()
    print(engine.search("北京天气怎么样？"))
    print(engine.search("上海天气呢？"))
    print(engine.search("武汉今天天气预报？"))
    print(engine.search("武汉明天天气预报呢？"))
    ans = engine.search("貂蝉是谁")
    print(ans)
    ans = engine.search("西施是谁")
    print(ans)
    ans = engine.search("你知道我是谁")
    print(ans)
    context = engine.contents
    print(context)
