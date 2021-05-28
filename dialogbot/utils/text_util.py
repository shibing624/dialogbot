# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import re

ch_pattern = re.compile(r"[\u4e00-\u9fa5]+")
remove_pattern = re.compile(r"好的")


def ch_count(text):
    """
    Count chinese number.
    :param text:
    :return:
    """
    text = remove_pattern.sub("", text)
    r = ch_pattern.findall(text)
    cnt = len("".join(r))
    return cnt
