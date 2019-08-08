# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: part-of-speech tagging

POS_WEIGHT = {
    "Ag": 1,  # 形语素
    "a": 0.5,  # 形容词
    "ad": 0.5,  # 副形词
    "an": 1,  # 名形词
    "b": 1,  # 区别词
    "c": 0.2,  # 连词
    "dg": 0.5,  # 副语素
    "d": 0.5,  # 副词
    "e": 0.5,  # 叹词
    "f": 0.5,  # 方位词
    "g": 0.5,  # 语素
    "h": 0.5,  # 前接成分
    "i": 0.5,  # 成语
    "j": 0.5,  # 简称略语
    "k": 0.5,  # 后接成分
    "l": 0.5,  # 习用语
    "m": 0.5,  # 数词
    "Ng": 1,  # 名语素
    "n": 1,  # 名词
    "nr": 1,  # 人名
    "ns": 1,  # 地名
    "nt": 1,  # 机构团体
    "nz": 1,  # 其他专名
    "o": 0.5,  # 拟声词
    "p": 0.3,  # 介词
    "q": 0.5,  # 量词
    "r": 0.2,  # 代词
    "s": 1,  # 处所词
    "tg": 0.5,  # 时语素
    "t": 0.5,  # 时间词
    "u": 0.5,  # 助词
    "vg": 0.5,  # 动语素
    "v": 1,  # 动词
    "vd": 1,  # 副动词
    "vn": 1,  # 名动词
    "w": 0.01,  # 标点符号
    "x": 0.5,  # 非语素字
    "y": 0.5,  # 语气词
    "z": 0.5,  # 状态词
    "un": 0.3  # 未知词
}

# n 名词
#     nr 人名
#         nr1 汉语姓氏
#         nr2 汉语名字
#         nrj 日语人名
#         nrf 音译人名
#     ns 地名
#     　nsf 音译地名
#     nt 机构团体名
#     nz 其它专名
#     nl 名词性惯用语
#     ng 名词性语素
#
# t 时间词
# 　　tg 时间词性语素
#
# s 处所词
#
# f 方位词
#
# v 动词
#     vd 副动词
#     vn 名动词
#     vshi 动词“是”
#     vyou 动词“有”
#     vf 趋向动词
#     vx 形式动词
#     vi 不及物动词（内动词）
#     vl 动词性惯用语
#     vg 动词性语素
# a 形容词
#     ad 副形词
#     an 名形词
#     ag 形容词性语素
#     al 形容词性惯用语
# b 区别词
#     bl 区别词性惯用语
# z 状态词
# r 代词
#     rr 人称代词
#     rz 指示代词
#         rzt 时间指示代词
#         rzs 处所指示代词
#         rzv 谓词性指示代词
#     ry 疑问代词
#         ryt 时间疑问代词
#         rys 处所疑问代词
#         ryv 谓词性疑问代词
#     rg 代词性语素
# m 数词
#     mq 数量词
# q 量词
#     qv 动量词
#     qt 时量词
# d 副词
# p 介词
#     pba 介词“把”
#     pbei 介词“被”
# c 连词
#     cc 并列连词
# u 助词
#     uzhe 着
#     ule 了 喽
#     uguo 过
#     ude1 的 底
#     ude2 地
#     ude3 得
#     usuo 所
#     udeng 等 等等 云云
#     uyy 一样 一般 似的 般
#     udh 的话
#     uls 来讲 来说 而言 说来
#
#     uzhi 之
#     ulian 连 （“连小学生都会”）
#
# e 叹词
# y 语气词(delete yg)
# o 拟声词
# h 前缀
# k 后缀
# x 字符串
#     xx 非语素字
#     xu 网址URL
# w 标点符号