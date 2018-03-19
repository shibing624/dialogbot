# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
import logging


def get_logger(name, log_file=None):
    """
    logger
    :param name: 模块名称
    :param log_file: 日志文件，如无则输出到标准输出
    :return:
    """
    formatter = logging.Formatter('[%(levelname)1.1s %(asctime)s %(module)s:%(lineno)d] %(message)s',
                                  datefmt='%m%d%Y %I:%M:%sS')
    if not log_file:
        handle = logging.StreamHandler()
    else:
        handle = logging.FileHandler(log_file)
    handle.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.addHandler(handle)
    logger.setLevel(logging.DEBUG)
    return logger


def segment(sentence):
    """
    切词
    :param sentence:
    :return: list
    """
    import jieba
    jieba.default_logger.setLevel(logging.ERROR)
    return jieba.lcut(sentence)


def segment_pos(sentence):
    """
    切词
    :param sentence:
    :return: list
    """
    import jieba
    from jieba import posseg
    jieba.default_logger.setLevel(logging.ERROR)
    return posseg.lcut(sentence)
